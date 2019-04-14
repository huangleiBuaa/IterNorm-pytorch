import argparse
import os
import shutil
import time
import warnings

import extension as ext
import torchvision.transforms as transforms

has_DALI = True
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    warnings.warn("Please install DALI from https://www.github.com/NVIDIA/DALI to enable DALI data loader")
    has_DALI = False
    Pipeline = object
    DALIClassificationIterator = object


def add_arguments(parser: argparse.ArgumentParser):
    group = ext.dataset.add_arguments(parser)
    group.add_argument('--dali', default=has_DALI, type=ext.utils.str2bool, metavar='BOOL',
                       help="Use NVIDIA DALI to accelerate data load.")
    group.set_defaults(dataset='ImageNet', batch_size=[256, 200])
    return group


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        # let user decide which pipeline works him bets for RN version he runs
        scale = [0.08, 1.0]
        ratio = [3. / 4., 4. / 3.]
        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoderRandomCrop(device=dali_device, output_type=types.RGB,
                                                    random_aspect_ratio=ratio, random_area=scale, num_attempts=100)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from
            # full-sized ImageNet without additional reallocations
            self.decode = ops.nvJPEGDecoderRandomCrop(device="mixed", output_type=types.RGB,
                                                      device_memory_padding=211025920, host_memory_padding=140544512,
                                                      random_aspect_ratio=ratio, random_area=scale, num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu", output_dtype=types.FLOAT, output_layout=types.NCHW,
                                            crop=(crop, crop), image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, local_rank=0, world_size=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=False)
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu", output_dtype=types.FLOAT, output_layout=types.NCHW,
                                            crop=(crop, crop), image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


class ImageNetDataLoader(DALIClassificationIterator):
    def __next__(self):
        data = super(ImageNetDataLoader, self).__next__()
        if isinstance(data, list):
            inputs = data[0]["data"]
            targets = data[0]["label"].squeeze().long()
            return inputs, targets
        else:
            return data

    def __len__(self):
        return int(self._size / self.batch_size)


def dail_loader(args, test=False, local_rank=0, world_size=1):
    logger = ext.get_logger()
    args.dataset_root = os.path.expanduser(args.dataset_root)
    root = os.path.join(args.dataset_root, args.dataset)
    assert os.path.exists(root), 'Please assign the correct dataset root path with --dataset-root <PATH>'
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')
    crop_size = 224
    if len(args.batch_size) == 0:
        args.batch_size = [256, 200]
    elif len(args.batch_size) == 1:
        args.batch_size.append(args.batch_size[0])
    if test:
        train_loader = None
    else:
        logger('==> Load ImageNet train dataset:')
        pipe = HybridTrainPipe(batch_size=args.batch_size[0], num_threads=args.workers, device_id=local_rank,
                               data_dir=train_dir, crop=crop_size)
        pipe.build()
        train_loader = ImageNetDataLoader([pipe], size=int(pipe.epoch_size("Reader") / world_size), auto_reset=True,
                                          stop_at_epoch=True)

    logger('==> Load ImageNet val dataset:')
    pipe = HybridValPipe(batch_size=args.batch_size[1], num_threads=args.workers, device_id=local_rank,
                         data_dir=val_dir, crop=crop_size, size=256)
    pipe.build()
    val_loader = ImageNetDataLoader([pipe], size=int(pipe.epoch_size("Reader") / world_size), auto_reset=True)
    return train_loader, val_loader


def set_dataset(cfg, test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                       normalize, ]
    val_transform = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize, ]
    if not test:
        train_loader = ext.dataset.get_dataset_loader(cfg, train_transform, None, True)
    else:
        train_loader = None
    val_loader = ext.dataset.get_dataset_loader(cfg, val_transform, None, False)
    return train_loader, val_loader


def setting(cfg, test=False):
    if has_DALI and cfg.dali:
        return dail_loader(cfg, test=test)
    return set_dataset(cfg, test=test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test Data Loader')
    add_arguments(parser)
    args = parser.parse_args()
    print('==> args: ', args)
    train_loader_, val_loader_ = dail_loader(args)
    print('len of train_loader', len(train_loader_))
    total = 0
    start_time = time.time()
    for i, (inputs, targets) in enumerate(train_loader_, 1):
        # inputs = data[0]["data"]
        # targets = data[0]["label"].squeeze().cuda().long()
        total += targets.size(0)
        print('Load train data [{}/{}]: {}, {}'.format(i, len(train_loader_), inputs.size(), targets.size()), end='\r')
    print('\nTrain Read {} images, use {:.2f}s'.format(total, time.time() - start_time))

    print('len of val_loader', len(val_loader_))
    total = 0
    start_time = time.time()
    for i, (inputs, targets) in enumerate(val_loader_, 1):
        # inputs = data[0]["data"]
        # targets = data[0]["label"].squeeze().cuda().long()
        total += targets.size(0)
        print('Load val data [{}/{}]: {}, {}'.format(i, len(val_loader_), inputs.size(), targets.size()), end='\r')
    print('\nTrain Read {} images, use {:.2f}s'.format(total, time.time() - start_time))
