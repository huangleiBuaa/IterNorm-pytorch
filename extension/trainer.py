import argparse
import torch
import os
import time
import random
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler

from .logger import get_logger


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Train Option')
    group.add_argument('-n', '--epochs', default=90, type=int, metavar='N', help='The total number of training epochs.')
    group.add_argument('--start-epoch', default=-1, type=int, metavar='N',
                       help='manual epoch number (useful on restarts)')
    group.add_argument('-o', '--output', default='./results', metavar='PATH',
                       help='The root path to store results (default ./results)')
    group.add_argument('-t', '--test', action='store_true', help='Only test model on validation set?')
    group.add_argument('--seed', default=-1, type=int, help='manual seed')
    # group.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    return


def setting(cfg: argparse.Namespace):
    cudnn.benchmark = True
    logger = get_logger()
    logger('==> args: {}'.format(cfg))
    logger('==> the results path: {}'.format(cfg.output))
    if not hasattr(cfg, 'seed') or cfg.seed < 0:
        cfg.seed = int(time.time())
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    logger('==> seed: {}'.format(cfg.seed))
    logger('==> PyTorch version: {}, cudnn version: {}'.format(torch.__version__, cudnn.version()))
    git_version = os.popen('git log --pretty=oneline | head -n 1').readline()[:-1]
    logger('==> git version: {}'.format(git_version))
    return

#
# class Trainer(object):
#
#     def __init__(self, num_model=1):
#         # config
#         self.parser = argparse.ArgumentParser(description='Trainer')
#         self.add_arguments()
#         self.args = self.parser.parse_args()
#
#         self.num_model = num_model
#
#         self.model_name = ''
#         self.model = torch.nn.Module() if self.num_model == 1 else []
#         self.quantization_cfg = quantization.setting(self.args)
#
#         self.result_path = ''
#         self.logger = None
#         self.vis = visualization.Visualization(False)
#
#         self.device = None
#         self.num_gpu = 0
#
#         self.train_transform = []
#         self.val_transform = []
#         self.train_loader = None
#         self.val_loader = None
#
#         self.lr_schedulers = []
#         self.optimizer = None
#         self.start_epoch = self.args.start_epoch if hasattr(self.args, 'start_epoch') else -1
#         self.criterion = None
#
#         self.start_time = time.time()
#         self.global_steps = 0
#         # self.set_model()
#         # self.set_optimizer()
#         # self.set_dataset()
#         # self.set_device()
#         # self.resume()
#         # self.set_lr_scheduler()
#         return
#
#     def train(self):
#         self.logger('\n++++++++++  train start (time: {})   ++++++++++'.format(
#             time.strftime("%y-%m-%d %H:%M:%S", time.localtime(time.time()))))
#         if self.args.evaluate:
#             self.validate()
#             return
#         self.start_time = time.time()
#         self.global_steps = 0
#         for epoch in range(self.start_epoch + 1, self.args.epochs):
#             if self.args.lr_method != 'auto':
#                 for i in range(len(self.lr_schedulers)):
#                     self.lr_schedulers[i].step(epoch=epoch)
#             self.train_epoch(epoch)
#             value = self.validate(epoch)
#             if self.args.lr_method == 'auto':
#                 for i in range(len(self.lr_schedulers)):
#                     self.lr_schedulers[i].step(value, epoch=epoch)
#             # self.save_checkpoint(epoch)
#         self.save()
#         now_date = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
#         new_log_filename = '{}_{}.txt'.format(self.model_name, now_date)
#         self.logger('\n==> Network training completed. Copy log file to {}'.format(new_log_filename))
#         self.logger.save(new_log_filename)
#         self.logger('\n++++++++++ train finished (time: {}) ++++++++++'.format(
#             time.strftime("%y-%m-%d %H:%M:%S", time.localtime(time.time()))))
#
#     def train_epoch(self, epoch):
#         return NotImplementedError
#
#     def validate(self, epoch=-1):
#         return NotImplementedError
#
#     def set_device(self):
#         self.device = torch.device('cuda')
#         self.num_gpu = torch.cuda.device_count()
#         if self.num_model == 1:
#             if self.num_gpu > 1:
#                 self.model = torch.nn.DataParallel(self.model)
#             self.model.cuda()
#         else:
#             for i in range(self.num_model):
#                 if self.num_gpu > 1:
#                     self.model[i] = torch.nn.DataParallel(self.model[i])
#                 self.model[i].cuda()
#         self.logger('==> use {:d} GPUs with cudnn {}'.format(self.num_gpu, cudnn.version()))
#
#     def set_dataset(self):
#         if not self.args.evaluate:
#             self.train_loader = dataset.get_dataset_loader(self.args, self.train_transform, train=True)
#             self.logger('==> Train Data Transforms: {}'.format(self.train_transform))
#         self.val_loader = dataset.get_dataset_loader(self.args, self.val_transform, train=False)
#         self.logger('==> Val Data Transforms: {}'.format(self.val_transform))
#         self.logger('==> Dataset: {}, image size: {}, classes: {}'.format(
#             self.args.dataset, self.args.im_size, self.args.dataset_classes))
#         return
