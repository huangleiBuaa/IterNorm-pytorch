#!/usr/bin/env python3
import time
import os
import sys
import shutil
import argparse

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

sys.path.append('..')
import models
import extension as ext


class ClassificationSmall:
    def __init__(self):
        self.cfg = self.add_arguments()
        self.model_name = self.cfg.arch + ext.normailzation.setting(self.cfg) + '_' + self.cfg.dataset

        self.result_path = os.path.join(self.cfg.output, self.model_name)
        os.makedirs(self.result_path, exist_ok=True)
        self.logger = ext.logger.setting('log.txt', self.result_path, self.cfg.test, bool(self.cfg.resume))
        ext.trainer.setting(self.cfg)
        self.model = models.__dict__[self.cfg.arch](**self.cfg.arch_cfg)
        self.logger('==> model [{}]: {}'.format(self.model_name, self.model))
        self.optimizer = ext.optimizer.setting(self.model, self.cfg)
        self.scheduler = ext.scheduler.setting(self.optimizer, self.cfg)

        self.saver = ext.checkpoint.Checkpoint(self.model, self.cfg, self.optimizer, self.scheduler, self.result_path,
                                               not self.cfg.test)
        self.saver.load(self.cfg.load)

        # dataset loader
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        val_transform = [transforms.ToTensor(), normalize, ]
        if self.cfg.augmentation:
            train_transform = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        else:
            train_transform = []
        train_transform.extend([transforms.ToTensor(), normalize, ])
        self.train_loader = ext.dataset.get_dataset_loader(self.cfg, train_transform, train=True)
        self.val_loader = ext.dataset.get_dataset_loader(self.cfg, val_transform, train=False)

        self.device = torch.device('cuda')
        self.num_gpu = torch.cuda.device_count()
        self.logger('==> use {:d} GPUs'.format(self.num_gpu))
        if self.num_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()

        self.best_acc = 0
        if self.cfg.resume:
            saved = self.saver.resume(self.cfg.resume)
            self.cfg.start_epoch = saved['epoch']
            self.best_acc = saved['best_acc']
        self.criterion = nn.CrossEntropyLoss()

        self.vis = ext.visualization.setting(self.cfg, self.model_name,
                                             {'train loss': 'loss', 'test loss': 'loss', 'train accuracy': 'accuracy',
                                              'test accuracy': 'accuracy'})
        return

    def add_arguments(self):
        model_names = sorted(
            name for name in models.__dict__ if not name.startswith("__") and callable(models.__dict__[name]))
        parser = argparse.ArgumentParser('Small Scale Image Classification')
        parser.add_argument('-a', '--arch', metavar='ARCH', default='simple', choices=model_names,
                            help='model architecture: ' + ' | '.join(model_names) + '\t(Default: simple)')
        parser.add_argument('--arch-cfg', metavar='DICT', default={}, type=ext.utils.str2dict,
                            help='The extra model architecture configuration.')
        parser.add_argument('-A', '--augmentation', type=ext.utils.str2bool, default=True, metavar='BOOL',
                            help='Use data augmentation? (default: True)')
        ext.trainer.add_arguments(parser)
        parser.set_defaults(epochs=200)
        ext.dataset.add_arguments(parser)
        parser.set_defaults(dataset='cifar10', workers=4)
        ext.scheduler.add_arguments(parser)
        parser.set_defaults(lr_method='steps', lr_steps=[100, 150], lr=0.1)
        ext.optimizer.add_arguments(parser)
        parser.set_defaults(optimizer='sgd', weight_decay=1e-4)
        ext.logger.add_arguments(parser)
        ext.checkpoint.add_arguments(parser)
        ext.visualization.add_arguments(parser)
        ext.normailzation.add_arguments(parser)
        args = parser.parse_args()
        if args.resume:
            args = parser.parse_args(namespace=ext.checkpoint.Checkpoint.load_config(args.resume))
        return args

    def train(self):
        if self.cfg.test:
            self.validate()
            return
        # train model
        for epoch in range(self.cfg.start_epoch + 1, self.cfg.epochs):
            if self.cfg.lr_method != 'auto':
                self.scheduler.step()
            self.train_epoch(epoch)
            accuracy, val_loss = self.validate(epoch)
            self.saver.save_checkpoint(epoch=epoch, best_acc=self.best_acc)
            if self.cfg.lr_method == 'auto':
                self.scheduler.step(val_loss)
        # finish train
        now_date = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
        self.logger('==> end time: {}'.format(now_date))
        new_log_filename = '{}_{}_{:5.2f}%.txt'.format(self.model_name, now_date, self.best_acc)
        self.logger('\n==> Network training completed. Copy log file to {}'.format(new_log_filename))
        shutil.copy(self.logger.filename, os.path.join(self.result_path, new_log_filename))
        return

    def train_epoch(self, epoch):
        self.logger('\nEpoch: {}, lr: {:.2g}, weight decay: {:.2g} on model {}'.format(epoch,
            self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[0]['weight_decay'], self.model_name))
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        progress_bar = ext.ProgressBar(len(self.train_loader))
        for i, (inputs, targets) in enumerate(self.train_loader, 1):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # compute output
            outputs = self.model(inputs)
            losses = self.criterion(outputs, targets)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            # measure accuracy and record loss
            train_loss += losses.item() * targets.size(0)
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(targets.view_as(pred)).sum().item()
            total += targets.size(0)
            if i % 10 == 0 or i == len(self.train_loader):
                progress_bar.step('Loss: {:.5g} | Accuracy: {:.2f}%'.format(train_loss / total, 100. * correct / total),
                    10)
        train_loss /= total
        accuracy = 100. * correct / total
        self.vis.add_value('train loss', train_loss)
        self.vis.add_value('train accuracy', accuracy)
        self.logger(
            'Train on epoch {}: average loss={:.5g}, accuracy={:.2f}% ({}/{}), time: {}'.format(epoch, train_loss,
                accuracy, correct, total, progress_bar.time_used()))
        return

    def validate(self, epoch=-1):
        test_loss = 0
        correct = 0
        total = 0
        progress_bar = ext.ProgressBar(len(self.val_loader))
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, targets).item() * targets.size(0)
                prediction = outputs.max(1, keepdim=True)[1]
                correct += prediction.eq(targets.view_as(prediction)).sum().item()
                total += targets.size(0)
                progress_bar.step('Loss: {:.5g} | Accuracy: {:.2f}%'.format(test_loss / total, 100. * correct / total))
        test_loss /= total
        accuracy = correct * 100. / total
        self.vis.add_value('test loss', test_loss)
        self.vis.add_value('test accuracy', accuracy)
        self.logger('Test on epoch {}: average loss={:.5g}, accuracy={:.2f}% ({}/{}), time: {}'.format(epoch, test_loss,
            accuracy, correct, total, progress_bar.time_used()))
        if not self.cfg.test and accuracy > self.best_acc:
            self.best_acc = accuracy
            self.saver.save_model('best.pth')
            self.logger('==> best accuracy: {:.2f}%'.format(self.best_acc))
        return accuracy, test_loss


if __name__ == '__main__':
    Cs = ClassificationSmall()
    Cs.train()
