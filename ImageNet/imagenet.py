#!/usr/bin/env python3
import os
import time
import argparse
import shutil
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np

sys.path.append('..')
import models
import data_loader
import extension as ext
from extension.progress_bar import format_time


class ClassificationLarge:
    def __init__(self):
        self.args = self.add_arguments()
        self.best_prec1 = 0
        self.model_name = self.args.arch + ext.normailzation.setting(self.args) + '_{}'.format(self.args.optimizer)
        if not self.args.resume:
            self.args.output = os.path.join(self.args.output, self.model_name)
        self.logger = ext.logger.setting('log.txt', self.args.output, self.args.test, bool(self.args.resume))
        ext.trainer.setting(self.args)
        self.model = models.__dict__[self.args.arch](**self.args.arch_cfg)
        self.logger('==> Model [{}]: {}'.format(self.model_name, self.model))
        self.optimizer = ext.optimizer.setting(self.model, self.args)
        self.scheduler = ext.scheduler.setting(self.optimizer, self.args)
        self.device = torch.device('cuda')
        self.num_gpus = torch.cuda.device_count()
        self.logger('==> The number of gpus: {}'.format(self.num_gpus))
        self.saver = ext.checkpoint.Checkpoint(self.model, self.args, self.optimizer, self.scheduler, self.args.output,
                                               not self.args.test)
        self.saver.load(self.args.load)
        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        self.model = self.model.cuda()
        if self.args.resume:
            saved = self.saver.resume(self.args.resume)
            self.args.start_epoch = saved['epoch']
            self.args.best_prec1 = saved['best_prec1']

        self.train_loader, self.val_loader = data_loader.setting(self.args, self.args.test)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.vis = ext.visualization.setting(self.args, self.model_name,
                                             {'train loss': 'loss', 'train top-1': 'accuracy',
                                              'train top-5': 'accuracy', 'test loss': 'loss', 'test top-1': 'accuracy',
                                              'test top-5': 'accuracy', 'epoch loss': 'epoch_loss',
                                              'loss average': 'epoch_loss'})
        return

    def add_arguments(self):
        parser = argparse.ArgumentParser('ImageNet Classification')
        ext.normailzation.add_arguments(parser)
        data_loader.add_arguments(parser)
        ext.trainer.add_arguments(parser)
        ext.optimizer.add_arguments(parser)
        ext.visualization.add_arguments(parser)
        ext.logger.add_arguments(parser)
        ext.checkpoint.add_arguments(parser)
        ext.scheduler.add_arguments(parser)
        model_names = sorted(name for name in models.__dict__ if
                             name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
        parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names,
                            help='model architecture: {' + ' | '.join(model_names) + '} (default: resnet18)')

        parser.add_argument('-ac', '--arch-cfg', metavar='DICT', default={}, type=ext.utils.str2dict,
                            help='The extra configure for model architecture')
        parser.set_defaults(dataset='ImageNet')
        parser.set_defaults(lr_method='step', lr=0.1, lr_step=30, weight_decay=1e-4)
        parser.set_defaults(epochs=90)
        parser.set_defaults(workers=10)
        args = parser.parse_args()
        if args.resume is not None:
            args = parser.parse_args(namespace=ext.checkpoint.Checkpoint.load_config(args.resume))
        return args

    def train(self):
        if self.args.test:
            self.validate()
            return
        self.logger('\n++++++++++++++++++ Begin Train ++++++++++++++++++')
        used_times = []
        for epoch in range(self.args.start_epoch + 1, self.args.epochs):
            epoch_start_time = time.time()
            # adjust_learning_rate(optimizer, epoch)
            if self.args.lr_method != 'auto':
                self.scheduler.step(epoch)
            self.logger('Model {} [{}/{}]: lr={:.3g}, weight decay={:.2g}, time: {}'.format(self.model_name, epoch,
                                                                                            self.args.epochs,
                                                                                            self.optimizer.param_groups[
                                                                                                0]['lr'],
                                                                                            self.optimizer.param_groups[
                                                                                                0]['weight_decay'],
                                                                                            time.asctime()))
            # train for one epoch
            self.train_epoch(epoch)

            # evaluate on validation set
            prec1, val_loss = self.validate(epoch)

            if self.args.lr_method == 'auto':
                self.scheduler.step(val_loss, epoch)
            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.best_prec1
            self.best_prec1 = max(prec1, self.best_prec1)
            self.saver.save_checkpoint('checkpoint.pth', epoch=epoch, best_prec1=self.best_prec1, arch=self.args.arch)
            if is_best:
                self.saver.save_model('best.pth')
            used_times.append(time.time() - epoch_start_time)
            self.logger('Epoch [{}/{}] use: {}, average: {}, expect: {}\n'.format(epoch, self.args.epochs,
                                                                                  format_time(used_times[-1]),
                                                                                  format_time(np.mean(used_times)),
                                                                                  format_time((
                                                                                                      self.args.epochs - 1 - epoch) * np.mean(
                                                                                      used_times))))

        now_date = time.strftime("%y-%m-%d_%H:%M:%S", time.localtime(time.time()))
        new_log_filename = '{}_{}_{:.2f}%.txt'.format(self.model_name, now_date, self.best_prec1)
        self.logger('\n==> Network training completed. Copy log file to {}'.format(new_log_filename))
        shutil.copy(self.logger.filename, os.path.join(self.args.output, new_log_filename))
        return

    def train_epoch(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.model.train()
        self.vis.clear('epoch_loss')
        end = time.time()
        for i, (inputs, targets) in enumerate(self.train_loader, 1):
            # measure data loading time
            # if self.args.gpu is not None:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            data_time.update(time.time() - end)

            # compute output
            output = self.model(inputs)
            loss = self.criterion(output, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            # logger
            is_log = i % self.args.print_f == 0 or i == len(self.train_loader)
            self.logger('Epoch: [{0}][{1:5d}/{2:5d}]  '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
                        'Loss {loss.val:.2f} ({loss.avg:.2f})  '
                        'Prec@1 {top1.val:5.2f} ({top1.avg:5.2f})  '
                        'Prec@5 {top5.val:5.2f} ({top5.avg:5.2f})  '
                        ''.format(epoch, i, len(self.train_loader), batch_time=batch_time, data_time=data_time,
                                  loss=losses, top1=top1, top5=top5), end='\n' if is_log else '\r', is_log=is_log)
            if is_log:
                self.vis.add_value('epoch loss', losses.val)
                self.vis.add_value('loss average', losses.ravg)

        self.vis.add_value('train loss', losses.avg)
        self.vis.add_value('train top-1', top1.avg)
        self.vis.add_value('train top-5', top5.avg)
        return

    def validate(self, epoch=-1):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (inputs, targets) in enumerate(self.val_loader, 1):
                # if self.args.gpu is not None:
                #     inputs = inputs.cuda(self.args.gpu, non_blocking=True)
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(None, non_blocking=True)

                # compute output
                output = self.model(inputs)
                loss = self.criterion(output, targets)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1[0], inputs.size(0))
                top5.update(prec5[0], inputs.size(0))

                # measure elapsed time
                torch.cuda.synchronize()
                batch_time.update(time.time() - end)
                end = time.time()

                is_log = i % self.args.print_f == 0 or i == len(self.val_loader)
                self.logger('Test: [{0:3d}/{1:3d}]  '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                            'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                            'Prec@1 {top1.val:5.2f} ({top1.avg:5.2f})  '
                            'Prec@5 {top5.val:5.2f} ({top5.avg:5.2f})  '
                            ''.format(i, len(self.val_loader), batch_time=batch_time, loss=losses, top1=top1,
                                      top5=top5), end='\n' if is_log else '\r', is_log=is_log)

        self.logger(' * Prec@1 {top1.avg:5.2f} Prec@5 {top5.avg:5.2f} on epoch {epoch}'.format(top1=top1, top5=top5,
                                                                                               epoch=epoch))

        self.vis.add_value('test loss', losses.avg)
        self.vis.add_value('test top-1', top1.avg)
        self.vis.add_value('test top-5', top5.avg)
        return top1.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.9):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.ravg = 0
        self.momentum = momentum

    def reset(self):
        self.ravg = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        if self.count == 0:
            self.ravg = self.val
        else:
            self.ravg = self.momentum * self.ravg + (1. - self.momentum) * val
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, prediction = output.topk(maxk, 1, True, True)
        prediction = prediction.t()
        correct = prediction.eq(target.view(1, -1).expand_as(prediction))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    ImageNet = ClassificationLarge()
    ImageNet.train()
