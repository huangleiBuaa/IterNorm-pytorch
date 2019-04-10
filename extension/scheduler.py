import argparse

from torch.optim.lr_scheduler import *

from .utils import str2list
from .logger import get_logger

_methods = ['fix', 'step', 'steps', 'ploy', 'auto', 'exp', 'user', 'cos', '1cycle']


def add_arguments(parser: argparse.ArgumentParser):
    # train learning rate
    group = parser.add_argument_group('Learning rate scheduler Option:')
    group.add_argument('--lr-method', default='step', choices=_methods, metavar='METHOD',
                       help='The learning rate scheduler: {' + ', '.join(_methods) + '}')
    group.add_argument('--lr', default=0.1, type=float, metavar='LR', help='The initial learning rate (default: 0.1)')
    group.add_argument('--lr-step', default=30, type=int,
                       help='Every some epochs, the learning rate is multiplied by a factor (default: 30)')
    group.add_argument('--lr-gamma', default=0.1, type=float, help='The learning rate decay factor. (default: 0.1)')
    group.add_argument('--lr-steps', default=[], type=str2list, help='the step values for learning rate policy "steps"')
    return group


def setting(optimizer, args, lr_func=None, **kwargs):
    lr_method = args.lr_method
    if lr_method == 'fix':
        scheduler = StepLR(optimizer, args.epochs, args.lr_gamma)
    elif lr_method == 'step':
        scheduler = StepLR(optimizer, args.lr_step, args.lr_gamma)
    elif lr_method == 'steps':
        scheduler = MultiStepLR(optimizer, args.lr_steps, args.lr_gamma)
    elif lr_method == 'ploy':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lambda _epoch: (1. - _epoch / args.epochs) ** args.lr_gamma)
    elif lr_method == 'auto':
        scheduler = ReduceLROnPlateau(optimizer, factor=args.lr_gamma, patience=args.lr_step, verbose=True)
    elif lr_method == 'exp':
        scheduler = ExponentialLR(optimizer, args.lr_gamma)
    elif lr_method == 'user':
        scheduler = LambdaLR(optimizer, lr_func)
    elif lr_method == 'cos':
        scheduler = CosineAnnealingLR(optimizer, args.lr_step, args.lr_gamma)
    elif lr_method == '1cycle':
        gamma = (args.lr_gamma - args.lr) / args.lr_step

        def adjust(epoch):
            if epoch < args.lr_step * 2:
                return (args.lr_gamma - gamma * abs(epoch - args.lr_step)) / args.lr
            else:
                return (args.epochs - epoch) / (args.epochs - args.lr_step * 2)

        scheduler = LambdaLR(optimizer, adjust)
    else:
        raise NotImplementedError('Learning rate scheduler {} is not supported!'.format(lr_method))
    LOG = get_logger()
    LOG('==> Scheduler: {}'.format(scheduler))
    return scheduler
