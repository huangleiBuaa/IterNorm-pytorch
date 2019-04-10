import torch
import time
import random

weight_3d_shapes = [(64, 3, 7, 7), (64, 3, 11, 11), (64, 64, 1, 1), (64, 64, 3, 3), (64, 256, 1, 1), (128, 64, 1, 1),
                    (128, 64, 3, 3), (128, 128, 3, 3), (128, 256, 1, 1), (128, 512, 1, 1), (192, 64, 5, 5),
                    (256, 64, 1, 1), (256, 128, 1, 1), (256, 128, 3, 3), (256, 256, 3, 3), (256, 384, 3, 3),
                    (256, 512, 1, 1), (256, 1024, 1, 1), (384, 192, 3, 3), (512, 128, 1, 1), (512, 256, 1, 1),
                    (512, 256, 1, 1), (512, 256, 3, 3), (512, 512, 3, 3), (512, 1024, 1, 1), (512, 2048, 1, 1),
                    (1024, 256, 1, 1), (1024, 512, 1, 1), (2048, 512, 1, 1), (2048, 1024, 1, 1)]

weight_2d_shapes = [(1000, 512), (1000, 2048), (1000, 4096), (4096, 4096), (4096, 9216)]

input_3d_shapes = [(3, 224, 244), (64, 112, 112), (64, 56, 56), (96, 55, 55), (128, 112, 112), (128, 56, 56),
                   (128, 28, 28), (256, 56, 56), (256, 28, 28), (256, 27, 27), (256, 14, 14), (256, 7, 7),
                   (384, 13, 13), (512, 28, 28), (512, 14, 14), (512, 7, 7), (1024, 14, 14), (2048, 7, 7)]
input_2d_shapes = [(512 * 7 * 7,), (256 * 7 * 7,), (4096,), (1000,)]


def rand_shapes(use="all", where='weight'):
    if where.startswith('w'):
        if use == 'all':
            return random.choice(weight_3d_shapes + weight_2d_shapes)
        elif use == '3d':
            return random.choice(weight_3d_shapes)
        else:
            return random.choice(weight_2d_shapes)
    else:
        if use == 'all':
            return random.choice(input_3d_shapes + input_2d_shapes)
        elif use == '3d':
            return random.choice(input_3d_shapes)
        else:
            return random.choice(input_2d_shapes)


def check(x: torch.Tensor, y: torch.Tensor, eps=1e-6, msg='Check Failed!'):
    err = (x - y).abs() / x.abs().max()
    err = err.max()
    if err > eps:
        x = x.view(-1)
        y = y.view(-1)
        err_idx = (x - y).abs().topk(min(8, x.numel()))[1]
        print('')
        print('idx: {}'.format(err_idx.data.cpu()))
        print('x:   {}'.format(x[err_idx].data.cpu()))
        print('y:   {}'.format(y[err_idx].data.cpu()))
        print('Error {} > eps={}'.format(err, eps))
        raise Exception(msg)


class Meter:
    def __init__(self):
        self.t1 = 0
        self.t2 = 0
        self.cnt1 = 0
        self.cnt2 = 0

    def run1(self, func, *args):
        self.cnt1 += 1
        st = time.time()
        output = func(*args)
        torch.cuda.synchronize()
        self.t1 += time.time() - st
        return output

    def run2(self, func, *args):
        self.cnt2 += 1
        st = time.time()
        output = func(*args)
        torch.cuda.synchronize()
        self.t2 += time.time() - st
        return output

    def print(self, info='', name1='benchmark', name2='test'):
        self.t1 /= self.cnt1
        self.t2 /= self.cnt2
        unit = 's'
        if self.t1 < 1.0:
            self.t1 *= 1000
            self.t2 *= 1000
            unit = 'ms'
            if self.t1 < 1.0:
                self.t1 *= 1000
                self.t2 *= 1000
                unit = 'us'

        print('{}: {}: {:.2f} {}, {}: {:.2f} {} ({:.2f}x)'.format(info, name1, self.t1, unit, name2, self.t2, unit,
                                                                  self.t1 / self.t2))
