import argparse
import os
import torch.nn as nn
import torch
import torch.nn.functional as F


class Shortcut(nn.Module):
    def __init__(self, block: nn.Module, shortcut=None):
        super(Shortcut, self).__init__()
        self.block = block
        self.shortcut = shortcut
        self.weight = nn.Parameter(torch.ones(1))  # self.weight.data.fill_(0.1)

    def forward(self, x):
        if self.shortcut is not None:
            return self.block(x) + self.shortcut(x)
        y = self.block(x)
        if x.size()[2:4] != y.size()[2:4]:
            x = F.adaptive_avg_pool2d(x, y.size()[2:4])
        # x = x * self.weight
        if x.size(1) >= y.size(1):
            y += x[:, :y.size(1), :, :]
        else:
            y[:, :x.size(1), :, :] += x
        return y


class sign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        weight_f, ctx.slope, ctx.back_way = inputs
        weight_b = weight_f.sign()
        ctx.save_for_backward(weight_f)
        return weight_b

    @staticmethod
    def backward(ctx, *grads):
        grad, = grads
        weight_f, = ctx.saved_variables
        if ctx.back_way == 0:
            # based on HardTanh
            grad[weight_f.abs() >= 1.] *= ctx.slope
        elif ctx.back_way == 1:
            # based on polynomial function
            grad[weight_f.abs() >= 1.] *= ctx.slope
            grad[0. <= weight_f < 1.] *= 2 - 2 * weight_f[0. <= weight_f < 1.]
            grad[-1 < weight_f < 0.] *= 2 + 2 * weight_f[-1 < weight_f < 0.]
        return grad


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Scale(nn.Module):
    def __init__(self, init_value=0.1):
        super(Scale, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1))
        self.init_value = init_value
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.fill_(self.init_value)

    def forward(self, input: torch.Tensor):
        return input * self.weight

    def extra_repr(self):
        return 'init_value={:.5g}'.format(self.init_value)


def str2num(s: str):
    s.strip()
    try:
        value = int(s)
    except ValueError:
        try:
            value = float(s)
        except ValueError:
            if s == 'True':
                value = True
            elif s == 'False':
                value = False
            elif s == 'None':
                value = None
            else:
                value = s
    return value


def str2bool(v):
    if not isinstance(v, str):
        return bool(v)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def str2dict(s) -> dict:
    if s is None:
        return {}
    if not isinstance(s, str):
        return s
    s = s.split(',')
    d = {}
    for ss in s:
        if ss == '':
            continue
        ss = ss.split('=')
        assert len(ss) == 2
        key = ss[0].strip()
        value = str2num(ss[1])
        d[key] = value
    return d


def str2list(s: str) -> list:
    if not isinstance(s, str):
        return list(s)
    items = []
    s = s.split(',')
    for ss in s:
        if ss == '':
            continue
        items.append(str2num(ss))
    return items


def str2tuple(s: str) -> tuple:
    return tuple(str2list(s))


def extend_list(l: list, size: int):
    while len(l) < size:
        l.append(l[-1])
    return l[:size]


def path(p: str):
    return os.path.expanduser(p)
