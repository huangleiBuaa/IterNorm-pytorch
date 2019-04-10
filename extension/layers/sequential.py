import torch.nn as nn
from collections import OrderedDict


def Sequential(*args):
    """
    Return a nn.Sequential object which ignore the parts not belong to nn.Module, such as None.
    """
    modules = []
    for m in args:
        if isinstance(m, nn.Module):
            modules.append(m)
    return nn.Sequential(*modules)


def NamedSequential(**kwargs):
    """
    Return a nn.Sequential object which ignore the parts not belong to nn.Module, such as None.
    """
    modules = []
    for k, v in kwargs.items():
        if isinstance(v, nn.Module):
            modules.append((k, v))
    return nn.Sequential(OrderedDict(modules))


if __name__ == '__main__':
    print(Sequential(nn.Conv2d(32, 3, 1, 1), nn.BatchNorm2d(32), None, nn.ReLU()))
    print(NamedSequential(conv1=nn.Conv2d(32, 3, 1, 1), bn=nn.BatchNorm2d(32), q=None, relu=nn.ReLU()))
