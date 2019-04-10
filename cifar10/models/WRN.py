"""
wide residual network
"""
import extension as my

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['WideResNet', 'WRN_28_10', 'WRN_40_10']


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, drop_ratio=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = my.Norm(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = my.Norm(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=drop_ratio) if drop_ratio > 0 else None
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        y = self.relu1(self.bn1(x))
        z = self.relu2(self.bn2(self.conv1(y)))
        if self.dropout is not None:
            z = self.dropout(z)
        z = self.conv2(z)
        if self.shortcut is None:
            return x + z
        else:
            return z + self.shortcut(y)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=1, num_classes=10, dropout=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], 3, 1, 1, bias=False)
        # 1st block
        self.block1 = self._make_layer(n, block, nChannels[0], nChannels[1], 1, dropout)
        # 2nd block
        self.block2 = self._make_layer(n, block, nChannels[1], nChannels[2], 2, dropout)
        # 3rd block
        self.block3 = self._make_layer(n, block, nChannels[2], nChannels[3], 2, dropout)
        # global average pooling and classifier
        self.bn = my.Norm(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(8, 1)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, n, block, in_planes, out_planes, stride, dropout):
        layers = []
        for i in range(int(n)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = self.pool(out)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def WRN_28_10(**kwargs):
    return WideResNet(28, 10, **kwargs)


def WRN_40_10(**kwargs):
    return WideResNet(40, 10, **kwargs)


if __name__ == '__main__':
    wrn = WideResNet(28, widen_factor=10)
    x = torch.randn(2, 3, 32, 32)
    y = wrn(x)
    print(wrn)
