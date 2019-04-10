import torch.nn as nn

import extension as my

__all__ = ['vgg']


class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        bias = True
        self.net = nn.Sequential(  # 2 x 128C3 - MP2
            nn.Conv2d(3, 128, 3, 1, 1, bias=bias), my.Norm(128), nn.ReLU(True), nn.Conv2d(128, 128, 3, 1, 1, bias=bias),
            nn.MaxPool2d(2, 2), # 2 x 256C3 - MP2
            my.Norm(128), nn.ReLU(True), nn.Conv2d(128, 256, 3, 1, 1, bias=bias), my.Norm(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=bias), nn.MaxPool2d(2, 2),  # 2 x 512C3 - MP2
            my.Norm(256), nn.ReLU(True), nn.Conv2d(256, 512, 3, 1, 1, bias=bias), my.Norm(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=bias), nn.MaxPool2d(2, 2), my.View(512 * 4 * 4),  # 1024FC
            # nn.BatchNorm1d(512 * 4 * 4),
            # my.quantizer(512 * 4 * 4, nn.ReLU(True)),
            # my.Linear(512 * 4 * 4, 1024, bias=bias),
            # Softmax
            nn.BatchNorm1d(512 * 4 * 4), nn.ReLU(True), nn.Linear(512 * 4 * 4, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
        return

    def forward(self, x):
        return self.net(x)


def vgg():
    return VGG()
