import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class CenterNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, dim=4, frozen=False, affine=True, *args, **kwargs):
        super(CenterNorm, self).__init__()
        self.frozen = frozen
        self.num_features = num_features
        self.momentum = momentum
        self.dim = dim
        self.shape = [1 for _ in range(dim)]
        self.shape[1] = self.num_features
        self.affine = affine
        if self.affine:
            self.bias = Parameter(torch.Tensor(*self.shape))
        self.register_buffer('running_mean', torch.zeros(self.shape))
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.zeros_(self.bias)
        self.running_mean.zero_()

    def forward(self, input: torch.Tensor):
        assert input.size(1) == self.num_features and self.dim == input.dim()
        if self.training and not self.frozen:
            mean = input.mean(0, keepdim=True)
            for d in range(2, self.dim):
                mean = mean.mean(d, keepdim=True)
            output = input - mean
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
        else:
            output = input - self.running_mean
        if self.affine:
            output = output + self.bias
        return output

    def extra_repr(self):
        return '{num_features}, momentum={momentum}, frozen={frozen}, affine={affine}'.format(**self.__dict__)


if __name__ == '__main__':
    cn = CenterNorm(32)
    print(cn)
    print(cn.running_mean.size())
    x = torch.randn(3, 32, 64, 64) + 1.
    print(x.mean())
    y = cn(x)
    print(y.mean())
    print(cn.running_mean.size())
