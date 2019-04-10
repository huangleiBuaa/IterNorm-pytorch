import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class GroupBatchNorm(nn.Module):
    def __init__(self, num_features, num_groups=1, num_channels=0, dim=4, eps=1e-5, momentum=0.1, affine=True, mode=0,
                 *args, **kwargs):
        """"""
        super(GroupBatchNorm, self).__init__()
        if num_channels > 0:
            assert num_features % num_channels == 0
            num_groups = num_features // num_channels
        assert num_features % num_groups == 0
        self.num_features = num_features
        self.num_groups = num_groups
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.mode = mode
        self.shape = [1] * dim
        self.shape[1] = num_features

        if self.affine:
            self.weight = Parameter(torch.Tensor(*self.shape))
            self.bias = Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_groups))
        self.register_buffer('running_var', torch.ones(num_groups))
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor):
        training = self.mode > 0 or (self.mode == 0 and self.training)
        assert input.dim() == self.dim and input.size(1) == self.num_features
        sizes = input.size()
        reshaped = input.view(sizes[0] * sizes[1] // self.num_groups, self.num_groups, *sizes[2:self.dim])
        output = F.batch_norm(reshaped, self.running_mean, self.running_var, training=training, momentum=self.momentum,
                              eps=self.eps)
        output = output.view_as(input)
        if self.affine:
            output = output * self.weight + self.bias
        return output

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'mode={mode}'.format(**self.__dict__)


if __name__ == '__main__':
    GBN = GroupBatchNorm(64, 16, momentum=1)
    print(GBN)
    # print(GBN.weight)
    # print(GBN.bias)
    x = torch.randn(4, 64, 32, 32) * 2 + 1
    print('x mean = {}, var = {}'.format(x.mean(), x.var()))
    y = GBN(x)
    print('y size = {}, mean = {}, var = {}'.format(y.size(), y.mean(), y.var()))
    print(GBN.running_mean, GBN.running_var)
