import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Scale(nn.Module):
    def __init__(self, num_features, dim=4):
        super(Scale, self).__init__()
        self.num_features = num_features
        shape = [1 for _ in range(dim)]
        shape[1] = self.num_features

        self.weight = Parameter(torch.Tensor(*shape))
        self.bias = Parameter(torch.Tensor(*shape))

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.uniform_(self.weight)
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        return input * self.weight + self.bias

    def extra_repr(self):
        return '{}'.format(self.num_features)


if __name__ == '__main__':
    s = Scale(4)
    x = torch.ones(3, 4, 5, 6)
    print(s.weight.size())
    nn.init.constant_(s.weight, 2)
    nn.init.constant_(s.bias, 1)
    y = s(x)
    print(y, y.size())
