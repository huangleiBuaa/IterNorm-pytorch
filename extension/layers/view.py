import torch


class View(torch.nn.Module):
    """
    reshape input tensor to a new tensor with <new_size> by use torch.view()
    size is not include batch_size
    """

    def __init__(self, *new_size: int):
        super(View, self).__init__()
        self.new_size = new_size

    def forward(self, x: torch.Tensor):
        y = x.view(x.size(0), *self.new_size)
        return y

    def __repr__(self):
        return 'view{}'.format(self.new_size)
