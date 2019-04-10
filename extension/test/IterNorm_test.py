from extension.normailzation.iterative_normalization import IterNorm
import torch
from extension.test.test_util import *


class IterNorm_py(IterNorm):
    def forward(self, X: torch.Tensor):
        # change NxCxHxW to Cx(NxHxW), i.e., d*m
        x = X.transpose(0, 1).contiguous().view(self.num_groups, self.num_features // self.num_groups, -1)
        g, d, m = x.size()
        if self.training:
            # calculate centered activation by subtracted mini-batch mean
            mean = x.mean(-1, keepdim=True)
            x_c = x - mean
            # calculate covariance matrix
            Sigma = x_c.matmul(x_c.transpose(1, 2)) / m + self.eps * torch.eye(d, dtype=X.dtype, device=X.device)
            # Sigma = torch.eye(d).to(X)
            # torch.baddbmm(self.eps, Sigma, 1./m, x_c, x_c.transpose(1, 2))
            # reciprocal of trace of Sigma: shape [g, 1, 1]
            rTr = x_c.new_empty(g, 1, 1)
            for i in range(g):
                rTr[i] = 1. / Sigma[i].trace()
            sigma_norm = Sigma * rTr
            P = [None] * (self.T + 1)
            P[0] = torch.eye(d).to(X).expand(g, d, d)
            for k in range(self.T):
                P[k + 1] = 0.5 * (3 * P[k] - torch.matrix_power(P[k], 3).matmul(
                    sigma_norm))  # P[k + 1] = P[k].clone()  # torch.baddbmm(1.5, P[k + 1], -0.5, torch.matrix_power(P[k], 3), sigma_norm)
            sigma_inv = P[self.T] * rTr.sqrt()
            self.running_mean = self.momentum * mean + (1. - self.momentum) * self.running_mean
            self.running_wm = self.momentum * sigma_inv + (1. - self.momentum) * self.running_wm
        else:
            x_c = x - self.running_mean
            sigma_inv = self.running_wm
        x_hat = sigma_inv.matmul(x_c)
        X_hat = x_hat.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()

        # affine
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat


def test_IterNorm(test_number=100):
    device = torch.device('cuda')
    torch.set_default_dtype(torch.float64)
    batch_size = 16
    eps = 1e-6 if torch.get_default_dtype() == torch.float64 else 1e-4

    fm = Meter()
    bm = Meter()

    for i in range(test_number):
        torch.cuda.empty_cache()
        shape = batch_size, *rand_shapes('3d', 'input')
        T = random.randint(1, 10)
        num_channels = 2 ** random.randint(3, 6)
        print('run test [{}/{}], input shape: {}, T: {}, num_channels={}      '.format(i + 1, test_number, shape, T,
            num_channels), end='\r')
        if shape[1] == 3:
            continue
        x1 = torch.randn(shape, device=device)
        x2 = x1.data.clone()
        x1.requires_grad_()
        x2.requires_grad_()
        g = torch.randn(shape, device=device)
        n1 = IterNorm_py(shape[1], num_channels=num_channels, T=T, dim=len(shape)).to(device)
        n2 = IterNorm(shape[1], num_channels=num_channels, T=T, dim=len(shape)).to(device)

        r1 = fm.run1(n1, x1)
        bm.run1(lambda: torch.autograd.backward(r1, g))

        r2 = fm.run2(n2, x2)
        bm.run2(lambda: torch.autograd.backward(r2, g))

        # z = r1.transpose(0, 1).contiguous().view(shape[1], -1)
        # z = z.matmul(z.t()) / z.size(1) - torch.eye(shape[1], device=device)
        # print('\n', z)
        # assert (z.abs() < 1e-4).sum().item() == 0
        #
        # z = r2.transpose(0, 1).contiguous().view(shape[1], -1)
        # z = z.matmul(z.t()) / z.size(1) - torch.eye(shape[1], device=device)
        # print(z)
        # assert (z.abs() < 1e-4).sum().item() == 0

        check(r1, r2, eps=eps)
        # check(x1.grad, x2.grad, eps=eps)
        check(n1.running_mean, n2.running_mean)
        check(n1.running_wm, n2.running_wm)

        del r1, r2, x1, x2, g, n1, n2

    print('\n\033[32mPass {} test!\033[0m'.format(test_number))
    fm.print('IterNorm  forward', 'py', 'c++')
    bm.print('IterNorm backward', 'py', 'c++')


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    print("############# Test IterNorm #############")
    print('seed = {}'.format(seed))
    test_IterNorm(1000)
