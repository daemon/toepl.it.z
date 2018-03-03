import math

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim

def expand_toeplitz(diag, lower_diags, upper_diags):
    pattern = torch.cat([upper_diags, diag, lower_diags], 0)
    d = lower_diags.size(0)
    columns = []
    for i in range(d + 1):
        columns.append(pattern[d - i:d - i + d + 1])
    return torch.stack(columns, 0)

class ToeplitzBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.diag = nn.Parameter(torch.Tensor([0]))
        self.lower_diags = nn.Parameter(torch.Tensor(dim - 1).zero_())
        self.upper_diags = nn.Parameter(torch.Tensor(dim - 1).zero_())

    def diagonals(self):
        return [self.diag + 1, self.lower_diags, self.upper_diags]

    def forward(self, x):
        return torch.matmul(expand_toeplitz(*self.diagonals()), x)

class ToeplitzNet(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        self.dim = dim
        self.blocks = [ToeplitzBlock(dim) for _ in range(n_layers)]
        self._blocklist = nn.ModuleList(self.blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def decompose(self):
        return [expand_toeplitz(*block.diagonals()) for block in self.blocks]

    def matrix(self):
        matrices = self.decompose()
        a = matrices[0]
        for i in range(1, len(matrices)):
            a = torch.matmul(matrices[i], a)
        return a

def toeplitz_loss(x, x_recon):
    return (x - x_recon).norm() / x.size(0)

def generate_input(dim, noise_std=0):
    eye = torch.eye(dim)
    eye += torch.Tensor(*eye.size()).normal_() * noise_std
    return eye

def td(target, n_matrices, use_cuda=True, steps=30000, log=False):
    assert target.dim() == 2 and target.size(0) == target.size(1)
    if use_cuda:
        target = target.cuda()
    target = Variable(target)
    net = ToeplitzNet(target.size(0), n_matrices)
    if use_cuda:
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), 1E-2)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=200)

    for step in range(steps):
        optimizer.zero_grad()
        input = generate_input(net.dim)
        if use_cuda:
            input = input.cuda()

        input = Variable(input)
        output = net(input)
        loss = toeplitz_loss(target, output)
        loss.backward()
        if log:
            print("Step: {}, Loss: {:.5}".format(step, loss.cpu().data[0]))
        
        optimizer.step()
        scheduler.step(loss)
    error = loss.cpu().data[0]
    return net.decompose(), net.matrix(), error

def _approx_test(matrix=None, n_approx=None, steps=3500):
    if matrix is None:
        test_n = 50
        matrix = (torch.Tensor(test_n, test_n).uniform_() - 0.5) * 10
    if n_approx is None:
        n_approx = int(matrix.size(0) / 1.5)
    print(td(matrix, n_approx, log=True, steps=steps), matrix)

def main():
    a = torch.Tensor([[3, 2, 4, 5, 1,   2],
                      [1, 3, 3, 7, -10, 1],
                      [2, 3, 1, 0, 2,   2],
                      [2, 0, 0, 3, 6,   5],
                      [9, 1, 1, 3, -1,  5],
                      [-5, 1, 1, 2, 3,  2]])
    _approx_test(n_approx=15, steps=6000)

if __name__ == "__main__":
    main()
