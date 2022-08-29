import torch
import torch.nn as nn
import math


class StochasticGates(nn.Module):
    def __init__(self, size, sigma, gate_init=None):
        super().__init__()
        self.size = size

        if gate_init is None:
            mus = 0.5 * torch.ones(size)
        else:
            mus = torch.from_numpy(gate_init)

        self.mus    = nn.Parameter(mus, requires_grad=True)
        self.sigma  = sigma

    def forward(self, x):
        """
        self.training holds the model state: train() or eval()
        """
        # at train(): Get a gaussian samples ~ N(0,sigma^2), at val(): zero
        gaussian         = self.sigma * torch.randn(self.mus.size()) * self.training

        shifted_gaussian = self.mus + gaussian.to(x.device)

        # Use the trick, create a diff' gating
        z = self.make_bernoulli(shifted_gaussian)

        # Make the input with the gates
        gated_x = x * z

        return gated_x

    @staticmethod
    def make_bernoulli(z):
        return torch.clamp(z, 0.0, 1.0)

    def get_loss(self):
        return torch.sum((1 + torch.erf((self.mus / self.sigma) / math.sqrt(2))) / 2)

    def get_gates(self):
        return self.make_bernoulli(self.mus)


if __name__ == '__main__':

    model_gate = StochasticGates(size=100, sigma=0.5, lamb=1)
    data = torch.randn(100)
    output = model_gate(data)

    print(f"{data.shape=}, {output.shape=}")
