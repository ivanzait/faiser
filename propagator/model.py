"""Residual stepper z_{t+dt} = z_t + f(z_t, cond). The acceleration (Lorentz) part is local to the cell; the neighbour
input needed for streaming will enter later as an extra argument of `forward`."""
import torch
from torch import nn


class Stepper(nn.Module):
    def __init__(self, dim, cond_dim, hidden=256, layers=3):
        super().__init__()
        d = [dim + cond_dim] + [hidden] * layers
        self.net = nn.Sequential(*[m for i in range(layers) for m in (nn.Linear(d[i], d[i + 1]), nn.GELU())],
                                 nn.Linear(hidden, dim))

    def forward(self, z, cond):
        return z + self.net(torch.cat([z, cond], -1))

    def rollout(self, z, conds):
        """conds: (B, H, cond_dim) -> (B, H, dim). The model is applied repeatedly, with the same dt per step."""
        out = []
        for h in range(conds.shape[1]):
            z = self(z, conds[:, h]); out.append(z)
        return torch.stack(out, 1)
