"""3-D conv autoencoder for 32^3 VDFs: encoder g -> z (d numbers), decoder z -> g_hat (>= 0)."""
import torch
from torch import nn


class AE(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(1, 16, 3, 2, 1), nn.GELU(),      # 32 -> 16
            nn.Conv3d(16, 32, 3, 2, 1), nn.GELU(),     # 16 -> 8
            nn.Conv3d(32, 64, 3, 2, 1), nn.GELU(),     # 8 -> 4
            nn.Flatten(), nn.Linear(64 * 4 ** 3, d))
        self.dec_in = nn.Linear(d, 64 * 4 ** 3)
        self.dec = nn.Sequential(
            nn.GELU(), nn.ConvTranspose3d(64, 32, 4, 2, 1), nn.GELU(),   # 4 -> 8
            nn.ConvTranspose3d(32, 16, 4, 2, 1), nn.GELU(),               # 8 -> 16
            nn.ConvTranspose3d(16, 1, 4, 2, 1), nn.Softplus())            # 16 -> 32, non-negative

    def encode(self, g):                      # g: (B, 32, 32, 32)
        return self.enc(g[:, None])

    def decode(self, z):
        return self.dec(self.dec_in(z).view(-1, 64, 4, 4, 4))[:, 0]

    def forward(self, g):
        return self.decode(self.encode(g))
