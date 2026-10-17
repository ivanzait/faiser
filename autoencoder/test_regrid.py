"""python autoencoder/test_regrid.py: mass is conserved, a Gaussian lands where it should, outside fraction is right."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regrid import regrid

VLIM, VLEN = 6e5, 120


def gaussian(V, s):
    ax = np.linspace(-VLIM, VLIM, VLEN, endpoint=False) + VLIM / VLEN
    Z, Y, X = np.meshgrid(ax, ax, ax, indexing='ij')
    return np.exp(-((X - V[0]) ** 2 + (Y - V[1]) ** 2 + (Z - V[2]) ** 2) / (2 * s ** 2))


def test_mass_and_shape():
    V, s = np.array([1e4, -2e4, 3e4]), 6e4
    cube = gaussian(V, s)
    g, n, out = regrid(cube, VLIM, VLEN, V, s)
    dvp = 10 / 32
    assert abs(g.sum() * dvp ** 3 + out - 1) < 1e-9                       # sum g dv'^3 = 1 - outside
    assert out < 1e-4                                                      # +-5 sigma box
    ax = (np.arange(32) + 0.5) * dvp - 5
    ref = np.exp(-0.5 * (ax[:, None, None] ** 2 + ax[None, :, None] ** 2 + ax[None, None, :] ** 2)) / (2 * np.pi) ** 1.5
    assert np.abs(g - ref).max() / ref.max() < 2e-2                        # the standard Gaussian in v'
    _, _, out_small = regrid(cube, VLIM, VLEN, V, s, box=1.0)
    assert abs(out_small - (1 - 0.6827 ** 3)) < 2e-2                       # density outside +-1 sigma


if __name__ == '__main__':
    test_mass_and_shape(); print('ok')
