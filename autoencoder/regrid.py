"""Conservative regrid of a dense VDF cube onto a box in v' = (v - V) / v_th. cube[iz, iy, ix], velocities in m/s."""
import numpy as np


def axis_weights(vlim, vlen, centre, vth, n_grid, box):
    """(W [n_grid, vlen], edges') for one axis: W maps source cells to target bins without losing mass."""
    dv = 2 * vlim / vlen
    lo, hi = -vlim + dv * np.arange(vlen), -vlim + dv * (np.arange(vlen) + 1)
    tgt = centre + vth * np.linspace(-box, box, n_grid + 1)
    over = np.minimum(hi[None, :], tgt[1:, None]) - np.maximum(lo[None, :], tgt[:-1, None])
    return np.clip(over, 0, None) / dv


def regrid(cube, vlim, vlen, V, vth, n_grid=32, box=5.0):
    """(g [n_grid]*3, n, outside): g = vth^3 f / n averaged over each target bin (so sum g dv'^3 = 1 - outside),
    n = density from the full cube, outside = density fraction that falls outside the box."""
    dv = 2 * vlim / vlen
    n = float(cube.sum()) * dv ** 3
    Wx, Wy, Wz = (axis_weights(vlim, vlen, V[i], vth, n_grid, box) for i in range(3))
    mass = np.einsum('kz,ly,mx,zyx->klm', Wz, Wy, Wx, cube, optimize=True) * dv ** 3
    dvp = 2 * box / n_grid
    g = mass / (n * dvp ** 3)
    return g.astype(np.float32), n, 1.0 - float(mass.sum() / n)
