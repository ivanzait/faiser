"""Basic visualization building blocks, in two layers:

  axes-level    plot_vdf(ax, ...)                draws ONE thing on a given Axes, returns the artist
  figure-level  original_vs_reconstruction(...)  builds a layout, calls the axes-level ones, returns the figure

Figure-level functions never save; the caller does fig.savefig(...).
Velocities are given in m/s (vlim) and shown in km/s. VDF cubes follow cube[iz, iy, ix].
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# summing a cube over `axis` leaves this (horizontal, vertical) pair of velocity axes
_PLANE = {0: ('vx', 'vy'), 1: ('vx', 'vz'), 2: ('vy', 'vz')}


def project(cube, axis=0):
    """Sum a VDF cube along `axis`. Negative values (Gibbs ringing) are clipped so a log scale can show it."""
    return np.sum(np.maximum(cube, 0), axis=axis)


def plot_vdf(ax, cube, vlim, axis=0, norm=None,  title=None):    
    img = project(cube, axis)
    vkm = vlim / 1e3
    im = ax.imshow(img, origin='lower', interpolation='nearest', cmap='Spectral_r', norm=norm,
                   extent=[-vkm, vkm, -vkm, vkm])
    xlabel, ylabel = _PLANE[axis]
    ax.set(xlabel=f'{xlabel} [km/s]', ylabel=f'{ylabel} [km/s]', title=title)
    return im


def original_vs_reconstruction(cube, rec, vlim, axis=0, decades=8, labels=('original', 'reconstruction')):
    peak = project(cube, axis).max()
    norm = LogNorm(peak * 10.0 ** -decades, peak)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for a, c, label in zip(ax, (cube, rec), labels):
        im = plot_vdf(a, c, vlim, axis, norm, title=label)
    fig.colorbar(im, ax=list(ax), shrink=0.8, label=f'f summed over v{"zyx"[axis]}')
    return fig
