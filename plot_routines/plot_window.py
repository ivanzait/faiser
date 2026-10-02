"""Figure for one cell: original VDF, Hermite spectrum before / after the spectral window, reconstruction."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def spectrum(coeffs, order):
    """P[l, m] = sum_n C[l,m,n]^2: the whole tetrahedron on one plane (NaN where there is none)."""
    p = np.zeros((order, order))
    for (l, m, n), c in coeffs.items():
        p[l, m] += c * c
    return np.where(p > 0, p, np.nan)


def plot_window(cube, rec, coeffs, coeffs_w, order, vlim, window):
    """cube: original VDF; rec: its windowed + cropped reconstruction; coeffs / coeffs_w: coefficient
    dicts before / after the window (name `window`). Returns the figure; the caller saves it."""
    proj = lambda c: np.sum(np.maximum(c, 0), axis=0)          # sum over z -> the (vy, vx) plane
    top, vkm = proj(cube).max(), vlim / 1e3
    p0, p1 = spectrum(coeffs, order), spectrum(coeffs_w, order)
    vdf_kw  = dict(origin='lower', interpolation='nearest', cmap='Spectral_r', norm=LogNorm(top * 1e-8, top), extent=[-vkm, vkm, -vkm, vkm])
    spec_kw = dict(origin='lower', interpolation='nearest', cmap='viridis', norm=LogNorm(np.nanmax(p0) * 1e-16, np.nanmax(p0)))

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)
    for a, img, kw, title in [(ax[0], proj(cube), vdf_kw, 'original VDF'),
                              (ax[1], p0, spec_kw, 'Hermite spectrum (power summed over n)'),
                              (ax[2], p1, spec_kw, f'spectrum after {window} window'),
                              (ax[3], proj(rec), vdf_kw, f'reconstruction: {window} window + bbox')]:
        fig.colorbar(a.imshow(img, **kw), ax=a, shrink=0.8)
        a.set_title(title)
    for a in (ax[0], ax[3]):
        a.set(xlabel='vx [km/s]', ylabel='vy [km/s]')
    for a in (ax[1], ax[2]):
        a.set(xlabel='m', ylabel='l', xticks=range(0, order, 4), yticks=range(0, order, 4))
    return fig
