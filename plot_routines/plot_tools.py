"""Plotting tools for VDF cubes, cube[iz, iy, ix] in m/s (vlim), shown in km/s.

  reductions    project, central_slice, reduce_cube             3D -> 2D
  axes-level    plot_vdf_1d, plot_vdf_2d, plot_spectrum          draw on given Axes, return the artists
  figure-level  vdf_1d, vdf_2d, vdf_figure                       build a figure, return it; the caller saves it

Figure-level functions take a list of cubes, e.g. [cube] or [cube, rec]: one line (1D) or one row (2D) per cube.
vel_bounds is get_vdf_bounding_box()['vel'] = ((vz), (vy), (vx)), indexed like the cube axes.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tools import vdf_tools as vt


################################
########  REDUCTIONS   #########
################################

def project(cube, axis=0):
    """Sum along `axis`. Negative values (ringing) are clipped so a log scale can show the result."""
    return np.sum(np.maximum(cube, 0), axis=axis)


def central_slice(cube, axis=0):
    """Plane through the centre of the velocity mesh (cell vlen//2) normal to `axis`, negatives clipped."""
    return np.maximum(np.take(cube, cube.shape[axis] // 2, axis=axis), 0)


def reduce_cube(cube, axis=0, reduce='slice'):
    """3D -> 2D: 'slice' gives central_slice, 'sum' gives project."""
    return central_slice(cube, axis) if reduce == 'slice' else project(cube, axis)


################################
########    HELPERS    #########
################################

def _v_km(cube, vlim):
    """Cell-centre velocity axis in km/s."""
    vlen = cube.shape[0]
    return vt.velocity_axis(vlim, vlen, 2 * vlim / vlen) / 1e3


def _plane(axis):
    """Cube axes (horizontal, vertical) of an image reduced along `axis`."""
    h, v = sorted({0, 1, 2} - {axis}, reverse=True)
    return h, v


def _name(cube_axis):
    return f'v{"zyx"[cube_axis]}'


def _zoom(ax, axis, vel_bounds):
    """Limit an image reduced along `axis` to the box; None keeps the whole velocity mesh."""
    if vel_bounds is None:
        return
    h, v = _plane(axis)
    ax.set_xlim(*(np.array(vel_bounds[h]) / 1e3))
    ax.set_ylim(*(np.array(vel_bounds[v]) / 1e3))


def _contour_sp_th(ax, cube, vlim, axis, sp_th):
    """sp_th as a black contour on a central slice."""
    v = _v_km(cube, vlim)
    ax.contour(v, v, central_slice(cube, axis), levels=[sp_th], colors='k', linewidths=0.8)


################################
########  AXES-LEVEL   #########
################################

def plot_vdf_1d(axes, cube, vlim, label=None, **plot_kw):
    """Three 1D cuts through the centre of the velocity mesh, along vx, vy, vz, one per Axes in `axes`.
    Returns the Line2D artists."""
    c = cube.shape[0] // 2
    v = _v_km(cube, vlim)
    cuts = (cube[c, c, :], cube[c, :, c], cube[:, c, c])          # along vx, vy, vz
    return [ax.plot(v, cut, label=label, **plot_kw)[0] for ax, cut in zip(axes, cuts)]


def plot_vdf_2d(ax, cube, vlim, axis=0, reduce='slice', norm=None, title=None):
    """Image of the cube reduced along `axis` (reduce='slice': central plane, 'sum': projection)."""
    vkm = vlim / 1e3
    # origin='lower': row 0 of the image is the most negative velocity, so it must sit at the bottom;
    # 'nearest': no resampling, which on a log norm smears zeros and peaks into speckle when the image is shrunk
    im = ax.imshow(reduce_cube(cube, axis, reduce), origin='lower', interpolation='nearest', cmap='Spectral_r',
                   norm=norm, extent=[-vkm, vkm, -vkm, vkm])
    h, v = _plane(axis)
    ax.set(xlabel=f'{_name(h)} [km/s]', ylabel=f'{_name(v)} [km/s]', title=title)
    return im


def plot_spectrum(ax, coeffs, order, norm=None, title=None):
    """Hermite power P[l, m] = sum_n C[l,m,n]^2 from a {(l, m, n): c} dict (blank where there is none)."""
    p = np.zeros((order, order))
    for (l, m, n), c in coeffs.items():
        p[l, m] += c * c
    im = ax.imshow(np.where(p > 0, p, np.nan), origin='lower', cmap='viridis', norm=norm)
    ax.set(xlabel='m', ylabel='l', title=title, xticks=range(0, order, 4), yticks=range(0, order, 4))
    return im


################################
######## FIGURE-LEVEL  #########
################################

def vdf_1d(cubes, vlim, labels, sp_th=None, vel_bounds=None):
    """1 x 3 figure of 1D cuts through the mesh centre, one line per cube, |f| on a log scale (negative ringing
    shows). sp_th, if given, is a dotted line and sets the y range from sp_th/1e10 to 10x the first cube's peak."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for c, label in zip(cubes, labels):
        plot_vdf_1d(axes, np.abs(c), vlim, label=label)
    for ax, cube_axis in zip(axes, (2, 1, 0)):                    # along vx, vy, vz
        ax.set(yscale='log', xlabel=f'{_name(cube_axis)} [km/s]',
               title=f'along {_name(cube_axis)}, other two at the mesh centre')
        if sp_th is not None:
            ax.axhline(sp_th, color='k', ls=':', lw=0.8, label='sp_th')
            ax.set_ylim(sp_th * 1e-10, cubes[0].max() * 10)
        if vel_bounds is not None:
            ax.set_xlim(*(np.array(vel_bounds[cube_axis]) / 1e3))
    axes[0].set_ylabel('f')
    axes[0].legend()
    return fig


def vdf_2d(cubes, vlim, labels, reduce='slice', sp_th=None, vel_bounds=None, decades=8):
    """Grid of images, one row per cube, columns reduced along vz, vy, vx (reduce='slice': central planes with
    sp_th, if given, as a contour; 'sum': projections). Each column shares one LogNorm: the first cube's peak in
    that column, `decades` below it."""
    what = 'slice at' if reduce == 'slice' else 'summed over'
    fig, ax = plt.subplots(len(cubes), 3, figsize=(15, 4.5 * len(cubes)), constrained_layout=True, squeeze=False)
    for axis in range(3):
        peak = reduce_cube(cubes[0], axis, reduce).max()
        norm = LogNorm(peak * 10.0 ** -decades, peak)
        for row, (c, label) in enumerate(zip(cubes, labels)):
            a = ax[row, axis]
            im = plot_vdf_2d(a, c, vlim, axis, reduce, norm, title=f'{label}, {what} {_name(axis)}')
            if sp_th is not None and reduce == 'slice':
                _contour_sp_th(a, c, vlim, axis, sp_th)
            _zoom(a, axis, vel_bounds)
        fig.colorbar(im, ax=list(ax[:, axis]), shrink=0.8, location='bottom')
    return fig


def vdf_figure(cubes, vlim, labels, dim=1, sp_th=None, vel_bounds=None):
    """Through the centre of the velocity mesh: dim=1 -> vdf_1d, dim=2 -> vdf_2d central slices."""
    if dim == 1:
        return vdf_1d(cubes, vlim, labels, sp_th=sp_th, vel_bounds=vel_bounds)
    return vdf_2d(cubes, vlim, labels, reduce='slice', sp_th=sp_th, vel_bounds=vel_bounds)
