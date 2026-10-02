"""
Presentation-quality plots explaining the adaptive Hermite algorithm:
the tetrahedral spectrum filling in level by level, and the corresponding
VDF reconstruction improving as levels are added.

Two building blocks:
  spectrum_slice(coeffs, order)       -- 2D fill-state of the (l, m) plane at n=0
  crop_coeffs(coeffs, s_max)          -- coefficients up to total order s_max

One combined figure:
  plot_algorithm_explainer(...)       -- top row: spectrum filling in;
                                          bottom row: matching reconstruction

Run directly for a demo on a real cell:
    python3 scripts/plots_presentation.py
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
import vdf_tools as vt

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 14,
    'figure.facecolor': 'white',
    # LaTeX-like look (Computer Modern) without depending on a LaTeX install
    'font.family': 'serif',
    'font.serif': ['cmr10', 'Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False,
})


# ---
# Building blocks
# ---

def crop_coeffs(coeffs, s_max):
    """Coefficients with total order l+m+n <= s_max."""
    return {k: v for k, v in coeffs.items() if sum(k) <= s_max}


def spectrum_slice(coeffs, order):
    """
    The (l, m) plane at n=0, as a dense (order, order) array. Since every
    filled cell here has l+m <= (l+m+n) <= s_max, this slice shows exactly
    the same growing triangle as the tetrahedral fill in 3D, in 2D.
    """
    grid = np.full((order, order), np.nan)   # NaN = never reached -> shown as background
    for (l, m, n), c in coeffs.items():
        if n == 0:
            grid[l, m] = c
    return grid


def reconstruct_at_order(coeffs, s_max, vlim, vlen, vth, u, bbox_idx=None):
    trunc = crop_coeffs(coeffs, s_max)
    order = max((sum(k) for k in trunc), default=0) + 1
    rec = vt.reconstruct_vdf_adaptive(trunc, vlim, vlen, order, vth, u)
    if bbox_idx is not None:
        rec = vt.apply_bounding_box(rec, bbox_idx)
    return rec, len(trunc)


# ---
# Combined explainer figure
# ---

def plot_algorithm_explainer(cube, coeffs, vlim, vlen, vth, u, levels,
                              sp_th=1e-15, proj_axis=1, title='', save=None,
                              bbox_idx=None, deltas=None):
    """
    Two-row figure: for each s in `levels`,
      top row    -- spectrum fill state, (l, m) plane at n=0, up to total order s
      bottom row -- VDF reconstruction using only coefficients with total order <= s

    A rightmost column shows the true VDF for reference.

    bbox_idx : optional VDF bounding box (vt.get_vdf_bounding_box(...)['idx']).
               When given, each reconstruction is cropped to it -- cuts the
               Hermite-ringing lobes that sit outside the true VDF support,
               without needing the full per-voxel mask.
    deltas   : optional {s: delta} dict, as returned by adaptive_transform
               (the cumulative Parseval delta through level s). When given,
               each bottom-row panel is labeled with its delta.
    """
    n_cols = len(levels) + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(3.1 * n_cols, 6.6))

    full_order = max(sum(k) for k in coeffs) + 1
    cmap_spec = plt.get_cmap('pink_r').copy()
    cmap_spec.set_bad(color='0.92')   # unfilled bins: light gray, not white (reads as "no data" not "zero")

    # C[0,0,0] dwarfs every higher-order coefficient (it's essentially the
    # density), so a linear norm washes everything else to background gray.
    # Plotting |C| on a plain log scale (pink_r is sequential, not
    # diverging -- it has no consistent way to show sign anyway) keeps both
    # the DC term and the small structure-carrying coefficients readable,
    # with a clean colorbar instead of SymLogNorm's crowded zero-crossing.
    all_c = np.abs(np.array(list(coeffs.values())))
    vabs = all_c.max()
    vmin = max(np.sort(all_c)[len(all_c) // 20], vabs * 1e-6)   # ~5th-percentile magnitude
    spec_norm = LogNorm(vmin=vmin, vmax=vabs)

    true_proj = np.sum(cube, axis=proj_axis)
    vmax_proj = true_proj.max()

    for col, s in enumerate(levels):
        grid = np.abs(spectrum_slice(coeffs, full_order))
        grid_shown = np.where(np.array([[l + m <= s for m in range(full_order)]
                                         for l in range(full_order)]), grid, np.nan)
        ax = axes[0, col]
        im_spec = ax.imshow(grid_shown, origin='lower', cmap=cmap_spec, norm=spec_norm)
        ax.set_title(f's $\\leq$ {s}')
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel('Hermite spectrum\n(l, m) plane, n=0', fontsize=12)

        rec, n_coeffs = reconstruct_at_order(coeffs, s, vlim, vlen, vth, u, bbox_idx=bbox_idx)
        proj = np.sum(rec, axis=proj_axis)
        ax = axes[1, col]
        ax.imshow(proj, origin='lower', cmap='Spectral_r',
                  norm=LogNorm(vmin=sp_th, vmax=vmax_proj))
        ax.set_title(f'{n_coeffs} coeffs')
        ax.set_xticks([]); ax.set_yticks([])
        if deltas is not None and s in deltas:
            ax.set_xlabel(f'$\\delta$ = {deltas[s]:.3f}', fontsize=12)
        if col == 0:
            ax.set_ylabel('reconstructed VDF', fontsize=12)

    axes[0, -1].axis('off')
    cbar_ax = axes[0, -1].inset_axes([0.0, 0.0, 0.1, 1.0])   # 1/10th width, full height, left-aligned
    cbar = fig.colorbar(im_spec, cax=cbar_ax)
    cbar.set_label('$|C_{l,m,0}|$', fontsize=11)

    ax = axes[1, -1]
    ax.imshow(true_proj, origin='lower', cmap='Spectral_r',
              norm=LogNorm(vmin=sp_th, vmax=vmax_proj))
    ax.set_title('original VDF')
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"saved -> {save}")
    return fig


# ---
# One-picture summary: delta vs harmonics, simple vs complex VDF
# ---

def plot_delta_summary(cases, sp_th=1e-15, proj_axis=1, title='', save=None):
    """
    Single hero figure: a VDF thumbnail for each case, plus their Parseval
    delta vs harmonic order on one shared log-y axis -- the whole adaptive-
    order idea in one picture: simple (near-Maxwellian) VDFs converge almost
    immediately, complex ones need many harmonics before delta drops.

    cases : list of dicts, each {'label': str, 'cube': ndarray, 'deltas': dict}
            'deltas' as returned by adaptive_transform (run with
            tolerance=0.0 so every level up to max_order is present).
    """
    n = len(cases)
    fig = plt.figure(figsize=(3.6 * n + 5.5, 4.4))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [2.3])

    colors = plt.get_cmap('tab10').colors
    ax_delta = fig.add_subplot(gs[0, -1])

    for i, case in enumerate(cases):
        ax = fig.add_subplot(gs[0, i])
        proj = np.sum(case['cube'], axis=proj_axis)
        ax.imshow(proj, origin='lower', cmap='Spectral_r',
                  norm=LogNorm(vmin=sp_th, vmax=proj.max()))
        ax.set_title(case['label'], fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])

        s_vals = sorted(case['deltas'])
        d_vals = [case['deltas'][s] for s in s_vals]
        ax_delta.plot(s_vals, d_vals, 'o-', color=colors[i % len(colors)],
                      label=case['label'], markersize=4)

    ax_delta.set_yscale('log')
    ax_delta.set_xlabel('harmonic order s')
    ax_delta.set_ylabel(r'Parseval $\delta$')
    ax_delta.set_title('Adaptive stopping')
    ax_delta.legend()
    ax_delta.grid(alpha=0.3, which='both')

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"saved -> {save}")
    return fig


# ---
# Demo
# ---

if __name__ == '__main__':
    import pytools as pt

    BULK_FILE = "/Users/ivanzait/Downloads/bulk.0000055.vlsv"
    CELL_COORDS = [0, 0, 0]
    MAX_ORDER = 20
    LEVELS = [0, 5, 10, 15]
    SP_TH = 1e-15
    PLOTDIR = os.path.join(os.path.dirname(__file__), '..', 'ml_corrector', 'plots')

    reader = pt.vlsvfile.VlsvReader(BULK_FILE)
    vlim, vlen, dv = vt.get_vdf_parameters(reader)
    cellid = vt.get_nearest_vdf_cellid(reader, coords=CELL_COORDS)
    cube = vt.build_cube(cellid, reader, vlim, vlen, dv).astype(np.float64)
    u = vt.get_drift_velocity_cube(cube, vlim, vlen)
    vth = vt.get_thermal_velocity_cube(cube, vlim, vlen, u)

    coeffs, order_used, deltas = vt.adaptive_transform(
        cube, vlim, vlen, vth, u, max_order=MAX_ORDER, tolerance=0.0)
    box = vt.get_vdf_bounding_box(cube, vlim, vlen, sp_th=SP_TH)
    bbox_idx = box['idx'] if box is not None else None

    os.makedirs(PLOTDIR, exist_ok=True)
    save_path = os.path.join(PLOTDIR, f'algorithm_explainer_{cellid}.png')
    plot_algorithm_explainer(
        cube, coeffs, vlim, vlen, vth, u, levels=LEVELS,
        title=f'Adaptive tetrahedral Hermite transform -- cell {cellid}',
        save=save_path, bbox_idx=bbox_idx, deltas=deltas)
