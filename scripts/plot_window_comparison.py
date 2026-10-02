"""
Does spectral windowing (vt.apply_spectral_window) reduce the Gibbs ringing
left over from truncating the Hermite coefficient series at max_order?

Windowing smooths the sharp truncation edge in coefficient space (order
s > max_order set to exactly zero), which is what produces Gibbs ringing in
the reconstruction -- a mid-slope "kink" independent of the VDF's own edge.

Main metric: eps_log against the TRUE VDF, cropped to the VDF's bounding box
(+1 cell margin, so the reconstruction has room to decay into view instead
of being cut off mid-slope).

Two outputs:
  1. window_metric_<cellid>.png  -- eps_log vs max order, one curve per window
  2. window_tails_<cellid>.png   -- VDF projections + 1-D cuts at a chosen
     order, one panel per window

Usage
-----
python3 scripts/plot_window_comparison.py   # run from the repo root
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
import vdf_tools as vt

# ===
# CONFIG -- edit these
# ===

BULK_FILE   = "/Users/ivanzait/Downloads/bulk.0000055.vlsv"
CELL_COORDS = [0, 0, 0]
ORDERS      = [4, 8, 12, 14, 16, 20]
PLOT_ORDER  = 20
SP_TH       = 1e-15
WINDOWS = {
    'no window':      None,
    'lanczos':        'lanczos',
    'raised cosine':  'raised_cosine',
}
PROJ_AXIS = 1
PLOTDIR = os.path.join(os.path.dirname(__file__), '..', 'ml_corrector', 'plots')


def eps_log(true, rec, sp_th):
    lt, lr = np.log(np.maximum(true, sp_th)), np.log(np.maximum(rec, sp_th))
    return float(np.sqrt(np.mean((lt - lr) ** 2)))


def main():
    import pytools as pt
    reader = pt.vlsvfile.VlsvReader(BULK_FILE)
    vlim, vlen, dv = vt.get_vdf_parameters(reader)
    cellid = vt.get_nearest_vdf_cellid(reader, coords=CELL_COORDS)
    cube = vt.build_cube(cellid, reader, vlim, vlen, dv).astype(np.float64)
    u = vt.get_drift_velocity_cube(cube, vlim, vlen)
    vth = vt.get_thermal_velocity_cube(cube, vlim, vlen, u)

    crop_box = vt.get_vdf_bounding_box(cube, vlim, vlen, sp_th=SP_TH, margin=1)
    print(f"cell {cellid}  crop box idx={crop_box['idx']}")

    results = {name: [] for name in WINDOWS}
    recs_at_plot_order = {}
    for order in ORDERS:
        coeffs, _, _ = vt.adaptive_transform(cube, vlim, vlen, vth, u, max_order=order, tolerance=0.0)
        for name, window in WINDOWS.items():
            c = coeffs if window is None else vt.apply_spectral_window(coeffs, order, window=window)
            rec = vt.reconstruct_vdf_adaptive(c, vlim, vlen, order, vth, u)
            rec_boxed = vt.apply_bounding_box(rec, crop_box['idx'])
            results[name].append(eps_log(cube, rec_boxed, SP_TH))
            if order == PLOT_ORDER:
                recs_at_plot_order[name] = rec_boxed

    print(f"\n{'order':>6}" + ''.join(f'{name:>16}' for name in WINDOWS))
    for i, order in enumerate(ORDERS):
        print(f"{order:6d}" + ''.join(f'{results[name][i]:16.4f}' for name in WINDOWS))

    os.makedirs(PLOTDIR, exist_ok=True)

    # --- figure 1: metric vs order ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name in WINDOWS:
        ax.plot(ORDERS, results[name], 'o-', label=name)
    ax.set_xlabel('max order')
    ax.set_ylabel('eps_log (vs true VDF, boxed)')
    ax.set_title(f'Spectral windowing vs reconstruction fidelity -- cell {cellid}')
    ax.legend()
    fig.tight_layout()
    save1 = os.path.join(PLOTDIR, f'window_metric_{cellid}.png')
    fig.savefig(save1, dpi=150)
    plt.close(fig)
    print(f"\nsaved -> {save1}")

    # --- figure 2: tail inspection at PLOT_ORDER, with a 1-D cut below each panel ---
    n_cols = len(WINDOWS) + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 7),
                              gridspec_kw={'height_ratios': [1, 0.7]})
    true_proj = np.sum(cube, axis=PROJ_AXIS)
    vmax = true_proj.max()
    cut_row = int(np.argmax(true_proj) // true_proj.shape[1])

    panels = [('original VDF', true_proj)] + \
             [(f'{name}\norder={PLOT_ORDER}', np.sum(recs_at_plot_order[name], axis=PROJ_AXIS))
              for name in WINDOWS]

    for col, (label, proj) in enumerate(panels):
        ax_img = axes[0, col]
        ax_img.imshow(proj, origin='lower', cmap='Spectral_r', norm=LogNorm(vmin=SP_TH, vmax=vmax))
        ax_img.axhline(cut_row, color='k', lw=0.7, ls='--', alpha=0.5)
        ax_img.set_title(label)
        ax_img.set_xticks([]); ax_img.set_yticks([])

        ax_cut = axes[1, col]
        ax_cut.plot(true_proj[cut_row, :], color='0.6', lw=1.2, label='original' if col > 0 else None)
        ax_cut.plot(proj[cut_row, :], color='C0' if col > 0 else 'k', lw=1.5)
        ax_cut.axhline(SP_TH, color='r', lw=0.7, ls=':')
        ax_cut.set_yscale('log')
        ax_cut.set_ylim(SP_TH * 0.5, vmax * 2)
        ax_cut.set_xticks([])
        if col == 0:
            ax_cut.set_ylabel('f(v)  [1-D cut]')
        else:
            ax_cut.set_yticklabels([])
            ax_cut.legend(fontsize=8, loc='upper right')

    fig.suptitle(f'Reconstruction tails, boxed -- cell {cellid}')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save2 = os.path.join(PLOTDIR, f'window_tails_{cellid}.png')
    fig.savefig(save2, dpi=150)
    plt.close(fig)
    print(f"saved -> {save2}")


if __name__ == '__main__':
    main()
