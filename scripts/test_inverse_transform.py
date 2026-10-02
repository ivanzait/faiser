"""
Round-trip test of the adaptive Hermite transform on a real cell: forward
(vt.adaptive_transform) then inverse (vt.reconstruct_vdf_adaptive), compared
against the original VDF cube by relative L2 norm, with a projection plot.

Usage
-----
python3 scripts/test_inverse_transform.py   # run from the repo root
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
import pytools as pt
import vdf_tools as vt

# ===
# CONFIG
# ===

BULK_FILE   = "/Users/ivanzait/Downloads/bulk.0000055.vlsv"
CELL_COORDS = [0, 0, 0]   # nearest cell with a VDF to this point
MAX_ORDER   = 20
TOLERANCE   = 0.05
SP_TH       = 1e-15

PLOTDIR = os.path.join(os.path.dirname(__file__), '..', 'plots')


def main():
    reader = pt.vlsvfile.VlsvReader(BULK_FILE)
    vlim, vlen, dv = vt.get_vdf_parameters(reader)

    cellid = vt.get_nearest_vdf_cellid(reader, coords=CELL_COORDS)
    cube = vt.build_cube(cellid, reader, vlim, vlen, dv).astype(np.float64)
    u    = vt.get_drift_velocity_cube(cube, vlim, vlen)
    vth  = vt.get_thermal_velocity_cube(cube, vlim, vlen, u)

    coeffs, order_used, deltas = vt.adaptive_transform(
        cube, vlim, vlen, vth, u, max_order=MAX_ORDER, tolerance=TOLERANCE)
    rec = vt.reconstruct_vdf_adaptive(coeffs, vlim, vlen, order_used, vth, u)

    eps_rel = float(np.linalg.norm(cube - rec) / np.linalg.norm(cube))
    log_true = np.log(np.maximum(cube, SP_TH))
    log_rec  = np.log(np.maximum(rec, SP_TH))
    eps_log  = float(np.sqrt(np.mean((log_true - log_rec) ** 2)))

    print(f"cell {cellid}  vlen={vlen}  vlim={vlim:.3e}  vth={vth:.3e}")
    print(f"order_used={order_used}  n_coeffs={len(coeffs)}  "
          f"final Parseval delta={deltas[order_used - 1]:.5f}")
    print(f"eps_rel (L2, f-space) = {eps_rel:.5f}")
    print(f"eps_log (log-space)   = {eps_log:.5f}")

    os.makedirs(PLOTDIR, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    vmax = max(np.sum(cube, axis=ax).max() for ax in (0, 1, 2))
    for row, (title, data) in enumerate([('original', cube), ('reconstructed', rec)]):
        for col, ax_sum in enumerate([0, 1, 2]):
            ax = axes[row, col]
            proj = np.sum(data, axis=ax_sum)
            ax.imshow(proj, origin='lower', cmap='Spectral_r',
                      norm=LogNorm(vmin=SP_TH, vmax=vmax))
            if row == 0:
                ax.set_title(f'sum axis={ax_sum}')
            if col == 0:
                ax.set_ylabel(title, fontsize=10)
    fig.suptitle(f"cell {cellid}  order_used={order_used}  "
                 f"eps_rel={eps_rel:.4f}  eps_log={eps_log:.4f}")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_path = os.path.join(PLOTDIR, f'inverse_transform_{cellid}.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved -> {save_path}")


if __name__ == '__main__':
    main()
