"""
Reconstruct the VDF for cell 795501 (bulk.0001125, BIE run) from two coefficient
sets -- the ground-truth adaptive-Hermite spectrum and a CAE's reconstruction of
that spectrum -- and compare both against the original VDF cube.

vlim/dv are not stored in the npz (only u, vth, and the coefficients are). vlim
was recovered by matching the forward transform of C[0,0,0] against the stored
coefficient (see chat): vlim=6.24e6 with u and vth recomputed from the raw cube
(NOT the stored v_means/v_ths, which predate the get_thermal_velocity_cube axis
fix) reproduce hermite_coeffs[0,0,0] to 7 significant figures.

Usage
-----
python3 scripts/check_vae_reconstruction.py   # run from the repo root
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
from data_processing import vdf_tools as vt

# ===
# CONFIG
# ===

NPZ_PATH = os.path.join(os.path.dirname(__file__), '..', 'vae_data',
                         'bulk.0001125_vdf_init_HN20.npz')
CAE_PATH = os.path.join(os.path.dirname(__file__), '..', 'vae_data',
                         'CAE_test_dilated_reconst_herm_BIE1125_cid795501.npy')

VLIM  = 6.24e6      # recovered by matching forward C[0,0,0]; see module docstring
SP_TH = 1e-15

PLOTDIR = os.path.join(os.path.dirname(__file__), '..', 'ml_corrector', 'plots')


def l2_rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def main():
    d = np.load(NPZ_PATH)
    cube = d['vdfs'][0].astype(np.float64)
    vlen = cube.shape[0]
    vlim = VLIM
    dv = 2 * vlim / vlen

    order_used = int(d['order_used'][0])
    cellid = int(d['cellids'][0])

    # recompute u, vth from the raw cube at the recovered vlim -- these match
    # the stored hermite_coeffs (verified: C[0,0,0] to 7 sig figs), unlike the
    # stored v_means/v_ths which predate the get_thermal_velocity_cube axis fix.
    u   = vt.get_drift_velocity_cube(cube, vlim, vlen)
    vth = vt.get_thermal_velocity_cube(cube, vlim, vlen, u)

    h_true = d['hermite_coeffs'][0]           # (order, order, order), tetrahedral
    h_cae  = np.load(CAE_PATH)                 # CAE's reconstruction of h_true

    print(f"cell {cellid}  vlen={vlen}  vlim={vlim:.3e}  dv={dv:.1f}  "
          f"u={u}  vth={vth:.6e}  order_used={order_used}")

    order = h_true.shape[0]
    rec_true = vt.reconstruct_vdf(h_true, vlim, vlen, order, vth, u)
    rec_cae  = vt.reconstruct_vdf(h_cae,  vlim, vlen, order, vth, u)

    eps_true = l2_rel(cube, rec_true)
    eps_cae  = l2_rel(cube, rec_cae)
    eps_cae_vs_true = l2_rel(rec_true, rec_cae)

    print(f"\nL2 relative error vs original VDF:")
    print(f"  ground-truth spectrum reconstruction : {eps_true:.4f}")
    print(f"  CAE spectrum reconstruction           : {eps_cae:.4f}")
    print(f"  CAE reconstruction vs true reconstruction (not vs original): {eps_cae_vs_true:.4f}")

    os.makedirs(PLOTDIR, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    panels = [('original', cube), ('reconstructed (true coeffs)', rec_true),
              ('reconstructed (CAE coeffs)', rec_cae)]
    vmax = max(np.sum(cube, axis=ax).max() for ax in (0, 1, 2))
    for row, (title, data) in enumerate(panels):
        for col, ax_sum in enumerate([0, 1, 2]):
            ax = axes[row, col]
            proj = np.sum(data, axis=ax_sum)
            ax.imshow(proj, origin='lower', cmap='Spectral_r',
                      norm=LogNorm(vmin=SP_TH, vmax=vmax))
            if row == 0:
                ax.set_title(f'sum axis={ax_sum}')
            if col == 0:
                ax.set_ylabel(title, fontsize=10)
    fig.suptitle(f"cell {cellid}  eps_rel(true)={eps_true:.4f}  eps_rel(CAE)={eps_cae:.4f}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(PLOTDIR, f'vae_reconstruction_{cellid}.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved -> {save_path}")


if __name__ == '__main__':
    main()
