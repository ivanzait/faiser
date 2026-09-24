import os, sys
import numpy as np
import matplotlib.pyplot as plt



"""
Stage 1: cubic vs tetrahedral Hermite transform, side by side, same order.
"""

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
from data_processing import vdf_tools as vt
from adaptive_hermite import cubic_transform

# ===
# CONFIG -- edit these
# ===

BULK_FILE   = "/Users/ivanzait/Downloads/bulk.0000055.vlsv"
CELL_COORDS = [0, 0, 0]   # None = center of box; nearest cell with a VDF is used
ORDER       = 20          # indices l,m,n run 0..ORDER-1 for both transforms
SP_TH       = 1e-15
PLOTDIR = os.path.join(os.path.dirname(__file__), '..', 'ml_corrector', 'plots')


def reconstruct(coeffs, vlim, vlen, vth, u, order):
    """Dense reconstruction from a coefficient dict -- independent of whatever
    internal metric each transform used to stop/report accuracy."""
    dv   = 2.0 * vlim / vlen
    v_ax = vt.velocity_axis(vlim, vlen, dv)
    Hx   = vt.hermite_basis(v_ax, order, vth, u[0])
    Hy   = vt.hermite_basis(v_ax, order, vth, u[1])
    Hz   = vt.hermite_basis(v_ax, order, vth, u[2])
    rec = np.zeros((vlen, vlen, vlen), dtype=np.float64)
    for (l, m, n), c in coeffs.items():
        rec += c * np.einsum('x,y,z->zyx', Hx[n], Hy[m], Hz[l])
    return rec


def accuracy(cube, coeffs, vlim, vlen, vth, u, order, sp_th):
    """Honest accuracy against the true cube, from a full reconstruction --
    not the incremental Parseval bookkeeping each transform keeps internally."""
    rec = reconstruct(coeffs, vlim, vlen, vth, u, order)
    eps_rel = float(np.linalg.norm(cube - rec) / np.linalg.norm(cube))
    log_true = np.log(np.maximum(cube, sp_th))
    log_rec  = np.log(np.maximum(rec, sp_th))
    eps_log  = float(np.sqrt(np.mean((log_true - log_rec) ** 2)))
    return eps_rel, eps_log


def main():
    import pytools as pt
    
    reader = pt.vlsvfile.VlsvReader(BULK_FILE)
    
    vlim, vlen, dv = vt.get_vdf_parameters(reader)
    cellid = vt.get_nearest_vdf_cellid(reader, coords=CELL_COORDS)
    cube = vt.build_cube(cellid, reader, vlim, vlen, dv)
    u    = vt.get_drift_velocity_cube(cube, vlim, vlen)
    vth  = vt.get_thermal_velocity_cube(cube, vlim, vlen, u)
    

    # tolerance=0 -> never satisfied -> full sweep to ORDER-1, same coverage as cubic_transform(N=ORDER)
    coeffs_tetra, s_stop = vt.adaptive_transform(
        cube, vlim, vlen, vth, u, max_order=ORDER, tolerance=0.1)
    coeffs_cubic, eps_cubic, _ = cubic_transform(cube, vlim, vlen, vth, u, ORDER)

    eps_rel_tetra, eps_log_tetra = accuracy(cube, coeffs_tetra, vlim, vlen, vth, u, ORDER, SP_TH)
    eps_rel_cubic, eps_log_cubic = accuracy(cube, coeffs_cubic, vlim, vlen, vth, u, ORDER, SP_TH)

    print(f"\n{'':12s}  {'n_coeffs':>8}  {'eps_rel':>8}  {'eps_log':>8}")
    print(f"{'tetrahedral':12s}  {len(coeffs_tetra):8d}  {eps_rel_tetra:8.5f}  {eps_log_tetra:8.5f}")
    print(f"{'cubic':12s}  {len(coeffs_cubic):8d}  {eps_rel_cubic:8.5f}  {eps_log_cubic:8.5f}")


    os.makedirs(PLOTDIR, exist_ok=True)
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    vdf_proj = np.sum(cube, axis=1)   # xz projection, matches spectra's axis=1 sum below
    im0 = axes[0].imshow(vdf_proj, origin='lower',
                          extent=[-vlim, vlim, -vlim, vlim],
                          cmap='Spectral', norm=LogNorm(vmin=max(vdf_proj.min(), 1e-15),
                                                         vmax=vdf_proj.max()))
    axes[0].set_title(f'VDF  (cell {cellid})')

    panels = [(axes[1], coeffs_tetra, 'tetrahedral', eps_rel_tetra, eps_log_tetra),
              (axes[2], coeffs_cubic, 'cubic', eps_rel_cubic, eps_log_cubic)]
    for ax, coeffs, title, eps_rel, eps_log in panels:
        h_cube = np.zeros((ORDER, ORDER, ORDER))
        for k, c in coeffs.items():
            h_cube[k] = c
        ax.imshow(np.sum(h_cube, axis=1))
        ax.set_title(f'{title}  (n={len(coeffs)})\neps_rel={eps_rel:.4f}  eps_log={eps_log:.4f}')
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save_path = os.path.join(PLOTDIR, f'cubic_vs_tetra_{cellid}.png')
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved -> {save_path}")


if __name__ == '__main__':
    main()
