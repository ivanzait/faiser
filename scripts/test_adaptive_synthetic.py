import os, sys
import numpy as np

"""
Synthetic-VDF validation for adaptive_transform's log-space stopping
criterion (no pytools/.vlsv data required).

Two cases, both drifting Maxwellian core:
  - "maxwellian": pure Maxwellian, no tail -- should stop at very low
    order (log_delta collapses fast, nothing left to resolve).
  - "core_halo": core + a much weaker (1e-3), hotter (3x vth) halo
    population, with values below a sparse-storage floor zeroed out --
    mimics a real suprathermal tail sitting near the instrument's sparse
    threshold. This is deliberately hard: it should NOT stop early, and
    log_delta should decrease smoothly/monotonically as more harmonics
    resolve the tail, rather than plateauing (which is the "insensitive"
    failure mode the linear/Parseval criterion had).
"""

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
from data_processing import vdf_tools as vt

VLIM   = 6.0e5     # m/s
VLEN   = 41
U      = np.array([1.0e5, 0.0, 0.0])   # drifting core
VTH    = 1.2e5
MAX_ORDER = 20      # matches HN used in data_processing/vdfs_for_initialization.py


def build_maxwellian_cube(vlim, vlen, u, vth):
    dv = 2.0 * vlim / vlen
    v_ax = vt.velocity_axis(vlim, vlen, dv)
    X, Y, Z = np.meshgrid(v_ax, v_ax, v_ax, indexing='xy')  # cube[z,y,x]
    cube = np.exp(-((X - u[0])**2 + (Y - u[1])**2 + (Z - u[2])**2) / (2 * vth**2))
    return (cube / cube.max()).astype(np.float64)


def build_core_halo_cube(vlim, vlen, u, vth, halo_frac=1e-3, halo_vth_mult=3.0,
                          sparse_frac=1e-4):
    """
    core: drifting Maxwellian, n=1
    halo: much weaker, hotter population -- a supra-thermal tail sitting
          orders of magnitude below the core, same qualitative shape as
          observed core+halo/strahl VDFs.
    Values below `sparse_frac` of the peak are zeroed, mimicking a real
    instrument's sparse-storage threshold (real VDF cubes are exactly zero
    outside the populated region, not asymptotically small).
    """
    dv = 2.0 * vlim / vlen
    v_ax = vt.velocity_axis(vlim, vlen, dv)
    X, Y, Z = np.meshgrid(v_ax, v_ax, v_ax, indexing='xy')

    core = np.exp(-((X - u[0])**2 + (Y - u[1])**2 + (Z - u[2])**2) / (2 * vth**2))
    vth_halo = vth * halo_vth_mult
    halo = halo_frac * np.exp(-((X - u[0])**2 + (Y - u[1])**2 + (Z - u[2])**2) / (2 * vth_halo**2))

    cube = core + halo
    cube /= cube.max()
    cube[cube < sparse_frac] = 0.0
    return cube.astype(np.float64)


def reconstruct(coeffs, vlim, vlen, vth, u, order):
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
    """Honest accuracy from an independent full reconstruction -- not the
    incremental bookkeeping adaptive_transform keeps internally."""
    rec = reconstruct(coeffs, vlim, vlen, vth, u, order)
    eps_rel = float(np.linalg.norm(cube - rec) / np.linalg.norm(cube))
    log_true = np.log(np.maximum(cube, sp_th))
    log_rec  = np.log(np.maximum(rec, sp_th))
    eps_log  = float(np.sqrt(np.mean((log_true - log_rec) ** 2)))
    return eps_rel, eps_log


def sweep(label, cube, sp_th):
    print(f"\n=== {label} (sp_th={sp_th:g}) ===")
    print(f"{'tolerance':>10}  {'s_stop':>6}  {'n_coeffs':>8}  {'eps_rel':>8}  {'eps_log':>8}")
    for tolerance in [0.5, 0.3, 0.2, 0.15, 0.1, 0.05]:
        coeffs, s_stop = vt.adaptive_transform(
            cube, VLIM, VLEN, VTH, U, max_order=MAX_ORDER,
            sp_th=sp_th, tolerance=tolerance)
        eps_rel, eps_log = accuracy(cube, coeffs, VLIM, VLEN, VTH, U, MAX_ORDER, sp_th)
        print(f"{tolerance:10.3f}  {s_stop:6d}  {len(coeffs):8d}  {eps_rel:8.5f}  {eps_log:8.5f}")


def main():
    sweep("maxwellian (easy, no tail)",
          build_maxwellian_cube(VLIM, VLEN, U, VTH), sp_th=1e-15)
    sweep("core+halo (hard, tail near sparse floor)",
          build_core_halo_cube(VLIM, VLEN, U, VTH), sp_th=5e-5)


if __name__ == '__main__':
    main()
