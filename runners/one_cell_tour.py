"""One cell, there and back: read VDF -> Hermite transform -> reconstruct -> three 1D cuts through the
center of velocity space, original vs reconstructed.

    python3 runners/one_cell_tour.py

SQRT_F=True expands sqrt(f) instead of f (see vdf_tools.adaptive_transform). DIM picks 1D cuts or 2D slices.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/ivanzait/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytools as pt
from data_processing import vdf_tools as vt
from plot_routines import plot_tools


plt.rcParams['text.usetex'] = False   # importing analysator switches it on; there is no latex here

runid = 'BID'
tstep = 1120

BULK_FILE = f"/turso/group/spacephysics/vlasiator/data/L0/2D/{runid}/bulk.{tstep:07d}.vlsv"
CELL_ID   = 180041
MAX_ORDER = 40
TOLERANCE = 0.005
SQRT_F    = True
DIM       = 1        # cuts figure: 1 = 1D cuts, 2 = 2D central slices (both through the centre of the velocity mesh)
STEM      = os.path.join(os.path.dirname(__file__), '..', 'plots', f'one_cell_tour_{runid}_tstep{tstep}_cell{CELL_ID}')
OUTFILE_CUTS = f'{STEM}_cuts_{DIM}D.png'
OUTFILE_PROJ = f'{STEM}_projections.png'


def main():
    reader = pt.vlsvfile.VlsvReader(BULK_FILE)
    vlim, vlen, dv = vt.get_vdf_parameters(reader)
    sp_th = vt.get_sparse_threshold(reader)

    cube = vt.build_cube(CELL_ID, reader, vlim, vlen, dv).astype(np.float64)
    box  = vt.get_vdf_bounding_box(cube, vlim, vlen)
    bbox_ind = box['idx']

    u   = vt.get_drift_velocity_cube(cube, vlim, vlen)
    vth = vt.get_thermal_velocity_cube(cube, vlim, vlen, u)

    coeffs, order, deltas = vt.adaptive_transform(
        cube, vlim, vlen, vth, u, max_order=MAX_ORDER, tolerance=TOLERANCE, sqrt_f=SQRT_F)
    rec = vt.reconstruct_vdf_adaptive(coeffs, vlim, vlen, order, vth, u, sqrt_f=SQRT_F)
    rec = vt.apply_bounding_box(rec, bbox_ind)

    print(f'order_used={order}  n_coeffs={len(coeffs)}  Parseval delta={deltas[order - 1]:.4f}')
    print(f'u={u / 1e3} km/s  vth={vth / 1e3:.1f} km/s')
    print(f'f max={cube.max():.3e}  sp_th={sp_th:.1e}  cells>=sp_th: {np.count_nonzero(cube >= sp_th)}  cells>0: {np.count_nonzero(cube > 0)}')
    print(f'density: original={cube.sum() * dv**3:.4e}  reconstructed={rec.sum() * dv**3:.4e}'
          f'  rel err={rec.sum() / cube.sum() - 1:+.2e}')

    # metrics in the terms that matter after Vlasiator's sparsification: what survives sp_th
    core = cube >= sp_th
    fake_halo = ~core & (rec >= sp_th)                                  # would become content blocks in Vlasiator
    # dlog = np.abs(np.log10(np.maximum(rec[core], 1e-300) / cube[core]))
    print(f'negative mass / positive mass: {-rec[rec < 0].sum() / rec[rec > 0].sum():.2e}')
    print(f' cells where halo gets above sp_th  (orig < sp_th, rec >= sp_th): {fake_halo.sum()}  mass frac {rec[fake_halo].sum() / cube.sum():.2e}')

    title = f'{runid} t={tstep} cell {CELL_ID}  order {order}  sqrt_f={SQRT_F}'
    os.makedirs(os.path.dirname(STEM), exist_ok=True)
    cubes, labels = [cube, rec], ['original', 'reconstruction']
    for out, fig in ((OUTFILE_CUTS, plot_tools.vdf_figure(cubes, vlim, labels, dim=DIM, sp_th=sp_th,
                                                          vel_bounds=box['vel'])),
                     (OUTFILE_PROJ, plot_tools.vdf_2d(cubes, vlim, labels, reduce='sum', vel_bounds=box['vel']))):
        fig.suptitle(title)
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f'saved -> {out}')

if __name__ == '__main__':
    main()
