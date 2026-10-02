"""One cell: adaptive Hermite transform + spectral window + bounding box.
Plots the original VDF, the Hermite spectrum before / after the window, and the reconstruction."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/ivanzait/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytools as pt
from data_processing import vdf_tools as vt
from plot_routines.plot_window import plot_window

plt.rcParams['text.usetex'] = False   # importing analysator switches it on; there is no latex here

BULK_FILE = "/wrk-vakka/group/spacephysics/vlasiator/2D/BIE/bulk.0001125.vlsv"
CELL_ID   = 795501
MAX_ORDER = 20
TOLERANCE = 0.05
WINDOW    = 'lanczos'      # or 'raised_cosine'
OUTFILE   = os.path.join(os.path.dirname(__file__), '..', 'plots', f'run_one_cell_{CELL_ID}.png')


def main():
    reader = pt.vlsvfile.VlsvReader(BULK_FILE)
    vlim, vlen, dv = vt.get_vdf_parameters(reader)
    sp_th = vt.get_sparse_threshold(reader)

    cube = vt.build_cube(CELL_ID, reader, vlim, vlen, dv).astype(np.float64)
    u    = vt.get_drift_velocity_cube(cube, vlim, vlen)
    vth  = vt.get_thermal_velocity_cube(cube, vlim, vlen, u)

    coeffs, order, deltas = vt.adaptive_transform(
        cube, vlim, vlen, vth, u, max_order=MAX_ORDER, tolerance=TOLERANCE)
    coeffs_w = vt.apply_spectral_window(coeffs, order, window=WINDOW)
    box = vt.get_vdf_bounding_box(cube, vlim, vlen, sp_th=sp_th, margin=1)['idx']
    print(f'order_used={order}  n_coeffs={len(coeffs)}  final Parseval delta={deltas[order - 1]:.4f}')

    recs = {}
    for name, c in [('no window', coeffs), (WINDOW, coeffs_w)]:
        recs[name] = vt.apply_bounding_box(vt.reconstruct_vdf_adaptive(c, vlim, vlen, order, vth, u), box)
        eps_log = np.sqrt(np.mean((np.log(np.maximum(cube, sp_th)) - np.log(np.maximum(recs[name], sp_th))) ** 2))
        print(f'{name:10s} eps_log = {eps_log:.4f}')

    fig = plot_window(cube, recs[WINDOW], coeffs, coeffs_w, order, vlim, WINDOW)
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    fig.savefig(OUTFILE, dpi=130)
    print(f'saved -> {OUTFILE}')


if __name__ == '__main__':
    main()
