"""Summary of a file written by runners/run_multi_cell.py, then original vs reconstructed VDF of one of its cells.

    python3 runners/multi_cell_view.py [data/run_TSI_tstep30.npz]

The summary uses only the stored arrays (order_used, delta_final), no reconstruction. The cell is CELL, or a random
one if CELL is None: its spectrum is reconstructed and cropped to its box, the original VDF is taken from the bulk file. 
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/ivanzait/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytools as pt
from tools import vdf_tools as vt
from plot_routines import plot_tools
plt.rcParams['text.usetex'] = False   # importing analysator switches it on; there is no latex here

NPZ_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'run_TSI_tstep30.npz')
PLOTDIR  = os.path.join(os.path.dirname(__file__), '..', 'plots')
CELL     = None      # cell id to plot; None: a random cell of the file
DIM      = 1         # 1: 1D cuts, 2: 2D slices, both through the centre of the velocity mesh
ZOOM     = True      # zoom to the box of everything stored for the cell (halo included)


def n_coeffs(order):
    """Number of coefficients in levels 0..order-1 (tetrahedral truncation)."""
    return order * (order + 1) * (order + 2) // 6


def main(npz_file):
    stem = os.path.splitext(os.path.basename(npz_file))[0]
    print('STEM',stem)
    d = np.load(npz_file)
    runid, tstep, bulk_file = str(d['runid']), int(d['tstep']), str(d['bulk_file'])
    vlim, vlen, sqrt_f = float(d['vlim']), int(d['vlen']), bool(d['sqrt_f'])
    max_order, tol = int(d['max_order']), float(d['tolerance'])
    cellids, orders, deltas = d['cellids'], d['order_used'], d['delta_final']

    # ---- summary over the file, from the stored arrays only
    print(f'{runid} t={tstep}: {len(cellids)} cells, max_order={max_order}, tolerance={tol}, sqrt_f={sqrt_f}')
    # print(f'  order_used    min {orders.min()}  median {np.median(orders):.0f}  max {orders.max()}  '
    #       f'| at max_order (not converged): {(orders == max_order).sum()} of {len(orders)}')
    # print(f'  delta_final   median {np.median(deltas):.3g}  max {deltas.max():.3g}  '
    #       f'| below tolerance: {(deltas < tol).sum()} of {len(deltas)}')
    # print(f'  coefficients  {sum(n_coeffs(int(o)) for o in orders)} in total, '
    #       f'{np.median([n_coeffs(int(o)) for o in orders]):.0f} per cell (median)')

    ###  INVERSE HERMITE: one cell 
    cell_id = int(CELL) if CELL is not None else int(np.random.default_rng().choice(cellids))
    i = int(np.flatnonzero(cellids == cell_id)[0])
    o = int(orders[i])
    coeffs = dict(np.ndenumerate(d['hermite_coeffs'][i][:o, :o, :o]))      # levels beyond order_used are zero
    rec = vt.reconstruct_vdf_adaptive(coeffs, vlim, vlen, o, float(d['v_ths'][i]), d['v_means'][i], sqrt_f=sqrt_f)
    rec = vt.apply_bounding_box(rec, d['bbox_idx'][i])

    ##
    reader = pt.vlsvfile.VlsvReader(bulk_file)
    sp_th = vt.get_sparse_threshold(reader)
    dv = 2 * vlim / vlen
    cube =  vt.build_cube(cell_id, reader, vlim, vlen, dv).astype(np.float64)

    print(f"\ncell {cell_id} (row {i}): order_used={o}, delta_final={deltas[i]:.3g} ") ## cell, order, delta    
    #print(f"u={np.round(d['v_means'][i] / 1e3, 1)} km/s, vth={float(d['v_ths'][i]) / 1e3:.1f} km/s") ## velocities
    
    for k, v in vt.reconstruction_metrics(cube, rec, sp_th, dv, n_coeffs=n_coeffs(o)).items():
        print(f'  {k:15s} {v:.4g}' if isinstance(v, float) else f'  {k:15s} {v}')

    # Parseval delta is the L2 error of the EXPANDED function (sqrt(f) if sqrt_f), uncropped: rebuild that directly
    if sqrt_f:
        g     = np.sqrt(np.clip(cube, 0, None))
        g_rec = vt.reconstruct_vdf_adaptive(coeffs, vlim, vlen, o, np.sqrt(2) * float(d['v_ths'][i]), d['v_means'][i],
                                            sqrt_f=False)                  # the sqrt(f) series itself, not squared
        print(f'  {"l2_err_sqrt_f":15s} {np.linalg.norm(g_rec - g) / np.linalg.norm(g):.4g}'
              f'   (same quantity as delta_final = {deltas[i]:.4g})')

    vel_bounds = vt.get_vdf_bounding_box(cube, vlim, vlen)['vel'] if ZOOM else None
    
    cubes, labels = [cube, rec], ['original', f'reconstruction (order {o}, sqrt_f={sqrt_f})']
    title = f'{runid}   t = {tstep}   cell {cell_id}'
    
    os.makedirs(PLOTDIR, exist_ok=True)
    
    for name, fig in ((f'cuts_{DIM}D', plot_tools.vdf_figure(cubes, vlim, labels, dim=DIM, sp_th=sp_th,vel_bounds=vel_bounds)),
                      ('projections', plot_tools.vdf_2d(cubes, vlim, labels, reduce='sum', vel_bounds=vel_bounds)) 
                      ):
        fig.suptitle(title)
        out = os.path.join(PLOTDIR, f'{stem}_cell{cell_id}_{name}.png')
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f'saved -> {out}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else NPZ_FILE)
