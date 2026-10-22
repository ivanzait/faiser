"""Original vs reconstructed VDF for a cell saved by runners/run_one_cell.py: metrics and plots.

    python3 runners/one_cell_view.py [data/one_cell_run_TSI_tstep30_cell200.npz]

Everything comes from the npz: coefficients, u, vth, sqrt_f, vlim, vlen, bounding box, cell id, and the bulk
file, run id and time step. The spectrum may be saved sparse (a pickled {(l, m, n): c} dict) or dense (flag `dense`); both
are handled. The reconstruction is rebuilt from the coefficients and cropped to the box; the original VDF and the
sparsity threshold are re-read from the bulk file.

Plots: DIM=1 1D cuts or DIM=2 2D central slices through the centre of the velocity mesh, plus projections,
all zoomed to the box (ZOOM).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MACHINE = 'turso'
if MACHINE == 'turso':
    sys.path.insert(0, "/home/ivanzait/analysator")            # same analysator as run_one_cell.py
else:
    sys.path.insert(0, "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytools as pt
from tools import vdf_tools as vt
from plot_routines import plot_tools
plt.rcParams['text.usetex'] = False   # importing analysator switches it on; there is no latex here

NPZ_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'one_cell_run_TSI_tstep30_cell200.npz')
PLOTDIR  = os.path.join(os.path.dirname(__file__), '..', 'plots')
DIM      = 1         # 1: 1D cuts, 2: 2D slices, both through the centre of the velocity mesh
ZOOM     = True      # zoom to the box of everything stored for the cell (halo included)


def load_coeffs(hermite_coeffs, dense, order_used):
    """({(l, m, n): c}, order) for reconstruct_vdf_adaptive. A dense spectrum is a (MAX_ORDER,)*3 cube, zero-padded
    beyond order_used, so its side is the order to use; a sparse one is a pickled dict reconstructed at order_used."""
    if dense:
        return dict(np.ndenumerate(hermite_coeffs)), hermite_coeffs.shape[0]
    return hermite_coeffs.item(), order_used


def main(npz_file):
    stem = os.path.splitext(os.path.basename(npz_file))[0]
    d = np.load(npz_file, allow_pickle=True)                    # a sparse spectrum is a pickled dict
    if 'bulk_file' not in d.files:
        sys.exit(f"{npz_file} has no 'bulk_file': written by an older run_one_cell.py, run that again")
    runid, tstep, bulk_file = str(d['runid']), int(d['tstep']), str(d['bulk_file'])
    dense, order_used = bool(d['dense']), int(d['order_used'])
    coeffs, order = load_coeffs(d['hermite_coeffs'], dense, order_used)
    vlim, vlen, cell_id = float(d['vlim']), int(d['vlen']), int(d['cellids'])
    dv = 2 * vlim / vlen
    sqrt_f = bool(d['sqrt_f']) if 'sqrt_f' in d.files else False   # files from before sqrt_f are plain-f spectra

    rec = vt.reconstruct_vdf_adaptive(coeffs, vlim, vlen, order, float(d['v_ths']), d['v_means'], sqrt_f=sqrt_f)
    rec = vt.apply_bounding_box(rec, d['bbox_idx'])

    reader = pt.vlsvfile.VlsvReader(bulk_file)
    sp_th = vt.get_sparse_threshold(reader)
    cube = vt.build_cube(cell_id, reader, vlim, vlen, dv).astype(np.float64)

    print(f"{runid} t={tstep} cell {cell_id}: {'dense' if dense else 'sparse'} spectrum, order_used={order_used}, "
          f"sqrt_f={sqrt_f}, u={np.round(d['v_means'] / 1e3, 1)} km/s, vth={float(d['v_ths']) / 1e3:.1f} km/s")
    n_coeffs = order_used * (order_used + 1) * (order_used + 2) // 6       # levels 0..order_used-1, tetrahedral
    for k, v in vt.reconstruction_metrics(cube, rec, sp_th, dv, n_coeffs=n_coeffs).items():
        print(f'  {k:15s} {v:.4g}' if isinstance(v, float) else f'  {k:15s} {v}')

    vel_bounds = vt.get_vdf_bounding_box(cube, vlim, vlen)['vel'] if ZOOM else None
    cubes, labels = [cube, rec], ['original', f'reconstruction (order {order_used}, sqrt_f={sqrt_f})']
    title = f'{runid}   t = {tstep}   cell {cell_id}'
    os.makedirs(PLOTDIR, exist_ok=True)
    for name, fig in ((f'cuts_{DIM}D', plot_tools.vdf_figure(cubes, vlim, labels, dim=DIM, sp_th=sp_th,
                                                             vel_bounds=vel_bounds)),
                      ('projections', plot_tools.vdf_2d(cubes, vlim, labels, reduce='sum', vel_bounds=vel_bounds))):
        fig.suptitle(title)
        out = os.path.join(PLOTDIR, f'{stem}_{name}.png')
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f'saved -> {out}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else NPZ_FILE)
