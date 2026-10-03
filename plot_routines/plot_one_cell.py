"""Original vs reconstructed VDF for a cell saved by runners/run_one_cell.py.

    python plot_routines/plot_one_cell.py [data/one_cell_run_BIE_tstep1125.npz]

Run id and time step come from the file name; coefficients, u, vth, vlim, vlen, bounding box and cell id come
from the npz. The spectrum may be saved sparse (a pickled {(l, m, n): c} dict) or dense (a plain array, flag
`dense` in the npz); both are handled. The reconstruction is rebuilt from the saved coefficients, the original
VDF is re-read from the bulk file.
"""
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/ivanzait/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytools as pt
from data_processing import vdf_tools as vt
from plot_routines.plot_tools import original_vs_reconstruction

plt.rcParams['text.usetex'] = False   # importing analysator switches it on; there is no latex here

NPZ_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'one_cell_run_BIE_tstep1125.npz')
BULK_DIR = "/turso/group/spacephysics/vlasiator/data/L0/2D"     # <BULK_DIR>/<runid>/bulk.<tstep, 7 digits>.vlsv
AXIS     = 0                                                    # axis summed over (0: the vx-vy plane)
PLOTDIR  = os.path.join(os.path.dirname(__file__), '..', 'plots')


def load_coeffs(hermite_coeffs, dense, order_used):
    """({(l, m, n): c}, order) for reconstruct_vdf_adaptive. A dense spectrum is a (MAX_ORDER,)*3 cube, zero-padded
    beyond order_used, so its side is the order to use; a sparse one is a pickled dict reconstructed at order_used."""
    if dense:
        return dict(np.ndenumerate(hermite_coeffs)), hermite_coeffs.shape[0]
    return hermite_coeffs.item(), order_used


def main(npz_file):
    stem = os.path.splitext(os.path.basename(npz_file))[0]
    m = re.fullmatch(r'one_cell_run_(.+)_tstep(\d+)', stem)
    if m is None:
        sys.exit(f"cannot read run id and time step from '{stem}' (expected one_cell_run_<runid>_tstep<N>)")
    runid, tstep = m.group(1), int(m.group(2))

    d = np.load(npz_file, allow_pickle=True)                    # a sparse spectrum is a pickled dict
    if 'dense' not in d.files:
        sys.exit(f"{npz_file} has no 'dense' flag: written by an older run_one_cell.py, run that again")
    dense, order_used = bool(d['dense']), int(d['order_used'])
    coeffs, order = load_coeffs(d['hermite_coeffs'], dense, order_used)
    print(f"{runid} t={tstep}: {'dense' if dense else 'sparse'} spectrum, order_used={order_used}")
    vlim, vlen, cell_id = float(d['vlim']), int(d['vlen']), int(d['cellids'])

    rec = vt.reconstruct_vdf_adaptive(coeffs, vlim, vlen, order, float(d['v_ths']), d['v_means'])
    rec = vt.apply_bounding_box(rec, d['bbox_idx'])

    reader = pt.vlsvfile.VlsvReader(f'{BULK_DIR}/{runid}/bulk.{tstep:07d}.vlsv')
    cube = vt.build_cube(cell_id, reader, vlim, vlen, 2 * vlim / vlen)

    fig = original_vs_reconstruction(cube, rec, vlim, axis=AXIS,
                                     labels=('original', f'reconstruction (order {order_used} + bbox)'))
    fig.suptitle(f'{runid}   t = {tstep}   cell {cell_id}')
    os.makedirs(PLOTDIR, exist_ok=True)
    out = os.path.join(PLOTDIR, f'{stem}_original_vs_reconstruction.png')
    fig.savefig(out, dpi=130)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else NPZ_FILE)
