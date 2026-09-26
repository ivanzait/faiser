"""
Run the adaptive Hermite transform (rel_threshold-based stopping) on N
random cells from a bulk file, to check how order-at-convergence and
eps_rel behave across the domain.

Usage
-----
python3 scripts/run_random_cells.py   # run from the repo root
"""

import os, sys
import numpy as np
sys.path.insert(0, "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator")
import pytools as pt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
import vdf_tools as vt
# from data_processing.vdf_tools import (get_vdf_parameters, build_cube,
#                         get_drift_velocity_cube, get_thermal_velocity_cube,
#                         get_sparse_threshold, adaptive_transform)
# from data_processing.adaptive_hermite import adaptive_transform, total_coeffs

# ===
# CONFIG
# ===
BULKDIR       = '/Users/ivanzait/Downloads/'
BULKFILE      = 'bulk.0000055.vlsv'
N_CELLS       = 10
H_ORDER     = 22
REL_THRESHOLD = 0.1
SEED          = 1812


def main():
    
    fpath = os.path.join(BULKDIR, BULKFILE)
    reader = pt.vlsvfile.VlsvReader(fpath)
    
    vlim, vlen, dv = vt.get_vdf_parameters(reader)
    sp_th = vt.get_sparse_threshold(reader)

    cells_with_vdf = reader.read(mesh='SpatialGrid', tag='CELLSWITHBLOCKS')
    n_total = len(cells_with_vdf)    
    print(f"File: {fpath}")
    print(f"vlen={vlen}  vlim={vlim/1e3:.0f} km/s  sp_th={sp_th:.1e}")
    print(f"Cells with VDF: {n_total}\n")

    rng = np.random.default_rng(SEED)
    idxs = rng.choice(n_total, size=N_CELLS, replace=False)

    for idx in sorted(idxs):
        cid = int(cells_with_vdf[idx]); print(cid)
        coords = reader.get_cell_coordinates(cid)
        cube = vt.build_cube(cid, reader, vlim, vlen, dv)
        u    = vt.get_drift_velocity_cube(cube, vlim, vlen)
        vth  = vt.get_thermal_velocity_cube(cube, vlim, vlen, u)
        sparse_mask = cube >= sp_th                        
        Hspectra, s_stop, delta = vt.adaptive_transform(cube, vlim, vlen, vth, u, max_order=H_ORDER, tolerance=0.01)
        print(s_stop)


if __name__ == '__main__':
    main()
