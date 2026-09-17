"""
Run the adaptive Hermite transform (rel_threshold-based stopping) on N
random cells from a bulk file, to check how order-at-convergence and
eps_rel behave across the domain.

Usage
-----
python3 hermite_ml/run_random_cells.py   # run from the repo root
"""

import os, sys
import numpy as np
import pytools as pt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vdf_tools import (get_vdf_parameters, build_cube,
                       get_drift_velocity_cube, get_thermal_velocity_cube,
                       get_sparse_threshold)
from hermite_ml.adaptive_hermite import adaptive_transform, total_coeffs

# ===
# CONFIG
# ===
BULKDIR       = 'reconnection_2d_beta025'
BULKFILE      = 'bulk.0000024.vlsv'
N_CELLS       = 10
MAX_ORDER     = 22
REL_THRESHOLD = 0.04
SEED          = 42


def main():
    fpath = os.path.join(BULKDIR, BULKFILE)
    reader = pt.vlsvfile.VlsvReader(fpath)
    vlim, vlen, dv = get_vdf_parameters(reader)
    sp_th = get_sparse_threshold(reader)

    cells_with_vdf = reader.read(mesh='SpatialGrid', tag='CELLSWITHBLOCKS')
    n_total = len(cells_with_vdf)
    print(f"File: {fpath}")
    print(f"vlen={vlen}  vlim={vlim/1e3:.0f} km/s  sp_th={sp_th:.1e}")
    print(f"Cells with VDF: {n_total}\n")

    rng = np.random.default_rng(SEED)
    idxs = rng.choice(n_total, size=N_CELLS, replace=False)

    print(f"{'idx':>4} {'cid':>5} {'x[km]':>8} {'z[km]':>8} {'vth[km/s]':>10} "
          f"{'s_stop':>7} {'n_coef':>7} {'eps_rel':>8} {'eps_log':>8}")
    print("-" * 78)

    results = []
    for idx in sorted(idxs):
        cid = int(cells_with_vdf[idx])
        coords = reader.get_cell_coordinates(cid)
        cube = build_cube(cid, reader, vlim, vlen, dv)
        u    = get_drift_velocity_cube(cube, vlim, vlen)
        vth  = get_thermal_velocity_cube(cube, vlim, vlen, u)
        sparse_mask = cube >= sp_th

        coeffs, eps_final, s_stop, history = adaptive_transform(
            cube, vlim, vlen, vth, u,
            max_order=MAX_ORDER,
            rel_threshold=REL_THRESHOLD,
            even_first=False,
            sp_th=sp_th,
            sparse_mask=sparse_mask,
            verbose=False,
        )
        eps_log_final = history[-1][3] if history else float('nan')
        results.append((idx, cid, coords, vth, s_stop, len(coeffs), eps_final, eps_log_final))
        print(f"{idx:4d} {cid:5d} {coords[0]/1e3:8.1f} {coords[2]/1e3:8.1f} "
              f"{vth/1e3:10.2f} {s_stop:7d} {len(coeffs):7d} "
              f"{eps_final:8.5f} {eps_log_final:8.5f}")

    print("-" * 78)
    s_stops = [r[4] for r in results]
    n_coefs = [r[5] for r in results]
    print(f"s_stop:  min={min(s_stops)}  max={max(s_stops)}  mean={np.mean(s_stops):.1f}")
    print(f"n_coefs: min={min(n_coefs)}  max={max(n_coefs)}  mean={np.mean(n_coefs):.0f}  "
          f"(vs {total_coeffs(MAX_ORDER)} at MAX_ORDER)")


if __name__ == '__main__':
    main()
