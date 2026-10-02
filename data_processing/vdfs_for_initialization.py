"""
Build a VDF-initialization dataset: adaptive Hermite decomposition (baked
into a dense (HN,HN,HN) cube, so every cell's hermite array has the same
shape and they stack into one array) for a set of cells, saved into one
compressed npz.

Parallelized the same way as profile_bulk.py: one process per cell, each
with its own file reader (readers aren't picklable/fork-safe to share).

NOTE on SAVE_DENSE_VDF: the raw dense VDF cube is ~vlen^3 floats/cell --
at vlen=240 that's ~55MB/cell, so saving it for many cells adds up fast
(e.g. ~1TB for all ~18k VDF cells of the BIE run). Keep N_CELLS small
while SAVE_DENSE_VDF=True, or set SAVE_DENSE_VDF=False (keeping only the
hermite coefficients + moments, which is what the compression is for) to
scale up to more cells.

Usage
-----
python3 data_processing/vdfs_for_initialization.py   # run from the repo root

Tunable parameters are in the CONFIG block below.
"""

import os, sys
import multiprocessing as mp
import numpy as np

sys.path.insert(0, "/home/ivanzait/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytools as pt
from data_processing import vdf_tools as vt

# ===
# CONFIG -- edit these
# ===

BULK_FILE = "/wrk-vakka/group/spacephysics/vlasiator/2D/BIE/bulk.0001125.vlsv"
OUTDIR    = os.path.join(os.path.dirname(__file__), '..', 'data_initialization')
HN        = 20      # fixed Hermite order -> every cell's hermite cube has shape (HN,HN,HN)
TOLERANCE = 0.05     # adaptive_transform stops early once the Parseval delta < this

GLOBAL_RUN = True    # skip inner-boundary cells with no real VDF -- see profile_bulk.py
SAVE_DENSE_VDF = True   # see NOTE above before raising N_CELLS much past a few dozen

N_WORKERS = max(1, (os.cpu_count() or 2) - 2)   # 1 = serial
N_CELLS   = 20       # None = all cells with a VDF; an int = random subsample of that size
SEED      = 0
LOG_EVERY = 50

print('N_WORKERS:', N_WORKERS)

_reader = None   # one reader and one set of mesh parameters per process
_params = None
_sp_th = None    # sparsity threshold, defines the VDF support for the bounding box


def _init_worker():
    global _reader, _params, _sp_th
    _reader = pt.vlsvfile.VlsvReader(BULK_FILE)
    _params = vt.get_vdf_parameters(_reader)
    _sp_th = vt.get_sparse_threshold(_reader)

def process_cell(cid):
    """Decompose one cell; returns a result dict, or None if it can't be processed."""
    reader = _reader
    vlim, vlen, dv = _params
    cube = vt.build_cube(cid, reader, vlim, vlen, dv)

    u = vt.get_drift_velocity_cube(cube, vlim, vlen)
    vth = vt.get_thermal_velocity_cube(cube, vlim, vlen, u)
    coeffs, order_used, deltas = vt.adaptive_transform(
        cube, vlim, vlen, vth, u, max_order=HN, tolerance=TOLERANCE)
    h_cube = vt.coeffs_into_cube(coeffs, HN)
    coords = reader.get_cell_coordinates(cid)

    # Tight bounding box of the VDF support, index space ((iz0,iz1),(iy0,iy1),(ix0,ix1)).
    # Stored so a reconstruction can be cropped to it (apply_bounding_box); the box can't
    # be recomputed from the coefficients. (0,0) box for an empty VDF.
    box = vt.get_vdf_bounding_box(cube, vlim, vlen, sp_th=_sp_th)
    bbox_idx = box['idx'] if box is not None else ((0, 0), (0, 0), (0, 0))

    row = dict(cellid=int(cid), coords=coords, u=u, vth=float(vth),
              hermite=h_cube, order_used=int(order_used),
              delta_final=float(deltas[order_used - 1]), bbox_idx=bbox_idx)
    if SAVE_DENSE_VDF:
        row['vdf'] = cube
    
    return row


def main():
    
    _init_worker()
    cells = np.atleast_1d(_reader.read(mesh="SpatialGrid", tag="CELLSWITHBLOCKS", name="proton"))

    if GLOBAL_RUN:
        skip = vt.get_inner_boundary_cells(_reader)
        n_before = len(cells)
        cells = cells[~np.isin(cells, skip)]
        n_removed = n_before - len(cells)
        if n_removed:
            print(f"GLOBAL_RUN: skipping {n_removed} inner-boundary cells ")

    if N_CELLS is not None and N_CELLS < len(cells):
        cells = np.random.default_rng(SEED).choice(cells, size=N_CELLS, replace=False)    
    cells = [int(c) for c in cells]
    
    if N_WORKERS > 1:
        pool = mp.Pool(N_WORKERS, initializer=_init_worker)
        results = pool.imap_unordered(process_cell, cells, chunksize=1)
    else:
        pool = None
        results = map(process_cell, cells)

    cellid_ar, coords_ar, vdf_ar = [], [], []
    v_mean_ar, v_th_ar, hermite_ar, order_used_ar, bbox_ar = [], [], [], [], []
    
    for i, row in enumerate(results, start=1):
        cellid_ar.append(row['cellid'])
        coords_ar.append(row['coords'])
        v_mean_ar.append(row['u'])
        v_th_ar.append(row['vth'])
        hermite_ar.append(row['hermite'])
        order_used_ar.append(row['order_used'])
        bbox_ar.append(row['bbox_idx'])
        if SAVE_DENSE_VDF:
            vdf_ar.append(row['vdf'])
        if i % LOG_EVERY == 0:
            print(f"  {i}/{len(cells)} processed")
    if pool is not None:
        pool.close()
        pool.join()


    os.makedirs(OUTDIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(BULK_FILE))[0]
    out_path = os.path.join(OUTDIR, f"{stem}_vdf_init_HN{HN}.npz")

    save_kwargs = dict(
        cellids=np.array(cellid_ar), coords=np.array(coords_ar),
        v_means=np.array(v_mean_ar), v_ths=np.array(v_th_ar),
        hermite_coeffs=np.array(hermite_ar), order_used=np.array(order_used_ar),
        bbox_idx=np.array(bbox_ar),   # (N, 3, 2): per axis (z, y, x), inclusive index bounds
        vlim=_params[0], vlen=_params[1],
    )
    
    if SAVE_DENSE_VDF:
        save_kwargs['vdfs'] = np.array(vdf_ar)
    np.savez_compressed(out_path, **save_kwargs)
    print(f"saved -> {out_path}")


if __name__ == '__main__':
    main()
