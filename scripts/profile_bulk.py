"""
Profile the adaptive Hermite transform over the VDFs of a bulk file.

For every cell with a proton VDF it records, separately:
  - t_read       : building the dense velocity cube from the file
  - t_moments    : bulk velocity u and thermal velocity vth
  - t_transform  : vt.adaptive_transform
plus order_used, n_coeffs, final Parseval delta and the number of non-empty voxels
(a cheap proxy for how complex the cell is). Results are written to a CSV, flushed
periodically so a long run can be inspected (or interrupted) safely.

Parallel mode (N_WORKERS > 1) spreads cells over processes, each with its own file
reader. Per-cell times are still measured inside the worker, but they are inflated
by contention (memory bandwidth, shared CPU). For clean per-cell timings use
N_WORKERS = 1; use parallel mode for total throughput.

Usage
-----
python3 scripts/profile_bulk.py   # run from the repo root

Tunable parameters are in the CONFIG block below.
"""

import os, sys, csv, time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytools as pt
from data_processing import vdf_tools as vt

# ===
# CONFIG -- edit these
# ===

BULK_FILE = "/Users/ivanzait/Downloads/bulk.0000055.vlsv"
MAX_ORDER = 20
TOLERANCE = 0.05

N_WORKERS = max(1, (os.cpu_count() or 2) - 2)   # 1 = serial (clean per-cell timings)
N_CELLS = None      # None = all cells with a VDF; an int = random subsample of that size
SEED    = 0
LOG_EVERY = 50      # progress line every N cells

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'profiling')

FIELDS = ['cellid', 'x', 'y', 'z', 'nonzero_voxels', 'vth', 'order_used', 'n_coeffs',
          'delta_final', 't_read', 't_moments', 't_transform']

_reader = None   # one reader and one set of mesh parameters per process
_params = None   # (vlim, vlen, dv), read once in _init_worker


def _init_worker():
    global _reader, _params
    _reader = pt.vlsvfile.VlsvReader(BULK_FILE)
    _params = vt.get_vdf_parameters(_reader)


def process_cell(cid):
    """Transform one cell; returns a result row, or None if the VDF cannot be read."""
    reader = _reader
    vlim, vlen, dv = _params

    t0 = time.perf_counter()
    try:
        cube = vt.build_cube(cid, reader, vlim, vlen, dv)
    except ValueError:
        return None
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    u   = vt.get_drift_velocity_cube(cube, vlim, vlen)
    vth = vt.get_thermal_velocity_cube(cube, vlim, vlen, u)
    t_moments = time.perf_counter() - t0

    t0 = time.perf_counter()
    coeffs, order_used, deltas = vt.adaptive_transform(
        cube, vlim, vlen, vth, u, max_order=MAX_ORDER, tolerance=TOLERANCE)
    t_transform = time.perf_counter() - t0

    x, y, z = reader.get_cell_coordinates(cid)
    return dict(cellid=int(cid), x=x, y=y, z=z,
                nonzero_voxels=int(np.count_nonzero(cube)), vth=float(vth),
                order_used=int(order_used), n_coeffs=len(coeffs),
                delta_final=float(deltas[order_used - 1]),
                t_read=t_read, t_moments=t_moments, t_transform=t_transform)


def main():
    t_init = time.perf_counter()
    _init_worker()
    print(f"\nreader + mesh parameters ready in {time.perf_counter() - t_init:.1f} s (per process)")
    cells = np.atleast_1d(_reader.read(mesh="SpatialGrid", tag="CELLSWITHBLOCKS", name="proton"))
    vlen = _params[1]
    if N_CELLS is not None and N_CELLS < len(cells):
        cells = np.random.default_rng(SEED).choice(cells, size=N_CELLS, replace=False)
    cells = [int(c) for c in cells]
    print(f"\n{len(cells)} cells | workers={N_WORKERS} | max_order={MAX_ORDER} "
          f"tolerance={TOLERANCE} | {BULK_FILE}")

    os.makedirs(OUTDIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(BULK_FILE))[0]
    csv_path = os.path.join(OUTDIR, f"{stem}_order{MAX_ORDER}_tol{TOLERANCE}.csv")

    if N_WORKERS > 1:
        pool = mp.Pool(N_WORKERS, initializer=_init_worker)
        results = pool.imap_unordered(process_cell, cells, chunksize=1)   # heavy-tailed cost
    else:
        pool = None
        results = map(process_cell, cells)

    rows, skipped = [], 0
    t_all = time.perf_counter()
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for i, row in enumerate(results, start=1):
            if row is None:
                skipped += 1
            else:
                rows.append(row)
                writer.writerow(row)
            if i % LOG_EVERY == 0:
                fh.flush()
                elapsed = time.perf_counter() - t_all
                eta = elapsed / i * (len(cells) - i)
                print(f"  {i:5d}/{len(cells)}  elapsed {elapsed/60:6.1f} min  ETA {eta/60:6.1f} min")
    if pool is not None:
        pool.close()
        pool.join()

    wall = time.perf_counter() - t_all
    if not rows:
        print("no cells processed")
        return

    tt = np.array([r['t_transform'] for r in rows])
    tr = np.array([r['t_read'] for r in rows])
    tm = np.array([r['t_moments'] for r in rows])
    ou = np.array([r['order_used'] for r in rows])
    nc = np.array([r['n_coeffs'] for r in rows])
    n_dense = vlen ** 3
    busy = tt.sum() + tr.sum() + tm.sum()

    print(f"\nprocessed {len(rows)} cells ({skipped} skipped) in {wall/60:.2f} min wall time")
    print(f"sum of per-cell times {busy/60:.2f} min -> average concurrency {busy/wall:.1f} "
          f"with {N_WORKERS} worker(s) (not a speedup: per-cell times inflate under load; "
          f"compare wall time against an N_WORKERS = 1 run)")
    print(f"{'':14s} {'total [s]':>10} {'mean [s]':>9} {'median':>8} {'p95':>8} {'max':>8}")
    for name, t in [('transform', tt), ('read cube', tr), ('moments', tm)]:
        print(f"{name:14s} {t.sum():10.1f} {t.mean():9.3f} {np.median(t):8.3f} "
              f"{np.percentile(t, 95):8.3f} {t.max():8.3f}")
    print(f"\norder_used: mean {ou.mean():.1f}, median {np.median(ou):.0f}, "
          f"min {ou.min()}, max {ou.max()}, hit max_order: {(ou == MAX_ORDER).sum()} cells")
    print(f"coefficients per cell: mean {nc.mean():.0f} | dense cube {n_dense} "
          f"-> mean compression {n_dense / nc.mean():.0f}x")
    print(f"\nCSV -> {csv_path}")


if __name__ == '__main__':
    main()
