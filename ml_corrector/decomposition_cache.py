"""
Disk cache for the expensive part of per-cell dataset building: the full
Hermite decomposition (adaptive_transform at FULL_ORDER + reconstruct).

This is the dominant cost of build_dataset() in data_processing.py
(~2.7s adaptive_transform + ~1.1s reconstruct per cell at FULL_ORDER=22,
so ~8min for the 128-cell reconnection_2d_beta025 dataset). Everything
downstream (voxel sampling, patch extraction, C_low subset) is cheap
(<1s/cell) and depends on config knobs (N_PER_CELL, S_LOW, SEED) that are
worth iterating on WITHOUT paying the coefficient-computation cost again --
hence caching coeffs+f_rec_full separately from the sampled training arrays.

Cache key is (BULKDIR simulation id, BULKFILE basename, FULL_ORDER) -- one
.npz per combination, keyed internally by cell_id, stored under
hermite_ml/data/<simulation_id>/. Invalidate by deleting the file (or
bumping FULL_ORDER) if the underlying bulk file or transform logic changes.

Usage
-----
    path = cache_path_for(bulkdir, bulkfile, full_order)
    cache = load_cache(path)              # {} if no cache file yet
    coeffs, f_rec_full, hit = get_or_compute(
        cid, cube, vlim, vlen, dv, vth, u, sp_th, sparse_mask, full_order, cache)
    ...
    save_cache(path, cache)               # call once after the loop
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_processing.adaptive_hermite import (adaptive_transform, reconstruct,
                                         coeffs_to_array, array_to_coeffs)


def cache_path_for(bulkdir, bulkfile, full_order, data_dir=None):
    """
    bulkdir identifies the simulation (e.g. 'reconnection_2d_beta025') and
    becomes the cache subfolder name: hermite_ml/data/<bulkdir>/full_decomp_
    <bulkfile>_order<full_order>.npz. `data_dir` overrides the base
    'hermite_ml/data' directory if given.
    """
    data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'data')
    simulation_id = os.path.basename(os.path.normpath(bulkdir))
    sim_dir = os.path.join(data_dir, simulation_id)
    os.makedirs(sim_dir, exist_ok=True)
    safe_name = os.path.splitext(os.path.basename(bulkfile))[0]
    return os.path.join(sim_dir, f'full_decomp_{safe_name}_order{full_order}.npz')


def load_cache(path):
    """Returns {cell_id: {'coeffs_flat', 'f_rec_full', 'u', 'vth'}}, or {} if absent."""
    if not os.path.exists(path):
        return {}
    npz = np.load(path)
    cell_ids = npz['cell_ids']
    cache = {}
    for i, cid in enumerate(cell_ids):
        cache[int(cid)] = dict(
            coeffs_flat=npz['coeffs_flat'][i],
            f_rec_full=npz['f_rec_full'][i],
            u=npz['u'][i],
            vth=float(npz['vth'][i]),
        )
    print(f"  [cache] loaded {len(cache)} cells from {path}")
    return cache


def save_cache(path, cache):
    if not cache:
        return
    cell_ids = np.array(sorted(cache.keys()), dtype=np.int64)
    coeffs_flat = np.stack([cache[c]['coeffs_flat'] for c in cell_ids])
    f_rec_full  = np.stack([cache[c]['f_rec_full'] for c in cell_ids]).astype(np.float32)
    u   = np.stack([cache[c]['u'] for c in cell_ids])
    vth = np.array([cache[c]['vth'] for c in cell_ids], dtype=np.float64)
    np.savez_compressed(path, cell_ids=cell_ids, coeffs_flat=coeffs_flat,
                        f_rec_full=f_rec_full, u=u, vth=vth)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  [cache] saved {len(cache)} cells -> {path} ({size_mb:.1f} MB)")


def get_or_compute(cid, cube, vlim, vlen, dv, vth, u, sp_th, sparse_mask,
                   full_order, cache):
    """
    Returns (coeffs_dict, f_rec_full, cache_hit). Mutates `cache` in place on
    a miss (caller must call save_cache() after the loop to persist).
    """
    if cid in cache:
        entry = cache[cid]
        coeffs = array_to_coeffs(entry['coeffs_flat'], full_order)
        return coeffs, entry['f_rec_full'], True

    coeffs, eps_rel, s_stop, hist = adaptive_transform(
        cube, vlim, vlen, vth, u, max_order=full_order, even_first=False,
        sp_th=sp_th, sparse_mask=sparse_mask, track_log_eps=False, verbose=False)
    f_rec_full = reconstruct(coeffs, vlim, vlen, vth, u, turning_point_cutoff=False)

    cache[cid] = dict(
        coeffs_flat=coeffs_to_array(coeffs, full_order),
        f_rec_full=f_rec_full.astype(np.float32),
        u=np.asarray(u, dtype=np.float64),
        vth=float(vth),
    )
    return coeffs, f_rec_full, False
