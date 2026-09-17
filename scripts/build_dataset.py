"""
Parallel, multi-timestep dataset builder for the Hermite corrector.

Two-stage pipeline per bulk file:
  1. DECOMPOSITION (expensive, ~3.8s/cell at FULL_ORDER=22): full Hermite
     transform + reconstruct. Parallelized ONE PROCESS PER CELL (embarrassingly
     parallel, CPU-bound, no shared state) via multiprocessing.Pool. Each
     worker opens its own VlsvReader (readers aren't picklable/fork-safe to
     share) and returns (cid, coeffs_flat, f_rec_full, u, vth) for the
     parent to assemble into ONE decomposition_cache.py cache dict and save
     once -- avoids write contention from many processes hitting the same
     cache file.
  2. FEATURE EXTRACTION (cheap, <1s/cell): voxel sampling, 3x3x3 patches,
     C_low, axis_spectra, per-cell adaptive lambda -- serial, reusing the
     now-complete decomposition cache.

Validation split strategy -- see ml_corrector/EXPERIMENT_LOG.md's cross-timestep test:
splitting by CELL within one snapshot (Exp 1-10) only measures
within-snapshot generalization and hid a real overfitting problem (the
model scored 62% held-out within its training snapshot but -28% on an
unseen timestep). This script defaults to splitting by TIMESTEP: entire
--train-bulkfiles vs entirely separate --val-bulkfiles, so the saved
val.npz is a genuine cross-timestep generalization check, not just a
held-out-cells check. (An optional --val-fraction-within-train can also
carve out a same-snapshot dev set from the training files, e.g. for early
stopping diagnostics -- kept separate from the true cross-timestep val.)

Usage
-----
python3 scripts/build_dataset.py \
    --train-bulkfiles bulk.0000024.vlsv bulk.0000050.vlsv bulk.0000080.vlsv \
    --val-bulkfiles bulk.0000111.vlsv \
    --n-workers 8 \
    --output-name multi_snapshot_v1

Or via the wrapper (edit config at the top of the script):
    bash scripts/build_dataset.sh
"""

import argparse, os, sys, time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_processing.vdf_tools import (get_vdf_parameters, build_cube, get_drift_velocity_cube,
                       get_thermal_velocity_cube, get_sparse_threshold, velocity_axis)
from ml_corrector.corrector_model import (coeffs_dict_to_vector, pad_field, extract_patches,
                                        axis_spectra)
from data_processing.adaptive_hermite import (adaptive_transform, reconstruct,
                                         coeffs_to_array, array_to_coeffs)
from ml_corrector.decomposition_cache import cache_path_for, load_cache, save_cache


# ===
# Stage 1: parallel decomposition
# ===

def _decompose_one_cell(args):
    """
    Worker: full Hermite decomposition for ONE (bulkfile, cell_id). Runs in
    its own process -- opens a fresh VlsvReader (not shared across processes).
    Returns None for empty cells, else a plain-data tuple (picklable).
    """
    bulkdir, bulkfile, cid, full_order = args
    import pytools as pt
    fpath = os.path.join(bulkdir, bulkfile)
    reader = pt.vlsvfile.VlsvReader(fpath)
    vlim, vlen, dv = get_vdf_parameters(reader)
    sp_th = get_sparse_threshold(reader)

    cube = build_cube(cid, reader, vlim, vlen, dv)
    if cube.sum() < 1e-30:
        return None
    u = get_drift_velocity_cube(cube, vlim, vlen)
    vth = get_thermal_velocity_cube(cube, vlim, vlen, u)
    sparse_mask = cube >= sp_th

    coeffs, eps_rel, s_stop, hist = adaptive_transform(
        cube, vlim, vlen, vth, u, max_order=full_order, even_first=False,
        sp_th=sp_th, sparse_mask=sparse_mask, track_log_eps=False, verbose=False)
    f_rec_full = reconstruct(coeffs, vlim, vlen, vth, u, turning_point_cutoff=False)

    return (cid, coeffs_to_array(coeffs, full_order), f_rec_full.astype(np.float32),
           np.asarray(u, dtype=np.float64), float(vth))


def build_decomposition_cache_parallel(bulkdir, bulkfile, cell_ids, full_order,
                                       n_workers, use_cache=True, verbose=True):
    """
    Ensures a full decomposition_cache.py cache exists for every cid in
    cell_ids, computing missing ones in parallel (n_workers processes) and
    saving ONCE at the end. Returns the cache dict.
    """
    cache_path = cache_path_for(bulkdir, bulkfile, full_order)
    cache = load_cache(cache_path) if use_cache else {}
    missing = [cid for cid in cell_ids if cid not in cache]

    if missing:
        if verbose:
            print(f"  [{bulkfile}] decomposing {len(missing)}/{len(cell_ids)} "
                  f"missing cells with {n_workers} workers ...")
        t0 = time.time()
        tasks = [(bulkdir, bulkfile, cid, full_order) for cid in missing]
        with mp.Pool(n_workers) as pool:
            for result in pool.imap_unordered(_decompose_one_cell, tasks):
                if result is None:
                    continue
                cid, coeffs_flat, f_rec_full, u, vth = result
                cache[cid] = dict(coeffs_flat=coeffs_flat, f_rec_full=f_rec_full,
                                  u=u, vth=vth)
        if verbose:
            print(f"  [{bulkfile}] decomposed {len(missing)} cells in "
                  f"{time.time()-t0:.1f}s ({(time.time()-t0)/max(len(missing),1):.2f}s/cell)")
        if use_cache:
            save_cache(cache_path, cache)
    elif verbose:
        print(f"  [{bulkfile}] all {len(cell_ids)} cells already cached")

    return cache


# ===
# Stage 2: feature-array extraction (serial, cheap)
# ===

def build_feature_arrays(bulkdir, bulkfile, cell_ids, cache, s_low, full_order,
                         n_per_cell, lambda_max, lambda_power, ref_var, rng):
    """
    Returns per-cell dict: cid -> {'X', 'y', 'lam', 'mean_sq_cell', 'lambda_cell'}.
    Requires `cache` to already contain every cid (see stage 1).
    """
    import pytools as pt
    fpath = os.path.join(bulkdir, bulkfile)
    reader = pt.vlsvfile.VlsvReader(fpath)
    vlim, vlen, dv = get_vdf_parameters(reader)
    sp_th = get_sparse_threshold(reader)
    log_sp_th = float(np.log(sp_th))
    v_ax = velocity_axis(vlim, vlen, dv)
    Vz, Vy, Vx = np.meshgrid(v_ax, v_ax, v_ax, indexing='ij')

    data = {}
    for cid in cell_ids:
        if cid not in cache:
            continue  # was an empty cell in stage 1
        entry = cache[cid]
        coeffs = array_to_coeffs(entry['coeffs_flat'], full_order)
        f_rec_full = entry['f_rec_full']
        u = entry['u']
        vth = entry['vth']

        cube = build_cube(cid, reader, vlim, vlen, dv)
        sparse_mask = cube >= sp_th

        C_low = np.array(coeffs_dict_to_vector(coeffs, s_low), dtype=np.float32)
        spec = axis_spectra(coeffs, full_order)
        total_power_cell = float(spec[0].sum())
        spec_norm = spec / max(total_power_cell, 1e-30)
        log_spec_flat = np.log10(np.maximum(spec_norm, 1e-20)).astype(np.float32).flatten()
        cell_features = np.concatenate([C_low, log_spec_flat])

        log_true = np.log(np.maximum(cube, sp_th))
        log_rec  = np.log(np.maximum(f_rec_full, sp_th))
        delta_true = (log_true - log_rec).astype(np.float32)
        log_rec_padded = pad_field(log_rec.astype(np.float32), log_sp_th)

        mean_sq_cell = float(np.mean(delta_true[sparse_mask] ** 2))
        lambda_cell = lambda_max / (1.0 + (mean_sq_cell / ref_var) ** lambda_power)

        idx_inside = np.argwhere(sparse_mask)
        if len(idx_inside) == 0:
            continue
        n_take = min(n_per_cell, len(idx_inside))
        take = idx_inside[rng.choice(len(idx_inside), size=n_take, replace=False)]

        iz, iy, ix = take[:, 0], take[:, 1], take[:, 2]
        x_hat = (Vx[iz, iy, ix] - u[0]) / vth
        y_hat = (Vy[iz, iy, ix] - u[1]) / vth
        z_hat = (Vz[iz, iy, ix] - u[2]) / vth

        patches = extract_patches(log_rec_padded, iz, iy, ix)
        C_rep = np.tile(cell_features, (n_take, 1))
        X = np.concatenate([patches, C_rep,
                           x_hat[:, None], y_hat[:, None], z_hat[:, None]],
                           axis=1).astype(np.float32)
        y = delta_true[iz, iy, ix]
        lam = np.full(n_take, lambda_cell, dtype=np.float32)

        data[cid] = dict(X=X, y=y, lam=lam, mean_sq_cell=mean_sq_cell,
                         lambda_cell=lambda_cell)
    return data


def get_cell_ids(bulkdir, bulkfile):
    import pytools as pt
    reader = pt.vlsvfile.VlsvReader(os.path.join(bulkdir, bulkfile))
    return [int(c) for c in reader.read(mesh='SpatialGrid', tag='CELLSWITHBLOCKS')]


# ===
# Orchestration
# ===

def build_split(bulkdir, bulkfiles, s_low, full_order, n_per_cell,
                lambda_max, lambda_power, ref_var, n_workers, seed,
                val_fraction_within=0.0, use_cache=True):
    """
    Builds concatenated (X, y, lam) arrays across all cells of `bulkfiles`.
    If val_fraction_within > 0, ALSO carves out that fraction of cells
    (per bulkfile) into a second "internal dev" split -- useful for a
    same-snapshot early-stopping signal separate from true cross-timestep
    val. Returns (main_dict, dev_dict_or_None).
    """
    rng = np.random.default_rng(seed)
    main_X, main_y, main_lam = [], [], []
    dev_X, dev_y, dev_lam = [], [], []

    for bulkfile in bulkfiles:
        cell_ids = get_cell_ids(bulkdir, bulkfile)
        print(f"[{bulkfile}] {len(cell_ids)} cells with VDF")

        cache = build_decomposition_cache_parallel(
            bulkdir, bulkfile, cell_ids, full_order, n_workers, use_cache)

        if val_fraction_within > 0:
            shuffled = rng.permutation(cell_ids)
            n_dev = max(1, int(len(shuffled) * val_fraction_within))
            dev_cells = set(shuffled[:n_dev].tolist())
            main_cells = [c for c in cell_ids if c not in dev_cells]
        else:
            main_cells = cell_ids
            dev_cells = set()

        data = build_feature_arrays(bulkdir, bulkfile, cell_ids, cache,
                                    s_low, full_order, n_per_cell,
                                    lambda_max, lambda_power, ref_var, rng)
        for cid, d in data.items():
            if cid in dev_cells:
                dev_X.append(d['X']); dev_y.append(d['y']); dev_lam.append(d['lam'])
            else:
                main_X.append(d['X']); main_y.append(d['y']); main_lam.append(d['lam'])

    main = dict(X=np.concatenate(main_X, axis=0),
               y=np.concatenate(main_y, axis=0),
               lam=np.concatenate(main_lam, axis=0))
    dev = None
    if val_fraction_within > 0 and dev_X:
        dev = dict(X=np.concatenate(dev_X, axis=0),
                  y=np.concatenate(dev_y, axis=0),
                  lam=np.concatenate(dev_lam, axis=0))
    return main, dev


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bulkdir', default='reconnection_2d_beta025')
    ap.add_argument('--train-bulkfiles', nargs='+', required=True,
                    help='Bulk files used for TRAINING (feature arrays saved to train.npz)')
    ap.add_argument('--val-bulkfiles', nargs='+', default=[],
                    help='Bulk files held out ENTIRELY for cross-timestep validation '
                         '(saved to val.npz). Should not overlap --train-bulkfiles.')
    ap.add_argument('--val-fraction-within-train', type=float, default=0.0,
                    help='Optionally also carve out this fraction of cells from the '
                         'TRAINING bulkfiles as a same-snapshot dev set (dev.npz), '
                         'e.g. for early-stopping diagnostics. Default 0 (off) -- '
                         'the real signal should come from --val-bulkfiles instead.')
    ap.add_argument('--s-low', type=int, default=2)
    ap.add_argument('--full-order', type=int, default=22)
    ap.add_argument('--n-per-cell', type=int, default=3000)
    ap.add_argument('--lambda-max', type=float, default=20.0)
    ap.add_argument('--lambda-power', type=float, default=2.0)
    ap.add_argument('--ref-var', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-workers', type=int, default=8)
    ap.add_argument('--no-cache', action='store_true',
                    help='Ignore/rebuild the decomposition_cache.py cache from scratch')
    ap.add_argument('--output-name', default='dataset',
                    help='Output subfolder under hermite_ml/datasets/')
    args = ap.parse_args()

    overlap = set(args.train_bulkfiles) & set(args.val_bulkfiles)
    if overlap:
        print(f"WARNING: bulkfiles in both train and val: {overlap} -- "
              f"this defeats the cross-timestep validation purpose.")

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'ml_corrector', 'datasets', args.output_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Building TRAIN split from {args.train_bulkfiles} ===")
    train_data, dev_data = build_split(
        args.bulkdir, args.train_bulkfiles, args.s_low, args.full_order,
        args.n_per_cell, args.lambda_max, args.lambda_power, args.ref_var,
        args.n_workers, args.seed, args.val_fraction_within_train,
        use_cache=not args.no_cache)
    np.savez_compressed(os.path.join(out_dir, 'train.npz'), **train_data)
    print(f"train.npz: {train_data['X'].shape[0]} samples, "
          f"dim={train_data['X'].shape[1]}")

    if dev_data is not None:
        np.savez_compressed(os.path.join(out_dir, 'dev.npz'), **dev_data)
        print(f"dev.npz (same-snapshot, within train bulkfiles): "
              f"{dev_data['X'].shape[0]} samples")

    if args.val_bulkfiles:
        print(f"\n=== Building VAL split (cross-timestep) from {args.val_bulkfiles} ===")
        val_data, _ = build_split(
            args.bulkdir, args.val_bulkfiles, args.s_low, args.full_order,
            args.n_per_cell, args.lambda_max, args.lambda_power, args.ref_var,
            args.n_workers, args.seed, 0.0, use_cache=not args.no_cache)
        np.savez_compressed(os.path.join(out_dir, 'val.npz'), **val_data)
        print(f"val.npz: {val_data['X'].shape[0]} samples")
    else:
        print("\nNo --val-bulkfiles given -- no cross-timestep val.npz written "
              "(corrector_train.py will need dev.npz or its own split instead).")

    # Save the config used, for reproducibility / for corrector_train.py to read back.
    np.savez(os.path.join(out_dir, 'config.npz'),
             s_low=args.s_low, full_order=args.full_order,
             n_per_cell=args.n_per_cell, lambda_max=args.lambda_max,
             lambda_power=args.lambda_power, ref_var=args.ref_var,
             seed=args.seed, train_bulkfiles=args.train_bulkfiles,
             val_bulkfiles=args.val_bulkfiles, bulkdir=args.bulkdir)
    print(f"\nSaved dataset -> {out_dir}")


if __name__ == '__main__':
    main()
