"""
Cross-timestep validation: evaluate the trained corrector (fit on
BULKFILE_TRAIN's snapshot) on a DIFFERENT bulk file/timestep it has never
seen. Tests whether the learned correction generalizes across the
simulation's time evolution, not just across held-out cells of the same
snapshot (which is all corrector_eval.py checks).

Usage
-----
python3 scripts/corrector_validate.py
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import torch
torch.set_num_threads(4)
import pytools as pt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_processing.vdf_tools import (get_vdf_parameters, build_cube, get_drift_velocity_cube,
                       get_thermal_velocity_cube, get_sparse_threshold,
                       velocity_axis)
from ml_corrector.corrector_model import (HermiteCorrectorMLP, coeffs_dict_to_vector,
                                        pad_field, extract_patches, axis_spectra)
from ml_corrector.hermite_plots import plot_cube_comparison, plot_cube_1d_cuts
from ml_corrector.decomposition_cache import cache_path_for, load_cache, save_cache, get_or_compute

ML_CORRECTOR_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml_corrector')
BULKDIR    = 'reconnection_2d_beta025'
BULKFILE_VAL = 'bulk.0000111.vlsv'   # NOT the training file (bulk.0000024.vlsv)
S_LOW      = 2
MODEL_PATH = os.path.join(ML_CORRECTOR_DIR, 'corrector_weights.pt')
PLOTDIR    = os.path.join(ML_CORRECTOR_DIR, 'plots')
USE_CACHE  = True
N_PLOT     = 4   # plot this many cells: worst, best, median, and one lobe-like


def main():
    os.makedirs(PLOTDIR, exist_ok=True)
    fpath = os.path.join(BULKDIR, BULKFILE_VAL)
    print(f"Validation file (UNSEEN timestep): {fpath}")
    reader = pt.vlsvfile.VlsvReader(fpath)
    vlim, vlen, dv = get_vdf_parameters(reader)
    sp_th = get_sparse_threshold(reader)
    log_sp_th = float(np.log(sp_th))
    v_ax = velocity_axis(vlim, vlen, dv)
    Vz, Vy, Vx = np.meshgrid(v_ax, v_ax, v_ax, indexing='ij')

    cells_with_vdf = reader.read(mesh='SpatialGrid', tag='CELLSWITHBLOCKS')
    cell_ids = [int(c) for c in cells_with_vdf]
    print(f"Cells with VDF in {BULKFILE_VAL}: {len(cell_ids)}")

    ckpt = torch.load(MODEL_PATH)
    model = HermiteCorrectorMLP(n_features=ckpt['n_features'], hidden=ckpt['hidden'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    full_order = ckpt.get('full_order', 22)
    print(f"Loaded model: s_low={ckpt.get('s_low', S_LOW)}  full_order={full_order}  "
          f"n_features={ckpt['n_features']}  (trained on bulk.0000024.vlsv)")

    cache_path = cache_path_for(BULKDIR, BULKFILE_VAL, full_order)
    cache = load_cache(cache_path) if USE_CACHE else {}

    print(f"\n{'cid':>5} {'eps_log_base':>13} {'eps_log_corr':>13} {'improvement':>12}")
    print("-" * 48)

    results = []
    for cid in cell_ids:
        cube = build_cube(cid, reader, vlim, vlen, dv)
        u    = get_drift_velocity_cube(cube, vlim, vlen)
        vth  = get_thermal_velocity_cube(cube, vlim, vlen, u)
        sparse_mask = cube >= sp_th
        if cube.sum() < 1e-30:
            continue

        coeffs, f_rec_full, hit = get_or_compute(
            cid, cube, vlim, vlen, dv, vth, u, sp_th, sparse_mask, full_order, cache)
        C_low = np.array(coeffs_dict_to_vector(coeffs, S_LOW), dtype=np.float32)

        spec = axis_spectra(coeffs, full_order)
        total_power_cell = float(spec[0].sum())
        spec_norm = spec / max(total_power_cell, 1e-30)
        log_spec_flat = np.log10(np.maximum(spec_norm, 1e-20)).astype(np.float32).flatten()
        cell_features = np.concatenate([C_low, log_spec_flat])

        log_true = np.log(np.maximum(cube, sp_th))
        log_base = np.log(np.maximum(f_rec_full, sp_th))
        eps_base = np.sqrt(np.mean((log_true[sparse_mask] - log_base[sparse_mask])**2))

        log_base_padded = pad_field(log_base.astype(np.float32), log_sp_th)
        idx = np.argwhere(sparse_mask)
        iz, iy, ix = idx[:, 0], idx[:, 1], idx[:, 2]
        x_hat = (Vx[iz, iy, ix] - u[0]) / vth
        y_hat = (Vy[iz, iy, ix] - u[1]) / vth
        z_hat = (Vz[iz, iy, ix] - u[2]) / vth
        patches = extract_patches(log_base_padded, iz, iy, ix)
        C_rep = np.tile(cell_features, (len(idx), 1))
        X = np.concatenate([patches, C_rep,
                           x_hat[:, None], y_hat[:, None], z_hat[:, None]],
                           axis=1).astype(np.float32)
        with torch.no_grad():
            delta_pred = model(torch.from_numpy(X)).numpy()

        log_corr_masked = log_base[iz, iy, ix] + delta_pred
        eps_corr = np.sqrt(np.mean((log_true[iz, iy, ix] - log_corr_masked)**2))

        improvement = (eps_base - eps_corr) / eps_base * 100 if eps_base > 0 else float('nan')
        results.append((cid, eps_base, eps_corr, improvement, log_corr_masked, iz, iy, ix, cube, f_rec_full, u))
        print(f"{cid:5d} {eps_base:13.4f} {eps_corr:13.4f} {improvement:11.1f}%")

    if USE_CACHE:
        save_cache(cache_path, cache)

    base_list = [r[1] for r in results]
    corr_list = [r[2] for r in results]
    print("-" * 48)
    print(f"mean eps_log_base = {np.mean(base_list):.4f}")
    print(f"mean eps_log_corr = {np.mean(corr_list):.4f}")
    print(f"mean improvement  = {(1 - np.mean(corr_list)/np.mean(base_list))*100:.1f}%")
    print(f"n_cells evaluated = {len(results)}")

    # plot a representative spread: worst, best, median, and a lobe-like (base~0) cell
    by_base = sorted(results, key=lambda r: r[1])
    picks = {
        'lowest_base (lobe-like)': by_base[0],
        'median_base': by_base[len(by_base)//2],
        'highest_base (hardest)': by_base[-1],
    }
    for label, r in picks.items():
        cid, eps_base, eps_corr, improvement, log_corr_masked, iz, iy, ix, cube, f_rec_full, u = r
        sparse_mask = cube >= sp_th
        base_cube_full = np.zeros_like(cube)
        base_cube_full[sparse_mask] = np.maximum(f_rec_full[sparse_mask], 0.0)
        corr_cube_full = np.zeros_like(cube)
        corr_cube_full[iz, iy, ix] = np.exp(log_corr_masked)
        coords_label = f"cid={cid} ({label}) eps_base={eps_base:.3f} eps_corr={eps_corr:.3f}"
        plot_cube_comparison(
            cube, [base_cube_full, corr_cube_full],
            [f'base (full, order={full_order})', 'corrected'],
            vlim, sp_th=sp_th, title=coords_label,
            save=os.path.join(PLOTDIR, f'validate_2d_{cid}.png'))
        plot_cube_1d_cuts(
            cube, [base_cube_full, corr_cube_full],
            [f'base (full, order={full_order})', 'corrected'],
            vlim, vlen, u, title=coords_label,
            save=os.path.join(PLOTDIR, f'validate_1d_{cid}.png'))
        print(f"  plotted {label}: cid={cid}")


if __name__ == '__main__':
    main()
