"""
Hermite VDF diagnostic -- run interactively to explore one bulk file.

Usage
-----
python3 scripts/run_diagnostic.py   # run from the repo root

Tunable parameters are in the CONFIG block below.
Generates three PNG files in ml_corrector/plots/.
"""

import os, sys
import numpy as np
import pytools as pt

# import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_processing.vdf_tools import (get_vdf_parameters, build_cube,
                       get_drift_velocity_cube, get_thermal_velocity_cube,
                       to_log_shifted, get_sparse_threshold)
from data_processing.adaptive_hermite import (adaptive_transform, cubic_transform,
                                          coeffs_to_array, total_coeffs, level_sizes)
from ml_corrector.hermite_plots import (plot_parseval_convergence,
                                       plot_spectra_slices,
                                       plot_vdf_comparison,
                                       plot_vdf_residuals,
                                       plot_vdf_1d_cuts)

# ===
# CONFIG -- edit these
# ===

BULKDIR  = 'reconnection_2d_beta025'
BULKFILE = 'bulk.0000024.vlsv'    # which file to analyse

# Pick a cell by index into the list of cells-with-VDF
# (0 = first cell, -1 = last, etc.)
CELL_IDX = 55   # cid=672, x=-22.6 z=-248.9 km -- current sheet center (non-Maxwellian)

MAX_ORDER     = 22      # hard ceiling on Hermite order
REL_THRESHOLD = 0.04    # stop as soon as eps_rel < this (sound now: eps_rel
                        # is monotonic after the velocity-grid fix)
EVEN_FIRST    = False   # sequential (default); True = even levels first

# Orders for the VDF reconstruction comparison plot
COMPARE_ORDERS = [4,10,20]

PLOTDIR = os.path.join(os.path.dirname(__file__), '..', 'ml_corrector', 'plots')

# ===
# MAIN
# ===

def main():
    os.makedirs(PLOTDIR, exist_ok=True)
    fpath = os.path.join(BULKDIR, BULKFILE)

    # 1. Open file and read velocity mesh parameters
    print(f"\n{'='*60}")
    print(f"File : {fpath}")
    reader = pt.vlsvfile.VlsvReader(fpath)
    vlim, vlen, dv = get_vdf_parameters(reader)
    print(f"vlen={vlen}  vlim={vlim/1e3:.0f} km/s  dv={dv/1e3:.2f} km/s")

    SP_TH = get_sparse_threshold(reader)   # read from file, do NOT hardcode
    print(f"sp_th (from file config) = {SP_TH:.3e}")

    n_tot  = total_coeffs(MAX_ORDER)
    n_even = sum(n for s, n in level_sizes(MAX_ORDER) if s % 2 == 0)
    print(f"Max order {MAX_ORDER}: {n_tot} tetrahedral coeffs  "
          f"({n_even} even-only,  {MAX_ORDER**3} cube-style)")

    # 2. Pick a cell
    cells_with_vdf = reader.read(mesh='SpatialGrid', tag='CELLSWITHBLOCKS')
    print(f"\nCells with VDF in file: {len(cells_with_vdf)}")

    cid    = int(cells_with_vdf[CELL_IDX])
    coords = reader.get_cell_coordinates(cid)
    print(f"Selected cell {cid}  "
          f"x={coords[0]/1e3:.1f}  z={coords[2]/1e3:.1f} km")

    # 3. Build VDF cube
    print("\nBuilding VDF cube ... ", end='', flush=True)
    cube = build_cube(cid, reader, vlim, vlen, dv)
    u    = get_drift_velocity_cube(cube, vlim, vlen)
    vth  = get_thermal_velocity_cube(cube, vlim, vlen, u)
    print(f"done.  vth={vth/1e3:.1f} km/s  "
          f"u=[{u[0]/1e3:.1f}, {u[1]/1e3:.1f}, {u[2]/1e3:.1f}] km/s")
    print(f"  n = {cube.sum() * (2*vlim/vlen)**3:.3e} m^-3  "
          f"  non-zero cells: {np.count_nonzero(cube)}/{vlen**3}")

    log_cube    = to_log_shifted(cube, SP_TH)   # kept for reference / logging only
    sparse_mask = cube >= SP_TH                 # exact structural support mask

    # 4. Adaptive Hermite transform -- expand f directly (not log_cube).
    # Stop as soon as eps_rel < REL_THRESHOLD -- sound now that the
    # velocity-grid fix makes eps_rel a genuine, monotonic convergence measure.
    print(f"\nComputing adaptive Hermite transform in f-space "
          f"(max_order={MAX_ORDER}, rel_threshold={REL_THRESHOLD}) ...")
    print(f"  {'s':>3}  {'parity':>5}  {'n_new':>6}  {'n_cum':>7}  "
          f"{'eps_rel':>8}  {'eps_log':>8}")
    print("  " + "-"*55)

    coeffs, eps_final, s_stop, history = adaptive_transform(
        cube, vlim, vlen, vth, u,
        max_order=MAX_ORDER,
        rel_threshold=REL_THRESHOLD,
        even_first=EVEN_FIRST,
        sp_th=SP_TH,
        sparse_mask=sparse_mask,
        verbose=False,
    )

    # print history ourselves -- cleaner than verbose=True
    n_cum = 0
    for s, parity, eps_rel, eps_log, n_new in history:
        n_cum  += n_new
        marker = '  <-- STOP' if s == s_stop else ''
        print(f"  {s:3d}  {parity:>5}  {n_new:6d}  {n_cum:7d}  "
              f"{eps_rel:8.5f}  {eps_log:8.5f}{marker}")

    print(f"\nStopped at s={s_stop},  eps_rel={eps_final:.5f},  "
          f"coefficients used: {len(coeffs)}/{n_tot}")

    # 4b. Cubic transform comparison
    # cubic_transform(N) and adaptive_transform(max_order) now share the same
    # index convention: indices l,m,n and total order s all run 0..N-1 /
    # 0..max_order-1. So calling cubic_transform with N=MAX_ORDER covers
    # exactly the same index range as the tetrahedral run above.
    print(f"\n{'='*60}")
    print(f"Cubic transform comparison (N per axis, N^3 coefficients):")
    print(f"  {'N':>4}  {'N^3':>6}  {'eps_cubic':>10}  |  tetra s<=N-1: {'n_tetra':>7}  eps_tetra(log)")
    print("  " + "-"*60)
    _, _, cubic_hist = cubic_transform(cube, vlim, vlen, vth, u, MAX_ORDER)
    for n_max, n_coeffs, eps_c in cubic_hist:
        # comparable tetrahedral: same max per-axis order N-1 == total order s
        s_tetra = n_max - 1
        tetra_eps_entry = next(
            (el for s, p, er, el, n in history if s == s_tetra and p == 'even'), None)
        n_tetra = total_coeffs(s_tetra)
        if tetra_eps_entry is not None:
            print(f"  {n_max:4d}  {n_coeffs:6d}  {eps_c:10.5f}  |  "
                  f"s<={s_tetra}: {n_tetra:7d}  {tetra_eps_entry:.5f}")
        else:
            print(f"  {n_max:4d}  {n_coeffs:6d}  {eps_c:10.5f}  |  "
                  f"s<={s_tetra}: {n_tetra:7d}  (not computed)")
    print(f"{'='*60}")

    # 5. Plot: Parseval convergence
    cell_label = f"cell {cid}  x={coords[0]/1e3:.0f} z={coords[2]/1e3:.0f} km"
    save_parseval = os.path.join(PLOTDIR, f'parseval_{cid}.png')
    print(f"\nPlotting Parseval convergence -> {save_parseval}")
    plot_parseval_convergence(history,
                              title=cell_label,
                              save=save_parseval)

    # 6. Plot: Hermite spectra slices
    save_spectra = os.path.join(PLOTDIR, f'spectra_{cid}.png')
    print(f"Plotting spectra slices -> {save_spectra}")
    plot_spectra_slices(coeffs, MAX_ORDER, title=cell_label, save=save_spectra)

    # 7. Compute truncated coefficient sets for comparison
    print(f"\nComputing reconstructions at orders {COMPARE_ORDERS} ...")
    compare_coeffs = []
    for N in COMPARE_ORDERS:
        trunc = {k: v for k, v in coeffs.items() if sum(k) <= N}
        compare_coeffs.append(trunc)
        n_trunc = len(trunc)
        from data_processing.adaptive_hermite import parseval_check, log_eps_check
        eps_f   = parseval_check(cube, trunc, 2 * vlim / vlen)
        eps_lg  = log_eps_check(cube, trunc, vlim, vlen, vth, u,
                                 sp_th=SP_TH, sparse_mask=sparse_mask)
        print(f"  N={N:2d}: {n_trunc:4d} coeffs,  "
              f"eps_rel(f)={eps_f:.4f}   eps_log={eps_lg:.4f}")

    # 8. Plot: VDF comparison
    labels = [f'N={N}' for N in COMPARE_ORDERS]

    save_vdf = os.path.join(PLOTDIR, f'vdf_comparison_{cid}.png')
    print(f"\nPlotting VDF comparison -> {save_vdf}")
    plot_vdf_comparison(cube, compare_coeffs, labels,
                        vlim, vlen, vth, u, sp_th=SP_TH,
                        title=cell_label, save=save_vdf)

    save_res = os.path.join(PLOTDIR, f'vdf_residuals_{cid}.png')
    print(f"Plotting residuals -> {save_res}")
    plot_vdf_residuals(cube, compare_coeffs, labels,
                       vlim, vlen, vth, u, sp_th=SP_TH,
                       title=cell_label, save=save_res)

    # 9. Plot: 1-D cuts through int-dz projection
    save_cuts = os.path.join(PLOTDIR, f'vdf_1d_cuts_{cid}.png')
    print(f"Plotting 1-D cuts -> {save_cuts}")
    plot_vdf_1d_cuts(cube, compare_coeffs, labels,
                     vlim, vlen, vth, u, sp_th=SP_TH,
                     title=cell_label, save=save_cuts)

    print(f"\n{'='*60}")
    print("Done. Plots saved to:", PLOTDIR)


if __name__ == '__main__':
    main()
