"""
Adaptive Hermite transform + spectral window on ONE cell, with one figure:

  row 1 (velocity space, sum over PROJ_AXIS + a 1-D marginal)
      original VDF | reconstruction (no window) | reconstruction (windowed) | 1-D cut
  row 2 (Hermite space)
      spectrum | spectrum after window | power per level + window | Parseval delta

Spectrum panels show log10 of P[l,m] = sum_n C[l,m,n]^2, i.e. every
coefficient collapsed onto the (l,m) plane, so the whole tetrahedron is
visible on one log scale instead of a single C[l,m,n=0] slice.

The window is applied with max_order = order_used (the level adaptive_transform
actually stopped at), so it tapers to zero exactly where the series was cut.

Usage
-----
python3 scripts/run_one_cell.py                        # run from the repo root
python3 scripts/run_one_cell.py --order 10 --window raised_cosine
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, "/home/ivanzait/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytools as pt
from data_processing import vdf_tools as vt

# importing analysator turns on text.usetex; this script needs no LaTeX
plt.rcParams['text.usetex'] = False

# ===
# CONFIG -- edit these (or override on the command line)
# ===

BULK_FILE = "/turso/group/spacephysics/vlasiator/data/L0/2D/BIE/bulk.0001125.vlsv"
CELL_ID   = 795501
MAX_ORDER = 20
TOLERANCE = 0.05       # adaptive_transform stops once the Parseval delta < this
WINDOW    = 'lanczos'  # 'lanczos' or 'raised_cosine'
BBOX_MARGIN = 1        # cells added around the VDF support box

PROJ_AXIS = 0          # cube[iz,iy,ix]: 0 -> sum over z, shows the (vy,vx) plane
DECADES   = 8          # dynamic range of the log colour scales
PLOTDIR   = os.path.join(os.path.dirname(__file__), '..', 'ml_corrector', 'plots')


def reconstruct(coeffs, order, vlim, vlen, vth, u):
    """
    Dense VDF from a coefficient dict via ONE contraction (coeffs_into_cube +
    einsum with optimize=True). vt.reconstruct_vdf_adaptive does the same sum
    as one full-cube einsum per coefficient, which is far slower at vlen=240.
    """
    dv = 2.0 * vlim / vlen
    v_ax = vt.velocity_axis(vlim, vlen, dv)
    Hx, Hy, Hz = (vt.hermite_basis(v_ax, order, vth, u[i]) for i in range(3))
    h_cube = vt.coeffs_into_cube(coeffs, order)
    return np.einsum('lmn,nx,my,lz->zyx', h_cube, Hx, Hy, Hz, optimize=True)


def power_lm(coeffs, order):
    """P[l,m] = sum_n C[l,m,n]^2, NaN where no coefficient exists (outside the tetrahedron)."""
    p = np.zeros((order, order))
    seen = np.zeros((order, order), dtype=bool)
    for (l, m, n), c in coeffs.items():
        p[l, m] += c * c
        seen[l, m] = True
    p[~seen] = np.nan
    return p


def power_per_level(coeffs, order):
    p = np.zeros(order)
    for (l, m, n), c in coeffs.items():
        p[l + m + n] += c * c
    return p


def errors(cube, rec, sp_th):
    """(relative L2 inside the VDF support, eps_log over the whole cube, fraction of support voxels < 0)."""
    mask = cube >= sp_th
    rel_l2 = np.linalg.norm((rec - cube)[mask]) / np.linalg.norm(cube[mask])
    lt, lr = np.log(np.maximum(cube, sp_th)), np.log(np.maximum(rec, sp_th))
    eps_log = float(np.sqrt(np.mean((lt - lr) ** 2)))
    return float(rel_l2), eps_log, float((rec[mask] < 0).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bulk', default=BULK_FILE)
    ap.add_argument('--cellid', type=int, default=CELL_ID)
    ap.add_argument('--order', type=int, default=MAX_ORDER, help='max Hermite order (ceiling)')
    ap.add_argument('--tolerance', type=float, default=TOLERANCE)
    ap.add_argument('--window', default=WINDOW, choices=['lanczos', 'raised_cosine'])
    ap.add_argument('--bbox-margin', type=int, default=BBOX_MARGIN)
    ap.add_argument('--no-bbox', action='store_true', help='do not crop the reconstructions to the VDF bounding box')
    args = ap.parse_args()

    reader = pt.vlsvfile.VlsvReader(args.bulk)
    cells = np.atleast_1d(reader.read(mesh="SpatialGrid", tag="CELLSWITHBLOCKS", name="proton"))
    if args.cellid not in set(int(c) for c in cells):
        raise SystemExit(f"cell {args.cellid} has no stored proton VDF in {args.bulk}")

    vlim, vlen, dv = vt.get_vdf_parameters(reader)
    sp_th = vt.get_sparse_threshold(reader)
    x, y, z = reader.get_cell_coordinates(args.cellid)
    print(f"cell {args.cellid} at (x,y,z) = ({x:.3e}, {y:.3e}, {z:.3e}) m | vlen={vlen} sp_th={sp_th:g}")

    cube = vt.build_cube(args.cellid, reader, vlim, vlen, dv).astype(np.float64)
    u = vt.get_drift_velocity_cube(cube, vlim, vlen)
    vth = vt.get_thermal_velocity_cube(cube, vlim, vlen, u)
    print(f"u = {u}  vth = {vth:.4g}")

    coeffs, order_used, deltas = vt.adaptive_transform(
        cube, vlim, vlen, vth, u, max_order=args.order, tolerance=args.tolerance)
    coeffs_w = vt.apply_spectral_window(coeffs, order_used, window=args.window)
    print(f"adaptive_transform: order_used={order_used} (ceiling {args.order}), "
          f"{len(coeffs)} coefficients, Parseval delta = {deltas[order_used - 1]:.4f}")

    rec   = reconstruct(coeffs,   order_used, vlim, vlen, vth, u)
    rec_w = reconstruct(coeffs_w, order_used, vlim, vlen, vth, u)
    # Crop the reconstructions to the bounding box of the TRUE VDF support
    # (cells >= sp_th, plus a margin), which removes the ringing lobes that
    # sit outside it. The box comes from the original cube, so in production
    # it has to be stored per cell (see bbox_idx in vdfs_for_initialization).
    box = None
    if not args.no_bbox:
        box = vt.get_vdf_bounding_box(cube, vlim, vlen, sp_th=sp_th, margin=args.bbox_margin)
        rec, rec_w = (vt.apply_bounding_box(r, box['idx']) for r in (rec, rec_w))
        n_box = int(np.prod([hi - lo + 1 for lo, hi in box['idx']]))
        print(f"bbox idx (z,y,x) = {box['idx']}  ({n_box} of {vlen**3} voxels, "
              f"{box['n_active']} above sp_th)")
    e, e_w = errors(cube, rec, sp_th), errors(cube, rec_w, sp_th)
    print(f"{'':14s} {'rel L2 (support)':>17} {'eps_log':>9} {'frac<0':>8}")
    print(f"{'no window':14s} {e[0]:17.4f} {e[1]:9.4f} {e[2]:8.3f}")
    print(f"{args.window:14s} {e_w[0]:17.4f} {e_w[1]:9.4f} {e_w[2]:8.3f}")

    # ---------- figure ----------
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)

    proj = lambda c: np.sum(np.maximum(c, 0.0), axis=PROJ_AXIS)
    p_true = proj(cube)
    norm = LogNorm(vmin=p_true.max() * 10.0 ** -DECADES, vmax=p_true.max())
    vlim_kms = vlim / 1e3
    extent = [-vlim_kms, vlim_kms, -vlim_kms, vlim_kms]
    plane = {0: ('vx', 'vy'), 1: ('vx', 'vz'), 2: ('vy', 'vz')}[PROJ_AXIS]
    tag = '' if box is None else ' + bbox'
    # projection plane -> which cube axes are shown as columns (x) and rows (y)
    col_ax, row_ax = {0: (2, 1), 1: (2, 0), 2: (1, 0)}[PROJ_AXIS]
    for ax, img, title in [
        (axes[0, 0], p_true,    'original VDF'),
        (axes[0, 1], proj(rec),   f'reconstruction, no window (order {order_used}){tag}'),
        (axes[0, 2], proj(rec_w), f'reconstruction, {args.window} window{tag}'),
    ]:
        im = ax.imshow(img, origin='lower', cmap='Spectral_r', norm=norm, extent=extent)
        ax.set_title(title)
        ax.set_xlabel(f'{plane[0]} [km/s]')
        ax.set_ylabel(f'{plane[1]} [km/s]')
        if box is not None:   # outline the crop box (cell edges = centre -/+ dv/2)
            (c0, c1), (r0, r1) = box['vel'][col_ax], box['vel'][row_ax]
            h = dv / 2e3
            ax.add_patch(plt.Rectangle((c0 / 1e3 - h, r0 / 1e3 - h), (c1 - c0) / 1e3 + 2 * h,
                                       (r1 - r0) / 1e3 + 2 * h, fill=False, ec='k', ls='--', lw=0.8))
    fig.colorbar(im, ax=axes[0, :3].tolist(), shrink=0.8, label='f summed along one axis')

    # 1-D marginal along vx (sum over z,y) -- tails and ringing show up here
    v_kms = vt.velocity_axis(vlim, vlen, dv) / 1e3
    ax = axes[0, 3]
    ax.plot(v_kms, cube.sum(axis=(0, 1)), color='0.55', lw=3, label='original')
    ax.plot(v_kms, np.maximum(rec.sum(axis=(0, 1)), 0),   lw=1.2, label='no window')
    ax.plot(v_kms, np.maximum(rec_w.sum(axis=(0, 1)), 0), lw=1.2, label=args.window)
    ax.set_yscale('log')
    peak = cube.sum(axis=(0, 1)).max()
    ax.set_ylim(peak * 10.0 ** -DECADES, peak * 2)
    ax.set_xlabel('vx [km/s]')
    ax.set_title(f'1-D marginal f(vx)  (negatives clipped{tag})')
    ax.legend(fontsize=8)

    # Hermite spectra: before / after window on one shared log scale
    p0, p1 = power_lm(coeffs, order_used), power_lm(coeffs_w, order_used)
    vmax = np.nanmax(p0)
    snorm = LogNorm(vmin=vmax * 10.0 ** -(2 * DECADES), vmax=vmax)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('white')
    for ax, p, title in [(axes[1, 0], p0, 'Hermite spectrum  P[l,m] = sum_n C^2'),
                         (axes[1, 1], p1, f'spectrum after {args.window} window')]:
        im = ax.imshow(np.ma.masked_invalid(np.where(p > 0, p, np.nan)), origin='lower',
                       cmap=cmap, norm=snorm, extent=[-0.5, order_used - 0.5] * 2)
        ax.set_xlabel('m'); ax.set_ylabel('l'); ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.colorbar(im, ax=axes[1, :2].tolist(), shrink=0.8, label='power')

    # power per level s = l+m+n, before/after, with the window itself
    ax = axes[1, 2]
    s = np.arange(order_used)
    ax.semilogy(s, power_per_level(coeffs, order_used),   'o-', label='no window')
    ax.semilogy(s, power_per_level(coeffs_w, order_used), 's-', label=args.window)
    ax.set_xlabel('level  s = l+m+n'); ax.set_ylabel('power at level s')
    ax.set_title('power per level  /  window w(s)')
    ax2 = ax.twinx()
    ax2.plot(s, [coeffs_w[k] / coeffs[k] if coeffs[k] != 0 else np.nan
                 for k in [next(kk for kk in coeffs if sum(kk) == lv) for lv in s]],
             'k--', lw=1, label='w(s)')
    ax2.set_ylabel('w(s)'); ax2.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='lower left')

    # Parseval delta per level -- why the adaptive loop stopped where it did
    ax = axes[1, 3]
    lv = sorted(deltas)
    ax.semilogy(lv, [deltas[k] for k in lv], 'o-')
    ax.axhline(args.tolerance, color='r', ls=':', label=f'tolerance {args.tolerance}')
    ax.axvline(order_used - 1, color='k', ls='--', lw=0.8, label=f'stopped at s={order_used - 1}')
    ax.set_xlabel('level s'); ax.set_ylabel('Parseval delta')
    ax.set_title('adaptive stopping')
    ax.legend(fontsize=8)

    fig.suptitle(f"cell {args.cellid}  ({os.path.basename(args.bulk)})   order_used={order_used}   "
                 f"eps_log: {e[1]:.3f} -> {e_w[1]:.3f} with {args.window}   "
                 f"rel L2: {e[0]:.3f} -> {e_w[0]:.3f}", fontsize=12)

    os.makedirs(PLOTDIR, exist_ok=True)
    out = os.path.join(PLOTDIR, f'run_one_cell_{args.cellid}{"_nobbox" if box is None else ""}.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == '__main__':
    main()
