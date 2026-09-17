"""
Plotting routines for Hermite VDF analysis.

Functions
---------
plot_parseval_convergence  -- eps_rel vs order, even/odd coloured
plot_power_per_level       -- bar chart: power added per level
plot_spectra_slices        -- 2-D heatmaps of coefficient planes
plot_vdf_comparison        -- original / reconstructed / diff side by side
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, SymLogNorm
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_processing.vdf_tools import to_log_shifted, velocity_axis
from data_processing.adaptive_hermite import reconstruct


# ---
# 1.  Parseval convergence curve
# ---

def plot_parseval_convergence(history, threshold=None, title='', save=None):
    """
    Plot both eps metrics vs Hermite level s.

    Parameters
    ----------
    history   : list of (s, parity, eps_rel, eps_log, n_new_coeffs)
    threshold : float | None  -- draw a horizontal line at this value
                                 (interpreted against eps_log, the default
                                 stopping metric)
    title     : str
    save      : str | None    -- filename to save (None = show interactively)
    """
    s_even       = [s for s, p, *_ in history if p == 'even']
    eps_rel_even = [er for _, p, er, el, n in history if p == 'even']
    eps_log_even = [el for _, p, er, el, n in history if p == 'even']
    s_odd        = [s for s, p, *_ in history if p == 'odd']
    eps_rel_odd  = [er for _, p, er, el, n in history if p == 'odd']
    eps_log_odd  = [el for _, p, er, el, n in history if p == 'odd']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # left: eps_log (honest, relative accuracy) vs s -- solid;
    #       eps_rel (f-space, peak-dominated) vs s -- faint, for comparison
    ax = axes[0]
    ax.plot(s_even, eps_log_even, 'o-', color='royalblue', label='eps_log even', linewidth=2)
    if s_odd:
        ax.plot(s_odd, eps_log_odd, 's--', color='tomato', label='eps_log odd', linewidth=2)
    ax.plot(s_even, eps_rel_even, 'o:', color='royalblue', alpha=0.35, label='eps_rel even', linewidth=1)
    if s_odd:
        ax.plot(s_odd, eps_rel_odd, 's:', color='tomato', alpha=0.35, label='eps_rel odd', linewidth=1)
    if threshold is not None:
        ax.axhline(threshold, color='grey', linestyle=':', label=f'threshold={threshold}')
    ax.set_xlabel('Hermite level  s = l+m+n')
    ax.set_ylabel('eps  (solid=log-space, faint=f-space)')
    ax.set_title(f'Convergence  {title}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # right: cumulative f-space power fraction (still the free/exact Parseval quantity)
    ax = axes[1]
    all_s   = [s for s, *_ in history]
    all_eps_rel = [er for _, _, er, _, _ in history]
    power_frac = [1 - er**2 for er in all_eps_rel]
    colors = ['royalblue' if p == 'even' else 'tomato'
              for _, p, *_ in history]
    ax.bar(all_s, [pf - (power_frac[i-1] if i > 0 else 0)
                   for i, pf in enumerate(power_frac)],
           color=colors, alpha=0.7, label='dP per level')
    ax.plot(all_s, power_frac, 'k-', linewidth=1.5, label='cumulative P/P_total')
    ax.set_xlabel('Hermite level  s')
    ax.set_ylabel('Parseval power fraction (f-space)')
    ax.set_title('Power captured per level')
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='royalblue', alpha=0.7, label='even'),
        Patch(color='tomato',    alpha=0.7, label='odd'),
        plt.Line2D([0],[0], color='k', label='cumulative'),
    ])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
        print(f"  saved: {save}")
    else:
        plt.show()
    plt.close(fig)


# ---
# 2.  2-D heatmaps of spectra
# ---

def _coeffs_to_matrix(coeffs, max_order, fixed_idx, fixed_val):
    """
    Extract a 2-D slice of C[l,m,n].
    fixed_idx : 0='l', 1='m', 2='n'
    fixed_val : value of the fixed index
    Returns 2-D array shape (max_order+1, max_order+1).
    """
    mat = np.zeros((max_order + 1, max_order + 1))
    for (l, m, n), c in coeffs.items():
        key = [l, m, n]
        if key[fixed_idx] == fixed_val:
            remaining = [k for i, k in enumerate(key) if i != fixed_idx]
            r0, r1 = remaining
            if r0 <= max_order and r1 <= max_order:
                mat[r0, r1] = c
    return mat


def plot_spectra_slices(coeffs, max_order, title='', save=None):
    """
    Show three 2-D slices of the Hermite spectra:
      C[l, m, 0]  (n=0 plane)
      C[l, 0, n]  (m=0 plane)
      C[0, m, n]  (l=0 plane)
    and a 1-D power spectrum (total power per level s).
    """
    max_s  = max(l + m + n for l, m, n in coeffs)
    levels = sorted(set(l + m + n for l, m, n in coeffs))
    power_per_level = {}
    for (l, m, n), c in coeffs.items():
        s = l + m + n
        power_per_level[s] = power_per_level.get(s, 0.0) + c * c

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Hermite spectra  {title}', fontsize=12)

    planes = [
        (axes[0, 0], 2, 0, 'C[l, m, n=0]', 'l', 'm'),
        (axes[0, 1], 1, 0, 'C[l, m=0, n]', 'l', 'n'),
        (axes[1, 0], 0, 0, 'C[l=0, m, n]', 'm', 'n'),
    ]
    for ax, fixed_idx, fixed_val, plane_label, xlabel, ylabel in planes:
        mat = _coeffs_to_matrix(coeffs, max_order, fixed_idx, fixed_val)
        vmax = np.abs(mat).max()
        vmax = vmax if vmax > 0 else 1.0
        im = ax.imshow(mat, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[-0.5, max_order + 0.5, -0.5, max_order + 0.5])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(plane_label)
        fig.colorbar(im, ax=ax, shrink=0.8)

    # 1-D power per level
    ax = axes[1, 1]
    s_vals = sorted(power_per_level)
    p_vals = [power_per_level[s] for s in s_vals]
    colors = ['royalblue' if s % 2 == 0 else 'tomato' for s in s_vals]
    ax.bar(s_vals, p_vals, color=colors, alpha=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('level  s = l+m+n')
    ax.set_ylabel('power  sum C[l,m,n]^2  at level s')
    ax.set_title('Power per Hermite level')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='royalblue', alpha=0.8, label='even'),
                       Patch(color='tomato',    alpha=0.8, label='odd')])
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
        print(f"  saved: {save}")
    else:
        plt.show()
    plt.close(fig)


# ---
# 3.  VDF comparison: original / reconstructed / diff
# ---

def _project_cube(cube):
    """Return three 2-D projections (sum over each axis)."""
    return [np.sum(cube, axis=0),   # sum_z -> (y, x) plane
            np.sum(cube, axis=1),   # sum_y -> (z, x) plane
            np.sum(cube, axis=2)]   # sum_x -> (z, y) plane


def plot_cube_comparison(cube, cube_sets, labels, vlim, sp_th=1e-15,
                         title='', save=None):
    """
    Compare original VDF against multiple ALREADY-COMPUTED cubes (e.g. the
    corrector's base/corrected reconstructions) -- unlike plot_vdf_comparison,
    takes raw (vlen,vlen,vlen) arrays directly instead of Hermite coeffs,
    since a corrected reconstruction isn't a coefficient-space object.

    Parameters
    ----------
    cube       : ndarray (vlen,vlen,vlen)   original VDF
    cube_sets  : list of ndarray            e.g. [f_rec_low, f_corrected]
    labels     : list of str                e.g. ['base (S_low=2)', 'corrected']
    """
    n_rec     = len(cube_sets)
    proj_axes = ['int dz (xy)', 'int dy (xz)', 'int dx (yz)']
    n_cols    = 1 + n_rec
    n_rows    = len(proj_axes)

    fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                            hspace=0.35, wspace=0.3)
    fig.suptitle(f'VDF before/after correction  {title}', fontsize=12, y=1.01)

    ext = [-vlim / 1e3, vlim / 1e3, -vlim / 1e3, vlim / 1e3]  # km/s

    orig_projs = _project_cube(cube)
    rec_cubes  = [np.maximum(c, 0.0) for c in cube_sets]  # clip any negatives

    vmax_global = max(p.max() for p in orig_projs)
    vmin_global = max(sp_th, vmax_global * 1e-10)   # keep LogNorm safe (no zeros)

    def _clip(arr):
        return np.clip(arr, vmin_global, None)

    for row, proj_label in enumerate(proj_axes):
        ax = fig.add_subplot(gs[row, 0])
        im = ax.imshow(_clip(orig_projs[row]), origin='lower', extent=ext,
                       cmap='Spectral_r',
                       norm=LogNorm(vmin=vmin_global, vmax=vmax_global))
        ax.set_title(f'Original\n{proj_label}' if row == 0 else proj_label)
        ax.set_xlabel('v [km/s]')
        ax.set_ylabel('v [km/s]')
        fig.colorbar(im, ax=ax, shrink=0.85)

        for col, (rcube, label) in enumerate(zip(rec_cubes, labels), start=1):
            rec_projs = _project_cube(rcube)
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(_clip(rec_projs[row]), origin='lower', extent=ext,
                           cmap='Spectral_r',
                           norm=LogNorm(vmin=vmin_global, vmax=vmax_global))
            ax.set_title(f'{label}\n{proj_label}' if row == 0 else proj_label)
            ax.set_xlabel('v [km/s]')
            fig.colorbar(im, ax=ax, shrink=0.85)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
        print(f"  saved: {save}")
    else:
        plt.show()
    plt.close(fig)


def plot_cube_1d_cuts(cube, cube_sets, labels, vlim, vlen, u,
                      title='', save=None):
    """
    1-D slices through the peak of the int-dz projection, for
    ALREADY-COMPUTED cubes (see plot_cube_comparison).
    """
    linestyles = ['--', '-.', ':']
    colors_rec = ['tab:blue', 'tab:orange', 'tab:green']

    v_km = velocity_axis(vlim, vlen, 2 * vlim / vlen) / 1e3

    proj_orig = cube.sum(axis=0)
    iy_pk, ix_pk = np.unravel_index(proj_orig.argmax(), proj_orig.shape)
    cut_vx = proj_orig[iy_pk, :]
    cut_vy = proj_orig[:, ix_pk]

    rec_cuts_vx, rec_cuts_vy = [], []
    for c in cube_sets:
        proj_rec = c.sum(axis=0)
        rec_cuts_vx.append(proj_rec[iy_pk, :])
        rec_cuts_vy.append(proj_rec[:, ix_pk])

    all_orig_pos = [v for c in [cut_vx, cut_vy] for v in c if v > 0]
    ymin = min(all_orig_pos) * 0.3 if all_orig_pos else 1e-20
    ymax = max(cut_vx.max(), cut_vy.max()) * 3

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'1-D cuts before/after correction  {title}', fontsize=11)

    for ax, cut_orig, rec_cuts, ax_label, u_comp in [
        (axes[0], cut_vx, rec_cuts_vx, 'vx  [km/s]', u[0] / 1e3),
        (axes[1], cut_vy, rec_cuts_vy, 'vy  [km/s]', u[1] / 1e3),
    ]:
        with np.errstate(divide='ignore', invalid='ignore'):
            orig_plot = np.where(cut_orig > 0, cut_orig, np.nan)
        ax.semilogy(v_km, orig_plot, 'k-', linewidth=2, label='original', zorder=10)

        for i, (rc, label) in enumerate(zip(rec_cuts, labels)):
            ls  = linestyles[i % len(linestyles)]
            col = colors_rec[i % len(colors_rec)]
            with np.errstate(divide='ignore', invalid='ignore'):
                rc_pos = np.where(rc > 0, rc, np.nan)
            ax.semilogy(v_km, rc_pos, linestyle=ls, color=col,
                        linewidth=1.8, label=label)
            with np.errstate(divide='ignore', invalid='ignore'):
                rc_neg = np.where(rc < 0, -rc, np.nan)
            ax.semilogy(v_km, rc_neg, linestyle=ls, color=col,
                        linewidth=0.8, alpha=0.3)

        ax.axvline(u_comp, color='grey', linestyle=':', linewidth=1,
                   label=f'u={u_comp:.1f} km/s')
        ax.set_xlabel(ax_label)
        ax.set_ylabel('int dz  f  [s^3/m^6 * m/s]')
        ax.set_ylim(ymin, ymax)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
        print(f"  saved: {save}")
    else:
        plt.show()
    plt.close(fig)


def plot_vdf_comparison(cube, coeffs_sets, labels, vlim, vlen, vth, u,
                        sp_th=1e-15, title='', save=None):
    """
    Compare original VDF against multiple reconstructions.

    Parameters
    ----------
    cube        : ndarray (vlen,vlen,vlen)   original VDF
    coeffs_sets : list of dicts              {(l,m,n): value}  -- one per reconstruction
    labels      : list of str                e.g. ['N=4', 'N=8', 'N=10']
    vlim, vlen, vth, u : mesh parameters
    sp_th       : sparsity threshold
    """
    n_rec     = len(coeffs_sets)
    proj_axes = ['int dz (xy)', 'int dy (xz)', 'int dx (yz)']
    n_cols    = 1 + n_rec
    n_rows    = len(proj_axes)

    fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                            hspace=0.35, wspace=0.3)
    fig.suptitle(f'VDF comparison  {title}', fontsize=12, y=1.01)

    ext = [-vlim / 1e3, vlim / 1e3, -vlim / 1e3, vlim / 1e3]  # km/s

    orig_projs = _project_cube(cube)

    # Exact sparse mask from the original cube -- structural info, not learned.
    sparse_mask = cube >= sp_th

    rec_cubes = []
    for cset in coeffs_sets:
        f_rec = reconstruct(cset, vlim, vlen, vth, u, sparse_mask=sparse_mask)
        rec_cubes.append(np.maximum(f_rec, 0.0))   # clip Gibbs negatives

    vmax_global = max(p.max() for p in orig_projs)
    vmin_global = max(sp_th, vmax_global * 1e-10)   # keep LogNorm safe (no zeros)

    def _clip(arr):
        return np.clip(arr, vmin_global, None)

    for row, proj_label in enumerate(proj_axes):
        # original
        ax = fig.add_subplot(gs[row, 0])
        im = ax.imshow(_clip(orig_projs[row]), origin='lower', extent=ext,
                       cmap='Spectral_r',
                       norm=LogNorm(vmin=vmin_global, vmax=vmax_global))
        ax.set_title(f'Original\n{proj_label}' if row == 0 else proj_label)
        ax.set_xlabel('v [km/s]')
        ax.set_ylabel('v [km/s]')
        fig.colorbar(im, ax=ax, shrink=0.85)

        # reconstructions
        for col, (rcube, label) in enumerate(zip(rec_cubes, labels), start=1):
            rec_projs = _project_cube(rcube)
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(_clip(rec_projs[row]), origin='lower', extent=ext,
                           cmap='Spectral_r',
                           norm=LogNorm(vmin=vmin_global, vmax=vmax_global))
            ax.set_title(f'{label}\n{proj_label}' if row == 0 else proj_label)
            ax.set_xlabel('v [km/s]')
            fig.colorbar(im, ax=ax, shrink=0.85)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
        print(f"  saved: {save}")
    else:
        plt.show()
    plt.close(fig)


def plot_vdf_residuals(cube, coeffs_sets, labels, vlim, vlen, vth, u,
                       sp_th=1e-15, title='', save=None):
    """
    Show |original - reconstructed| for each reconstruction.
    One row per projection, one column per reconstruction.
    """
    proj_axes = ['int dz (xy)', 'int dy (xz)', 'int dx (yz)']
    n_cols    = len(coeffs_sets)
    n_rows    = len(proj_axes)

    fig = plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(f'|Residual|  {title}', fontsize=12, y=1.01)

    ext = [-vlim / 1e3, vlim / 1e3, -vlim / 1e3, vlim / 1e3]

    orig_projs = _project_cube(cube)
    sparse_mask = cube >= sp_th   # exact structural mask, not learned

    for col, (cset, label) in enumerate(zip(coeffs_sets, labels)):
        f_rec    = reconstruct(cset, vlim, vlen, vth, u, sparse_mask=sparse_mask)
        rec_cube = np.maximum(f_rec, 0.0)   # clip Gibbs negatives
        rec_projs = _project_cube(rec_cube)

        for row, (op, rp, proj_label) in enumerate(
                zip(orig_projs, rec_projs, proj_axes)):
            diff = np.abs(op - rp)
            ax   = fig.add_subplot(gs[row, col])
            im   = ax.imshow(diff, origin='lower', extent=ext, cmap='inferno')
            ax.set_title(f'{label}  {proj_label}' if row == 0 else proj_label)
            ax.set_xlabel('v [km/s]')
            if col == 0:
                ax.set_ylabel('v [km/s]')
            fig.colorbar(im, ax=ax, shrink=0.85)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
        print(f"  saved: {save}")
    else:
        plt.show()
    plt.close(fig)


# ---
# 5.  1-D cuts through the int-dz projection
# ---

def plot_vdf_1d_cuts(cube, coeffs_sets, labels, vlim, vlen, vth, u,
                     sp_th=1e-15, title='', save=None):
    """
    1-D slices through the peak of the int-dz (vx-vy) projection.

    Parameters
    ----------
    cube        : ndarray (vlen,vlen,vlen)  original VDF
    coeffs_sets : list of dicts             {(l,m,n): value}
    labels      : list of str               e.g. ['N=4', 'N=10']
    Line styles: solid black = original, '--' = first rec, '-.' = second, ':' = third
    """
    linestyles = ['--', '-.', ':']
    colors_rec = ['tab:blue', 'tab:orange', 'tab:green']

    v_km = velocity_axis(vlim, vlen, 2 * vlim / vlen) / 1e3

    # int-dz projection: sum over z-axis -> shape (vy, vx)
    proj_orig = cube.sum(axis=0)   # cube[iz, iy, ix] -> sum_z

    # peak of the projection
    iy_pk, ix_pk = np.unravel_index(proj_orig.argmax(), proj_orig.shape)

    # 1-D slices of original
    cut_vx = proj_orig[iy_pk, :]   # fixed vy=vy_pk, vary vx
    cut_vy = proj_orig[:, ix_pk]   # fixed vx=vx_pk, vary vy

    # Exact sparse mask from the original cube removes Hermite-basis ringing
    # (Gibbs-type artifacts) outside the true VDF support -- this is
    # structural information (Vlasiator's velocity mesh is block-sparse),
    # not something learned or predicted.
    sparse_mask = cube >= sp_th

    rec_cuts_vx = []
    rec_cuts_vy = []
    for cset in coeffs_sets:
        f_rec    = reconstruct(cset, vlim, vlen, vth, u, sparse_mask=sparse_mask)
        proj_rec = f_rec.sum(axis=0)
        rec_cuts_vx.append(proj_rec[iy_pk, :])
        rec_cuts_vy.append(proj_rec[:, ix_pk])

    # y-axis floor: smallest positive original value (for log scale)
    all_orig_pos = [v for c in [cut_vx, cut_vy] for v in c if v > 0]
    ymin = min(all_orig_pos) * 0.3 if all_orig_pos else 1e-20
    ymax = max(cut_vx.max(), cut_vy.max()) * 3

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'1-D cuts of int dz VDF  {title}', fontsize=11)

    for ax, cut_orig, rec_cuts, ax_label, u_comp in [
        (axes[0], cut_vx, rec_cuts_vx, 'vx  [km/s]', u[0] / 1e3),
        (axes[1], cut_vy, rec_cuts_vy, 'vy  [km/s]', u[1] / 1e3),
    ]:
        # original -- solid black (only nonzero cells)
        with np.errstate(divide='ignore', invalid='ignore'):
            orig_plot = np.where(cut_orig > 0, cut_orig, np.nan)
        ax.semilogy(v_km, orig_plot, 'k-', linewidth=2, label='original', zorder=10)

        # reconstructions -- exact sparse_mask already zeroed everything outside
        # the true support, so any remaining negative values are genuine
        # in-support Gibbs ringing (near sharp internal features), plotted
        # faint; positive values plotted solid.
        for i, (rc, label) in enumerate(zip(rec_cuts, labels)):
            ls  = linestyles[i % len(linestyles)]
            col = colors_rec[i % len(colors_rec)]
            # positive values: plot normally
            with np.errstate(divide='ignore', invalid='ignore'):
                rc_pos = np.where(rc > 0, rc, np.nan)
            ax.semilogy(v_km, rc_pos, linestyle=ls, color=col,
                        linewidth=1.8, label=label)
            # negative values (in-support Gibbs oscillations): |rc| with lighter alpha
            with np.errstate(divide='ignore', invalid='ignore'):
                rc_neg = np.where(rc < 0, -rc, np.nan)
            ax.semilogy(v_km, rc_neg, linestyle=ls, color=col,
                        linewidth=0.8, alpha=0.3)

        ax.axvline(u_comp, color='grey', linestyle=':', linewidth=1,
                   label=f'u={u_comp:.1f} km/s')
        ax.set_xlabel(ax_label)
        ax.set_ylabel('int dz  f  [s^3/m^6 * m/s]')
        ax.set_ylim(ymin, ymax)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
        print(f"  saved: {save}")
    else:
        plt.show()
    plt.close(fig)
