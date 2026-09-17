"""
Adaptive Hermite transform for 3D VDFs.

Strategy: tetrahedral truncation (l+m+n = s, not l<N AND m<N AND n<N),
incremental Parseval check after each level -- adding a level costs only
one sum over the new coefficients.

Even-first mode: process s=0,2,4,... first; only add odd levels if needed.
This exploits the symmetry of reconnection geometry where distributions near
the X-point are nearly symmetric in the bulk frame -> even modes dominate.

Usage:
    from adaptive_hermite import adaptive_transform, reconstruct, parseval_check

    coeffs, eps, s_stop, history = adaptive_transform(
        cube, vlim, vlen, vth, u,
        rel_threshold=0.01, max_order=22, even_first=False, verbose=True
    )
    f_rec = reconstruct(coeffs, vlim, vlen, vth, u)

Two eps metrics, tracked at every level in `history` as
(s, parity, eps_rel, eps_log, n_new):
  eps_rel : f-space relative L2 error -- free via Bessel's equality, but
            dominated by the VDF peak, blind to tail accuracy.
  eps_log : log-space RMS residual -- honest relative accuracy across the
            full dynamic range. No incremental shortcut exists (log is
            nonlinear), so this requires an explicit full-grid reconstruction
            + comparison every level -- but that is O(vlen^3), ~7ms at
            vlen=60, so it is computed exactly rather than approximated.
            This is the default stopping_metric.
"""

import math
import numpy as np

from .vdf_tools import hermite_basis, to_log_shifted, from_log_shifted, velocity_axis


# ---
# Core helpers
# ---

def _precompute_basis(v_ax, max_order, vth, u):
    """
    Precompute Hermite basis functions for all three velocity axes.
    Returned arrays have shape (max_order+1, vlen).
    Computed ONCE and reused across all levels -- the dominant cost.
    """
    Hx = hermite_basis(v_ax, max_order + 1, vth, u[0])
    Hy = hermite_basis(v_ax, max_order + 1, vth, u[1])
    Hz = hermite_basis(v_ax, max_order + 1, vth, u[2])
    return Hx, Hy, Hz


def _compute_level(log_cube, s, Hx, Hy, Hz, dv):
    """
    Compute all Hermite coefficients C[l,m,n] with l+m+n == s.

    Number of (l,m,n) triples at level s: (s+1)(s+2)/2.
    Each coefficient is an O(vlen^3) inner product -- but precomputed basis
    vectors mean we only do the contraction, not the basis construction.

    Returns:
        new_coeffs  : dict {(l,m,n): float}
        level_power : sum C[l,m,n]^2 -- added to the running Parseval sum
    """
    new_coeffs = {}
    level_power = 0.0
    dv_vol = dv ** 3   # Riemann-sum cell volume for the continuous triple
                       # integral C[l,m,n] = int f(v) phi_l(vz)phi_m(vy)phi_n(vx) d^3v
                       # (3 independent velocity dimensions -> dv*dv*dv, not dv^1.5;
                       # verified numerically against the Maxwellian C000 theory
                       # value -- dv^1.5 was off by a factor of dv^1.5 = 4.6e5 at
                       # dv=6 km/s, dv^3 matches theory to ratio ~0.95)

    # cube axes convention from vdf_tools: cube[iz, iy, ix]
    # spectra convention:  C[l,m,n] contracts as z<->l, y<->m, x<->n
    for l in range(s + 1):
        for m in range(s + 1 - l):
            n = s - l - m
            # Inner product: C = sum f * Hz_l * Hy_m * Hx_n * dv^3  (Riemann sum)
            c = float(np.einsum('zyx,x,y,z->', log_cube,
                                Hx[n], Hy[m], Hz[l]) * dv_vol)
            new_coeffs[(l, m, n)] = c
            level_power += c * c

    return new_coeffs, level_power


def _parseval_total(log_cube, dv):
    """
    Total Parseval power from the raw log_cube -- O(vlen^3), computed once.
    Continuous-integral normalization: P = sum(data^2) * dv^3, approximating
    int f(v)^2 d^3v via a Riemann sum. Only approximately equal to
    sum C[l,m,n]^2 (Parseval/Bessel completeness), not an exact discrete
    identity -- but matches the physically meaningful C[l,m,n] = int f phi d^3v
    convention used in _compute_level / reconstruct.
    """
    return float(np.sum(log_cube ** 2)) * dv ** 3


def _eps(P_accum, P_total):
    """Relative L2 residual: sqrt(1 - P_accum/P_total)."""
    return math.sqrt(max(0.0, 1.0 - P_accum / P_total))


def _level_field(new_c, Hx, Hy, Hz, dv, vlen):
    """
    Reconstruct just the field contribution of one level's new coefficients
    (same normalization as reconstruct()): sum_c c * phi_lmn(v)  -- no extra
    dv factor, since C already carries the physical dv^3 from _compute_level
    (C = int f phi d^3v, so f(v) = sum C phi(v) directly, like a Fourier series).
    O(vlen^3) per level -- cheap for typical vlen (tens of ms total, see
    hermite_ml notes on why the log-space residual is not actually expensive
    despite Bessel's equality not applying there).
    """
    field = np.zeros((vlen, vlen, vlen), dtype=np.float64)
    for (l, m, n), c in new_c.items():
        field += c * np.einsum('x,y,z->zyx', Hx[n], Hy[m], Hz[l])
    return field


def _log_eps(cube, f_rec_accum, sp_th, sparse_mask=None):
    """
    Honest relative-accuracy metric computed in log-space.

    Bessel's equality (the cheap incremental sum-of-squares trick) only
    holds for the L2 norm in the SAME space the fit was done in (f-space
    here) -- log() is nonlinear, so there is no shortcut: this requires the
    full current reconstruction f_rec_accum and an explicit elementwise
    log-residual over the whole grid. That is O(vlen^3) per level, which is
    negligible for our grid sizes (~7ms/level at vlen=60), so we just do it
    honestly instead of approximating.

    If sparse_mask is given, f_rec_accum is clipped to the true support
    first (structural info from the sparse VDF mesh, not learned) so
    Hermite-basis ringing outside support does not pollute the metric.
    """
    f_rec = f_rec_accum
    if sparse_mask is not None:
        f_rec = np.where(sparse_mask, f_rec_accum, 0.0)
    log_true = np.log(np.maximum(cube, sp_th))
    log_rec  = np.log(np.maximum(f_rec, sp_th))
    return float(np.sqrt(np.mean((log_true - log_rec) ** 2)))


# ---
# Public API
# ---

def adaptive_transform(log_cube, vlim, vlen, vth, u,
                       max_order=14,
                       rel_threshold=None,
                       even_first=False,
                       sp_th=1e-15,
                       sparse_mask=None,
                       track_log_eps=True,
                       verbose=False):
    """
    Hermite transform, level by level s=0,1,2,...

    Since the velocity-grid alignment fix (see velocity_axis() -- the
    Hermite basis used to be evaluated off the true cell centers, breaking
    its discrete orthonormality and creating a spurious eps floor
    independent of order), eps_rel is now a genuinely monotonically
    decreasing, trustworthy convergence measure. So a simple threshold stop
    is sound again:

    rel_threshold : float or None
        If given, stop as soon as eps_rel < rel_threshold (checked after
        every level). If None (default), compute all levels s=0..max_order.

    Two eps metrics are recorded at every level in `history`:
      eps_rel : f-space relative L2 error, sqrt(1 - P_accum/P_total). Free
                (incremental sum of squared new coefficients, via Bessel's
                equality), and (post grid-fix) monotonically decreasing --
                this is what rel_threshold checks.
      eps_log : log-space RMS residual, sqrt(mean[(log(true)-log(rec))^2]).
                Still not monotonic at low order (truncated f-space fits can
                go negative over much of the domain) -- kept only as a
                diagnostic, not used for stopping. Requires a full
                reconstruction+compare every level (no incremental shortcut,
                log is nonlinear), O(vlen^3), ~7ms/level at vlen=60,
                negligible.

    Parameters
    ----------
    log_cube      : ndarray (vlen,vlen,vlen)  input VDF cube (f, not log(f))
    vlim          : float   half-range of velocity axis [m/s]
    vlen          : int     number of velocity cells per axis
    vth           : float   thermal velocity [m/s]
    u             : array   bulk drift velocity [ux,uy,uz] [m/s]
    max_order     : int     hard ceiling on total order s=l+m+n
    rel_threshold : float or None   stop when eps_rel < this (None = no stop)
    even_first    : bool    order in which levels are computed (even 0,2,4,...
                            then odd 1,3,5,... vs plain sequential); with
                            rel_threshold set, even_first is NOT recommended
                            (early exit could occur before odd modes are
                            tried, missing real structure) -- prefer
                            sequential (default).
    sp_th         : float   sparsity floor used for the log-space metric
    sparse_mask   : ndarray(bool), optional
                    exact true-support mask (e.g. log_cube>=sp_th) applied to
                    the running reconstruction before computing eps_log.
                    Structural info from the sparse VDF mesh, not learned.
    track_log_eps : bool    if True (default), maintain the running
                    reconstruction and eps_log at every level (see above).
                    This costs O(n_coeffs) full-grid field constructions
                    total (one per coefficient, via _level_field) -- at
                    max_order=22 (2300 coeffs) this dominates runtime, ~100x
                    more work than the coefficient computation itself. Set
                    False to skip it entirely when only final coeffs/eps_rel
                    are needed (e.g. building a high-order f_rec_full for a
                    downstream reconstruct() call) -- history then reports
                    eps_log=None for every level.
    verbose       : bool    print per-level progress

    Returns
    -------
    coeffs  : dict {(l,m,n): float}  all computed coefficients
    eps_rel : float  final f-space relative L2 error achieved
    s_stop  : int    total order at which computation stopped
    history : list of (s, parity, eps_rel, eps_log, n_new_coeffs)
    """
    dv   = 2.0 * vlim / vlen
    v_ax = velocity_axis(vlim, vlen, dv)

    P_total = _parseval_total(log_cube, dv)
    if P_total < 1e-30:
        # essentially zero VDF -- nothing to decompose
        return {}, 0.0, 0, []

    # Precompute all basis functions up to max_order -- reused at every level
    Hx, Hy, Hz = _precompute_basis(v_ax, max_order, vth, u)

    coeffs      = {}
    P_accum     = 0.0
    history     = []
    eps_rel     = 1.0
    eps_log     = (_log_eps(log_cube, np.zeros_like(log_cube), sp_th, sparse_mask)
                   if track_log_eps else None)
    f_rec_accum = np.zeros((vlen, vlen, vlen), dtype=np.float64) if track_log_eps else None

    def _process_level(s):
        nonlocal P_accum, eps_rel, eps_log, f_rec_accum
        new_c, lp = _compute_level(log_cube, s, Hx, Hy, Hz, dv)
        coeffs.update(new_c)
        P_accum  += lp
        eps_rel   = _eps(P_accum, P_total)

        if track_log_eps:
            f_rec_accum += _level_field(new_c, Hx, Hy, Hz, dv, vlen)
            eps_log      = _log_eps(log_cube, f_rec_accum, sp_th, sparse_mask)

        parity = 'even' if s % 2 == 0 else 'odd'
        history.append((s, parity, eps_rel, eps_log, len(new_c)))
        if verbose:
            print(f"  s={s:2d} ({parity:4s}): {len(new_c):4d} new coeffs | "
                  f"eps_rel={eps_rel:.5f} | eps_log={eps_log:.5f}")
        return rel_threshold is not None and eps_rel < rel_threshold

    s_stop = max_order
    if even_first:
        for s in range(0, max_order + 1, 2):
            if _process_level(s):
                s_stop = s
                return coeffs, eps_rel, s_stop, history
        for s in range(1, max_order + 1, 2):
            if _process_level(s):
                s_stop = s
                return coeffs, eps_rel, s_stop, history
    else:
        for s in range(max_order + 1):
            if _process_level(s):
                s_stop = s
                return coeffs, eps_rel, s_stop, history

    if verbose:
        print(f"[adaptive_hermite] Computed s=0..{max_order}, "
              f"final eps_rel={eps_rel:.5f}  final eps_log={eps_log:.5f}")
    return coeffs, eps_rel, s_stop, history


def reconstruct(coeffs, vlim, vlen, vth, u,
                 turning_point_cutoff=True, sparse_mask=None):
    """
    Reconstruct log_cube from a coefficient dict {(l,m,n): value}.

    Parameters
    ----------
    turning_point_cutoff : bool
        Used only when sparse_mask is None. If True (default), zero the
        reconstruction outside the quantum turning point
        v_turn = vth*sqrt(2*max_order+1) of the highest Hermite order used.
        Beyond v_turn, the Hermite basis functions are in their outermost
        oscillatory lobe -- any signal there is a known basis artifact
        (Gibbs-type ringing from expanding a compactly-supported /
        sharply-truncated VDF in a global Hermite basis), not physical
        content. This is only a partial fix: it removes the outermost lobe
        but not residual ringing from lower orders still inside v_turn(max).
    sparse_mask : ndarray(bool) (vlen,vlen,vlen), optional
        Exact mask of which velocity cells hold real (non-sparse) VDF data,
        e.g. `cube >= sp_th` from the ORIGINAL cube. Vlasiator's velocity
        mesh is itself block-sparse -- only non-empty blocks are stored --
        so this mask is structural information that comes for free with the
        VDF, not something that needs to be predicted or learned. When
        given, it takes precedence over turning_point_cutoff and removes
        ALL basis-ringing outside the true support exactly, not just the
        outermost lobe.
    """
    if not coeffs:
        return np.zeros((vlen, vlen, vlen), dtype=np.float32)

    max_order = max(l + m + n for l, m, n in coeffs)
    dv   = 2.0 * vlim / vlen
    v_ax = velocity_axis(vlim, vlen, dv)
    Hx, Hy, Hz = _precompute_basis(v_ax, max_order, vth, u)

    log_rec = np.zeros((vlen, vlen, vlen), dtype=np.float64)
    for (l, m, n), c in coeffs.items():
        # C already carries the physical dv^3 from _compute_level
        # (C = int f phi d^3v), so f(v) = sum C*phi(v) directly -- no dv factor.
        log_rec += c * np.einsum('x,y,z->zyx', Hx[n], Hy[m], Hz[l])

    if sparse_mask is not None:
        # Exact structural mask -- zero everything outside the true support.
        log_rec[~sparse_mask] = 0.0
    elif turning_point_cutoff:
        # Tetrahedral truncation (l+m+n <= max_order) allows any single index
        # to reach max_order (when the other two are zero), so the same bound
        # applies to all three axes.
        v_turn = vth * math.sqrt(2 * max_order + 1)
        vx = v_ax[np.newaxis, np.newaxis, :]
        vy = v_ax[np.newaxis, :, np.newaxis]
        vz = v_ax[:, np.newaxis, np.newaxis]
        outside = ((np.abs(vx - u[0]) > v_turn) |
                   (np.abs(vy - u[1]) > v_turn) |
                   (np.abs(vz - u[2]) > v_turn))
        log_rec[outside] = 0.0

    return log_rec.astype(np.float32)


def cubic_transform(data, vlim, vlen, vth, u, N):
    """
    Full cubic Hermite transform: all C[l,m,n] with l,m,n < N.

    Uses a single einsum over all indices -- O(N^3 * vlen^3) but computed
    as one batch rather than (N^3/6) separate inner products.

    Parameters
    ----------
    data  : ndarray (vlen,vlen,vlen)  input (f or log_cube)
    N     : int  order per axis  (total coefficients: N^3)

    Returns
    -------
    coeffs  : dict {(l,m,n): float}
    eps_rel : float
    history : list of (N_used, n_coeffs, eps_rel)  -- cumulative by cubic shell
    """
    dv   = 2.0 * vlim / vlen
    v_ax = velocity_axis(vlim, vlen, dv)

    Hx = hermite_basis(v_ax, N, vth, u[0])   # (N, vlen)
    Hy = hermite_basis(v_ax, N, vth, u[1])
    Hz = hermite_basis(v_ax, N, vth, u[2])

    # single batch einsum: C[l,m,n] = sum_{z,y,x} data[z,y,x]*Hz[l,z]*Hy[m,y]*Hx[n,x] * dv^3
    # (dv^3 = Riemann-sum cell volume, same convention as _compute_level/reconstruct)
    C_arr = np.einsum('zyx,lz,my,nx->lmn', data, Hz, Hy, Hx) * dv ** 3

    P_total = float(np.sum(data ** 2)) * dv ** 3
    coeffs  = {}
    history = []

    # build history: cumulative power as we include shells n_max = 1,2,...,N
    # "cubic shell n_max" = all (l,m,n) with l,m,n < n_max
    P_accum = 0.0
    for n_max in range(1, N + 1):
        # power added by this shell (all new triples where max(l,m,n)==n_max-1)
        # easiest: just recompute cumulative sum up to n_max
        C_slice = C_arr[:n_max, :n_max, :n_max]
        P_accum = float(np.sum(C_slice ** 2))
        eps = math.sqrt(max(0.0, 1.0 - P_accum / P_total))
        history.append((n_max, n_max ** 3, eps))

    # fill coeffs dict
    for l in range(N):
        for m in range(N):
            for n in range(N):
                coeffs[(l, m, n)] = float(C_arr[l, m, n])

    eps_rel = history[-1][2]
    return coeffs, eps_rel, history


def parseval_check(log_cube, coeffs, dv):
    """
    Recompute relative Parseval error (f-space) for a given coefficient set.
    Useful for post-hoc verification. Peak-dominated -- see log_eps_check
    for the honest relative-accuracy metric.
    """
    P_total = _parseval_total(log_cube, dv)
    P_accum = sum(c * c for c in coeffs.values())
    return _eps(P_accum, P_total)


def log_eps_check(cube, coeffs, vlim, vlen, vth, u, sp_th=1e-15, sparse_mask=None):
    """
    Recompute the log-space RMS residual (eps_log) for a given coefficient
    set, post-hoc. Companion to parseval_check -- unlike the f-space metric,
    this reflects relative accuracy in the tails, not just at the peak.
    """
    f_rec = reconstruct(coeffs, vlim, vlen, vth, u,
                         turning_point_cutoff=(sparse_mask is None),
                         sparse_mask=sparse_mask)
    return _log_eps(cube, f_rec.astype(np.float64), sp_th, sparse_mask=None)


def coeffs_to_array(coeffs, max_order):
    """
    Flatten coefficient dict to a 1-D numpy array, ordered by
    level s=l+m+n then lexicographic (l,m,n).
    Total length: (max_order+1)(max_order+2)(max_order+3)//6
    Coefficients absent from the dict are set to 0.
    """
    out = []
    for s in range(max_order + 1):
        for l in range(s + 1):
            for m in range(s + 1 - l):
                n = s - l - m
                out.append(coeffs.get((l, m, n), 0.0))
    return np.array(out, dtype=np.float32)


def array_to_coeffs(arr, max_order):
    """Inverse of coeffs_to_array."""
    coeffs = {}
    idx = 0
    for s in range(max_order + 1):
        for l in range(s + 1):
            for m in range(s + 1 - l):
                n = s - l - m
                coeffs[(l, m, n)] = float(arr[idx])
                idx += 1
    return coeffs


def level_sizes(max_order):
    """Return list of (s, n_coeffs_at_level_s) for s=0..max_order."""
    return [(s, (s + 1) * (s + 2) // 2) for s in range(max_order + 1)]


def total_coeffs(max_order):
    """Total number of coefficients up to max_order (tetrahedral number)."""
    return (max_order + 1) * (max_order + 2) * (max_order + 3) // 6


# Diagnostic entry point moved to hermite_ml/run_diagnostic.py
