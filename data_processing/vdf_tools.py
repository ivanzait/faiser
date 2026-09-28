"""
VDF reader and Hermite transform tools for Vlasiator bulk files.
Original by Ivan Zaitsev; patched:
  - get_vdf_parameters: dvx read from VLSV file (was hardcoded to 52000)
  - build_cube: accepts explicit vlim/vlen/dv parameters
"""

import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


################################
########  VDF READER  #########
################################

def get_vdf_parameters(reader):    
    extents = reader.get_velocity_mesh_size(pop="proton")
    nx = int(extents[0] * 4)
    vx_min, vy_min, vz_min, vx_max, vy_max, vz_max = reader.get_velocity_mesh_extent(pop="proton")
    dvx = (vx_max - vx_min) / nx
    dv  = dvx   # must be cubic
    vxmin = vx_min
    vlim, vlen = abs(vxmin), nx
    return vlim, vlen, dv


def get_cellid_at_coords(reader, coords=None):
    """
    Return the spatial cellid at given [x, y, z] coordinates.
    If coords is None, uses the center of the simulation box.
    """
    if coords is None:
        xmin, ymin, zmin, xmax, ymax, zmax = reader.get_spatial_mesh_extent()
        coords = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    return int(reader.get_cellid(coords))


def get_nearest_vdf_cellid(reader, coords=None, pop="proton"):
    """
    Return the cellid with a stored VDF closest to given [x, y, z]
    coordinates. If coords is None, uses the center of the simulation box.
    """
    if coords is None:
        xmin, ymin, zmin, xmax, ymax, zmax = reader.get_spatial_mesh_extent()
        coords = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    coords = np.asarray(coords, dtype=float)

    cids_w_vdf = np.atleast_1d(reader.read(mesh="SpatialGrid", tag="CELLSWITHBLOCKS", name=pop))
    vdf_coords = reader.get_cell_coordinates(cids_w_vdf)
    dist2 = np.sum((vdf_coords - coords[np.newaxis, :]) ** 2, axis=1)
    nearest = int(cids_w_vdf[np.argmin(dist2)])
    return nearest


def velocity_axis(vlim, vlen, dv):
    """
    Cell-CENTER velocity axis, consistent with build_cube's binning
    (ix = round((v+vlim-dv/2)/dv)  <=>  v_center[i] = -vlim + dv/2 + i*dv).

    Do NOT use np.linspace(-vlim, vlim, vlen) -- that places vlen points
    INCLUSIVE of both endpoints, giving spacing 2*vlim/(vlen-1), which does
    NOT match the actual grid cell centers (spacing dv=2*vlim/vlen) that
    build_cube used to bin the VDF. This off-by-one caused the discrete
    Hermite basis functions to be evaluated at the wrong points, breaking
    their normalization
    """
    
    return -vlim + dv / 2.0 + np.arange(vlen) * dv


def get_sparse_threshold(reader, pop="proton"):
    cfg = reader.get_config()
    return float(cfg[f'{pop}_sparse']['minValue'][0])


def get_inner_boundary_cells(reader):
    cfg = reader.get_config()
    boundary_cfg = cfg['copysphere']
    r_min = float(boundary_cfg['radius'][0])
    center = (float(boundary_cfg['centerX'][0]),
             float(boundary_cfg['centerY'][0]),
             float(boundary_cfg['centerZ'][0]))
    cellids = reader.read_variable('CellID')
    cellids = np.asarray(cellids)
    coords = reader.get_cell_coordinates(cellids)
    r = np.linalg.norm(coords - np.asarray(center), axis=1)
    return cellids[r < r_min].astype(np.int64)



def build_cube(cellid, reader, vlim, vlen, dv):
    """
    cube[iz, iy, ix] convention.
    """
    vdf = reader.read_velocity_cells(cellid, "proton")

    vcellids = np.fromiter(vdf.keys(), dtype=np.int64)
    vcoords  = reader.get_velocity_cell_coordinates(vcellids)
    vx, vy, vz = vcoords[:, 0], vcoords[:, 1], vcoords[:, 2]
    fvals = np.fromiter(vdf.values(), dtype=np.float32)

    cube = np.zeros((vlen, vlen, vlen), dtype=np.float32)

    ix = np.round((vx + vlim - dv / 2) / dv).astype(np.int32)
    iy = np.round((vy + vlim - dv / 2) / dv).astype(np.int32)
    iz = np.round((vz + vlim - dv / 2) / dv).astype(np.int32)

    # clip to valid range
    mask = (ix >= 0) & (ix < vlen) & (iy >= 0) & (iy < vlen) & (iz >= 0) & (iz < vlen)
    cube[iz[mask], iy[mask], ix[mask]] = fvals[mask]
    return cube

################################
######## HERMITE TOOLS #########
################################

def run_hermite(cellid, reader, vlim, vlen, dv, order, sp_th, outdir):
    cube = build_cube(cellid, reader, vlim, vlen, dv)
    u    = get_drift_velocity_cube(cube, vlim, vlen)
    vth  = get_thermal_velocity_cube(cube, vlim, vlen, u)    
    hermite_cube = get_hermite_spectra_cube(cube, vlim, vlen, order, vth, u)    
    return hermite_cube, u, vth


def hermite_basis(v_ax, order, vth, u):
    """
    Compute orthonormal Hermite basis functions φ_n(v) = H_n((v-u)/vth) * G / norm_n
    where H_n are physicist's Hermite polynomials weighted by Gaussian G = exp(-x²/2).
    Returns array of shape (order, len(v_ax)).
    """
    x = (v_ax - u) / vth
    H = np.zeros((order, len(x)))
    H[0] = np.exp(-0.5 * x**2)
    if order > 1:
        H[1] = 2 * x * np.exp(-0.5 * x**2)
    for n in range(2, order):
        H[n] = 2 * x * H[n - 1] - 2 * (n - 1) * H[n - 2]
    norm = np.array([math.sqrt((2**n) * math.factorial(n) * math.sqrt(math.pi) * vth)
                     for n in range(order)])
    return H / norm[:, None]


def get_drift_velocity_cube(cube, vlim, vlen):
    """Compute bulk drift velocity [ux, uy, uz] from VDF cube."""
    dv   = 2 * vlim / vlen
    v_ax = velocity_axis(vlim, vlen, dv)
    n    = cube.sum() * dv**3
    ux   = np.einsum('zyx,x->', cube, v_ax) * dv**3 / n
    uy   = np.einsum('zyx,y->', cube, v_ax) * dv**3 / n
    uz   = np.einsum('zyx,z->', cube, v_ax) * dv**3 / n
    return np.array([ux, uy, uz])


def get_thermal_velocity_cube(cube, vlim, vlen, u):
    """
    Compute isotropic thermal velocity from VDF cube.

    Uses 1-D axis projections (cube.sum over the other two axes) instead of
    building three dense (vlen,vlen,vlen) meshgrids -- avoids ~330MB of
    temporary arrays per call and is ~100x+ faster at vlen~240 (measured
    1.8s -> 0.01s on a BIE bulk file cell).

    This also fixes a pre-existing correctness bug: with three equal-length
    inputs, np.meshgrid(v_ax, v_ax, v_ax, indexing='xy') returns X varying
    along axis 1 (the cube's y-axis), Y varying along axis 0 (z-axis), and
    Z varying along axis 2 (x-axis) -- but the old code paired X with u[0]
    (=ux), Y with u[1] (=uy), Z with u[2] (=uz), i.e. the wrong
    axis/velocity-component pairing. For an anisotropic drift velocity
    (the normal case) this silently inflated vth -- measured 524 km/s vs.
    the correct 180 km/s on one BIE cell with u ~ [-659, 4.6, -143] km/s --
    which in turn mis-scales the Hermite basis itself and plausibly
    contributed to cells failing to converge even at high order.
    """
    dv   = 2 * vlim / vlen
    v_ax = velocity_axis(vlim, vlen, dv)
    n    = cube.sum() * dv**3
    # cube[iz, iy, ix]: sum over the two axes NOT being profiled
    proj_x = cube.sum(axis=(0, 1))   # -> (vlen,), indexed like v_ax for x
    proj_y = cube.sum(axis=(0, 2))   # -> (vlen,), indexed like v_ax for y
    proj_z = cube.sum(axis=(1, 2))   # -> (vlen,), indexed like v_ax for z
    Pxx = np.sum(proj_x * (v_ax - u[0])**2) * dv**3
    Pyy = np.sum(proj_y * (v_ax - u[1])**2) * dv**3
    Pzz = np.sum(proj_z * (v_ax - u[2])**2) * dv**3
    return np.sqrt((Pxx + Pyy + Pzz) / (3 * n))


def get_hermite_spectra_cube(cube, vlim, vlen, order, vth, u):
    """
    Compute full Hermite spectra (cube indexing: all l,m,n < order).
    Returns array of shape (order, order, order).
    """
    dv   = 2 * vlim / vlen
    v_ax = velocity_axis(vlim, vlen, dv)
    Hx   = hermite_basis(v_ax, order, vth, u[0])
    Hy   = hermite_basis(v_ax, order, vth, u[1])
    Hz   = hermite_basis(v_ax, order, vth, u[2])
    # cube[iz, iy, ix];  spectra[l, m, n]  with z↔l, y↔m, x↔n
    spectra = np.einsum('zyx,nx,my,lz->lmn', cube, Hx, Hy, Hz) * dv**3
    return spectra


def reconstruct_vdf(spectra, vlim, vlen, order, vth, u):
    """Reconstruct log_cube from spectra, clip negatives."""
    dv    = 2 * vlim / vlen
    v_ax  = velocity_axis(vlim, vlen, dv)
    Hx    = hermite_basis(v_ax, order, vth, u[0])
    Hy    = hermite_basis(v_ax, order, vth, u[1])
    Hz    = hermite_basis(v_ax, order, vth, u[2])
    cube  = np.einsum('lmn,nx,my,lz->zyx', spectra, Hx, Hy, Hz)
    cube[cube < 0] = 0
    return cube



def to_log_shifted(cube, sp_th):
    """log(f) - log(sp_th), with floor at sp_th."""
    cube_clipped = np.maximum(cube, sp_th)
    return np.log(cube_clipped) - np.log(sp_th)


def from_log_shifted(log_cube, sp_th):
    """Inverse of to_log_shifted."""
    cube = np.exp(log_cube + np.log(sp_th))
    cube[cube <= sp_th * (1 + 1e-6)] = 0
    return cube


###########################################################################
########## ADAPTIVE  HERMITE : PARSEVAL CHECK FOR TRUNCATION  #############
###########################################################################


def adaptive_transform(vdf, vlim, vlen, vth, u,
                       max_order=14,
                       tolerance=0.15,
                       ):
    """
    Level-by-level adaptive Hermite transform with a Parseval stopping
    check: stops as soon as the running reconstructed power covers enough
    of the VDF's total power, WITHOUT paying for orders beyond the one
    actually reached -- that early exit is the whole point of "adaptive".

    The original per-triple implementation called
    np.einsum('zyx,x,y,z->', vdf, Hx[n], Hy[m], Hz[l]) once per (l,m,n)
    coefficient -- each call redoing a full O(vlen^3) contraction from
    scratch with NO reuse between coefficients, even ones sharing the same
    n or m. That made the cost scale with the number of coefficients
    computed (~max_order^3/6), not with how many levels were actually
    needed.

    This version separates the contraction into three axis passes (same
    identity as get_hermite_spectra_cube: C[l,m,n] = sum_x sum_y sum_z
    vdf * Hx[n] * Hy[m] * Hz[l]) and CACHES each partial result the first
    time it is needed:
      Px[n]    = sum_x vdf[z,y,x] * Hx[n,x]          -> (vlen,vlen), O(vlen^3)
      Pxy[n,m] = sum_y Px[n][z,y] * Hy[m,y]          -> (vlen,),     O(vlen^2)
      C[l,m,n] = sum_z Pxy[n,m][z] * Hz[l,z] * dv^3  -> scalar,      O(vlen)
    Px[n] is the dominant O(vlen^3) cost and is computed AT MOST ONCE per
    distinct n -- so a cell that stops at order_used=7 only ever pays for
    n in [0,7), restoring the adaptive early-exit saving that the naive
    per-triple loop had in spirit but not in practice (it was already
    O(vlen^3) per COEFFICIENT, so even stopping early was still far more
    expensive than it needed to be). A cell that needs the full max_order
    pays about what one full get_hermite_spectra_cube call would, still
    ~max_order^2/6 cheaper than the old per-triple loop (Pxy is O(vlen^2),
    not O(vlen^3)). Measured ~3000x faster than the old loop for a cell
    that uses the full max_order=20 on a 240^3 cube, for numerically
    identical coefficients.
    """
    dv   = 2.0 * vlim / vlen
    v_ax = velocity_axis(vlim, vlen, dv)

    # Precompute all basis functions
    Hx = hermite_basis(v_ax, max_order , vth, u[0])
    Hy = hermite_basis(v_ax, max_order , vth, u[1])
    Hz = hermite_basis(v_ax, max_order , vth, u[2])

    dv_vol = dv ** 3
    coeffs, deltas = {}, {}
    P_total = float(np.sum(vdf**2) * dv_vol)
    P_accum = 0.0

    Px_cache = {}    # n -> sum_x vdf[z,y,x] * Hx[n,x], shape (vlen, vlen)
    Pxy_cache = {}   # (n, m) -> sum_y Px[n][z,y] * Hy[m,y], shape (vlen,)

    def get_Px(n):
        row = Px_cache.get(n)
        if row is None:
            row = np.einsum('zyx,x->zy', vdf, Hx[n])
            Px_cache[n] = row
        return row

    def get_Pxy(n, m):
        key = (n, m)
        row = Pxy_cache.get(key)
        if row is None:
            row = np.einsum('zy,y->z', get_Px(n), Hy[m])
            Pxy_cache[key] = row
        return row

    for s in range(max_order):
        for l in range(s + 1):
            for m in range(s + 1 - l):
                n = s - l - m
                c = float(np.dot(get_Pxy(n, m), Hz[l]) * dv_vol)
                coeffs[(l, m, n)] = c
                P_accum += c * c
        delta = np.sqrt(max(0.0, 1.0 - P_accum / P_total))   # Parseval delta
        deltas[s] = delta
        if s > 2 and delta < tolerance:
            return coeffs, s + 1, deltas

    return coeffs, s + 1, deltas



def coeffs_into_cube(coeffs, order):    
    h_cube = np.zeros((order, order, order))
    for (l, m, n), c in coeffs.items():
        h_cube[l, m, n] = c
    return h_cube


def reconstruct_vdf_adaptive(coeffs, vlim, vlen, order, vth, u ):
    """reconstruction from a coefficient dict // for adaptive algo """
    dv   = 2.0 * vlim / vlen
    v_ax = velocity_axis(vlim, vlen, dv)
    Hx   = hermite_basis(v_ax, order, vth, u[0])
    Hy   = hermite_basis(v_ax, order, vth, u[1])
    Hz   = hermite_basis(v_ax, order, vth, u[2])
    rec = np.zeros((vlen, vlen, vlen), dtype=np.float64)
    for (l, m, n), c in coeffs.items():
        rec += c * np.einsum('x,y,z->zyx', Hx[n], Hy[m], Hz[l])
    return rec



#############################
############# I/O ###########
#############################

def run_hermite_and_save(cellid, reader, vlim, vlen, dv, order, sp_th, outdir):
    cube = build_cube(cellid, reader, vlim, vlen, dv)
    u    = get_drift_velocity_cube(cube, vlim, vlen)
    vth  = get_thermal_velocity_cube(cube, vlim, vlen, u)

    log_cube     = to_log_shifted(cube, sp_th)
    hermite_cube = get_hermite_spectra_cube(log_cube, vlim, vlen, order, vth, u)

    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(
        os.path.join(outdir, f"cell_{cellid}.npz"),
        cube=cube, hermite_coeffs=hermite_cube,
        u=u, vth=vth, vlim=vlim, vlen=vlen, order=order, sp_th=sp_th,
    )
    
    return hermite_cube, u, vth


def load_and_plot(npz_path):
    data          = np.load(npz_path)
    cube          = data['cube']
    hermite_cube  = data['hermite_coeffs']
    u             = data['u']
    vth           = float(data['vth'])
    vlim          = float(data['vlim'])
    vlen          = int(data['vlen'])
    order         = int(data['order'])
    sp_th         = float(data['sp_th'])

    log_cube_rec = reconstruct_vdf_cube_nolip(hermite_cube, vlim, vlen, order, vth, u)
    cube_rec     = from_log_shifted(log_cube_rec, sp_th)

    plot_vdf_reconstruction(cube, cube_rec, vlim, vlen, sp_th)

    residual = np.abs(cube - cube_rec)
    rel_err  = np.linalg.norm(residual) / np.linalg.norm(cube)
    print(f"{npz_path}: rel L2 error = {rel_err:.4e}")
    return rel_err


#############################
####  PLOTTING ROUTINES  ####
#############################

def plot_vdf_2d(vdf_ar, vlim, vlen, sp_th=1e-15):
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    UP_TH = np.max(np.sum(vdf_ar, axis=0))

    for ax, proj, label in [
        (ax1, np.sum(vdf_ar, axis=0), 'yz'),
        (ax2, np.sum(vdf_ar, axis=1), 'xz'),
        (ax3, np.sum(vdf_ar, axis=2), 'xy'),
    ]:
        im = ax.imshow(proj, origin='lower', extent=[-vlim, vlim, -vlim, vlim],
                       cmap='Spectral', norm=LogNorm(vmin=sp_th, vmax=UP_TH))
        ax.set_title(label)
        plt.colorbar(im, ax=ax, orientation='horizontal',
                     label='f(v) [s^3/m^6]', location='top')
    fig.savefig("vdf_2d.png", dpi=300)


def plot_vdf_reconstruction(cube, cube_rec, vlim, vlen, sp_th):
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    UP_TH  = max(np.max(np.sum(cube, axis=0)), np.max(np.sum(cube_rec, axis=0)))
    labels = ['sum_z (xy)', 'sum_y (xz)', 'sum_x (yz)']

    for col, (ax_sum, label) in enumerate(zip([0, 1, 2], labels)):
        orig_proj = np.sum(cube,     axis=ax_sum)
        rec_proj  = np.sum(cube_rec, axis=ax_sum)
        diff_proj = np.abs(orig_proj - rec_proj)

        im0 = axes[0, col].imshow(orig_proj, origin='lower',
                                   extent=[-vlim, vlim, -vlim, vlim],
                                   cmap='Spectral',
                                   norm=LogNorm(vmin=sp_th, vmax=UP_TH))
        axes[0, col].set_title(f'Original ({label})')
        plt.colorbar(im0, ax=axes[0, col], orientation='horizontal', location='top')

        im1 = axes[1, col].imshow(rec_proj, origin='lower',
                                   extent=[-vlim, vlim, -vlim, vlim],
                                   cmap='Spectral',
                                   norm=LogNorm(vmin=sp_th, vmax=UP_TH))
        axes[1, col].set_title(f'Reconstructed ({label})')
        plt.colorbar(im1, ax=axes[1, col], orientation='horizontal', location='top')

        im2 = axes[2, col].imshow(diff_proj, origin='lower',
                                   extent=[-vlim, vlim, -vlim, vlim],
                                   cmap='inferno')
        axes[2, col].set_title(f'|Diff| ({label})')
        plt.colorbar(im2, ax=axes[2, col], orientation='horizontal', location='top')

    fig.tight_layout()
    fig.savefig("vdf_reconstruction.png", dpi=300)
    plt.close(fig)


