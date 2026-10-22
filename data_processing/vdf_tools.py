"""
VDF tools for Vlasiator bulk files and the adaptive Hermite transform.

  reading   get_vdf_parameters, build_cube, get_sparse_threshold, get_inner_boundary_cells,
            get_cellid_at_coords, get_nearest_vdf_cellid
  grid      velocity_axis, get_vdf_bounding_box, apply_bounding_box
  moments   get_drift_velocity_cube, get_thermal_velocity_cube
  Hermite   hermite_basis, adaptive_transform, coeffs_into_cube, reconstruct_vdf_adaptive

Cubes follow cube[iz, iy, ix] on a cubic velocity mesh of vlen cells per axis with cell centres
-vlim + dv/2 + i*dv (see velocity_axis). Velocities are in m/s.
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
    """
    Velocity mesh of the proton population as (vlim, vlen, dv): vlen = 4 x the number of blocks per
    axis (Vlasiator blocks are 4x4x4 velocity cells), vlim = |vx_min|, dv = cell size. Assumes a cubic
    mesh symmetric about 0.
    """
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
    Cell-centre velocity axis, v[i] = -vlim + dv/2 + i*dv, matching the binning in build_cube.

    Not np.linspace(-vlim, vlim, vlen): its spacing is 2*vlim/(vlen-1), so the points miss the cell
    centres and the discrete normalisation of the Hermite basis breaks.
    """
    
    return -vlim + dv / 2.0 + np.arange(vlen) * dv


def get_sparse_threshold(reader, pop="proton"):
    """
    Sparsity threshold `<pop>_sparse.minValue` from the run config. Vlasiator keeps a 4x4x4 velocity
    block only if some cell in it reaches this value (or a neighbouring block does); every cell of a kept
    block is stored, including values below the threshold (the halo).
    """
    cfg = reader.get_config()
    return float(cfg[f'{pop}_sparse']['minValue'][0])


def get_inner_boundary_cells(reader):
    """
    Spatial cell ids inside the inner boundary (the copysphere radius around its centre). Their VDFs
    are boundary conditions, not physics, and global runs skip them.
    """
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
    Dense VDF of one spatial cell: cube[iz, iy, ix] of shape (vlen, vlen, vlen), float32.

    Only the velocity cells stored in the file are filled; dropped blocks stay 0. Stored values below the
    sparsity threshold (the halo) are kept as they are, and tiny negatives (~ -1e-20) occur.
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


def get_vdf_bounding_box(cube, vlim, vlen):   
    """
    Axis-aligned box of everything stored for the cell (f > 0, halo included), used to crop a
    reconstruction to the data region with apply_bounding_box. The box cannot be recovered from the
    coefficients, so it has to be stored with them. For Vlasiator data it is block-aligned.

    Returns None for an empty cube, otherwise a dict:
        idx       ((iz0, iz1), (iy0, iy1), (ix0, ix1))   inclusive index bounds
        vel       the same bounds as cell-centre velocities [m/s]
        n_active  number of cells with f > 0
    """
    mask = cube > 0 
    if not mask.any():
        return None

    iz_idx, iy_idx, ix_idx = np.nonzero(mask)
    idx_bounds = tuple((int(i.min()), int(i.max())) for i in (iz_idx, iy_idx, ix_idx))

    dv = 2 * vlim / vlen
    v_ax = velocity_axis(vlim, vlen, dv)
    vel_bounds = tuple((float(v_ax[lo]), float(v_ax[hi])) for lo, hi in idx_bounds)

    return {'idx': idx_bounds, 'vel': vel_bounds, 'n_active': int(mask.sum())}


def apply_bounding_box(cube, idx_bounds):
    """
    Copy of `cube` with everything outside the index box `idx_bounds`
    (get_vdf_bounding_box()['idx']) set to 0.
    """
    out = np.zeros_like(cube)
    (iz0, iz1), (iy0, iy1), (ix0, ix1) = idx_bounds
    out[iz0:iz1 + 1, iy0:iy1 + 1, ix0:ix1 + 1] = cube[iz0:iz1 + 1, iy0:iy1 + 1, ix0:ix1 + 1]
    return out



################################
######## HERMITE TOOLS #########
################################




def hermite_basis(v_ax, order, vth, u):
    """
    Orthonormal Hermite functions sampled on `v_ax`, shape (order, len(v_ax)):

        phi_n(v) = H_n(x) exp(-x^2/2) / sqrt(2^n n! sqrt(pi) vth),   x = (v - u) / vth,

    with H_n the physicists' Hermite polynomials (H_n = 2x H_{n-1} - 2(n-1) H_{n-2}). They satisfy
    int phi_n phi_m dv = delta_nm in the continuum; on the grid this holds only while the oscillations
    of phi_n are resolved, i.e. vth spans several cells dv.
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
    """
    Bulk velocity [ux, uy, uz] [m/s]: first moment of the cube over the density n = sum(f) dv^3.
    """
    dv   = 2 * vlim / vlen
    v_ax = velocity_axis(vlim, vlen, dv)
    n    = cube.sum() * dv**3
    ux   = np.einsum('zyx,x->', cube, v_ax) * dv**3 / n
    uy   = np.einsum('zyx,y->', cube, v_ax) * dv**3 / n
    uz   = np.einsum('zyx,z->', cube, v_ax) * dv**3 / n
    return np.array([ux, uy, uz])


def get_thermal_velocity_cube(cube, vlim, vlen, u):
    """
    Isotropic thermal speed sqrt((Pxx + Pyy + Pzz) / (3 n)) [m/s] about the drift u = [ux, uy, uz],
    from the 1D projections of the cube onto each velocity axis.

    For a VDF with several populations this is their overall spread, not the width of any one of them,
    and the Hermite basis built on it is correspondingly wide.
    """
    dv   = 2 * vlim / vlen
    v_ax = velocity_axis(vlim, vlen, dv)
    n  = cube.sum() * dv**3
    # cube[iz, iy, ix]: sum over the two axes NOT being profiled
    proj_x = cube.sum(axis=(0, 1))   # -> (vlen,), indexed like v_ax for x
    proj_y = cube.sum(axis=(0, 2))   # -> (vlen,), indexed like v_ax for y
    proj_z = cube.sum(axis=(1, 2))   # -> (vlen,), indexed like v_ax for z
    Pxx = np.sum(proj_x * (v_ax - u[0])**2) * dv**3
    Pyy = np.sum(proj_y * (v_ax - u[1])**2) * dv**3
    Pzz = np.sum(proj_z * (v_ax - u[2])**2) * dv**3

    return np.sqrt((Pxx + Pyy + Pzz) / (3 * n))









###########################################################################
########## ADAPTIVE  HERMITE : PARSEVAL CHECK FOR TRUNCATION  #############
###########################################################################


def adaptive_transform(vdf, vlim, vlen, vth, u,
                       max_order=14,
                       tolerance=0.15,
                       sqrt_f=True,
                       ):
    """
    Adaptive Hermite transform of a VDF cube, level by level, with a Parseval stopping rule.

    Coefficients C[l, m, n] = sum vdf * Hz[l] * Hy[m] * Hx[n] * dv^3 (index order z, y, x, like the
    cube) are computed in levels s = l + m + n (tetrahedral truncation; levels 0..N-1 hold
    N(N+1)(N+2)/6 coefficients). After each level

        delta_s = sqrt(1 - sum C^2 / (sum vdf^2 dv^3))

    is the relative L2 error of the truncated series (Parseval). The loop stops at the first s > 2 with
    delta_s < tolerance, or at max_order; higher levels are never computed.

    sqrt_f=True expands g = sqrt(f) instead of f, with the basis scaled to sqrt(2)*vth (the width of the
    square root of a Maxwellian); reconstruct_vdf_adaptive(..., sqrt_f=True) squares the result back.
    The reconstruction is then non-negative, ringing outside the support drops from ~eps*f_max to
    ~eps^2*f_max, and delta measures the density misfit (int g^2 = n). Pass the physical vth in both
    modes; the scaling is done here. Coefficients from the two modes are not interchangeable.

    Cost: the contraction is split into axis passes, each cached the first time it is needed,
        Px[n]    = sum_x vdf * Hx[n]         (vlen, vlen)   O(vlen^3), once per n
        Pxy[n,m] = sum_y Px[n] * Hy[m]       (vlen,)        O(vlen^2)
        C[l,m,n] = sum_z Pxy[n,m] * Hz[l] * dv^3            O(vlen)
    so a cell that stops early pays only for the levels it reached.

    Caveats: delta is an L2 measure dominated by the peak, so it says little about the tails. When vth
    is only ~1-2 dv (cold VDFs) the sampled basis is not orthonormal and delta can drop to 0 spuriously.

    Returns
        coeffs      {(l, m, n): C} for every level up to the stopping one
        order_used  number of levels computed (s + 1); the `order` for reconstruct_vdf_adaptive
        deltas      {s: delta_s}
    """
    if sqrt_f:
        vdf = np.sqrt(np.clip(vdf, 0, None))    # Vlasiator cubes have tiny negatives (~-1e-20)
        vth = np.sqrt(2) * vth

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




def coeffs_into_cube(coeffs, hermite_order):
    """
    Dense (hermite_order,)*3 array from a {(l, m, n): c} dict, zero where a coefficient is absent.
    """
    h_cube = np.zeros([hermite_order, hermite_order, hermite_order])
    for k, v in coeffs.items():
        h_cube[k] = v
    return h_cube


def reconstruct_vdf_adaptive(coeffs, vlim, vlen, order, vth, u, sqrt_f=True):
    """
    VDF cube from a coefficient dict: f[z, y, x] = sum C[l, m, n] Hz[l](z) Hy[m](y) Hx[n](x), done as
    one einsum over the dense coefficient cube.

    `order`, `vth`, `u` and `sqrt_f` must match the adaptive_transform call that produced `coeffs`
    (order = its order_used, or the side of a zero-padded dense spectrum). sqrt_f=True rebuilds
    g = sqrt(f) on the sqrt(2)*vth basis and returns g**2, which is non-negative.
    """
    if sqrt_f:
        vth = np.sqrt(2) * vth
    dv = 2.0 * vlim / vlen
    v_ax = velocity_axis(vlim, vlen, dv)
    Hx, Hy, Hz = (hermite_basis(v_ax, order, vth, u[i]) for i in range(3))
    h_cube = coeffs_into_cube(coeffs, order)
    rec = np.einsum('lmn,nx,my,lz->zyx', h_cube, Hx, Hy, Hz, optimize=True)
    return rec**2 if sqrt_f else rec





