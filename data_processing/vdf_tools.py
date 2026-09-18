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
    """
    Read velocity mesh parameters from VLSV file.
    Returns (vlim, vlen, dv) — symmetric mesh assumed.
    """
    extents = reader.get_velocity_mesh_size(pop="proton")
    nx = int(extents[0] * 4)
    ny = int(extents[1] * 4)
    nz = int(extents[2] * 4)

    # Read actual velocity mesh extent from file — do NOT hardcode dv
    vx_min, vy_min, vz_min, vx_max, vy_max, vz_max = \
        reader.get_velocity_mesh_extent(pop="proton")

    dvx = (vx_max - vx_min) / nx
    dvy = (vy_max - vy_min) / ny
    dvz = (vz_max - vz_min) / nz
    dv  = dvx   # assume symmetric

    vxmin = vx_min
    vymin = vy_min
    vzmin = vz_min

    print("ns:", nx, ny, nz)
    print("dvs:", dvx, dvy, dvz)
    print("mins:", vxmin, vymin, vzmin)
    vlim, vlen = abs(vxmin), nx
    return vlim, vlen, dv


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

    log_cube     = to_log_shifted(cube, sp_th)
    hermite_cube = get_hermite_spectra_cube(log_cube, vlim, vlen, order, vth, u)
    
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
    """Compute isotropic thermal velocity from VDF cube."""
    dv   = 2 * vlim / vlen
    v_ax = velocity_axis(vlim, vlen, dv)
    n    = cube.sum() * dv**3
    X, Y, Z = np.meshgrid(v_ax, v_ax, v_ax, indexing='xy')  # cube[z,y,x]
    Pxx  = np.sum(cube * (X - u[0])**2) * dv**3
    Pyy  = np.sum(cube * (Y - u[1])**2) * dv**3
    Pzz  = np.sum(cube * (Z - u[2])**2) * dv**3
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


def reconstruct_vdf_cube(spectra, vlim, vlen, order, vth, u):
    """Reconstruct log_cube from spectra, clip negatives."""
    dv    = 2 * vlim / vlen
    v_ax  = velocity_axis(vlim, vlen, dv)
    Hx    = hermite_basis(v_ax, order, vth, u[0])
    Hy    = hermite_basis(v_ax, order, vth, u[1])
    Hz    = hermite_basis(v_ax, order, vth, u[2])
    cube  = np.einsum('lmn,nx,my,lz->zyx', spectra, Hx, Hy, Hz)
    cube[cube < 0] = 0
    return cube





def reconstruct_vdf_cube_nolip(spectra, vlim, vlen, order, vth, u):
    """Reconstruct log_cube from spectra, no clipping."""
    dv   = 2 * vlim / vlen
    v_ax = velocity_axis(vlim, vlen, dv)
    Hx   = hermite_basis(v_ax, order, vth, u[0])
    Hy   = hermite_basis(v_ax, order, vth, u[1])
    Hz   = hermite_basis(v_ax, order, vth, u[2])
    return np.einsum('lmn,nx,my,lz->zyx', spectra, Hx, Hy, Hz)


def to_log_shifted(cube, sp_th):
    """log(f) - log(sp_th), with floor at sp_th."""
    cube_clipped = np.maximum(cube, sp_th)
    return np.log(cube_clipped) - np.log(sp_th)


def from_log_shifted(log_cube, sp_th):
    """Inverse of to_log_shifted."""
    cube = np.exp(log_cube + np.log(sp_th))
    cube[cube <= sp_th * (1 + 1e-6)] = 0
    return cube


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
