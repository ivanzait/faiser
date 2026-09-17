"""
One-off script: generate the "feature evolution" schematic figures used in
the presentation (make_presentation.py) -- low-S coefficients, 1D axis
spectra, and the local 3x3x3 patch -- illustrated on a real lobe cell
(cid=16, near-Maxwellian) vs a real current-sheet cell (cid=672,
strongly non-Maxwellian), reusing the already-cached full decomposition.

Usage
-----
python3 hermite_ml/make_feature_figures.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytools as pt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vdf_tools import (get_vdf_parameters, build_cube, get_drift_velocity_cube,
                       get_thermal_velocity_cube, get_sparse_threshold, velocity_axis)
from hermite_ml.adaptive_hermite import array_to_coeffs
from hermite_ml.decomposition_cache import cache_path_for, load_cache
from hermite_ml.corrector_model import (coeffs_dict_to_vector, axis_spectra,
                                        pad_field, extract_patches)

BULKDIR   = 'reconnection_2d_beta025'
BULKFILE  = 'bulk.0000024.vlsv'
FULL_ORDER = 22
S_LOW = 2
CID_LOBE = 16     # near-Maxwellian, far from current sheet
CID_SHEET = 672   # current-sheet center, strongly non-Maxwellian
PLOTDIR = os.path.join(os.path.dirname(__file__), 'plots')


def main():
    os.makedirs(PLOTDIR, exist_ok=True)
    reader = pt.vlsvfile.VlsvReader(os.path.join(BULKDIR, BULKFILE))
    vlim, vlen, dv = get_vdf_parameters(reader)
    sp_th = get_sparse_threshold(reader)
    log_sp_th = float(np.log(sp_th))
    v_ax = velocity_axis(vlim, vlen, dv)

    cache_path = cache_path_for(BULKDIR, BULKFILE, FULL_ORDER)
    cache = load_cache(cache_path)

    cells = {}
    for cid, label in [(CID_LOBE, 'lobe (cid=16)'), (CID_SHEET, 'current sheet (cid=672)')]:
        entry = cache[cid]
        coeffs = array_to_coeffs(entry['coeffs_flat'], FULL_ORDER)
        cube = build_cube(cid, reader, vlim, vlen, dv)
        u, vth = entry['u'], entry['vth']
        f_rec_full = entry['f_rec_full']
        sparse_mask = cube >= sp_th
        cells[cid] = dict(label=label, coeffs=coeffs, cube=cube, u=u, vth=vth,
                          f_rec_full=f_rec_full, sparse_mask=sparse_mask)

    colors = {CID_LOBE: '#3B7DD8', CID_SHEET: '#D8543B'}

    # === Figure 1: low-S coefficients (S_low <= 2, 10 numbers) ===
    fig, ax = plt.subplots(figsize=(9, 5))
    labels_low = []
    for s in range(S_LOW + 1):
        for l in range(s + 1):
            for m in range(s + 1 - l):
                n = s - l - m
                labels_low.append(f'({l},{m},{n})')
    width = 0.38
    xpos = np.arange(len(labels_low))
    for i, cid in enumerate([CID_LOBE, CID_SHEET]):
        C_low = np.array(coeffs_dict_to_vector(cells[cid]['coeffs'], S_LOW))
        ax.bar(xpos + (i - 0.5) * width, C_low, width=width,
              label=cells[cid]['label'], color=colors[cid], alpha=0.85)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels_low, rotation=45, ha='right')
    ax.set_xlabel('Hermite index (l, m, n)')
    ax.set_ylabel(r'$C_{l,m,n}$')
    ax.set_title(f'Feature 1: low-order Hermite coefficients ($S_\\mathrm{{low}}$={S_LOW}, 10 numbers)')
    ax.axhline(0, color='gray', lw=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTDIR, 'feature_lowS.png'), dpi=150)
    plt.close(fig)

    # === Figure 2: 1D axis power spectra (3 x (FULL_ORDER+1) numbers) ===
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    axis_names = ['x (spec_x)', 'y (spec_y)', 'z (spec_z)']
    for cid in [CID_LOBE, CID_SHEET]:
        spec = axis_spectra(cells[cid]['coeffs'], FULL_ORDER)
        total_power = float(spec[0].sum())
        spec_norm = spec / max(total_power, 1e-30)
        log_spec = np.log10(np.maximum(spec_norm, 1e-20))
        for a in range(3):
            axes[a].plot(np.arange(FULL_ORDER + 1), log_spec[a], marker='o',
                        ms=3, color=colors[cid], label=cells[cid]['label'])
    for a in range(3):
        axes[a].set_title(axis_names[a])
        axes[a].set_xlabel('Hermite order n')
        axes[a].grid(alpha=0.3)
    axes[0].set_ylabel(r'$\log_{10}$(normalized power)')
    axes[0].legend(fontsize=9)
    fig.suptitle('Feature 2: 1-D Hermite-space power spectra (3x(order+1) = 69 numbers)')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTDIR, 'feature_axis_spectra.png'), dpi=150)
    plt.close(fig)

    # === Figure 3: local 3x3x3 patch ===
    cid = CID_SHEET
    cube = cells[cid]['cube']
    f_rec_full = cells[cid]['f_rec_full']
    sparse_mask = cells[cid]['sparse_mask']
    sp_th_log = log_sp_th

    log_rec = np.log(np.maximum(f_rec_full, sp_th))
    log_rec_padded = pad_field(log_rec.astype(np.float32), sp_th_log)

    # pick a voxel with a large true error (a Gibbs artifact site) inside mask
    log_true = np.log(np.maximum(cube, sp_th))
    err = np.abs(log_true - log_rec)
    err_masked = np.where(sparse_mask, err, -1)
    iz0, iy0, ix0 = np.unravel_index(np.argmax(err_masked), err_masked.shape)

    patch = extract_patches(log_rec_padded, np.array([iz0]), np.array([iy0]), np.array([ix0]))
    patch = patch.reshape(3, 3, 3)  # (dz, dy, dx)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    vmin, vmax = patch.min(), patch.max()
    for k, dz in enumerate([-1, 0, 1]):
        im = axes[k].imshow(patch[k], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[k].set_title(f'dz = {dz}' + ('  (center slice)' if dz == 0 else ''))
        axes[k].set_xticks([0, 1, 2]); axes[k].set_xticklabels([-1, 0, 1])
        axes[k].set_yticks([0, 1, 2]); axes[k].set_yticklabels([-1, 0, 1])
        axes[k].set_xlabel('dx')
        if k == 0:
            axes[k].set_ylabel('dy')
        if dz == 0:
            axes[k].scatter([1], [1], marker='x', color='red', s=120,
                           label='query voxel v')
            axes[k].legend(loc='upper right', fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label=r'$\log f_\mathrm{rec\_full}$')
    fig.suptitle(f'Feature 3: local 3x3x3 patch of log($f_\\mathrm{{rec\\_full}}$) '
                f'around a Gibbs-artifact voxel (27 numbers, cid={cid})')
    fig.savefig(os.path.join(PLOTDIR, 'feature_patch.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("Saved: feature_lowS.png, feature_axis_spectra.png, feature_patch.png -> ",
         PLOTDIR)


if __name__ == '__main__':
    main()
