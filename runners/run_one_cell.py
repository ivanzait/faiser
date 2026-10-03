import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/ivanzait/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytools as pt
from data_processing import vdf_tools as vt
from plot_routines.plot_window import plot_window

plt.rcParams['text.usetex'] = False   # importing analysator switches it on; there is no latex here

runid='BIE'
tstep=1125

BULK_FILE = f"/turso/group/spacephysics/vlasiator/data/L0/2D/{runid}/bulk.000{tstep}.vlsv"
CELL_ID   = 795501
MAX_ORDER = 20
TOLERANCE = 0.05
WINDOW    = 'lanczos'      # or 'raised_cosine'
OUTFILE   = os.path.join(os.path.dirname(__file__), '..', f'data/one_cell_run_{runid}_tstep{tstep}.npz')
DENSE_SPECTRA = False  # if True, save all coefficients up to MAX_ORDER, not just the nonzero ones  


def main():

    reader = pt.vlsvfile.VlsvReader(BULK_FILE)
    vlim, vlen, dv = vt.get_vdf_parameters(reader)
    sp_th = vt.get_sparse_threshold(reader)
    
    coord = reader.get_cell_coordinates(CELL_ID)
    cube = vt.build_cube(CELL_ID, reader, vlim, vlen, dv).astype(np.float64)
    u    = vt.get_drift_velocity_cube(cube, vlim, vlen)
    vth  = vt.get_thermal_velocity_cube(cube, vlim, vlen, u)

    coeffs, order, deltas = vt.adaptive_transform(
        cube, vlim, vlen, vth, u, max_order=MAX_ORDER, tolerance=TOLERANCE)
    coeffs = vt.apply_spectral_window(coeffs, order, window=WINDOW)
    
    if DENSE_SPECTRA==True:
        coeffs = vt.coeffs_into_cube(coeffs, MAX_ORDER)
    
    
    bbox = vt.get_vdf_bounding_box(cube, vlim, vlen, sp_th=sp_th, margin=1)['idx']
    print(f'order_used={order}  n_coeffs={len(coeffs)}  final Parseval delta={deltas[order - 1]:.4f}')
    

    save_kwargs = dict(
        cellids=np.array(CELL_ID), coords=np.array(coord),
        v_means=np.array(u), v_ths=np.array(vth),
        hermite_coeffs=np.array(coeffs), order_used=np.array(order),
        bbox_idx=np.array(bbox),   # (3, 2): per axis (z, y, x), inclusive index bounds
        vlim=vlim, vlen=vlen, dense=DENSE_SPECTRA
    )
    
    np.savez_compressed(OUTFILE, **save_kwargs)


if __name__ == '__main__':
    main()
