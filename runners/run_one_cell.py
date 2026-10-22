import numpy as np
import os, sys
sys.path.insert(0, "/home/ivanzait/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytools as pt
from tools import vdf_tools as vt

runid='TSI'
tstep=30


BULK_FILE = f"/turso/group/spacephysics/vlasiator/data/L0/1D/{runid}/bulk.{tstep:07d}.vlsv"
CELL_ID   = 200
MAX_ORDER = 33
TOLERANCE = 0.005
OUTFILE   = os.path.join(os.path.dirname(__file__), '..', f'data/one_cell_run_{runid}_tstep{tstep}_cell{CELL_ID}.npz')
SQRT_F    = True   # expand sqrt(f), see vdf_tools.adaptive_transform; saved in the npz for the reconstruction
DENSE_SPECTRA = False  # if True, save all coefficients up to MAX_ORDER, not just the nonzero ones  


def main():

    reader = pt.vlsvfile.VlsvReader(BULK_FILE)
    vlim,vlen,dv = vt.get_vdf_parameters(reader)
    
    coord = reader.get_cell_coordinates(CELL_ID)
    cube = vt.build_cube(CELL_ID, reader, vlim, vlen, dv).astype(np.float64)
    u    = vt.get_drift_velocity_cube(cube, vlim, vlen)
    vth  = vt.get_thermal_velocity_cube(cube, vlim, vlen, u)

    coeffs, order, deltas = vt.adaptive_transform(
        cube, vlim, vlen, vth, u, max_order=MAX_ORDER, tolerance=TOLERANCE, sqrt_f=SQRT_F)
    
    print(f'order_used={order}  n_coeffs={len(coeffs)}  final Parseval delta={deltas[order - 1]:.4f}')
    
    if DENSE_SPECTRA:
        coeffs = vt.coeffs_into_cube(coeffs, MAX_ORDER)
    
    
    bbox = vt.get_vdf_bounding_box(cube, vlim, vlen)['idx']
    
    
    save_kwargs = dict(
        cellids=np.array(CELL_ID), coords=np.array(coord),
        v_means=np.array(u), v_ths=np.array(vth),
        hermite_coeffs=np.array(coeffs), order_used=np.array(order),
        bbox_idx=np.array(bbox),   # (3, 2): per axis (z, y, x), inclusive index bounds
        vlim=vlim, vlen=vlen, dense=DENSE_SPECTRA, sqrt_f=SQRT_F, cube=cube,
        bulk_file=BULK_FILE, runid=runid, tstep=tstep,   # where the original is, for one_cell_view.py
    )
    
    np.savez_compressed(OUTFILE, **save_kwargs)


if __name__ == '__main__':
    main()
