import os, sys
import numpy as np
import sys
sys.path.insert(0, "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator")
import pytools as pt
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
from data_processing import vdf_tools as vt
#from adaptive_hermite import cubic_transform

BULK_FILE   = "/Users/ivanzait/Downloads/bulk.0000055.vlsv"
CELL_COORDS = [1e+6, 0, 1e+6]   # None = center of box; nearest cell with a VDF is used
ORDER       = 26          # indices l,m,n run 0..ORDER-1 for both transforms
SP_TH       = 1e-15

def main():    
    reader = pt.vlsvfile.VlsvReader(BULK_FILE)

    vlim, vlen, dv = vt.get_vdf_parameters(reader)
    cellid = vt.get_nearest_vdf_cellid(reader, coords=CELL_COORDS)
    cube = vt.build_cube(cellid, reader, vlim, vlen, dv)
    mask = cube < SP_TH
    u    = vt.get_drift_velocity_cube(cube, vlim, vlen)
    vth  = vt.get_thermal_velocity_cube(cube, vlim, vlen, u)

    spectra, s_stop, delta = vt.adaptive_transform(cube, vlim, vlen, vth, u, max_order=ORDER, tolerance=0.4)
    print('stopped at order:', s_stop)

    vdf_rec = vt.reconstruct_vdf_adaptive(spectra, vlim, vlen, ORDER, vth, u )
    vdf_rec[mask] = 0.0
    eps_rel = float(np.linalg.norm(cube - vdf_rec) / np.linalg.norm(cube))

    print(f"\n{'':12s}  {'n_coeffs':>8}  {'eps_rel':>8}  ")
    print( f"{'tetrahedral':12s}  {len(spectra):8d}  {eps_rel:8.5f}" )


if __name__ == '__main__':
    main()
