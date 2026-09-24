import numpy as np
import matplotlib.pyplot as plt
import pytools as pt
import vdf_tools as vt
from adaptive_hermite import adaptive_transform, coeffs_into_cube, plot_hcube

#reader = pt.vlsvfile.VlsvReader("/wrk-vakka/group/spacephysics/vlasiator/2D/BIE/restart.0001125.2025-06-06_22-31-27.vlsv")

#data_dir = "/scratch/project_2000203/zaitsevi/runs/reconnection_beta025_large"
# outdir = "/scratch/project_2000203/zaitsevi/faiser/data_initialization"

data_dir = "/Users/ivanzait/Downloads"
outdir = data_dir
reader = pt.vlsvfile.VlsvReader(data_dir + "/bulk.0000055.vlsv")

vlim, vlen, dv = vt.get_vdf_parameters(reader)
print('vlim, vlen, dv:', vlim, vlen, dv)

vdf_ar = []
coords_ar = []
cellid_ar = []
hermite_ar = []
v_mean_ar = [] ## hermite need mean (bulk) velocity
v_th_ar = [] ## hermite also need thermal velocity

#SP_TH = vt.get_sparse_threshold(reader, pop="proton") ## 1e-15 ## sparsity threshold
SP_TH =  1e-15 ## sparsity threshold
HN = 20  # HERMITE ORDER


# pick a single, complex VDF cell nearest to the center of the box to
# stress-test the tetrahedral transform
target_cellid = vt.get_nearest_vdf_cellid(reader, coords=[0,0,0])
print('target cellid (nearest to box center with a stored VDF):', target_cellid)

for cellid in [target_cellid]:  
    if not reader.cellid_has_vdf(cellid, "proton"):
        raise RuntimeError(f'cellid {cellid} has no VDF -- pick a different coordinate')
    try:
        cube = vt.build_cube(cellid, reader, vlim, vlen, dv)
    except ValueError as e:
        print(f'skipping cellid {cellid}: {e}')
        continue
    print('cube shape:', cube.shape)
    ### cube transform
    # hermite_cube, u, vth = vt.run_hermite(cellid, reader, vlim, vlen, dv, HN, SP_TH, outdir)
    
    
    ### tetrahedral transform
    u = vt.get_drift_velocity_cube(cube, vlim, vlen)
    vth = vt.get_thermal_velocity_cube(cube, vlim, vlen, u)
    # max_order is the TOTAL order s=l+m+n, not a per-axis cap -- a single
    # index can reach max_order itself (other two zero). To match the cubic
    # convention (order=HN -> indices 0..HN-1), cap total order at HN-1.
    coeffs, eps_rel, s_stop, history = adaptive_transform( cube, vlim, vlen, vth, u, max_order=HN , rel_threshold=0.03, even_first=False, sp_th=SP_TH, sparse_mask=None,
                                                            track_log_eps=True,verbose=False)

    h_cube = coeffs_into_cube(coeffs, HN)  # indices l,m,n run 0..HN-1, matching cubic order=HN
        
    coords = reader.get_cell_coordinates(cellid)
    vdf_ar.append(cube)
    v_mean_ar.append(u)
    v_th_ar.append(vth)
    hermite_ar.append(h_cube)
    coords_ar.append(coords)
    cellid_ar.append(cellid)
    
    np.savez_compressed(outdir+"/vdf_data.npz", vdfs=vdf_ar, coords=coords_ar, cellids=cellid_ar, v_means=v_mean_ar, v_ths=v_th_ar, hermite_coeffs=hermite_ar) 
    