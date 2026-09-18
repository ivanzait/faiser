import numpy as np
import matplotlib.pyplot as plt
import vdf_tools as vt
import pytools as pt

#reader = pt.vlsvfile.VlsvReader("/wrk-vakka/group/spacephysics/vlasiator/2D/BIE/restart.0001125.2025-06-06_22-31-27.vlsv")

data_dir = "/scratch/project_2000203/zaitsevi/runs/reconnection_beta025_large"
outdir = "/scratch/project_2000203/zaitsevi/faiser/data_initialization"
reader = pt.vlsvfile.VlsvReader(data_dir + "/bulk.0000011.vlsv")

vlim, vlen, dv = vt.get_vdf_parameters(reader)
print('vlim, vlen, dv:', vlim, vlen, dv)

vdf_ar = []
coords_ar = []
cellid_ar = []
hermite_ar = []
v_mean_ar = [] ## hermite need mean (bulk) velocity
v_th_ar = [] ## hermite also need thermal velocity

SP_TH = vt.get_sparse_threshold(reader, pop="proton") ## 1e-15 ## sparsity threshold
HN = 4  # HERMITE ORDER

# #for cellid in range(0, 1500 * 1200 + 100):
for cellid in range(0, 1031):
    if cellid%10==0:
        print(cellid)

    if not reader.cellid_has_vdf(cellid, "proton"):
        continue

    cube = vt.build_cube(cellid, reader, vlim, vlen, dv)
    print('cube shape:', cube.shape)

    hermite_cube, u, vth = vt.run_hermite(cellid, reader, vlim, vlen, dv, HN, SP_TH, outdir)
    print('hermite_cube shape:', hermite_cube.shape)
        
    coords = reader.get_cell_coordinates(cellid)
    vdf_ar.append(cube)
    v_mean_ar.append(u)
    v_th_ar.append(vth)
    hermite_ar.append(hermite_cube)
    coords_ar.append(coords)
    cellid_ar.append(cellid)
    
np.savez_compressed(outdir+"/vdf_data.npz", vdfs=vdf_ar, coords=coords_ar, cellids=cellid_ar, v_means=v_mean_ar, v_ths=v_th_ar, hermite_coeffs=hermite_ar) 