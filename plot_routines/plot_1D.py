import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../data_processing'))
import pytools as pt
import vdf_tools as vt


BULK_FILE = "/Users/ivanzait/Documents/Documents_LM4500/Codes/Vlasiator/runs/two_stream_2/out_dense/bulk.0000043.vlsv"
var = 'proton/vg_rho'

f = pt.vlsvfile.VlsvReader(BULK_FILE)
cellids = f.read_variable('CellID')
#f.list()
var_vals = f.read_variable(var)
print('len:', len(var_vals))


vlim, vlen, dv = vt.get_vdf_parameters(f)
fx_store = np.zeros( [len(cellids), vlen] )

for i, cellid in enumerate(cellids):
    if i%20 == 0:
        print(f"Processing cell {i}")

    vdf = f.read_velocity_cells(cellid, "proton")   
    
    fvals = np.fromiter(vdf.values(), dtype=np.float32)
    vcellids = np.fromiter(vdf.keys(), dtype=np.int64)
    vcoords  = f.get_velocity_cell_coordinates(vcellids)
    vx, vy, vz = vcoords[:, 0], vcoords[:, 1], vcoords[:, 2]

    ix = np.round((vx + vlim - dv / 2) / dv).astype(np.int32)    
    m = (ix >= 0) & (ix < vlen) 
    fx_store[i] = np.bincount(ix[m], weights=fvals[m], minlength=vlen) * dv**2
        
    
fig = plt.figure(figsize=(12, 9))
ax1 = fig.add_subplot(111)

#ax1.plot( cellids, color='black')
ax1.imshow(fx_store.T, cmap='plasma', aspect='auto')
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'plots', 'plot_1D.png'), dpi=300)


    
    