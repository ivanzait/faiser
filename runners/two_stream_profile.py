"""Two-stream test run: f(vx, x) integrated over vy, vz along the 1D box, with a bulk variable profile above it.

    python3 runners/two_stream_profile.py <time step>
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False   # must come after the pt import
import sys, os

MACHINE = 'turso'

if MACHINE == 'turso':
    sys.path.insert(0, "/home/ivanzait/proj/analysator_original/analysator") 
else:
    sys.path.insert(0, "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../data_processing'))
import pytools as pt
import vdf_tools as vt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False   # must come after the pt import


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("t", type=int, help="time step number")
args = parser.parse_args()
t = args.t


BULKPATH = "/home/ivanzait/proj/RUNS/two_stream_test/"
BULK_FILE = BULKPATH + "bulk."+str(t).zfill(7) +".vlsv"
var = 'proton/vg_rho'

f = pt.vlsvfile.VlsvReader(BULK_FILE)
cellids = f.read_variable('CellID')
#f.list()
var_vals = f.read_variable(var)

xmin=f.read_parameter("xmin")
xmax=f.read_parameter("xmax")
xlen=f.read_parameter("xcells_ini")   # also ycells_ini, zcells_ini
x_profile = np.linspace(xmin, xmax, xlen)

vlim, vlen, dv = vt.get_vdf_parameters(f)
v_axis=np.linspace(-vlim,vlim, vlen)
fx_store = np.zeros( [len(cellids), vlen] )


all_vids = np.arange(vlen**3, dtype=np.int64)
vc = f.get_velocity_cell_coordinates(all_vids)
lut = np.round((vc[:, 0] + vlim - dv / 2) / dv).astype(np.int32)


pop = "proton"
nb = vlen // 4
ilocal = np.arange(64) % 4

cwb  = f.read(name=pop,   tag="CELLSWITHBLOCKS", mesh="SpatialGrid")
bpc  = f.read(name=pop,   tag="BLOCKSPERCELL",   mesh="SpatialGrid")
bids = f.read(name=pop,   tag="BLOCKIDS",        mesh="SpatialGrid")
avgs = f.read(name=pop, tag="BLOCKVARIABLE",  mesh="SpatialGrid")

row = np.repeat(np.arange(len(cwb)), bpc)
ix = (bids % nb)[:, None] * 4 + ilocal[None, :]
flat = row[:, None] * vlen + ix

fx = np.bincount(flat.ravel(), weights=avgs.ravel(), minlength=len(cwb) * vlen).reshape(len(cwb), vlen) * dv**2
fx_store = fx[np.argsort(cwb)]

    
fig = plt.figure(figsize=(12, 9))
ax1 = fig.add_axes([0.1, 0.05, 0.8, 0.4])
ax2 = fig.add_axes([0.1, 0.5, 0.8, 0.4])

ax1.pcolormesh( x_profile, v_axis,  fx_store.T, cmap='plasma')
ax1.set_xlabel('x')
ax1.set_ylabel('$f(V_x,x)$')

ax2.plot(x_profile,var_vals)
ax2.set_xlabel('x')
ax2.set_ylabel(var)
ax2.set_ylim([1.6e6, 2.4e6])


plots_dir = os.path.join(BULKPATH, 'plots')
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, 'f_prof_'+ str(t).zfill(5)+ '.png'), dpi=300)




