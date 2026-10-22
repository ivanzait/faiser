"""Plot the raw VDF of one cell (three projections) and compare its moments with the ones stored in the bulk file.

    python3 plot_routines/check_vdf.py [runid] [tstep] [cellid]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/ivanzait/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pytools as pt
from tools import vdf_tools as vt
from plot_routines import plot_tools

plt.rcParams['text.usetex'] = False
BULK_DIR = "/turso/group/spacephysics/vlasiator/data/L0/2D"

runid, tstep, cid = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else ('BID', 1147, 795501)

reader = pt.vlsvfile.VlsvReader(f'{BULK_DIR}/{runid}/bulk.{tstep:07d}.vlsv')
vlim, vlen, dv = vt.get_vdf_parameters(reader)
cube = vt.build_cube(cid, reader, vlim, vlen, dv).astype(np.float64)
print(f'{runid} t={tstep} cell {cid} at {np.round(reader.get_cell_coordinates(cid) / 6.371e6, 2)} Re | vlen={vlen} vlim={vlim/1e3:.0f} km/s dv={dv/1e3:.1f} km/s')
print(f'f: min={cube.min():.3g} max={cube.max():.3g} | nonzero voxels {np.count_nonzero(cube)} of {cube.size}')

# moments from the cube, cube[iz, iy, ix]
v = vt.velocity_axis(vlim, vlen, dv)
n = cube.sum() * dv**3
u = np.array([(cube.sum(axis=(0, 1)) * v).sum(), (cube.sum(axis=(0, 2)) * v).sum(), (cube.sum(axis=(1, 2)) * v).sum()]) * dv**3 / n
vth = vt.get_thermal_velocity_cube(cube, vlim, vlen, u)
print(f'from cube : n = {n/1e6:.4g} cm^-3 | u = {np.round(u/1e3, 1)} km/s | vth = {vth/1e3:.1f} km/s')

rho = float(reader.read_variable('proton/vg_rho', cellids=cid))          # number density [m^-3]
vel = np.asarray(reader.read_variable('proton/vg_v', cellids=cid)).ravel()
print(f'stored    : n = {rho/1e6:.4g} cm^-3 | u = {np.round(vel/1e3, 1)} km/s')
print(f'negative voxels: {(cube < 0).sum()}, their sum / total = {cube[cube < 0].sum() / cube.sum():.2g}')

fig = plot_tools.vdf_2d([cube], vlim, ['original'], reduce='sum')
fig.suptitle(f'{runid} t={tstep} cell {cid}')
os.makedirs('plots', exist_ok=True)
out = f'plots/vdf_{runid}_tstep{tstep}_cell{cid}.png'
fig.savefig(out, dpi=130)
print('saved ->', out)
