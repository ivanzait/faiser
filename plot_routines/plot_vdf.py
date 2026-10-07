"""VDF of one or more cells of a Vlasiator bulk file: the three velocity-space projections and the reduced 1-D
distributions, with the moments of the cube in the title. For quick looks at local runs.

    python plot_routines/plot_vdf.py [path/to/bulk.0000123.vlsv] [--show]

Reading goes through vdf_tools (get_vdf_parameters, build_cube, moments), the 2-D panels through plot_tools.plot_vdf.
The cell is taken from CELL_IDS if given, otherwise it is the cell with a VDF nearest to each of CELL_COORDS [m].
One figure per cell, saved to PLOTDIR (and shown with --show). Velocities are m/s inside, km/s on the plots.
Protons only (that is what build_cube reads); the mesh is assumed symmetric about 0.
"""
import os, sys
import numpy as np
import matplotlib

SHOW = '--show' in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
if not SHOW:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

for _p in ("/home/ivanzait/analysator", "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator"):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytools as pt
from data_processing import vdf_tools as vt
from plot_routines.plot_tools import plot_vdf, project

plt.rcParams['text.usetex'] = False   # importing analysator switches it on; there may be no latex

BULK_FILE   = "/Users/ivanzait/Documents/Documents_LM4500/Codes/Vlasiator/runs/gyrations/gyration2/bulk.0000020.vlsv"
CELL_ID    = 1                  # explicit cell ids; if empty, CELL_COORDS is used
CELL_COORDS = [[0.0, 0.0, 0.0]]       # [x, y, z] in m: the nearest cell with a VDF is plotted
DECADES     = 8                       # colour / y range: this many decades below the peak
SHOW_CUTS   = True                    # second row: reduced 1-D distributions f(vx), f(vy), f(vz)
PLOTDIR     = os.path.join(os.path.dirname(__file__), '..', 'plots')

M_P, K_B = 1.67262192369e-27, 1.380649e-23

def moments(cube, vlim, vlen):
    """Density [m^-3], bulk velocity [m/s] (3,), thermal speed sqrt(kT/m) [m/s], temperature [K]."""
    dv = 2 * vlim / vlen
    n = float(cube.sum()) * dv ** 3
    u = vt.get_drift_velocity_cube(cube, vlim, vlen)
    vth = float(vt.get_thermal_velocity_cube(cube, vlim, vlen, u))
    return n, u, vth, M_P * vth ** 2 / K_B


def vdf_figure(cube, vlim, vlen, decades=DECADES, show_cuts=SHOW_CUTS):
 
    
    nrows=1
    peak = max(project(cube, a).max() for a in range(3))
    norm = LogNorm(peak * 10.0 ** -decades, peak)
    fig, axes = plt.subplots(nrows, 3, figsize=(13, 4.6 * nrows), constrained_layout=True, squeeze=False)
    for a in range(3):
        ax = axes[0, a]
        im = plot_vdf(ax, cube, vlim, axis=a, norm=norm)
        ax.set_aspect('equal')
    fig.colorbar(im, ax=list(axes[0]), shrink=0.8, label='f summed over v')
    return fig


def main(bulk_file):
    
    reader = pt.vlsvfile.VlsvReader(bulk_file)
    stem = os.path.splitext(os.path.basename(bulk_file))[0]
    
    vlim, vlen, dv = vt.get_vdf_parameters(reader)
    cube = vt.build_cube(CELL_ID, reader, vlim, vlen, dv)
    
    os.makedirs(PLOTDIR, exist_ok=True)
    fig = vdf_figure(cube, vlim, vlen)    
    out = os.path.join(PLOTDIR, f'vdf_{stem}2_cell{CELL_ID}.png')
    fig.savefig(out, dpi=130)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main(BULK_FILE)
