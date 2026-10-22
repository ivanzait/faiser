"""Original VDF of one cell, no transform: 1D cuts (DIM=1) or 2D central slices (DIM=2) through the centre of the
velocity mesh, sp_th marked, optionally zoomed to the box of everything stored for the cell.

    python3 runners/one_cell_view.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MACHINE = 'turso'
if MACHINE == 'turso':
    sys.path.insert(0, "/home/ivanzait/proj/analysator_original/analysator") 
else:
    sys.path.insert(0, "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytools as pt
from data_processing import vdf_tools as vt
from plot_routines import plot_tools
plt.rcParams['text.usetex'] = False   # importing analysator switches it on; there is no latex here

runid='BID'
tstep=1120
BULK_FILE = f"/turso/group/spacephysics/vlasiator/data/L0/2D/{runid}/bulk.000{tstep}.vlsv"
CELL_ID   = 180041
DIM       = 2        # 1: 1D cuts, 2: 2D slices, both through the centre of the velocity mesh
ZOOM      = True     # zoom to the box of everything stored for the cell (halo included)

def main():

    reader = pt.vlsvfile.VlsvReader(BULK_FILE)
    vlim, vlen, dv = vt.get_vdf_parameters(reader)
    sp_th = vt.get_sparse_threshold(reader)
    cube = vt.build_cube(CELL_ID, reader, vlim, vlen, dv).astype(np.float64)
    
    vel_bounds = vt.get_vdf_bounding_box(cube, vlim, vlen)['vel'] if ZOOM else None

    fig = plot_tools.vdf_figure([cube], vlim, ['original'], dim=DIM, sp_th=sp_th, vel_bounds=vel_bounds)
    fig.suptitle(f'{runid} t={tstep} cell {CELL_ID}')
    fig.savefig(os.path.join(os.path.dirname(__file__), '..',
                             f'plots/one_cell_run_{runid}_tstep{tstep}_cell{CELL_ID}_vdf_cuts_{DIM}D.png'))
    plt.close(fig)
    
if __name__ == "__main__":
    main()