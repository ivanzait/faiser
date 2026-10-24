"""Adaptive Hermite transform of many cells of one bulk file -> one npz per time step, data/run_<runid>_tstep<N>.npz.

    python3 runners/run_multi_cell.py

Arrays are stacked along cells: row i of every per-cell array belongs to cell cellids[i].
    cellids (N,), coords (N, 3), v_means (N, 3) [m/s], v_ths (N,) [m/s],
    order_used (N,), delta_final (N,)    levels computed and the Parseval delta at the last one
    hermite_coeffs (N, M, M, M)          dense spectra, M = MAX_ORDER; zero beyond each cell's order_used
    bbox_idx (N, 3, 2)                   per axis (z, y, x), inclusive index bounds of f > 0
    cubes (N, vlen, vlen, vlen)          original VDFs, float32, only if SAVE_CUBES (fine for tens of cells;
                                         for bigger runs keep only the spectra and re-read cubes from the vlsv)
and per file: vlim, vlen, sqrt_f, max_order, tolerance, bulk_file, runid, tstep.
"""
import numpy as np
import os, sys
sys.path.insert(0, "/home/ivanzait/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytools as pt
from tools import vdf_tools as vt

runid='TSI'
tstep=30

BULK_FILE = f"/turso/group/spacephysics/vlasiator/data/L0/1D/{runid}/bulk.{tstep:07d}.vlsv"
CELL_SELECTION = 'random'  # 'random': N_CELLS cells drawn from the cells with a VDF; 'all': every such cell
N_CELLS   = 3
SEED      = 7
MAX_ORDER = 33
TOLERANCE = 0.02
OUTDIR    = os.path.join(os.path.dirname(__file__), '..', 'data')   # one npz per time step
SQRT_F    = True   # expand sqrt(f), see vdf_tools.adaptive_transform; saved in the npz for the reconstruction
SAVE_CUBES = False  # also store the original cubes (~55 MB/cell uncompressed at vlen=240)


def process_cell(cell_id, reader, vlim, vlen, dv):
    
    cube = vt.build_cube(cell_id, reader, vlim, vlen, dv)                      # float32
    c64  = cube.astype(np.float64)
    u    = vt.get_drift_velocity_cube(c64, vlim, vlen)
    vth  = vt.get_thermal_velocity_cube(c64, vlim, vlen, u)
    coeffs, order, deltas = vt.adaptive_transform(c64, vlim, vlen, vth, u, max_order=MAX_ORDER, tolerance=TOLERANCE, sqrt_f=SQRT_F)
    print(f'  order_used={order}  n_coeffs={len(coeffs)}  final Parseval delta={deltas[order - 1]:.4f}')
    
    row = dict(cellid=cell_id, 
               coords=reader.get_cell_coordinates(cell_id), 
               u=u, vth=vth,
               order=order, 
               delta=deltas[order - 1], 
               coeffs=vt.coeffs_into_cube(coeffs, MAX_ORDER),
               bbox=vt.get_vdf_bounding_box(cube, vlim, vlen)['idx'])
    
    if SAVE_CUBES:
        row['cube'] = cube
    return row


def main( tstep):
    BULK_FILE = f"/turso/group/spacephysics/vlasiator/data/L0/1D/{runid}/bulk.{tstep:07d}.vlsv"
    reader = pt.vlsvfile.VlsvReader(BULK_FILE)
    vlim,vlen,dv = vt.get_vdf_parameters(reader)

    cells = np.sort(np.atleast_1d(reader.read(mesh="SpatialGrid", tag="CELLSWITHBLOCKS", name="proton")))
    if CELL_SELECTION == 'random':
        cells = np.sort(np.random.default_rng(SEED).choice(cells, size=min(N_CELLS, len(cells)), replace=False))

    rows = []
    for cell_id in (int(c) for c in cells):       
        print('processing cell', cell_id)
        rows.append(process_cell(cell_id, reader, vlim, vlen, dv))

    col = lambda k: np.array([r[k] for r in rows])                  # stack one field along cells
    save_kwargs = dict(
        cellids=col('cellid'), coords=col('coords'), v_means=col('u'), v_ths=col('vth'),
        order_used=col('order'), delta_final=col('delta'), hermite_coeffs=col('coeffs'), bbox_idx=col('bbox'),
        
        vlim=vlim, vlen=vlen, sqrt_f=SQRT_F, max_order=MAX_ORDER, tolerance=TOLERANCE,
        bulk_file=BULK_FILE, runid=runid, tstep=tstep,
    )
    if SAVE_CUBES:
        save_kwargs['cubes'] = col('cube')

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f'run_{runid}_tstep{tstep}.npz')
    np.savez_compressed(out, **save_kwargs)
    print(f'saved {len(rows)} cells -> {out}')


if __name__ == '__main__':
    for tstep in range(5, 30, 5):
        print('Time step', tstep)
        main( tstep )
