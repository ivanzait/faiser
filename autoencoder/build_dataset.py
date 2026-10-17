"""Step 1: dense 32^3 VDFs in v' = (v - V)/v_th from every bulk file of the dense two-stream run.

    python autoencoder/build_dataset.py [--config autoencoder/config.toml] [--max-files N] [--max-cells N]

Saves one npz: g [N, 32, 32, 32] (density-normalised, float32), n, V [N, 3], vth, outside (density fraction outside the
box), N_v (stored velocity cells of the original VDF), cellid, x [m], out_idx, t [s].
"""
import os, sys, glob, argparse, tomllib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, HERE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=os.path.join(HERE, 'config.toml'))
    p.add_argument('--max-files', type=int); p.add_argument('--max-cells', type=int)
    p.add_argument('--only-last', action='store_true', help='only the last output (all its cells)')
    a = p.parse_args()
    cfg = tomllib.load(open(a.config, 'rb'))['data']
    sys.path.insert(0, cfg['analysator'])
    import pytools as pt
    from data_processing import vdf_tools as vt
    from regrid import regrid

    files = sorted(glob.glob(os.path.join(cfg['run_dir'], cfg['files'])))
    files = files[-1:] if a.only_last else files[:a.max_files]
    rows = {k: [] for k in ('g', 'n', 'V', 'vth', 'outside', 'N_v', 'cellid', 'x', 'out_idx', 't')}
    for fi, f in enumerate(files):
        r = pt.vlsvfile.VlsvReader(f)
        vlim, vlen, dv = vt.get_vdf_parameters(r)
        cells = np.atleast_1d(r.read(mesh='SpatialGrid', tag='CELLSWITHBLOCKS', name='proton')).astype(int)
        nblk = np.atleast_1d(r.read(mesh='SpatialGrid', tag='BLOCKSPERCELL', name='proton')).astype(int)
        t = float(r.read_parameter('time'))
        for cid, nb in list(zip(cells, nblk))[:a.max_cells]:
            cube = vt.build_cube(int(cid), r, vlim, vlen, dv).astype(np.float64)
            V = np.asarray(vt.get_drift_velocity_cube(cube, vlim, vlen), float)
            vth = float(vt.get_thermal_velocity_cube(cube, vlim, vlen, V))
            g, n, out = regrid(cube, vlim, vlen, V, vth, cfg['n_grid'], cfg['box'])
            for k, v in zip(rows, (g, n, V, vth, out, 64 * nb, int(cid), r.get_cell_coordinates(int(cid))[0], fi, t)):
                rows[k].append(v)
        print(f'{os.path.basename(f)}  t={t:6.2f}  {len(rows["g"])} VDFs so far', flush=True)

    os.makedirs(os.path.dirname(os.path.join(HERE, '..', cfg['out_file'])), exist_ok=True)
    out = os.path.join(HERE, '..', cfg['out_file'])
    np.savez(out, **{k: np.array(v) for k, v in rows.items()}, box=cfg['box'], n_grid=cfg['n_grid'])
    print('saved ->', os.path.normpath(out))


if __name__ == '__main__':
    main()
