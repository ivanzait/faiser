"""Gyration runs -> npz of latent trajectories; npz -> (z_t, z_t+k, conditioning) training pairs."""
import os, sys, glob
import numpy as np

sys.path.insert(0, "/Users/ivanzait/Documents/Documents_LM4500/Codes/analysator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytools as pt
from data_processing import vdf_tools as vt
from propagator.latent import encode

Q_M = 9.5788e7   # proton q/m [C/kg]


def read_B(reader):
    """B of the one-cell field grid (the gyration runs have uniform B and no vg_b_vol output)."""
    return np.asarray(reader.read(name='fg_b', tag='VARIABLE', mesh='fsgrid'), float).reshape(3)


def build_run(run_dir, out, K=8, cell=1):
    """Encode every bulk file of one run: z[T, D], t[T], B[3] (B is uniform and constant: read from the last file)."""
    z, t = [], []
    for f in sorted(glob.glob(os.path.join(run_dir, 'bulk.*.vlsv'))):
        r = pt.vlsvfile.VlsvReader(f)
        vlim, vlen, dv = vt.get_vdf_parameters(r)
        z.append(encode(vt.build_cube(cell, r, vlim, vlen, dv), vlim, vlen, K))
        t.append(float(r.read_parameter('time')))
    np.savez(out, z=np.array(z), t=np.array(t), B=read_B(r), vlim=vlim, vlen=vlen, K=K)


def conditioning(B, dt):
    """Dimensionless inputs: B-hat, log|B|, theta = Omega dt as (sin, cos, theta)."""
    b = np.linalg.norm(B)
    th = Q_M * b * dt
    return np.concatenate([B / b, [np.log(b), np.sin(th), np.cos(th), th]])


def pairs(npz_files, stride=1, horizon=1):
    """(z_0, [z_1..z_h], cond) with `horizon` consecutive steps of `stride` frames; one sample per start frame."""
    Z0, ZH, C = [], [], []
    for f in npz_files:
        d = np.load(f)
        z, t, B = d['z'], d['t'], d['B']
        for i in range(len(z) - stride * horizon):
            Z0.append(z[i]); ZH.append(z[i + stride:i + stride * horizon + 1:stride])
            C.append(conditioning(B, t[i + stride] - t[i]))
    return np.array(Z0), np.array(ZH), np.array(C)
