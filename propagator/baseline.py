"""Analytic gyration baseline: rotate each run into its local magnetic frame, then rotate the coefficients about z by
s * Omega * (t - t0) and compare with the measured trajectory (s = +1 / -1 tried; ions turn left-handed about B).

    python propagator/baseline.py
"""
import os, sys, glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from propagator.data import Q_M
from propagator.frames import local_frame, generators, rotation_matrix, gyration_matrix

DATA = os.path.join(os.path.dirname(__file__), 'data')


def rel(a, b):
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)


def main():
    for f in sorted(glob.glob(os.path.join(DATA, '*.npz'))):
        d = np.load(f); z, t, B, K = d['z'], d['t'], d['B'], int(d['K'])
        G = generators(K)
        c = z[:, 4:] @ rotation_matrix(local_frame(B, z[:, :3].mean(0) * float(d['vlim']), np.exp(z[0, 3])), G).T   # local frame
        Om = Q_M * np.linalg.norm(B)
        out = []
        for s in (+1, -1):
            pred = np.stack([gyration_matrix(s * Om * (tk - t[0]), G) @ c[0] for tk in t])
            out.append(rel(pred, c))
        ident = rel(np.repeat(c[:1], len(t), 0), c)
        print(f'{os.path.basename(f)[:-4]:14s} identity max {ident.max():.3f} | s=+1 max {out[0].max():.3f} mean {out[0].mean():.3f}'
              f' | s=-1 max {out[1].max():.3f} mean {out[1].mean():.3f}')


if __name__ == '__main__':
    main()
