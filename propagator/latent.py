"""Latent state of a single cell: z = [u/vlim, log vth, Hermite coefficients with l+m+n < K]."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_processing import vdf_tools as vt


def index_set(K):
    return [(l, m, n) for s in range(K) for l in range(s + 1) for m in range(s + 1 - l) for n in [s - l - m]]


def encode(cube, vlim, vlen, K):
    u = vt.get_drift_velocity_cube(cube, vlim, vlen)
    vth = float(vt.get_thermal_velocity_cube(cube, vlim, vlen, u))
    coeffs, _, _ = vt.adaptive_transform(cube, vlim, vlen, vth, u, max_order=K, tolerance=0.0)  # tolerance 0: all K orders
    c = np.array([coeffs[i] for i in index_set(K)])
    return np.concatenate([u / vlim, [np.log(vth)], c])


def decode(z, vlim, vlen, K):
    u, vth, c = np.asarray(z[:3]) * vlim, float(np.exp(z[3])), z[4:]
    return vt.reconstruct_vdf_adaptive(dict(zip(index_set(K), c)), vlim, vlen, K, vth, u)
