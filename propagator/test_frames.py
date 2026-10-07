"""Checks for propagator/frames.py:  python propagator/test_frames.py   (or pytest)."""
import os, sys
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from propagator.frames import local_frame, generators, rotation_matrix, gyration_matrix
from propagator.latent import encode

K, VLIM, VLEN = 8, 1e6, 64
G = generators(K)


def gaussian_cube(cov):
    ax = np.linspace(-VLIM, VLIM, VLEN, endpoint=False) + VLIM / VLEN
    Z, Y, X = np.meshgrid(ax, ax, ax, indexing='ij')
    v = np.stack([X, Y, Z], -1)
    return np.exp(-0.5 * np.einsum('...i,ij,...j->...', v, np.linalg.inv(cov), v))


def test_frame():
    B, V = np.array([1.0, 2.0, 0.5]), np.array([3.0, -1.0, 2.0])
    for R in (local_frame(B, V), local_frame(B, None), local_frame(B, B * 5), local_frame([0, 0, 1.0])):
        assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(R[2], B / np.linalg.norm(B)) or R[2][2] == 1.0
    R = local_frame(B, V)
    assert np.allclose(R[0], np.cross(V, B) / np.linalg.norm(np.cross(V, B)))


def test_generators():
    assert np.allclose(G, -G.transpose(0, 2, 1))
    comm = G[0] @ G[1] - G[1] @ G[0]
    assert np.allclose(comm, G[2]) or np.allclose(comm, -G[2])


def test_rotation_acts_on_coefficients():
    cov = np.diag([1.6e5, 1.0e5, 0.7e5]) ** 2
    R = Rotation.from_rotvec([0.4, -0.7, 0.9]).as_matrix()
    z, z_rot = encode(gaussian_cube(cov), VLIM, VLEN, K), encode(gaussian_cube(R @ cov @ R.T), VLIM, VLEN, K)
    c, c_rot = z[4:], z_rot[4:]
    err = np.linalg.norm(rotation_matrix(R, G) @ c - c_rot) / np.linalg.norm(c_rot)
    err_T = np.linalg.norm(rotation_matrix(R.T, G) @ c - c_rot) / np.linalg.norm(c_rot)
    print(f'rotation matrix: rel.err {err:.2e}   (inverse convention: {err_T:.2e}),  rotation changes coeffs by '
          f'{np.linalg.norm(c - c_rot) / np.linalg.norm(c_rot):.2f}')
    assert err < 1e-3


if __name__ == '__main__':
    for f in (test_frame, test_generators, test_rotation_acts_on_coefficients):
        f(); print('ok', f.__name__)
