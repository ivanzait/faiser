"""Local magnetic frame and the action of rotations on Hermite coefficients.

Frame (rows of R): e1 = (V x B)/|V x B| (the -E direction under ideal Ohm), e2 = B-hat x e1 (along -(V x B) x B, sign
chosen so the frame is right-handed), e3 = B-hat. If |V_perp| is below `eps`*vth the direction is undefined (V ~ 0 or
V || B) and e1 is the projection of `ref` perpendicular to B instead.

With a common vth and u = 0 the Hermite basis is the 3-D isotropic oscillator basis: a rotation mixes coefficients of
the same total degree only, and is exp(-rotvec . G) with G the angular-momentum generators built from ladder operators.
Coefficient order follows latent.index_set: (l, m, n) = (z, y, x).
"""
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from propagator.latent import index_set


def local_frame(B, V=None, vth=1.0, eps=1e-2, ref=(1.0, 0.0, 0.0)):
    b = np.asarray(B, float) / np.linalg.norm(B)
    v = np.zeros(3) if V is None else np.asarray(V, float)
    e1 = np.cross(v, b)
    if np.linalg.norm(e1) < eps * vth:
        r = np.asarray(ref, float)
        if abs(r @ b) > 0.99:
            r = np.array([0.0, 1.0, 0.0])
        e1 = r - (r @ b) * b
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(b, e1), b])


def generators(K):
    """G[axis] (D, D) real antisymmetric, axis 0, 1, 2 = x, y, z: G_z = a_x^+ a_y - a_y^+ a_x and cyclic."""
    idx = index_set(K)
    pos = {k: i for i, k in enumerate(idx)}

    def ladder(up, down):
        """Matrix of a_up^+ a_down on the coefficient basis; coordinate 0, 1, 2 = x, y, z = tuple slot 2, 1, 0."""
        M = np.zeros((len(idx), len(idx)))
        for j, k in enumerate(idx):
            n = [k[2], k[1], k[0]]
            if n[down] == 0:
                continue
            amp = np.sqrt(n[down]) * np.sqrt(n[up] + 1)
            n[down] -= 1; n[up] += 1
            t = (n[2], n[1], n[0])
            if t in pos:
                M[pos[t], j] = amp
        return M

    return np.stack([ladder(p, q) - ladder(q, p) for p, q in [(1, 2), (2, 0), (0, 1)]])


def rotation_matrix(R, G):
    """D x D matrix acting on coefficient vectors for a frame change with orthonormal rows R (v_new = R @ v)."""
    return expm(-np.tensordot(Rotation.from_matrix(R).as_rotvec(), G, axes=1))   # sign verified in test_frames


def gyration_matrix(theta, G):
    """Coefficient rotation for a physical rotation of the VDF by `theta` about z (the B axis of the local frame)."""
    return expm(-theta * G[2])
