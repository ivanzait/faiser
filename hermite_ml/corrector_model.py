"""
Hermite reconstruction safeguard -- MLP corrector, patch-input redesign
(Exp 9, after the Exp 1-8 reset -- see EXPERIMENT_LOG.md for the lessons
carried forward into this version).

Goal: general artifact removal after Hermite reconstruction -- primarily
Gibbs-type sign-flip ringing that appears when a strongly non-Maxwellian
VDF is decomposed in the Hermite basis. No a-priori restriction to "near
the support edge": the local patch input lets the network find artifacts
wherever they occur, rather than us hand-specifying a region.

f_final(v) = f_rec_full(v) * exp(Delta_pred(v))   inside sparse_mask
           = 0                                     outside (structural)

Delta_pred(v) is predicted per-voxel from:
    input = [local patch of log(f_rec_full) around v (3x3x3=27 values),
             C_low (S_low<=2 -> 10 coeffs, global condition),
             x_hat, y_hat, z_hat (query position)]        (40 numbers)

Architecture is still a plain MLP (Linear+ReLU only) -- the patch is just
concatenated into the input vector, not processed by actual convolutions.
This keeps a from-scratch (non-PyTorch) reimplementation trivial: the
forward pass is a handful of matmul+bias+relu steps.
"""

import numpy as np
import torch
import torch.nn as nn


class HermiteCorrectorMLP(nn.Module):
    """
    Plain MLP: Linear -> ReLU -> Linear -> ReLU -> Linear.
    input_dim = n_features + 3  (patch + C_low + normalized velocity coords)
    """

    def __init__(self, n_features, hidden=64):
        super().__init__()
        input_dim = n_features + 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def export_weights(self):
        """
        Return raw (W, b) pairs for each Linear layer, as numpy arrays, in
        forward-pass order. Used for a from-scratch (non-PyTorch) inference
        reimplementation: y = relu(W0 @ x + b0); y = relu(W1 @ y + b1);
        out = W2 @ y + b2.
        """
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        return [(lin.weight.detach().cpu().numpy(),
                 lin.bias.detach().cpu().numpy()) for lin in linears]


def coeffs_dict_to_vector(coeffs, s_low):
    """
    Flatten a coefficient dict {(l,m,n): value} to a fixed-order vector,
    keeping only l+m+n <= s_low. Order: level s=0,1,...,s_low, then
    lexicographic (l,m,n) within each level -- matches
    adaptive_hermite.coeffs_to_array's ordering convention.
    """
    out = []
    for s in range(s_low + 1):
        for l in range(s + 1):
            for m in range(s + 1 - l):
                n = s - l - m
                out.append(coeffs.get((l, m, n), 0.0))
    return out


def n_coeffs_at_order(s_low):
    """Tetrahedral number: total coefficients with l+m+n <= s_low."""
    return (s_low + 1) * (s_low + 2) * (s_low + 3) // 6


def axis_spectra(coeffs, max_order):
    """
    Marginal POWER spectrum per axis, computed directly in Hermite
    coefficient space -- for each axis, sum C[l,m,n]^2 over the other two
    indices:
        spec_x[n] = sum_{l,m : l+m+n<=max_order} C[l,m,n]^2
        spec_y[m] = sum_{l,n : l+m+n<=max_order} C[l,m,n]^2
        spec_z[l] = sum_{m,n : l+m+n<=max_order} C[l,m,n]^2

    Pure reduction over the already-computed coefficient dict -- negligible
    cost (~2300 additions at max_order=22) compared to the ~2.7s
    adaptive_transform that produced `coeffs` in the first place. From
    EXPERIMENT_LOG.md Exp 8: cheap, strong "how non-Maxwellian is this cell
    overall" signal (nearly perfect lobe-cell recognition, eps_log_corr
    0.021) but carries no per-voxel spatial information on its own --
    combined with the Exp 9 local patch (which handles the spatial part but
    struggles to disambiguate trivial-vs-hard cells at the sp_th floor) in
    Exp 10.

    Returns
    -------
    ndarray (3, max_order+1) -- rows are [x, y, z] (indexed by n, m, l
    respectively, matching u[0],u[1],u[2]).
    """
    spec_x = np.zeros(max_order + 1, dtype=np.float32)  # indexed by n
    spec_y = np.zeros(max_order + 1, dtype=np.float32)  # indexed by m
    spec_z = np.zeros(max_order + 1, dtype=np.float32)  # indexed by l
    for (l, m, n), c in coeffs.items():
        c2 = c * c
        spec_x[n] += c2
        spec_y[m] += c2
        spec_z[l] += c2
    return np.stack([spec_x, spec_y, spec_z], axis=0)


# Fixed 3x3x3 offset pattern (27 neighbors incl. center), used by
# extract_patches. Order: z-major, then y, then x -- must stay consistent
# between training and inference.
_PATCH_OFFSETS = np.array(
    [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
    dtype=np.int64)  # (27, 3)
PATCH_SIZE = len(_PATCH_OFFSETS)  # 27


def pad_field(field, pad_value):
    """Pad a (vlen,vlen,vlen) field by 1 cell on every side with pad_value."""
    return np.pad(field, pad_width=1, mode='constant', constant_values=pad_value)


def extract_patches(field_padded, iz, iy, ix):
    """
    Vectorized 3x3x3 patch extraction around each (iz,iy,ix) center
    (in ORIGINAL, unpadded coordinates -- field_padded has +1 offset baked
    in on every axis from pad_field).

    Parameters
    ----------
    field_padded : ndarray (vlen+2, vlen+2, vlen+2)
    iz, iy, ix    : 1-D int arrays, same length (query centers)

    Returns
    -------
    ndarray (len(iz), 27) -- flattened patch values per query point, in
    _PATCH_OFFSETS order.
    """
    dz = _PATCH_OFFSETS[:, 0]
    dy = _PATCH_OFFSETS[:, 1]
    dx = _PATCH_OFFSETS[:, 2]
    zz = iz[:, None] + 1 + dz[None, :]
    yy = iy[:, None] + 1 + dy[None, :]
    xx = ix[:, None] + 1 + dx[None, :]
    return field_padded[zz, yy, xx]
