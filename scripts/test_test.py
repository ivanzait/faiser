import math
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite

file = 
d = np.load(filepath)

data_trunc = d["data_trunc"]
N = data_trunc.shape[0]
ORDER = 19
NB = ORDER + 1
idx = np.arange(N, dtype=np.float64)
TURNING_PT = np.sqrt(2 * ORDER + 1)

# --- center of mass, per axis ---
mx = data_trunc.sum(axis=(1, 2))
my = data_trunc.sum(axis=(0, 2))
mz = data_trunc.sum(axis=(0, 1))
com = np.array([
np.sum(idx * mx) / mx.sum(),
np.sum(idx * my) / my.sum(),
np.sum(idx * mz) / mz.sum(),
])
margin = np.minimum(com, (N - 1) - com)
scale_per_axis = margin / TURNING_PT # largest scale keeping the basis fully on-grid

print(f"center of mass: {com}")
print(f"margin to nearest wall: {margin}")
print(f"scale per axis (grid-limited): {scale_per_axis}")
print(f"(for comparison, old fixed scale was 14.0, centered at grid center 99.5)")


def build_H(center, scale, idx, NB):
    xs = (idx - center) / scale
    H = np.empty((NB, N))
    for n in range(NB):
        Hn = eval_hermite(n, xs)
        norm = 1.0 / np.sqrt((2.0 ** n) * math.factorial(n) * np.sqrt(np.pi))
        H[n] = norm * Hn * np.exp(-xs ** 2 / 2.0)
        dx = xs[1] - xs[0]
    return H, dx


Hx, dx_x = build_H(com[0], scale_per_axis[0], idx, NB)
Hy, dx_y = build_H(com[1], scale_per_axis[1], idx, NB)
Hz, dx_z = build_H(com[2], scale_per_axis[2], idx, NB)

for name, H, dx in [("x", Hx, dx_x), ("y", Hy, dx_y), ("z", Hz, dx_z)]:
    gram = (H * dx) @ H.T
    print(f"axis {name}: orthonormality max error = {np.abs(gram - np.eye(NB)).max():.2e}")

t0 = time.perf_counter()
t = np.tensordot(Hx, data_trunc, axes=([1], [0])) # (a, j, k)
t = np.tensordot(Hy, t, axes=([1], [1])) # (b, a, k)
coeffs = np.tensordot(Hz, t, axes=([1], [2])).transpose(2, 1, 0) * dx_x * dx_y * dx_z # (a,b,c)
t1 = time.perf_counter()
print(f"forward transform: {(t1 - t0) * 1000:.2f} ms")

t0 = time.perf_counter()
r = np.tensordot(coeffs, Hx, axes=([0], [0])) # (b, c, i)
r = np.tensordot(r, Hy, axes=([0], [0])) # (c, i, j)
recon = np.tensordot(r, Hz, axes=([0], [0])) # (i, j, k)
t1 = time.perf_counter()
print(f"inverse transform: {(t1 - t0) * 1000:.2f} ms")

err = data_trunc - recon
max_abs_err = np.abs(err).max()
rmse = np.sqrt(np.mean(err ** 2))
rel_l2 = np.linalg.norm(err) / np.linalg.norm(data_trunc)
print(f"\nmax|data|: {data_trunc.max():.4e}")
print(f"max abs error: {max_abs_err:.4e} ({max_abs_err / data_trunc.max() * 100:.4f}% of peak)")
print(f"RMSE: {rmse:.4e}")
print(f"relative L2 error: {rel_l2:.4e} ({rel_l2 * 100:.4f}%)")
print(f"\n(for comparison: fixed grid-center window gave 20.87% relative L2 error)")

a_idx, b_idx, c_idx = np.meshgrid(np.arange(NB), np.arange(NB), np.arange(NB), indexing='ij')
total_order = a_idx + b_idx + c_idx
max_total = total_order.max()
energy_by_order = np.array([np.sum(coeffs[total_order == k] ** 2) for k in range(max_total + 1)])
energy_by_order_norm = energy_by_order / energy_by_order.sum()
print(f"cumulative energy fraction through order 10: {energy_by_order_norm[:11].sum():.6f}")

np.savez("/Users/ttonteri/bimodal_gaussian_pdf/hermite_recentered_results.npz",
        data_trunc=data_trunc, recon=recon, err=err, coeffs=coeffs,
        energy_by_order_norm=energy_by_order_norm, com=com, scale_per_axis=scale_per_axis)
print("saved hermite_recentered_results.npz")