"""Leave-one-run-out training of the stepper in the local magnetic frame.

    python propagator/run_training_local.py [--train-strides 1 2] [--test-strides 1 2 3 4 5] [--horizon 2] [--epochs 2000]

Every run is rotated into its local frame (frames.local_frame), where gyration is a rotation about z by theta = Omega dt.
The stepper sees only theta-features [sin, cos, theta]; the direction of B has been removed by the frame.
Training pairs use frames `stride` apart (stride k = dt of k outputs = a different theta), so 3 runs give several thetas;
test strides outside the training ones probe extrapolation in theta. Relative rollout error per test stride:

  identity   z stays put
  rotation   the exact analytic rotation by theta (no learning): the error floor of the data
  free       z' = z + f(z, theta-features), everything learned
  residual   z' = Rot(theta) z + f(z, theta-features): the net only corrects the analytic rotation
"""
import os, sys, glob, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from propagator.data import Q_M
from propagator.frames import local_frame, generators, rotation_matrix, gyration_matrix
from propagator.model import Stepper

DATA = os.path.join(os.path.dirname(__file__), 'data')


def load_local(f):
    """(c_local [T, D], t [T], Omega, G) of one run, coefficients rotated into the local magnetic frame."""
    d = np.load(f); z, t, B, K = d['z'], d['t'], d['B'], int(d['K'])
    G = generators(K)
    R = local_frame(B, z[:, :3].mean(0) * float(d['vlim']), np.exp(z[0, 3]))
    return z[:, 4:] @ rotation_matrix(R, G).T, t, Q_M * np.linalg.norm(B), G


def pairs(runs, strides, horizon):
    """z0 [N, D], zH [N, H, D], theta [N], Rot [N, D, D] (analytic coefficient rotation for one step of theta)."""
    Z0, ZH, TH, RT = [], [], [], []
    for c, t, Om, G in runs:
        for s in strides:
            th = Om * (t[s] - t[0])                      # equally spaced outputs: one dt per stride
            Rot = gyration_matrix(-th, G)                # ions turn left-handed about B (verified in baseline.py)
            for i in range(len(c) - s * horizon):
                Z0.append(c[i]); ZH.append(c[i + s:i + s * horizon + 1:s]); TH.append(th); RT.append(Rot)
    T = lambda x: torch.tensor(np.array(x), dtype=torch.float32)
    return T(Z0), T(ZH), T(TH), T(RT)


def features(th):
    return torch.stack([torch.sin(th), torch.cos(th), th], -1)


def rollout(model, z0, th, Rot, horizon, use_rot):
    out, z, f = [], z0, features(th)
    for _ in range(horizon):
        base = torch.einsum('nij,nj->ni', Rot, z) if use_rot else z
        z = base + model.net(torch.cat([z, f], -1))
        out.append(z)
    return torch.stack(out, 1)


def rel_err(pred, true):
    return float(torch.linalg.norm(pred - true) / torch.linalg.norm(true))


def fit(tr, horizon, epochs, lr, use_rot, seed=0):
    torch.manual_seed(seed)
    z0, zH, th, Rot = tr
    model = Stepper(z0.shape[-1], 3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        loss = ((rollout(model, z0, th, Rot, horizon, use_rot) - zH) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model, loss.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-strides', type=int, nargs='+', default=[1, 2])
    p.add_argument('--test-strides', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--horizon', type=int, default=2)
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-3)
    a = p.parse_args()

    runs = {os.path.basename(f)[:-4]: load_local(f) for f in sorted(glob.glob(os.path.join(DATA, '*.npz')))}
    print(f'{len(runs)} runs | train strides {a.train_strides} | horizon {a.horizon} | epochs {a.epochs}')
    print(f'{"held out":14s} {"method":9s} ' + ' '.join(f'stride {s:<3d}' for s in a.test_strides) + '   train loss')

    for name in runs:
        others = [v for k, v in runs.items() if k != name]
        tr = pairs(others, a.train_strides, a.horizon)
        sd = tr[0].std()                                 # one global scale; the rotation is orthogonal, so it commutes
        tr = (tr[0] / sd, tr[1] / sd, tr[2], tr[3])
        fitted = {'free': fit(tr, a.horizon, a.epochs, a.lr, False), 'residual': fit(tr, a.horizon, a.epochs, a.lr, True)}
        rows = {m: [] for m in ('identity', 'rotation', 'free', 'residual')}
        for s in a.test_strides:
            z0, zH, th, Rot = pairs([runs[name]], [s], a.horizon)
            z0, zH = z0 / sd, zH / sd
            with torch.no_grad():
                rows['identity'].append(rel_err(z0[:, None].expand_as(zH), zH))
                rows['rotation'].append(rel_err(_rot_only(z0, Rot, a.horizon), zH))
                rows['free'].append(rel_err(rollout(fitted['free'][0], z0, th, Rot, a.horizon, False), zH))
                rows['residual'].append(rel_err(rollout(fitted['residual'][0], z0, th, Rot, a.horizon, True), zH))
        for m, r in rows.items():
            loss = f'{fitted[m][1]:.2e}' if m in fitted else ''
            print(f'{name:14s} {m:9s} ' + ' '.join(f'{x:10.3f}' for x in r) + f'   {loss}')


def _rot_only(z0, Rot, horizon):
    out, z = [], z0
    for _ in range(horizon):
        z = torch.einsum('nij,nj->ni', Rot, z); out.append(z)
    return torch.stack(out, 1)


if __name__ == '__main__':
    main()
