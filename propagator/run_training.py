"""Leave-one-run-out training of the gyration stepper.

    python propagator/run_training.py [--horizon 4] [--epochs 2000]

For each run in data/*.npz: train on the others, test on it. Compared on the held-out run (relative rollout error):
  identity      z stays put (what "learning nothing" scores)
  no-cond       stepper without the B / dt inputs (can only learn the average dynamics)
  with-cond     stepper conditioned on B-hat, log|B|, theta = Omega dt
Latent = Hermite coefficients only (u and vth are numerical noise in these runs), one global scale.
"""
import os, sys, glob, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from propagator.data import conditioning
from propagator.model import Stepper

DATA = os.path.join(os.path.dirname(__file__), 'data')


def load(f, horizon):
    """(z0 [N,D], zH [N,H,D], cond [N,C]) from one run; consecutive frames, so one dt per run."""
    d = np.load(f)
    z, t, B = d['z'][:, 4:], d['t'], d['B']
    n = len(z) - horizon
    z0 = z[:n]
    zH = np.stack([z[i + 1:i + 1 + horizon] for i in range(n)])
    c = np.stack([conditioning(B, t[i + 1] - t[i]) for i in range(n)])
    return z0, zH, c


def cat(runs):
    return [torch.tensor(np.concatenate(x), dtype=torch.float32) for x in zip(*runs)]


def rel_err(pred, true):
    return float(torch.linalg.norm(pred - true) / torch.linalg.norm(true))


def fit(tr, use_cond, horizon, epochs, lr, seed=0):
    torch.manual_seed(seed)
    z0, zH, c = tr
    cd = c.shape[-1] if use_cond else 0
    model = Stepper(z0.shape[-1], cd)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    cond = lambda c: c[:, None].expand(-1, horizon, -1) if use_cond else c[:, None, :0].expand(-1, horizon, -1)
    for _ in range(epochs):
        loss = ((model.rollout(z0, cond(c)) - zH) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model, cond, loss.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--horizon', type=int, default=4)
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-3)
    a = p.parse_args()

    files = sorted(glob.glob(os.path.join(DATA, '*.npz')))
    runs = {os.path.basename(f)[:-4]: load(f, a.horizon) for f in files}
    print(f'{len(runs)} runs, horizon {a.horizon}, epochs {a.epochs}\n')
    print(f'{"held out":16s} {"identity":>9s} {"no-cond":>9s} {"with-cond":>10s}   train loss (with-cond)')

    for name in runs:
        tr = cat([v for k, v in runs.items() if k != name])
        te = cat([runs[name]])
        sd = tr[0].std()                                  # one global scale for all components
        tr[0], tr[1], te[0], te[1] = (x / sd for x in (tr[0], tr[1], te[0], te[1]))
        res = {}
        for use_cond in (False, True):
            model, cond, loss = fit(tr, use_cond, a.horizon, a.epochs, a.lr)
            with torch.no_grad():
                res[use_cond] = (rel_err(model.rollout(te[0], cond(te[2])), te[1]), loss)
        ident = rel_err(te[0][:, None].expand_as(te[1]), te[1])
        print(f'{name:16s} {ident:9.3f} {res[False][0]:9.3f} {res[True][0]:10.3f}   {res[True][1]:.2e}')


if __name__ == '__main__':
    main()
