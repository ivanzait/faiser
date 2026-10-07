"""Leave-one-run-out training of the stepper on gyration runs.

    python propagator/train.py data/run_*.npz --test data/run_07.npz
"""
import sys, argparse
import numpy as np
import torch

sys.path.insert(0, __file__.rsplit('/', 2)[0])
from propagator.data import pairs
from propagator.model import Stepper


def rel_err(pred, true):
    return float(torch.linalg.norm(pred - true) / torch.linalg.norm(true))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('train', nargs='+'); p.add_argument('--test', required=True)
    p.add_argument('--horizon', type=int, default=4); p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-3)
    a = p.parse_args()

    tr = [torch.tensor(x, dtype=torch.float32) for x in pairs(a.train, horizon=a.horizon)]
    te = [torch.tensor(x, dtype=torch.float32) for x in pairs([a.test], horizon=a.horizon)]
    # conditioning for a rollout step is the same dt: repeat the single-step cond along the horizon
    conds = lambda c: c[:, None].expand(-1, a.horizon, -1)
    mu, sd = tr[1].reshape(-1, tr[1].shape[-1]).mean(0), tr[1].reshape(-1, tr[1].shape[-1]).std(0) + 1e-8

    model = Stepper(tr[0].shape[-1], tr[2].shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    for ep in range(a.epochs):
        loss = (((model.rollout(tr[0], conds(tr[2])) - tr[1]) / sd) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 200 == 0 or ep == a.epochs - 1:
            with torch.no_grad():
                print(f'{ep:5d} loss {loss.item():.2e}  test rollout rel.err {rel_err(model.rollout(te[0], conds(te[2])), te[1]):.3f}')


if __name__ == '__main__':
    main()
