"""Train the AE on the last output only and look at what it does.

    python autoencoder/build_dataset.py --only-last            # once: dense 32^3 VDFs of the last output
    python autoencoder/demo_last_step.py [--config autoencoder/config.toml]

Train cells: x below test_x_frac of the box; test cells: the rest (never seen). Prints relative L2 and moment errors
(density fraction, mean v', variance of v') and saves original / reconstruction / 1-D cuts for the best, median and
worst test cell to plots/ae_demo_last_step.png. Runs on CUDA, MPS or CPU.
"""
import os, sys, argparse, tomllib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import AE


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def moments(g, box):
    """(mass fraction, mean v' [3], variance of v' summed over axes) of g [B, 32, 32, 32] (axes z, y, x)."""
    n = g.shape[-1]; dv = 2 * box / n
    ax = (torch.arange(n, device=g.device) + 0.5) * dv - box
    m = g.sum((1, 2, 3)) * dv ** 3
    mx = torch.einsum('bzyx,x->b', g, ax) * dv ** 3 / m; my = torch.einsum('bzyx,y->b', g, ax) * dv ** 3 / m
    mz = torch.einsum('bzyx,z->b', g, ax) * dv ** 3 / m
    r2 = torch.einsum('bzyx,z->b', g, ax ** 2) + torch.einsum('bzyx,y->b', g, ax ** 2) + torch.einsum('bzyx,x->b', g, ax ** 2)
    return m, torch.stack([mx, my, mz], 1), r2 * dv ** 3 / m - (mx ** 2 + my ** 2 + mz ** 2)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', default=os.path.join(HERE, 'config.toml')); a = p.parse_args()
    cfg = tomllib.load(open(a.config, 'rb')); dcfg, acfg = cfg['data'], cfg['ae']
    d = np.load(os.path.join(HERE, '..', dcfg['out_file']))
    last = d['out_idx'] == d['out_idx'].max()
    g, x, box = d['g'][last], d['x'][last], float(d['box'])
    print(f'{len(g)} VDFs from output {d["out_idx"].max()} (t = {d["t"][last][0]:.1f} s)')

    test = x > acfg['test_x_frac'] * x.max() * (1 + 1 / len(x))
    dev = device(); torch.manual_seed(acfg['seed'])
    G = torch.tensor(g, device=dev); tr, te = G[~torch.tensor(test, device=dev)], G[torch.tensor(test, device=dev)]
    print(f'device {dev} | train {len(tr)} cells, test {len(te)} cells | latent d = {acfg["latent"]}')

    model = AE(acfg['latent']).to(dev); opt = torch.optim.Adam(model.parameters(), lr=acfg['lr'])
    for ep in range(acfg['epochs']):
        perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr), acfg['batch']):
            b = tr[perm[i:i + acfg['batch']]].sqrt()      # train in sqrt space: plain MSE on g collapses to 0 (g is 99% empty)
            loss = ((model(b) - b) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == acfg['epochs'] - 1:
            with torch.no_grad():
                print(f'epoch {ep:4d}  train loss {loss.item():.3e}  test rel.L2 {rel(recon(model, te), te).mean():.3f}')

    model.eval()
    with torch.no_grad():
        for name, s in (('train', tr), ('test', te)):
            r = recon(model, s); e = rel(r, s)
            m0, v0, s0 = moments(s, box); m1, v1, s1 = moments(r, box)
            print(f'{name:5s} rel.L2 mean {e.mean():.3f} max {e.max():.3f} | mass err {(m1 / m0 - 1).abs().mean():.2e} '
                  f'| |<v>| err {(v1 - v0).norm(dim=1).mean():.3f} (units of v_th) | var err {((s1 - s0) / s0).abs().mean():.2e}')
        rec = recon(model, te); z = model.encode(te.sqrt())
        mean_img = tr.mean(0, keepdim=True).expand_as(te)
        print(f'reference: predicting the train-mean VDF everywhere gives test rel.L2 {rel(mean_img, te).mean():.3f}')
    print(f'latent: {tuple(z.shape)} per cell {z.shape[1]} numbers vs {te[0].numel()} voxels')

    err = rel(rec, te).cpu().numpy(); order = np.argsort(err); pick = [order[0], order[len(order) // 2], order[-1]]
    te, rec = te.cpu().numpy(), rec.cpu().numpy(); ext = [-box, box, -box, box]
    fig, axs = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
    for row, (k, tag) in zip(axs, zip(pick, ('best', 'median', 'worst'))):
        pk = te[k].sum(0).max(); norm = LogNorm(pk * 1e-6, pk)
        row[0].imshow(te[k].sum(0), origin='lower', extent=ext, norm=norm, cmap='Spectral_r'); row[0].set_title(f'original ({tag}, err {err[k]:.2f})')
        row[1].imshow(rec[k].sum(0), origin='lower', extent=ext, norm=norm, cmap='Spectral_r'); row[1].set_title('AE reconstruction')
        row[2].imshow(rec[k].sum(0) - te[k].sum(0), origin='lower', extent=ext, cmap='RdBu_r', vmin=-pk * 0.5, vmax=pk * 0.5); row[2].set_title('difference')
        v = (np.arange(32) + 0.5) * 2 * box / 32 - box
        row[3].semilogy(v, te[k].sum((0, 1)), label='original'); row[3].semilogy(v, rec[k].sum((0, 1)), '--', label='AE'); row[3].legend(); row[3].set_title("f(v'_x)")
        row[3].set_ylim(te[k].sum((0, 1)).max() * 1e-6, None)
        for ax_ in row[:3]: ax_.set(xlabel="v'_x", ylabel="v'_y")
    out = os.path.join(HERE, '..', 'plots', 'ae_demo_last_step.png'); os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120); print('saved ->', os.path.normpath(out))


def recon(model, g):
    """g -> sqrt(g) -> AE -> squared: non-negative reconstruction of g."""
    return model(g.sqrt()) ** 2


def rel(pred, true):
    return (pred - true).flatten(1).norm(dim=1) / true.flatten(1).norm(dim=1)


if __name__ == '__main__':
    main()
