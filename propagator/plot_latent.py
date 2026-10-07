"""z(t) of the gyration runs: bulk velocity, thermal speed, the Hermite coefficients that change most, and the
bulk-velocity gyration check (|u|, u.B-hat constant; rotation rate vs Omega = (q/m)|B|).

    python propagator/plot_latent.py [propagator/data/gyration*.npz]
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from propagator.data import Q_M
from propagator.latent import index_set

OUT = os.path.join(os.path.dirname(__file__), '..', 'plots', 'latent_trajectories.png')


def main(files):
    fig, axes = plt.subplots(len(files), 4, figsize=(17, 3.2 * len(files)), constrained_layout=True, squeeze=False)
    for row, f in zip(axes, files):
        d = np.load(f); z, t, B = d['z'], d['t'], d['B']; K = int(d['K'])
        bh, Om = B / np.linalg.norm(B), Q_M * np.linalg.norm(B)
        u = z[:, :3] * float(d['vlim']) / 1e3                                   # km/s
        c = z[:, 4:]; idx = index_set(K)
        top = np.argsort(np.ptp(c, axis=0))[::-1][:6]
        for k in range(3):
            row[0].plot(t, u[:, k], label='uxyz'[k + 1])
        row[0].set(title=f'{os.path.basename(f)[:-4]}  B=({", ".join(f"{b*1e10:.2f}" for b in B)})e-10', ylabel='u [km/s]')
        row[1].plot(t, np.exp(z[:, 3]) / 1e3); row[1].set(title='vth', ylabel='km/s')
        for k in top:
            row[2].plot(t, c[:, k], label=str(idx[k]))
        row[2].set(title='6 most-varying coefficients'); row[2].legend(fontsize=6, ncol=2)
        upar, uperp = u @ bh, np.linalg.norm(u - np.outer(u @ bh, bh), axis=1)
        e1 = (u[0] - upar[0] * bh); e1 /= np.linalg.norm(e1) if np.linalg.norm(e1) > 0 else 1
        e2 = np.cross(bh, e1); phi = np.unwrap(np.arctan2(u @ e2, u @ e1))
        row[3].plot(t, phi, label='measured'); row[3].plot(t, phi[0] - Om * (t - t[0]), '--', label='-Omega t')
        row[3].plot(t, phi[0] + Om * (t - t[0]), ':', label='+Omega t')
        row[3].set(title=f'gyrophase  |u_perp| {uperp.mean():.1f}±{uperp.std():.2f}  u_par {upar.mean():.1f}±{upar.std():.2f}',
                   ylabel='rad'); row[3].legend(fontsize=7)
        print(f'{os.path.basename(f)}: Omega={Om:.4f} rad/s  phase change {phi[-1]-phi[0]:+.3f} rad  expected {Om*(t[-1]-t[0]):.3f}  '
              f'ptp(vth)/vth {np.ptp(np.exp(z[:,3]))/np.exp(z[0,3]):.2e}  ptp(coeffs) max {np.ptp(c, axis=0).max():.3e}  C000 {c[0,0]:.3e}')
    os.makedirs(os.path.dirname(OUT), exist_ok=True); fig.savefig(OUT, dpi=110); print('saved ->', OUT)


if __name__ == '__main__':
    main(sys.argv[1:] or sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'data', '*.npz'))))
