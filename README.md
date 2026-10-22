# Adaptive Hermite compression of Vlasiator proton VDFs

## Layout

```
tools/vdf_tools.py                  reading VDFs, moments, adaptive Hermite transform, reconstruction, metrics
plot_routines/plot_tools.py         1D cuts, 2D slices / projections, Hermite spectrum
plot_routines/check_vdf.py          raw VDF of one cell, moments vs the ones stored in the bulk file
runners/run_one_cell.py             transform one cell -> data/one_cell_run_<run>_tstep<N>_cell<ID>.npz
runners/one_cell_view.py            npz -> reconstruction, metrics, plots
runners/vdfs_for_initialization.py  many cells -> one dataset npz
runners/two_stream_profile.py       f(vx, x) profile of the two-stream test run
```

## Algorithm

- **Cube.** The VDF of a spatial cell is read into a dense `cube[iz, iy, ix]` on the velocity mesh. Vlasiator
  stores whole 4x4x4 blocks, so cells below the sparsity threshold inside kept blocks (the halo) are present;
  dropped blocks are 0.

- **Basis.** Orthonormal Hermite functions centred on the bulk velocity $u$ with scale $v_{th}$, both from the
  moments of the cube. Coefficients $C_{lmn}$ are grouped in levels $s = l+m+n$ (tetrahedral truncation), so
  levels $0 \dots N-1$ give $N(N+1)(N+2)/6$ coefficients.

- **Stopping rule.** The basis is orthonormal, so by Parseval the power left out after level $s$ is known
  without reconstructing anything:

  $$\delta_s = \sqrt{1 - \frac{\sum_{s' \le s} C_{lmn}^2}{\int f^2\, d^3v}}$$

  Levels are added one by one and the loop stops at the first $s > 2$ with $\delta_s <$ `tolerance`, or at
  `max_order`. Higher levels are never computed, so cells with simple VDFs are cheap.

- **$\sqrt{f}$ expansion** (`sqrt_f=True`, default). The transform expands $g = \sqrt{f}$ on a basis of scale
  $\sqrt{2}\,v_{th}$ and the reconstruction is squared back. It is then non-negative, ringing outside the support
  drops from $\sim\epsilon f_{max}$ to $\sim\epsilon^2 f_{max}$, and $\delta$ measures the density error
  ($\int g^2 = n$).

- **Bounding box.** The box of everything stored for the cell ($f > 0$, halo included) is saved with the
  coefficients, and the reconstruction is zeroed outside it.
