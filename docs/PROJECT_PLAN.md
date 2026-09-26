# Hermite VDF Transform + ML Corrector — Project Plan

**Goal**: fast adaptive Hermite decomposition of Vlasiator VDFs (per-cell order chosen by a
Parseval check), with an ML corrector for the truncation error. The Hermite core is
implemented and working; the ML part is being rewritten for the new outputs.

---

## Priorities

### Priority 1 — Performance evaluation of the Hermite transform

Question to answer: **how long does it take to transform all VDFs in a bulk file, and what
does it cost in accuracy and storage?** Reference file: `bulk.0000055.vlsv`, 2812 cells with
a proton VDF (`vlen=100`, `dv=30 km/s`).

- [ ] Time the full file with `vt.adaptive_transform` (serial first; then parallel over cells).
      Report total wall time, time per cell, and the distribution of `order_used`.
- [ ] Tolerance sweep on the whole file (e.g. 0.3, 0.2, 0.1, 0.05, 0.035): total time,
      mean/max `order_used`, mean coefficients per cell, compression ratio vs the
      dense `vlen^3` cube.
- [ ] Baseline: `cubic_transform`-equivalent cost at the same order (one cell is enough,
      extrapolate; cubic ~15 s/cell at order 20 vs ~3 s tetrahedral on cell 44704).
- [ ] Split results by cell difficulty (simple vs complex VDFs), not only file totals.
- [ ] Honest reconstruction error on a random sample of cells (independent full
      reconstruction, raw and masked), next to the Parseval delta used for stopping.
- [ ] Record the validity limit: `vth * sqrt(2*order + 1) < vlim`. Above it the basis is no
      longer orthonormal on the grid and the Parseval delta can report false convergence.
      Cap `max_order` per simulation accordingly.
- Deliverable: one script (`scripts/`) that prints the table above for a given bulk file.

### Priority 2 — Rewrite the ML part for the new transform output

The old pipeline assumed the previous `adaptive_transform` API (four return values,
`sparse_mask`, `track_log_eps`, epsilon metrics, index convention `0..max_order`) and a
fixed 2300-coefficient layout. None of that exists any more.

- [ ] Rewrite `ml_corrector/decomposition_cache.py` for the new output
      `coeffs, order_used, deltas` (store `order_used`, `deltas`, `u`, `vth` per cell).
- [ ] Rewrite `scripts/build_dataset.py` on top of the new cache.
- [ ] Rewrite `ml_corrector/corrector_model.py`: features for a per-cell variable order
      (pad to a common order or fix the input order), and decide what the MLP predicts
      (residual field vs missing high-order coefficients).
- [ ] Decide the evaluation protocol before training: held-out runs/time steps (not
      random cells), and the ablation raw vs masked vs masked + MLP at equal coefficient
      budget.

### Priority 3 — Later

- [ ] Re-add training, evaluation and cross-timestep validation scripts (all removed
      for now; start from scratch against the new API).
- [ ] Physics metrics: density, bulk velocity, temperature before/after, positivity.
- [ ] Optional: even/odd mode separation (dropped from the current algorithm).
- [ ] Rewrite `README.md` for the new scope.

---

## 1. Current algorithm (implemented in `data_processing/vdf_tools.py`)

The transform expands the VDF `f` (not `log f`) in the orthonormal Hermite basis, centred on
the cell's bulk velocity `u` and scaled by its thermal velocity `vth`:

```
C[l,m,n] = sum f[z,y,x] * Hz_l * Hy_m * Hx_n * dv^3        (Riemann sum)
```

**Index convention.** `order` is a count, like `N` in a cubic transform: indices `l,m,n` and
the total order `s = l+m+n` all run `0..order-1`. `adaptive_transform` returns
`order_used` as a count too, so `order_used` can be passed directly as `order` to a
reconstruction or a crop, with no `+/-1`.

**Truncation.** Total-degree (tetrahedral) truncation, `l+m+n < order`, computed level by
level in `s`. Level `s` adds `(s+1)(s+2)/2` coefficients; the total up to `order` is
`order*(order+1)*(order+2)/6`.

| order | tetrahedral | cube (order^3) |
|---|---|---|
| 4  | 20   | 64    |
| 10 | 220  | 1000  |
| 20 | 1540 | 8000  |
| 22 | 2024 | 10648 |

**Parseval stop.** With an orthonormal basis, the power in the coefficients equals the power
in the VDF. Track the accumulated power `P_accum = sum(C^2)` against
`P_total = sum(f^2) * dv^3` (computed once):

```
delta_parseval(s) = sqrt(1 - P_accum(s) / P_total)
```

`delta_parseval` decreases monotonically with `s`. The loop stops at the first level `s > 2`
where `delta_parseval < tolerance` (the first three levels are always computed). Cost is
negligible: no reconstruction is needed, only the coefficients that were computed anyway.

**Validity limit.** The identity holds only while the basis is orthonormal on the velocity
grid, i.e. while `vth * sqrt(2*order + 1) < vlim`. Beyond that the outermost lobes are cut
by the grid boundary, the accumulated power can exceed `P_total`, and the clamp at 0 makes
the delta look converged when it is not.

**Reference measurement** (cell 44704, `bulk.0000055.vlsv`, `max_order=20`):

| tolerance | order_used | coefficients | reconstruction error (raw) |
|---|---|---|---|
| 0.2–0.3 | 4  | 20   | 0.189 |
| 0.1     | 8  | 120  | 0.099 |
| 0.05    | 14 | 560  | 0.048 |
| 0.035   | 19 | 1330 | 0.032 |

The stopping delta matches the independently measured reconstruction error.

## 2. Physical motivation for Hermite modes

In the bulk frame (shift `u`, scale `vth`), low orders carry the fluid-like content:

| Total order s | Parity | Physical content |
|---|---|---|
| 0 | even | density |
| 1 | odd  | residual drift (about 0 by construction of `u`) |
| 2 | even | temperature anisotropy, pressure cross-terms |
| 3 | odd  | heat flux, skewness |
| 4 | even | kurtosis, flat-top, bi-Maxwellian structure |
| high | mixed | beams, crescents, ring distributions, asymmetric tails |

A pure Maxwellian is captured by `s = 0`; non-Maxwellian structure needs higher orders,
which is what the adaptive order exploits.

## 3. Known issues and caveats

- **Tails.** The Parseval delta is dominated by the peak and is blind to the tails. A
  reconstruction can have a small delta and a large log-space error. This is the motivation
  for the mask and for the ML corrector.
- **Gibbs ringing** outside the support and negative values in the raw reconstruction.
  The structural mask (`f >= sp_th`) removes the ringing outside the support.
- **Old ML results are obsolete.** `ml_corrector/EXPERIMENT_LOG.md` is kept for reference,
  but it describes the previous pipeline (log-space transform, epsilon metrics, old index
  convention). Do not compare new numbers against it directly.

## 4. Repository layout

```
faiser/
├── data_processing/
│   ├── vdf_tools.py                  # readers, Hermite basis, adaptive_transform
│   └── vdfs_for_initialization.py    # bulk file -> per-cell coefficients
├── ml_corrector/                     # to be rewritten (Priority 2)
│   ├── corrector_model.py
│   ├── decomposition_cache.py
│   ├── hermite_plots.py
│   └── EXPERIMENT_LOG.md             # history of the previous pipeline
├── scripts/
│   ├── test_adaptive.py              # cubic vs tetrahedral comparison
│   ├── run_random_cells.py
│   ├── build_dataset.py              # to be rewritten (Priority 2)
│   └── test_adapt_herm.ipynb
└── docs/
```

---
*Last updated: 2026-09-26*
