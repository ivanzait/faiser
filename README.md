# Hermite Corrector

An ML safeguard for Hermite transform of velocity
distribution functions (VDFs) from [Vlasiator](https://github.com/fmihpc/vlasiator)
hybrid-Vlasov simulations.


## Method

**Transformation.** `data_processing/` holds the core VDF/Hermite
primitives, kept separate from the ML-specific code so it can be shared
with (and stay compatible with) other tools working on the same
decomposition. `vdf_tools.py` reads a Vlasiator VDF into a dense
velocity-space cube; `adaptive_hermite.py` computes an adaptive
tetrahedral Hermite decomposition up to a configurable maximum order,
tracking both a f-space error (`eps_rel`, via Parseval) and log-space RMS error (`eps_log`).

**Correction.** Everything specific to the ML safeguard lives in
`ml_corrector/`. The MLP corrector (`ml_corrector/corrector_model.py`)
predicts a delta correction applied multiplicatively in log-space:

```
f_final(v) = f_rec_full(v) * exp(Δ_pred(v))    inside the sparsity mask
           = 0                                  outside (structural)
```

This guarantees positivity regardless of the network's

**Features.** The corrector's input combines three signals, each covering
a different blind spot of the others (see the presentation for the full
story of why all three were needed):

| Feature | Size | Role |
|---|---|---|
| Low-order coefficients (`S_low` ≤ 2) | 10 | coarse global condition, already computed |
| 1-D Hermite-space power spectra (`sum_(l,m) C[l,m,n]^2` per axis) | 3×23 = 69 | cheap, strong "how non-Maxwellian" signal |
| Local 3×3×3 patch of `log(f_rec_full)` | 27 | spatial context to localize/shape the correction |

**Regularization.** A per-cell adaptive penalty on the predicted
correction, `λ_cell = λ_max / (1 + (mean_sq_cell / ref_var)^power)`,
suppresses correction on cells that are already well-reconstructed
without limiting it on genuinely hard ones. This is currently the
weakest point of the design — `λ_max`, `power`, `ref_var` are tuned
empirically per dataset.


## Repository layout

```
data_processing/                 # core library: colleague-compatible, no ML dependency
  vdf_tools.py                   # VDF cube extraction, drift/thermal velocity, sparsity threshold
  adaptive_hermite.py            # tetrahedral Hermite transform + reconstruction
ml_corrector/                    # ML-corrector library
  corrector_model.py             # the MLP, feature helpers (axis spectra, patches)
  decomposition_cache.py         # disk cache for the expensive full decomposition
  EXPERIMENT_LOG.md              # full experimental log
  corrector_weights.pt           # example pretrained checkpoint (multi-snapshot training)
scripts/                         # all runnable entry points (import from the two folders above)
  build_dataset.py               # parallel (multiprocessing), multi-timestep dataset builder
  build_dataset.sh               # config wrapper around build_dataset.py
  corrector_train.py             # trains the MLP from a prebuilt dataset
  corrector_eval.py              # within-snapshot held-out evaluation + before/after plots
  corrector_validate.py          # cross-timestep evaluation on an entirely unseen bulk file
  run_diagnostic.py              # single-cell Hermite convergence / spectra / VDF plots
  run_random_cells.py            # adaptive-order convergence sweep over random cells
  make_feature_figures.py        # regenerates the low-S / spectra / patch illustration figures
  make_presentation.py           # assembles docs/hermite_corrector_presentation.pdf
docs/
  hermite_corrector_presentation.pdf
  PROJECT_PLAN.md                # original design doc (partly superseded by EXPERIMENT_LOG.md)
  figures/                       # a curated subset of generated plots, used above
```

## Setup

```bash
pip install -r requirements.txt
```

You will also need [analysator](https://github.com/fmihpc/analysator)
(imported here under its legacy name, `import pytools as pt`) and a set of
Vlasiator `bulk.*.vlsv` files from a reconnection run. Neither is included
in this repository. By default the scripts expect bulk files under
`reconnection_2d_beta025/` at the repo root (edit `BULKDIR` / `--bulkdir`
to point elsewhere). Generated caches, datasets and plots are written
under `ml_corrector/{data,datasets,plots}/` and are gitignored.


MIT — see [LICENSE](LICENSE).
