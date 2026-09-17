# Hermite VDF Corrector — Project Plan

**Goal**: adaptive Hermite decomposition of kinetic VDFs from Vlasiator bulk files,
with ML-based correction of truncation error, exploiting the even/odd symmetry structure
of reconnection geometry.

---

## 1. Physical Motivation

### Why Hermite in log-space?

The VDF spans ~10 orders of magnitude. Working in log(f) space:
- A pure Maxwellian becomes a quadratic → captured exactly by order-2 Hermite
- Non-Maxwellian features (beams, crescents, counter-streaming) appear only in higher orders
- Truncation error is O(1) in log-space vs O(10^8) in f-space → far more tractable for ML

The transform (from vdf_tools.py):
```
log_cube = log(cube) - log(sp_th)       # log-shifted, zeros floor
spectra  = Hermite_transform(log_cube)  # coefficients C[l,m,n]
```

### Hermite modes and their physical meaning

In the bulk frame (shifted by drift velocity u, scaled by thermal velocity vth):

| Total order s = l+m+n | Parity | Physical content |
|---|---|---|
| 0 | even | log-normalization ~ log(n) |
| 1 | odd  | residual drift (should be ~0 by construction of u) |
| 2 | even | temperature anisotropy, pressure cross-terms, agyrotropy |
| 3 | odd  | heat flux, skewness |
| 4 | even | kurtosis, bi-Maxwellian structure, flat-top |
| 5 | odd  | asymmetric beams |
| ... | | |
| high even | even | sharp temperature gradients, ring/crescent wings |
| high odd  | odd  | asymmetric non-Maxwellian tails, single beams |

### Number of coefficients per approach

For total-order truncation s = l+m+n ≤ N (triangle/tetrahedral):
- Level s has (s+1)(s+2)/2 terms
- Total up to N: (N+1)(N+2)(N+3)/6

| N  | Even-only | Total (even+odd) | Cube N³ (current code) |
|----|-----------|-----------------|------------------------|
| 4  |        22 |              35 |                     64 |
| 8  |       165 |             165→ C(10,3)=120... compute properly |
| 10 |       286 |             286 |                   1000 |
| 22 |      1222 |            2300 |                  10648 |

> Note: current vdf_tools.py uses cube indexing (l,m,n each 0..order-1 independently),
> i.e., includes ALL combinations including l=20,m=20,n=20 (total order 60).
> The adaptive transformer uses triangle/tetrahedral truncation instead.

---

## 2. Even/Odd Mode Separation

### Physical justification

In the bulk drift frame, parity of f(v) = f(-v) (symmetric distribution) → ONLY even modes.

In GEM-challenge reconnection:

- **X-point region**: nearly symmetric by geometry → even modes dominate
- **Reconnection exhaust**: bulk outflow (odd drift already removed by u-shift), but heating and 
  pressure anisotropy are even → even modes dominate for the non-trivial features too
- **Separatrices / current sheet edge**: crescent distributions and counter-streaming may have 
  significant odd content (agyrotropy ≠ 0 in a specific sense)
- **Lobe**: near-Maxwellian → even, essentially C[0,0,0] only

**Key insight**: for most reconnection-relevant structures (agyrotropy, bi-Maxwellian, flat-top,
ring distributions), the dominant non-Maxwellian signal is in EVEN modes.
Odd modes capture residual asymmetries after the u-shift.

### Practical strategy

```
Step 1: compute even modes only (levels s = 0, 2, 4, ...)
Step 2: check Parseval error ε_even
Step 3: if ε_even < threshold → stop (odd modes negligible)
Step 4: otherwise add odd modes level by level until ε < threshold
```

This allows the transformer to skip odd modes for symmetric cells (most of the domain),
saving ~50% computation and storage for those cells.

---

## 3. Parseval-Based Adaptive Stopping

### Parseval's theorem for the normalized Hermite basis

Since the Hermite functions φ_{l,m,n}(v) are orthonormal:

```
||log(f)||² = ∫∫∫ [log(f)]² d³v ≈ sum(log_cube²) * dv³   [total Parseval power]
            = Σ_{l,m,n} C[l,m,n]²                          [sum of squared coefficients]
```

**Reconstruction error** after including modes up to level s_max:
```
ε²(s_max) = Σ_{s > s_max} C[l,m,n]²
           = P_total - Σ_{s ≤ s_max} C[l,m,n]²
```

P_total is computed ONCE directly from the VDF (no Hermite needed).
As we compute more levels, ε decreases monotonically.

### Adaptive algorithm

```python
def adaptive_hermite(log_cube, vlim, vlen, vth, u,
                     rel_threshold=0.01, max_order=22,
                     even_first=True):
    """
    Returns Hermite coefficients and achieved relative error.
    Stops as soon as relative L2 reconstruction error < rel_threshold.
    """
    dv = 2 * vlim / vlen
    P_total = np.sum(log_cube**2) * dv**3   # Parseval total power
    
    spectra = {}   # dict: (l,m,n) -> coefficient value
    P_accum = 0.0
    
    levels = even_levels_then_odd(max_order) if even_first \
             else all_levels(max_order)
    
    for s in levels:
        # compute all C[l,m,n] with l+m+n = s
        new_coeffs = compute_level(log_cube, s, vlim, vlen, vth, u)
        for idx, val in new_coeffs.items():
            spectra[idx] = val
            P_accum += val**2

        eps_rel = np.sqrt(max(0, 1 - P_accum / P_total))
        
        if eps_rel < rel_threshold:
            return spectra, eps_rel, s   # done
    
    return spectra, eps_rel, max_order   # reached max order
```

### Per-cell compression statistics (expected)

| Cell location | Expected stopping order | Dominant parity | Coefficients used |
|---|---|---|---|
| Deep lobe | s=0 | even | 1 |
| Quiet lobe | s=2..4 | even | 22 |
| Current sheet center | s=4..8 | even | 22..165 |
| X-point vicinity | s=8..14 | even+odd | 165..560 |
| Separatrix / crescent | s=12..20 | even+odd | 364..1540 |

---

## 4. ML Corrector Architecture

### Problem framing

Given:
- `spectra_low` = Hermite coefficients up to N_low (e.g., 4) — computed cheaply
- `context` = local plasma state: E, B, u, vth
- Target: `spectra_full` = coefficients up to N_high (as computed by adaptive transformer)

Train MLP to predict `spectra_full` from `spectra_low + context`,
so that adaptive transformer can be bypassed at inference time.

### Input/output dimensionality

| Quantity | Size |
|---|---|
| spectra_low (N_low=4, even only) | 22 |
| spectra_low (N_low=4, all) | 35 |
| E_x, E_y, E_z | 3 |
| B_x, B_y, B_z | 3 |
| u_x, u_y, u_z, vth | 4 |
| **Total input** | **~45** |

Output: coefficients from level N_low+1 up to N_high (adaptive stopping order).
For N_high=10: ~85 values. For N_high=22: ~2265 values.

### Dimensionality reduction: PCA on output

For N_high=22 (~2300 output values), apply PCA:
```
from all training spectra_full: find top K=64..128 principal components
MLP predicts PCA coordinates (K values)
Reconstruct: spectra_full = pca_mean + U @ prediction
```

**Proposed architecture (znet.py `MultiLayerPerceptron`)**:
```
Input:  45 features   (normalized to 0-mean unit-variance per feature)
Hidden: [256, 256]    (tanh activation)
Output: 64..128 PCA coordinates  (targets min-max normalized to (-1,1))
```

### Even/odd ML split (optional refinement)

Train two separate small MLPs:
- `MLP_even`: predicts even-mode PCA coordinates
- `MLP_odd`: predicts odd-mode PCA coordinates, only called when even-only insufficient

At inference: call MLP_even always, call MLP_odd conditionally based on Parseval check.

---

## 5. Pipeline Steps

```
[Bulk files]
    │
    ▼
extract_data.py
  - for each bulk file: loop over cells with VDF
  - vdf_tools.build_cube() → cube
  - fix dvx: read from VLSV file (NOT hardcoded 52000!)
  - compute u, vth
  - adaptive_hermite() → spectra_full (ground truth)
  - spectra_low = spectra_full up to N_low
  - read E, B from bulk file for this cell
  - save: (X_features, Y_spectra_full) per cell
    │
    ▼ [training data .npz]
    │
    ▼
pca_fit.py
  - load all Y_spectra_full
  - fit PCA → K components
  - save pca_mean, pca_components
    │
    ▼
train.py
  - normalize X (z-score per feature)
  - project Y onto PCA → Y_pca
  - normalize Y_pca (min-max to (-1,1))
  - train MultiLayerPerceptron (znet.py)
  - save: model weights, X_norm_params, Y_pca_params
    │
    ▼
evaluate.py
  - apply model to test cells
  - reconstruct spectra_full from PCA
  - reconstruct log_cube → cube
  - compute L2 error vs ground truth
  - plot: original / reconstructed / ML-corrected side by side
```

---

## 6. Key Technical Decisions

### Fixes needed in vdf_tools.py

```python
# REPLACE hardcoded dvx in get_vdf_parameters():
vx_min, _, _, vx_max, _, _ = reader.get_velocity_mesh_extent(pop="proton")
nx = int(extents[0] * 4)
dvx = (vx_max - vx_min) / nx
vlim = vx_max
dv = dvx
```

### Fix needed in znet.py

```python
# update_biases: missing output layer
# CHANGE:
for i in range(len(self.B)-1):
# TO:
for i in range(len(self.B)):
```

### Level-by-level Hermite computation

Current `get_hermite_spectra_cube` computes the full cube (order × order × order).
For adaptive stopping, we need per-level computation:

```python
def compute_hermite_level(log_cube, s, vlim, vlen, vth, u):
    """Compute all C[l,m,n] with l+m+n == s."""
    v_ax = np.linspace(-vlim, vlim, vlen)
    dv = 2 * vlim / vlen
    # precompute basis functions for orders 0..s in each dimension
    Hx = hermite_basis(v_ax, s+1, vth, u[0])
    Hy = hermite_basis(v_ax, s+1, vth, u[1])
    Hz = hermite_basis(v_ax, s+1, vth, u[2])
    # sum only over (l,m,n) with l+m+n == s
    result = {}
    for l in range(s+1):
        for m in range(s+1-l):
            n = s - l - m
            c = np.einsum('zyx,x,y,z->', log_cube, Hx[n], Hy[m], Hz[l]) * dv**3
            result[(l,m,n)] = c
    return result
```

---

## 7. Open Questions

1. **What N_low to use?** N_low=4 captures density, temperature, heat flux (classic 10-moment fluid).
   N_low=2 captures only density and temperature (5-moment). Start with N_low=4.

2. **What threshold for adaptive stopping?** ε_rel=0.01 (1%) seems reasonable.
   Need to benchmark: what ε_rel does pure N_low=4 achieve in the reconnection region?

3. **How many PCA components needed?** Run PCA on training data and check cumulative variance.
   Likely K=32..64 explains >95% of variation in non-Maxwellian structure.

4. **Training data size**: currently ~8 cells/timestep with VDF (stride=32 in β=1 run).
   β=0.25 run has xline_stride=1 (all 64 x-cells at z=0) → much better.
   Need >500 samples ideally. If few: use data augmentation (velocity-space rotations, rescaling).

5. **Separability test**: check if training at X-point generalizes to separatrices.
   If not: train separate models per plasma region (X-point, exhaust, lobe).

---

## 8. File Structure

```
faiser/
├── hermite_plan.md              ← this file
├── vdf_tools.py                 ← (patched: dvx from file)
├── znet.py                      ← (patched: bias update fix)
├── hermite_ml/
│   ├── adaptive_hermite.py      ← level-by-level transform + Parseval stopping
│   ├── extract_data.py          ← bulk files → training .npz
│   ├── pca_fit.py               ← fit PCA on spectra_full
│   ├── train.py                 ← normalize + MLP.train() + save
│   └── evaluate.py             ← quality metrics + plots
└── hermite_data/
    ├── training_*.npz           ← extracted cell data
    ├── pca_model.npz            ← PCA components
    └── mlp_weights.npz          ← trained MLP weights
```

---

## 9. Implementation Order

1. `adaptive_hermite.py` — the core: level-by-level + Parseval + even/odd split
2. Run on one bulk file manually → check what N_stop looks like per cell type
3. `extract_data.py` — build training set from all available bulk files
4. `pca_fit.py` — fit PCA, check variance explained vs K
5. `train.py` — MLP training loop
6. `evaluate.py` — comparison plots

---
*Last updated: 2026-09-11*
