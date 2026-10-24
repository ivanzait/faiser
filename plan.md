# Plan

Goal: compress Vlasiator proton VDFs with the adaptive Hermite transform, reconstruct them with the hybrid
method below, and feed the reconstructed VDFs back into the solver.

## Where we are (2026-10-24)

- `adaptive_transform` / `reconstruct_vdf_adaptive` with `sqrt_f=True` (expand sqrt(f), basis scale sqrt(2)*vth).
  Parseval delta = exact L2 error of the sqrt(f) series (checked on TSI cells); the L2 error in f is 1.6-2.4x larger.
- Vlasiator's block rule (content block = some cell >= sp_th, plus +-1 block around it) reproduces the stored
  support of the original almost exactly (BID/BIE exactly, TSI to 0.01-0.2%; the rest is the spatial-neighbour
  part of the rule).
- **Hybrid reconstruction** (tested on 4 TSI t=30 cells, order 33):
  1. Cesaro (Fejer) mean of the series: level s = l+m+n weighted by (N - s) / N, N = order_used.
  2. Boundary: `apply_block_mask` with the content detected on the Cesaro mean, width 1 block (Vlasiator's rule).
  3. Halo (inside the mask, outside the Cesaro sp_th contour): Cesaro values.
  4. Core (Cesaro mean >= sp_th): plain series.

  ```python
  l, m, n = np.indices(c.shape)                                  # c: dense spectrum cut to order_used = N
  c_ces   = c * np.clip(1 - (l + m + n) / N, 0, None)
  rec     = reconstruct_vdf_adaptive(c,     ..., sqrt_f=True)
  rec_ces = reconstruct_vdf_adaptive(c_ces, ..., sqrt_f=True)
  hybrid  = np.where(rec_ces >= sp_th, rec, apply_block_mask(rec_ces, sp_th, width=1))
  ```

  Result vs plain series + bbox: density and L2 unchanged (core is the plain series); halo median
  |dlog10 f| 0.38-0.52 instead of 1.25-1.52; extra support ~2% of stored cells with mass ~3e-8
  (was 94-117% with mass ~1e-4); halo cells lost 8-14% (the Cesaro contour is slightly tighter than
  the true one; width 2 would keep them, 0.1-0.6% lost, at 21-24% extra support).
  Known leftover: small holes from the plain series' ringing inside the core near the contour
  (possible fix later: plain series only where rec_ces >= k*sp_th, k ~ 10).

## Steps

1. **Revise the code**
   - Move the hybrid into `tools/vdf_tools.py`: Cesaro weights, `apply_block_mask(..., detect=...)`
     (already there), and one function for the hybrid reconstruction.
   - Use it in `one_cell_view.py` / `multi_cell_view.py` (option: plain or hybrid).
   - The hybrid does not need the bbox: its Cesaro block mask already sets the boundary (bbox on top changes
     nothing on TSI; on BIE 795501 it trims extra cells 26% -> 15% but only 3.4e-5 of the mass). Remove the bbox
     from the reconstruction; keep `bbox_idx` in the files for now or drop it.
   - Fix `runners/run_multi_cell.py`: `BULK_FILE` is built once at import with tstep=30, so the loop over
     time steps reads bulk 30 every time. The current `data/run_TSI_tstep{5..25}.npz` all contain tstep 30
     data and must be regenerated.
   - Clean up what is left from the review (unused imports in `vdf_tools`, commented-out metrics).
   - The experiment scripts are in the session scratchpad (`block_mask_test.py`, `cesaro_test.py`,
     `hybrid_test.py`); they are temporary and will be lost, the recipe above is the reference.

2. **Check performance**
   - Time per cell: build_cube, transform, plain and Cesaro reconstructions, masking.
     The hybrid needs two reconstructions; both can share the basis functions.
   - Memory at vlen = 240 and the cost of the dense 3D einsum at order ~33.

3. **Test on another simulation: BIE (legendary cell 795501, two beams), plus BID 180041**
   - Same metrics: density, L2 in f, halo |dlog10|, support (lost / extra cells), compression.
   - Expect the two-beam limitation of a single Hermite centre to remain; check that the hybrid still
     helps the halo there.

4. **Plug into Vlasiator**
   - Reconstruct the VDFs with the hybrid method and write them back so the solver can run from them
     (initial condition / restart).
   - Compare the evolution against the original run.
