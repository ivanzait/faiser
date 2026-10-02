

## Adaptive Hermite transform


- **Stopping rule.** The basis is orthonormal, so by Plancherel the power left out after level $s$ is
  known without reconstructing anything:

  $$\delta_s = \sqrt{1 - \frac{\sum_{s' \le s} C_{lmn}^2}{\int f^2\, d^3v}}$$

  This is the relative L2 error of the truncated series. Levels are added one by one and the loop
  stops at the first $s > 2$ with $\delta_s <$ `tolerance`, or at `max_order`. Higher levels are never
  computed, so cells with simple VDFs are cheap.

  - **Coefficients** are grouped in levels $s = l+m+n$ (tetrahedral truncation), so levels
  $0 \dots N-1$ give $N(N+1)(N+2)/6$ coefficients.

- **Caveat.** $\delta$ is an f-space measure dominated by the peak of the VDF, so it says little
  about the tails. Some cells (e.g. two separated beams) never reach the tolerance and use all
  `max_order` levels.

  Two optional post-processing steps, both in `data_processing/vdf_tools.py`:

- `apply_spectral_window`: tapers the coefficients to zero with $s$ (Lanczos or raised cosine) to
  reduce Gibbs ringing from truncating the series.
- `get_vdf_bounding_box` / `apply_bounding_box`: zeroes the reconstruction outside the box
  containing the VDF support. The box comes from the true VDF, so it has to be stored with the coefficients.

