# Hermite corrector — experiment log

Running notebook of ML-corrector experiments (Variant B: MLP predicts a
log-space multiplicative correction on top of a low-order Hermite fit).
Started once the underlying pipeline was verified correct (velocity-grid
alignment bug fixed, dv^3 normalization fixed, sp_th read from file).

Setup recap: base = f_rec_low at S_low=2 (10 coeffs, sign-safe, eps_rel~0.27
but eps_log floor-dominated). Target: Delta_true(v) = log(max(cube,sp_th)) -
log(max(f_rec_low,sp_th)), inside sparse_mask only. MLP input = [C_low (10),
x_hat,y_hat,z_hat] (13-dim), plain Linear+ReLU (easy to reimplement from
scratch for inference later). f_final = f_rec_low * exp(Delta_pred) inside
mask, 0 outside (structural).

Data: bulk.0000024.vlsv (reconnection_2d_beta025), 128 cells, split 103
train / 25 val by CELL (seed=42).

---

## Exp 1 — baseline (uniform voxel sampling, unweighted MSE loss)

Config: S_low=2, 3000 samples/cell (uniform random inside sparse_mask),
hidden=64, 30 epochs, batch=4096, lr=1e-3, plain MSE loss.

Result:
- train_mse: 4.03 -> 1.69 (still decreasing at epoch 30, not converged)
- val_mse: 3.60 -> 1.52
- Held-out eval (eps_log, RMS log residual inside mask):
  mean base (f_rec_low alone) = 1.5123
  mean corrected              = 1.1014
  **mean improvement = 27.2%**

Per-cell breakdown: real improvement (20-60%) on non-Maxwellian cells with
large base error (eps_log_base > 1). But: on 2 near-pure-Maxwellian lobe
cells (cid 64, 2016) where eps_log_base = 0.0000 EXACTLY (S_low=2 already
perfect there, matches earlier finding that lobes need only s=0), the
corrector *degrades* the result, adding noise (eps_log_corr ~ 0.39). The
network, trained mostly on "hard" cells, doesn't learn "output ~0 when
nothing needs fixing" -- it has no mechanism to be locally conservative.

Hypotheses for the failure mode:
(a) uniform per-cell sample budget under-represents "trivial" cells in the
    global loss (an already-good cell contributes near-zero-target samples,
    easily outweighed by noisy targets from hard cells in the same batch)
(b) no regularization pulling Delta_pred toward 0 in the absence of signal
(c) plain MSE weights every masked voxel equally, including deep-tail
    voxels where cube is tiny and physically negligible but numerous

Next planned experiment: weight the loss by |cube - f_rec_low| (the raw
f-space residual of the base fit) -- cheap to compute since S_low=2
reconstruction is fast. Rationale: this should shift training emphasis onto
voxels where the base fit is most wrong in absolute (physically meaningful)
terms -- likely real structure (jets, heat flux) -- rather than treating
deep-noise-floor voxels (numerous, but physically negligible, currently
equal-weighted in the loss) the same as the peak/near-peak region.

---

## Exp 2 — weighted loss, weight = |cube - f_rec_low| (per-cell normalized) — FAILED

Same config as Exp 1, but loss = sum(w*(pred-target)^2)/sum(w), with
w = |cube - f_rec_low| / mean(|cube - f_rec_low| over the full masked
region of that cell). Same data split, same seed.

Training curves looked healthy on their OWN metric (val_wmse: 3.68 -> 0.96),
but the UNWEIGHTED val_mse_plain (directly comparable to Exp 1's val_mse)
got WORSE, not better: 3.79 -> 4.46 (peaking ~7.0 mid-training), vs Exp 1's
3.60 -> 1.52.

Held-out eval (same eps_log metric as Exp 1):
  mean base       = 1.5123  (unchanged, same base fit)
  mean corrected  = 1.9097
  **mean improvement = -26.3%**  (Exp 1 was +27.2%)

Every single cell got worse or only marginally better (best case +29.1% on
cid 1008, but most cells -15% to -140%). The two "trivial" lobe cells
(64, 2016) that Exp 1 merely nudged off zero (eps_log_corr ~0.39) are now
FAR worse (~1.07).

**Diagnosis (why this failed):** |cube - f_rec_low| is an f-space
(absolute, linear) quantity -- completely dominated by the peak. Weighting
by it means voxels near the peak get weight >>1 and deep-tail/noise-floor
voxels get weight ~0, so the network essentially stops being penalized for
getting the tail's log-value wrong at all. But eps_log (the evaluation
metric, and the metric we actually care about -- see hermite_ml notes on
why eps_rel/f-space metrics are peak-dominated and blind to tail accuracy
in the first place) is a UNIFORM average over every masked voxel, tail
included. So this weighting doesn't just fail to help eps_log, it actively
optimizes against it: training objective and evaluation objective point in
different directions here.

Lesson: any voxel importance-weighting scheme has to be defined in the SAME
space as the thing we're actually trying to get right (log-space relative
accuracy across the full dynamic range), not in raw f-space. An f-space
residual weight reintroduces exactly the peak-domination problem eps_log
was invented to get away from (see the very first "why eps_rel doesn't
resolve the tails" finding earlier in this project). Weighting by something
like |Delta_true| itself (the log-space residual, not the f-space one)
might make more sense if the goal is "focus on voxels most wrong in the
metric we evaluate" -- but that risks a different failure mode (circularly
telling the network to prioritize whatever it's currently worst at, which
for deep-tail noise could just mean chasing overfitted noise). Reverting to
Exp 1's unweighted loss as the working baseline; next ideas should target
the *lobe-cell degradation* failure mode specifically (regularization
toward Delta_pred=0, or more trivial-cell representation) rather than
reweighting voxels by an f-space quantity.

**Reverted.** train_corrector.py now has a WEIGHT_MODE flag ('none' default
= Exp 1 behavior, 'f_resid' = Exp 2, kept only for reference/comparison --
do not use for a real model). Retrained with WEIGHT_MODE='none' to restore
a working checkpoint: val_mse 3.60->1.63 (vs Exp 1's 1.52 -- small run-to-run
variation from unseeded torch init/shuffle, same ballpark), held-out eval
mean improvement = 24.3% (vs Exp 1's 27.2%, same story). corrector_weights.pt
on disk is this restored baseline, not the Exp 2 model.

---

## Exp 3 — output regularization: loss += lambda * mean(Delta_pred^2), lambda=0.1

Motivation: fix the lobe-cell degradation (Exp 1/1-revert: eps_log_corr
~0.33-0.39 on cid 64,2016 where eps_log_base=0.0 exactly) by adding an
explicit "predict 0 unless the fit term says otherwise" prior directly on
the network's output, rather than reweighting voxels (Exp 2's approach,
which failed because f-space weights are peak-dominated). WEIGHT_MODE
stayed 'none'.

Training: val_fit 3.61->1.62 (fit-only term, comparable to Exp 1's plain
val_mse -- consistent, regularization barely cost any fit accuracy at
lambda=0.1). val_reg grew from 0.02 to 0.165 (Delta_pred magnitudes
shrinking overall, as expected).

Held-out eval:
  mean base      = 1.5123  (unchanged)
  mean corrected = 1.1226
  **mean improvement = 25.8%**  (Exp 1-revert was 24.3% -- about the same)

Lobe cells (cid 64, 2016): eps_log_corr = 0.3325 (was 0.39 in Exp 1). A
real but modest improvement -- roughly 15% smaller spurious correction, far
from the ideal 0.0.

**Assessment:** lambda=0.1 is too weak to meaningfully fix the lobe-cell
failure mode; it shrinks Delta_pred everywhere a little, which nudges the
worst offenders slightly without specifically teaching "recognize
near-pure-C000 input -> output exactly 0". Overall eps_log improvement is
essentially unchanged (25.8% vs 24.3%), i.e. this lambda is small enough
that it did not meaningfully trade away accuracy on the hard cells either --
suggests there's room to push lambda higher before the fit term starts
losing on the hard cells. Trying lambda=1.0 next to see if the lobe-cell
fix becomes more pronounced, and whether/when the hard-cell improvement
starts degrading as the tradeoff sharpens.

---

## Exp 3b — same, lambda=1.0 (10x stronger)

Held-out eval:
  mean base      = 1.5123
  mean corrected = 1.2177
  **mean improvement = 19.5%**  (down from Exp 3's 25.8%, Exp 1's 24.3%)

Lobe cells (cid 64, 2016): eps_log_corr = 0.1920 (down from Exp 3's 0.33,
Exp 1's 0.39). Monotonic and substantial improvement on the specific
failure mode this regularizer targets.

**Clear trade-off, as expected from a single global lambda:**
| lambda | lobe eps_log_corr | mean improvement (25 cells) |
|--------|--------------------|------------------------------|
| 0 (Exp 1) | 0.39 | 24.3% |
| 0.1 (Exp 3) | 0.33 | 25.8% |
| 1.0 (Exp 3b) | 0.19 | 19.5% |

A single global lambda can't win both: pushing it up to fix 2/25 "trivial"
cells steadily costs the other 23 "hard" cells fit accuracy, because the
same scalar penalty applies uniformly regardless of whether a given cell
actually needs a large correction. Only 2/25 held-out cells are trivial, so
their improvement is outweighed in the mean by the degradation elsewhere.

**Idea for Exp 4 (not yet run):** make lambda PER-CELL, scaled by how much
correction that cell's own ground truth actually needs during training
(e.g. lambda_cell = LAMBDA_MAX / (1 + mean(Delta_true_cell^2)/ref_var) --
large lambda for cells where Delta_true is small everywhere, small lambda
for cells that genuinely need big corrections). This only requires ground
truth at TRAINING time (same as the existing supervised targets) -- it does
NOT need to be computed at inference time, since it only shapes the loss
during training, not the model's forward pass. This should let the
regularizer target the *specific* failure mode (trivial cells getting
spurious corrections) without dragging down hard-cell accuracy the way a
single global lambda does.

---

## Exp 4 — per-cell adaptive lambda: lambda_cell = LAMBDA_MAX/(1+mean_sq_cell/REF_VAR) — SUCCESS

First computed the distribution of mean(Delta_true^2) inside sparse_mask
across all 128 cells (S_low=2 base) to pick REF_VAR sensibly:
  min=0, p5=0, p10=0.33, p25=0.51, p50=3.22, p75=6.51, p90=9.77, p95=11.56,
  max=18.21
Several cells (48,16,32,64,2032,...) have mean_sq EXACTLY 0 (perfect
Maxwellians). Clear separation between the low-percentile "trivial" cells
and the median/upper range "hard" cells -- picked REF_VAR=0.5 (~p25) and
LAMBDA_MAX=5.0 (can afford to go higher than Exp 3b's global 1.0 since it
now only fires hard on the cells that actually need it).

Loss: reg = mean(lambda_cell_i * Delta_pred_i^2), lambda_cell computed ONCE
per cell from ground truth (training-time only, never used at inference).
Resulting per-sample lambda range: train [0.134, 5.000], val [0.140, 5.000]
-- confirms trivial cells get ~full LAMBDA_MAX, hardest cells get ~27x less.

Held-out eval:
  mean base      = 1.5123
  mean corrected = 1.1355
  **mean improvement = 24.9%**

| lambda scheme | lobe eps_log_corr | mean improvement (25 cells) |
|----------------|--------------------|-------------------------------|
| none (Exp 1) | 0.39 | 24.3% |
| global 0.1 (Exp 3) | 0.33 | 25.8% |
| global 1.0 (Exp 3b) | 0.19 | 19.5% |
| **per-cell adaptive (Exp 4)** | **0.19** | **24.9%** |

**This resolves the trade-off**: lobe-cell error matches Exp 3b's strong
global regularization (0.19, both cid 64 and 2016), while overall
improvement stays at the Exp 1/3 level (24.9%, not the 19.5% cost Exp 3b
paid). Per-cell breakdown confirms hard cells (cid 1008, 704, 1088, etc.)
keep large improvements (30-53%) essentially undiminished from Exp 1.

**Current best model.** corrector_weights.pt on disk is this Exp 4 model.
Next directions to consider: tune REF_VAR/LAMBDA_MAX further (this was a
first reasonable guess, not swept), try more epochs (still not fully
converged), or move on to a different axis of improvement (e.g. S_low
choice, network capacity, or the log-space voxel-weighting idea shelved
after Exp 2's f-space failure).

---

## Exp 5 — same as Exp 4, just EPOCHS 30 -> 150, with best-val checkpointing

Exp 4 stopped at epoch 29 still clearly mid-descent (val_fit was still
dropping every epoch, no plateau). Bumped EPOCHS to 150 and added
best-val-fit checkpoint tracking (train_corrector.py now saves the model
state from whichever epoch had the lowest val_fit, not just the final
epoch -- guards against overfitting in longer runs).

Convergence: val_fit fell essentially monotonically the entire run, with
only 3 minor non-improving epochs (90, 120, 140) out of 150 -- no
overfitting onset, train and val tracked down together throughout.
  epoch  29 (Exp 4 stopping point): val_fit=1.698
  epoch  50: val_fit=1.456
  epoch 100: val_fit=1.182
  epoch 148 (best): val_fit=1.082
Still had NOT clearly plateaued by epoch 150 -- more epochs would likely
help further (next thing to try if pursuing this axis further).

Held-out eval:
  mean base      = 1.5123
  mean corrected = 0.9149
  **mean improvement = 39.5%**  (up from Exp 4's 24.9%)

Lobe cells (cid 64, 2016): eps_log_corr = 0.1382 (down from Exp 4's 0.19).

**Every single held-out cell now shows positive improvement** -- the
run of negative-improvement cells seen in every prior experiment (Exp
1/3/3b/4 all had several cells go backwards, -2% to -40%) is gone. Best
result so far by a wide margin, and purely from training longer -- Exp 4
was simply undertrained at 30 epochs, not fundamentally limited.

**Current best model** (corrector_weights.pt). Given the fit curve hadn't
plateaued, extending training further (300+ epochs, or a learning-rate
schedule/decay) is the obvious next lever before trying anything else.

---

## Exp 6 — same as Exp 5, EPOCHS 150 -> 400, added CosineAnnealingLR (LR -> 1% of initial)

train_corrector.py now uses torch.optim.lr_scheduler.CosineAnnealingLR
(T_max=EPOCHS, eta_min=LR*0.01) alongside the existing best-val-fit
checkpointing.

Convergence: val_fit kept improving smoothly through ~epoch 320, then
visibly flattened as the LR decayed toward its floor -- last 80 epochs
(320->399) only moved val_fit from 0.983 to 0.978, essentially plateaued
(a few epochs even ticked up slightly, e.g. 340, 360, 380, consistent with
being near a noise floor rather than still descending).
  epoch 148 (Exp 5 best): val_fit=1.082
  epoch 228: val_fit=1.020
  epoch 320: val_fit=0.983
  epoch 357 (best): val_fit=0.979
  epoch 399 (final): val_fit=0.983

Held-out eval:
  mean base      = 1.5123
  mean corrected = 0.8789
  **mean improvement = 41.9%**  (up from Exp 5's 39.5%, smaller gain than
  the 30->150 epoch jump -- consistent with approaching a plateau)

Lobe cells (cid 64, 2016): eps_log_corr = 0.1169 (down from Exp 5's 0.138).

**Assessment:** this looks close to the ceiling for the current setup
(S_low=2, hidden=64, N_PER_CELL=3000, this LAMBDA_MAX/REF_VAR, this
13-dim input). Further epochs alone are unlikely to move the needle much
more -- the cosine-decayed LR has essentially bottomed out and val_fit has
flattened. To improve further, the next levers are architectural/data
ones rather than "train longer": more hidden units, more training cells
(only 128 available in this one bulk file -- could pull from other bulk
files or other timesteps), a higher S_low base, or revisiting the
log-space voxel-weighting idea (shelved after Exp 2's f-space failure).

---

## Exp 7 — pivot: base = f_rec_full (max_order=22) instead of f_rec_low — WORSE, but explains why

Motivation (user's idea): the full tetrahedral Hermite decomposition IS the
compression (60^3 velocity cells -> ~2300 coefficients, mostly implicitly
zero at high order). The corrector should be a single shared model doing
artifact-removal on top of that already-compressed representation, using
only the (already-stored) low-order coefficient subset C_low as compact
conditioning -- not a separate "reconstruct from 10 numbers" scheme.

Changes: Delta_true = log(cube) - log(f_rec_full), f_final =
f_rec_full*exp(Delta_pred). C_low (S_low=2, 10 coeffs) is now literally a
SUBSET of the same coefficient dict used to build f_rec_full, not a
separately-fit basis. Everything else (per-cell adaptive lambda, sampling,
architecture) unchanged.

**Performance fix needed first:** naive full-order (max_order=22)
adaptive_transform per cell was taking ~110s/cell (would be ~4h for 128
cells) because the existing eps_log tracking (`_level_field` called every
level) does one O(vlen^3) full-grid field construction PER COEFFICIENT --
2300 of them at order 22, dominant cost, useless here since we only need
the final coefficients/f_rec_full. Added `track_log_eps=False` option to
`adaptive_transform` (adaptive_hermite.py) to skip this bookkeeping
entirely when not needed -- cut per-cell cost to ~3.8s (~40x), making
dataset-building at FULL_ORDER=22 tractable (~8min for 128 cells).

REF_VAR recalibrated: mean(Delta_true_cell^2) against f_rec_full is roughly
half the scale seen against f_rec_low (median ~1.5-1.7 vs 3.2), so set
REF_VAR=0.3 (down from Exp4-6's 0.5), kept LAMBDA_MAX=5.0. Resulting
lambda_cell range: train [0.122, 5.0], similar shape to before.

Training: val_fit barely moved -- 3.65 -> 3.18 (~13% relative reduction),
nowhere near Exp 6's 3.60 -> 0.98 (~73% reduction) for the same 400
epochs/architecture. Plateaued early (epoch ~150) and stayed flat.

Held-out eval:
  mean base      = 1.6955  (NOTE: not directly comparable to Exp1-6's 1.5123
                    -- that was against f_rec_low, this is against f_rec_full,
                    a different quantity; f_rec_full's eps_log is naturally
                    somewhat different in aggregate)
  mean corrected = 1.5913
  **mean improvement = 6.1%**  (vs Exp 6's 41.9% against f_rec_low)

Lobe cells (cid 64, 2016): eps_log_corr = 0.2336 (worse than Exp 6's 0.117 --
almost 2x the spurious correction on cells that need none).

**Diagnosis:** log(cube) - log(f_rec_full) is dominated by the residual
Gibbs/sign-flip structure of the HIGH-order fit (see earlier finding: ~20-30%
of in-support voxels still flip sign even at order 22). That structure's
exact location/shape depends on the high-order coefficients (l+m+n up to
22) that C_low (l+m+n<=2) simply does not encode -- from C_low's point of
view this residual looks close to unpredictable noise, not a learnable
function of the input. The network can only learn a small generic
average-case correction (~6-11% per cell), unlike Exp 1-6 where Delta_true
against f_rec_low reflects large-scale MISSING PHYSICAL STRUCTURE (higher
moments, asymmetry) that correlates well with C_low since C_low literally
*is* the crude version of that same structure.

**Conclusion:** conditioning only on C_low works well for correcting a
LOW-order base (where the correction is large-scale, structurally related
to C_low) but works poorly for correcting a HIGH-order base (where the
residual is high-frequency artifact noise uncorrelated with the low-order
shape). If the "artifact removal on top of full compression" framing is to
be pursued further, the corrector would need higher-frequency information
as input too (e.g. some subset of the higher-order coefficients, or a
summary statistic of them like neg_frac or local oscillation energy) --
plain C_low is not enough signal for this specific residual.

**IMPORTANT -- current state on disk:** corrector_weights.pt right now
holds this Exp 7 (worse) model, since train_corrector.py just overwrote it.
Exp 6's checkpoint was NOT saved separately and is gone. If Exp 6's
f_rec_low-based approach is wanted again, S_LOW/FULL_ORDER/REF_VAR need to
be reverted in train_corrector.py (base back to f_rec_low, REF_VAR back to
0.5) and retrained from scratch.

---

## Exp 8 — add axis_spectra to the input (base still f_rec_full) -- mixed result, informative failure

User's idea to enrich the input without jumping to a CNN: for each axis,
compute a marginal POWER spectrum directly in Hermite-coefficient space
(NOT physical space) -- spec_x[n] = sum_{l,m: l+m+n<=FULL_ORDER} C[l,m,n]^2
(similarly spec_y[m], spec_z[l]). This is a pure reduction over the
already-computed coefficient dict (no extra reconstruction), giving
3*(FULL_ORDER+1)=69 numbers. Verified sum(spec_x) == sum(C^2) exactly
(sanity check passed). Values span ~20+ orders of magnitude (lobe cells:
higher orders ~1e-20 to 1e-27; hard cells: real power out to order 22), so
normalized per-cell by total power and log10-compressed (floor 1e-20)
before use -- added to corrector_model.axis_spectra().

New input: [C_low (10), log_spec_flat (69), x_hat,y_hat,z_hat (3)] = 82-dim
(up from Exp 7's 13-dim). n_features now derived from X_train.shape[1]-3
directly rather than hardcoded, so it can't drift out of sync.

Training: val_fit converged to 3.334 (best, epoch 343) -- WORSE than Exp 7's
3.181 for the same architecture/epochs/lambda schedule. More input
information made the fit-only loss slightly worse, not better.

Held-out eval:
  mean base      = 1.6955  (same base as Exp 7, f_rec_full)
  mean corrected = 1.6240
  **mean improvement = 4.2%**  (vs Exp 7's 6.1% -- slightly worse overall)

BUT: lobe cells (cid 64, 2016) dramatically improved: eps_log_corr = 0.0210
(vs Exp 7's 0.2336, and even better than Exp 6's 0.117 against a DIFFERENT
base). Near-ideal -- the network now clearly recognizes "this C_low+spectrum
pattern means a pure Maxwellian, apply ~0 correction" and does so almost
exactly. Every hard cell's improvement individually shrank a bit compared
to Exp 7 (e.g. cid 1008: 6.8%->5.5%, cid 704: 3.9%->5.5% mixed), netting
out slightly worse overall.

**Diagnosis:** the axis-marginal power spectrum answers "how much
oscillatory energy exists at order n along x" but throws away SIGN and
CROSS-TERM (joint l,m,n) information by squaring and summing. Two very
different 3-D coefficient patterns can have identical axis-marginal power
spectra while producing completely different local sign-flip locations in
the reconstructed field. So this feature is a good coarse "is this cell
trivial or not" signal (explaining the striking lobe-cell fix) but
essentially uninformative for the POINTWISE (per-voxel) question of
"where exactly does the base fit go wrong here" -- which is what limits
hard-cell improvement. This matches the CNN discussion: the missing
information is inherently spatial/local, not summarizable by a handful of
per-axis scalars without losing phase.

**Takeaway:** C_low+axis_spectra as input is currently NOT better than
plain C_low (Exp 7) for the main goal (fixing hard cells), and both remain
far below Exp 6's f_rec_low-based 41.9%. The lobe-cell result is a genuine
and useful side-finding though -- if the f_rec_full framing is pursued
further, this coarse "triviality" signal could be worth keeping (e.g. as
an auxiliary gate/feature) even if paired with a richer local-context
input (CNN or a similar spatial mechanism) for the hard-cell correction
itself.

corrector_weights.pt on disk is now this Exp 8 model.

---

## RESET -- corrector code wiped, starting the design over

train_corrector.py, eval_corrector.py, corrector_model.py, and
corrector_weights.pt deleted (along with the correction_*.png plots).
This log is kept as the record of what was tried and learned across
Exp 1-8; the Hermite pipeline itself (adaptive_hermite.py, vdf_tools.py --
velocity-grid fix, dv^3 normalization, track_log_eps optimization) is
UNCHANGED and still the verified-correct foundation everything below
should build on.

**Carried-forward lessons, so the next attempt doesn't re-litigate them:**
- Multiplicative log-space correction (f_final = f_base*exp(Delta_pred),
  0 outside sparse_mask) is the right structural form -- guarantees
  positivity regardless of what the network predicts, and the sparse_mask
  zeroing is exact/free (structural, from the sparse VDF mesh).
- Voxel-level weighting/features must live in LOG space, not f-space
  (Exp 2: f-space |cube-f_rec| weighting is peak-dominated, actively hurts
  eps_log -- the same reason eps_rel doesn't reflect tail accuracy).
- A single global output-regularization lambda trades hard-cell accuracy
  for trivial-cell cleanliness; a PER-CELL adaptive lambda (scaled by that
  cell's own ground-truth correction magnitude, training-time only) avoids
  the trade-off (Exp 4).
- Training needs way more epochs than seemed obvious -- Exp 4->5 (30->150
  epochs) alone nearly doubled the improvement (24.9%->39.5%); watch
  val_fit for a genuine plateau, don't stop early.
- Correcting a LOW-order base (f_rec_low, S_low~2) worked well (Exp 6:
  41.9% mean improvement) because the needed correction is large-scale
  structure that correlates with C_low. Correcting a HIGH-order base
  (f_rec_full, order~22) worked poorly (Exp 7-8: 4-6%) because the residual
  there is high-frequency Gibbs/sign-flip noise that depends on joint
  (l,m,n) phase information C_low (or even the axis-marginal power
  spectrum) doesn't carry -- that residual is inherently local/spatial, not
  summarizable by a handful of global scalars.
- Axis-marginal power spectra (sum_{l,m} C[l,m,n]^2 per axis) are a cheap,
  informative "how non-Maxwellian is this cell overall" signal (fixed the
  lobe-cell false-positive-correction problem almost perfectly, Exp 8) but
  carry ~no information about WHERE a specific voxel's correction should
  go, since squaring destroys sign/cross-term structure.

---

## Exp 9 -- patch input (concatenated 3x3x3 local neighborhood), general
## artifact-removal framing, no CNN

New design after the reset. Problem reformulated generally ("remove
reconstruction artifacts, wherever they occur") rather than narrowly
("fix only near the support edge") -- reasoning: a local patch input lets
the network discover where corrections are needed on its own, so we don't
need to hand-specify/restrict a region via e.g. a distance-to-boundary
transform. Base is still f_rec_full (order=22, the "real" compressed
representation); C_low (S_low<=2, 10 coeffs) kept as a global condition.

Key architectural change from Exp 7/8: input is
  [3x3x3 patch of log(f_rec_full) around the query voxel (27 values,
   padded with log(sp_th) at cube boundaries), C_low (10), x_hat,y_hat,z_hat]
  = 40 numbers total (37 features + 3 position). Still a plain MLP
  (Linear+ReLU only, no actual convolutions) -- the patch is just
  concatenated, keeping a from-scratch reimplementation trivial. New
  helpers in corrector_model.py: pad_field(), extract_patches() (vectorized
  fancy-indexing over 27 fixed offsets).

**New process rule going forward**: always run a short smoke test (~20
epochs) before committing to a full long run, to catch bugs/sanity-check
the loss trend cheaply. EPOCHS=20 for the first Exp 9 run below (dataset
build dominates wall time regardless of epoch count, ~8min, so a smoke
test costs almost nothing extra).

Smoke test (20 epochs, same LAMBDA_MAX=5.0/REF_VAR=0.3/architecture as
Exp 7-8 otherwise): val_fit fell 3.07 -> 1.16, EVERY epoch was a new best
(no plateau reached) -- already below Exp 7's 400-epoch final (3.18) and
Exp 8's (3.33).

Held-out eval (20 epochs only!):
  mean base      = 1.6955  (same base as Exp 7/8, f_rec_full)
  mean corrected = 0.9214
  **mean improvement = 45.7%**

This already beats Exp 6's 41.9% (which corrected the EASIER f_rec_low
base) after just 20 epochs against the HARDER f_rec_full target. Every
hard cell individually shows large, consistent gains (37-54%, vs Exp 7/8's
~4-10%) -- confirms the local-patch hypothesis: the Gibbs/sign-flip
residual IS locally predictable, C_low/axis-spectra just couldn't see it.

Lobe cells (cid 64, 2016): eps_log_corr = 0.1304 -- still worse than base
(0.0), similar to Exp 7. Expected at only 20 epochs: the per-cell lambda
regularization needs more training to pull trivial-cell predictions to
~0, same pattern seen in Exp 4->5->6 (lobe-cell fix kept improving over
hundreds of epochs). Not yet a concern -- next step is the full run.

**Status: full 400-epoch run queued next**, expecting both the hard-cell
gains to hold/improve further and the lobe-cell regression to resolve
(per the Exp 4-6 precedent of long per-cell-lambda training).

**Full run (400 epochs) -- big win.** val_fit: 3.49 -> 0.615 (best, epoch
332) -- clean descent, no overfitting (train/val tracked together the
whole way), far below every prior experiment's floor (Exp 6: 0.98,
Exp 7: 3.18, Exp 8: 3.33).

Held-out eval:
  mean base      = 1.6955  (f_rec_full, same as Exp 7/8)
  mean corrected = 0.7340
  **mean improvement = 56.7%**  -- new best by a wide margin, beating even
  Exp 6's 41.9% (which had the EASIER f_rec_low target)

Every hard cell: 50-67% improvement, tight and consistent (e.g. cid 1008:
67.3%, cid 1088: 67.4%, cid 1216: 66.3%, cid 704: 56.4%) -- much more
uniform than Exp 6's spread (5-60%). Confirms the local-patch input made
the Gibbs-residual-fixing problem genuinely learnable, as hypothesized.

Remaining issue: lobe cells (cid 64, 2016) still eps_log_corr=0.1266 (base
0.0) -- barely moved from the smoke test's 0.1304 despite 400 epochs vs 20.
Unlike Exp 4->6 (where more per-cell-lambda training steadily fixed the
equivalent problem), this isn't resolving with more epochs alone here --
worth a dedicated look (retune REF_VAR/LAMBDA_MAX for the new loss scale,
or check whether the patch input itself is somehow harder to zero out for
these cells). Minor in aggregate (2/25 cells, ~0.01 impact on the mean) but
worth fixing for physical correctness on Maxwellian-like cells.

**Current best model overall, by far.** corrector_weights.pt on disk is
this Exp 9 model.

---

## Post-Exp-9 lambda recalibration -- ruled out regularization strength

Diagnosed the lobe-cell failure directly: for cid 64/2016, the WORST
individual delta_pred (~0.65) occurs at voxels where log_base AND log_true
are BOTH already at the sp_th floor (~-32.2, i.e. delta_true~0 -- nothing
to fix). For hard cell 704, the worst voxels ALSO sit at log_base~floor,
but there log_true is far above it (~-27, needs a real +5.3 correction).
Same local "patch looks flat at the floor" signature means opposite things
depending on the cell -- the network isn't using C_low enough to arbitrate.

Tried: lambda_cell = LAMBDA_MAX/(1+(mean_sq_cell/REF_VAR)^LAMBDA_POWER)
(power-law, LAMBDA_POWER=2) with LAMBDA_MAX raised 5.0->20.0 -- this gives
lobe cells ~4x the regularization pressure while giving hard cells even
LESS interference than before (0.078 vs 0.294 for cid 704 at mean_sq=4.8).
Result: eps_log_corr for lobe cells UNCHANGED (0.1308 vs Exp 9's
0.1266/0.1304) despite the much stronger push. Also checked whether the
problematic near-floor voxels were simply under-sampled: they're 12.36% of
the mask (expected ~371 of the 3000 per-cell training samples) -- not rare
at all. Both hypotheses (regularization too weak, insufficient sampling)
ruled out -- this is a genuine architectural/information problem, not a
hyperparameter one.

---

## Exp 10 -- add axis_spectra alongside the patch -- SUCCESS, fixes the lobe-cell problem

Combined Exp 8's axis_spectra (global "how non-Maxwellian" signal, nearly
perfect lobe-cell recognition on its own but no spatial info) with Exp 9's
local patch (excellent spatial correction but can't arbitrate the
floor-ambiguity alone). Rationale: axis_spectra should let the network use
C_low-like global context to override the patch's ambiguous floor signal
specifically where needed.

New input: [patch (27), C_low (10), axis_spectra log10-normalized (69),
x_hat,y_hat,z_hat (3)] = 109 numbers (up from Exp 9's 40). Kept the
LAMBDA_MAX=20/LAMBDA_POWER=2/REF_VAR=0.3 regularization from the
recalibration attempt (harmless, just not sufficient alone).

Smoke test (30 epochs, per new process convention): val_fit=1.005 (best,
epoch 25) -- already better than Exp 9+recalibration's 30-epoch val_fit
(1.165) with the same lambda settings, patch alone.

Held-out eval (30 epochs only!):
  mean base      = 1.6955
  mean corrected = 0.8813
  **mean improvement = 48.0%**  (vs Exp 9's 45.7% at the same 20-30 epoch
  smoke-test stage -- comparable/better overall, AND:)

**Lobe cells (cid 64, 2016): eps_log_corr = 0.0136** -- ~10x better than
Exp 9's 0.1266-0.1308 (post-400-epochs) and unmoved-by-lambda-alone value.
Confirms the diagnosis: the network needed axis_spectra's explicit global
signal to resolve the floor-region ambiguity that C_low alone (even with
much stronger regularization) could not.

Every hard cell: consistent 45-54% improvement, similar spread to Exp 9.

**Status: full 400-epoch run queued next** -- expect both further hard-cell
gains (following Exp 9's own 30-epoch-to-400-epoch trajectory of
45.7%->56.7%) and the lobe-cell result to hold or improve further.

**Full run (400 epochs) -- new best by a wide margin, problem essentially
solved.** val_fit: 3.46 -> 0.465 (best, epoch 308) -- well below Exp 9's
0.615, clean descent, no overfitting.

Held-out eval:
  mean base      = 1.6955
  mean corrected = 0.6403
  **mean improvement = 62.2%**  (vs Exp 9's 56.7%)

**Lobe cells (cid 64, 2016): eps_log_corr = 0.0107** -- ~12x better than
Exp 9's 0.1266, and even slightly better than Exp 8's axis-spectra-only
0.021 (which had ~0% hard-cell improvement). Both halves of the problem
solved simultaneously by the same model.

Every hard cell substantially improved and much more uniform than Exp 9:
36-78% (several now >70%: cid 1088 77.7%, cid 1216 76.0%, cid 1120 75.4%,
cid 1008 73.7%). Only cid 640 lags at 36.9%, still a solid gain.

Summary across the whole reset (Exp 9-10):
| exp | input | mean improvement | lobe eps_log_corr |
|-----|-------|-------------------|---------------------|
| 7 | C_low only | 6.1% | 0.234 |
| 8 | C_low + axis_spectra | 4.2% | 0.021 |
| 9 | C_low + local patch | 56.7% | 0.127 |
| **10** | **C_low + patch + axis_spectra** | **62.2%** | **0.0107** |

The patch supplies spatial correction capability; axis_spectra supplies the
global "is this cell trivial" signal needed to arbitrate the patch's
otherwise-ambiguous sp_th-floor readings. Neither alone was sufficient;
combined, both problems are resolved together.

**Current best model, by a wide margin ON THE TRAINING SNAPSHOT.** See the
cross-timestep validation below -- this does NOT generalize.

---

## Cross-timestep validation (bulk.0000111.vlsv, never seen in training) -- FAILS

All prior held-out evaluation was within bulk.0000024.vlsv (same simulation
snapshot the model trained on, just different spatial cells). Tested the
Exp 10 model on ALL 128 cells of bulk.0000111.vlsv -- a much later, more
evolved timestep of the same reconnection_2d_beta025 run (clearly-developed
jets, per earlier session notes), never touched during training.

  mean eps_log_base = 2.1936  (vs 1.6955 for bulk.0000024's held-out cells
                                -- this timestep's cells are intrinsically
                                harder/more non-Maxwellian on average)
  mean eps_log_corr = 2.8074
  **mean improvement = -28.0%**  -- the corrector makes things WORSE on
  average on an unseen timestep.

Many individual cells regress catastrophically (-70% to -173%), while a
handful still improve (up to +52%). One thing DID generalize perfectly:
lobe-like cells (eps_log_base=0.0, e.g. cid 16/48/32/64/2032/2048/2000/2016)
get eps_log_corr=0.0107 EXACTLY -- identical to the training-snapshot lobe
result. The axis_spectra-driven "is this cell trivial" recognition
generalizes (makes sense -- "pure Maxwellian" is a simple, universal
pattern). The patch-driven "how to fix THIS specific non-Maxwellian
structure" does not -- it appears to have learned bulk.0000024's particular
Gibbs-ringing patterns rather than the general principle.

**Diagnosis:** the entire Exp 1-10 arc trained on ONE snapshot (128 cells,
one instant of one simulation). That's enough spatial diversity to split
into a meaningful train/val set (the held-out-cells eval was legitimate
for THAT snapshot), but apparently not enough to learn a genuinely
snapshot-independent correction -- the model overfit to bulk.0000024's
specific non-Maxwellian "signature" (whatever combination of jet
speeds/temperatures/asymmetries characterizes that instant) rather than
the general Gibbs-artifact-removal principle. This directly confirms the
user's concern about the architecture's weakest point (empirically-tuned,
single-snapshot-calibrated regularization) -- but the deeper issue is
single-snapshot TRAINING DATA, not just the regularization constants.

**Next step: multi-snapshot training.** Rebuild the dataset from SEVERAL
bulk files (different timesteps, ideally spanning early/mid/late
reconnection phases) instead of just bulk.0000024, so the model sees
diverse non-Maxwellian structure and (hopefully) learns the general
artifact pattern rather than one snapshot's specifics. decomposition_cache.py
already supports this per-file (one .npz per bulk file) with no changes
needed; train_corrector.py needs updating to pull cells from multiple
(BULKFILE, cell_id) pairs instead of one BULKFILE's full cell list.
