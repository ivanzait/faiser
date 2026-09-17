#!/bin/bash
# Build the Hermite-corrector training/validation dataset.
#
# Edit the config block below, then run:
#   bash hermite_ml/run_data_processing.sh
#
# Validation strategy: VAL_BULKFILES should be DIFFERENT timesteps than
# TRAIN_BULKFILES (cross-timestep generalization check -- see
# EXPERIMENT_LOG.md: cell-level splitting within one snapshot hid a real
# overfitting problem). VAL_FRACTION_WITHIN_TRAIN optionally also carves a
# same-snapshot dev set out of the training files, for early-stopping
# diagnostics only -- it is not a substitute for VAL_BULKFILES.

set -euo pipefail
cd "$(dirname "$0")/.."   # project root

# ===
# CONFIG -- edit these
# ===
BULKDIR="reconnection_2d_beta025"

# Timesteps for TRAINING -- spread across the reconnection evolution so the
# corrector sees diverse non-Maxwellian structure, not one snapshot's
# specific signature (see EXPERIMENT_LOG.md's cross-timestep-validation
# finding: single-snapshot training scored 62% held-out but -28% on an
# unseen timestep).
TRAIN_BULKFILES=(bulk.0000048.vlsv bulk.0000100.vlsv bulk.0000070.vlsv)

# Timestep(s) held out ENTIRELY for validation -- must not appear in
# TRAIN_BULKFILES.
VAL_BULKFILES=(bulk.0000114.vlsv)

VAL_FRACTION_WITHIN_TRAIN=0.0   # 0 = off; e.g. 0.15 to also get a same-snapshot dev.npz

S_LOW=2
FULL_ORDER=22
N_PER_CELL=3000
LAMBDA_MAX=20.0
LAMBDA_POWER=2.0
REF_VAR=0.3
SEED=42
N_WORKERS=8              # parallel decomposition workers (one process per cell);
                          # keep below total core count to leave headroom on a shared machine
OUTPUT_NAME="multi_snapshot_v1"

# ===
# RUN
# ===
PYTHON="${PYTHON:-python3}"
"$PYTHON" -u hermite_ml/data_processing.py \
    --bulkdir "$BULKDIR" \
    --train-bulkfiles "${TRAIN_BULKFILES[@]}" \
    --val-bulkfiles "${VAL_BULKFILES[@]}" \
    --val-fraction-within-train "$VAL_FRACTION_WITHIN_TRAIN" \
    --s-low "$S_LOW" \
    --full-order "$FULL_ORDER" \
    --n-per-cell "$N_PER_CELL" \
    --lambda-max "$LAMBDA_MAX" \
    --lambda-power "$LAMBDA_POWER" \
    --ref-var "$REF_VAR" \
    --seed "$SEED" \
    --n-workers "$N_WORKERS" \
    --output-name "$OUTPUT_NAME"
