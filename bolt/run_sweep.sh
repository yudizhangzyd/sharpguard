#!/usr/bin/env bash
# Parallel λ-sweep on 8 GPUs.
set -e -x

cd "$(dirname "$0")/.."

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/sharpguard-sweep"
mkdir -p "$OUT_DIR"

# Show what we have.
nvidia-smi -L || true
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "GPUs detected: $NUM_GPUS"

# shellcheck disable=SC2206
LAM_ARR=($LAMBDAS)

python experiments/run_sweep.py \
    --lambdas      "${LAM_ARR[@]}" \
    --out          "$OUT_DIR" \
    --n-train      "$N_TRAIN" \
    --epochs-clean "$EPOCHS_CLEAN" \
    --epochs-pois  "$EPOCHS_POIS" \
    --epochs-s3    "$EPOCHS_S3" \
    --poison-rate  "$POISON_RATE"

echo "Done."
ls -la "$OUT_DIR"
echo "---"
[ -f "$OUT_DIR/sweep_summary.json" ] && cat "$OUT_DIR/sweep_summary.json" | head -80
