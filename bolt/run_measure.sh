#!/usr/bin/env bash
#
# Bolt run: execute the Stage 1 sharpness measurement.
# Required env vars (set in boltconfig.yaml or via --env on submit):
#   CLEAN_MODEL       HF id or local path of the clean OpenVLA checkpoint
#   BACKDOORED_MODEL  HF id or local path of the BadVLA-poisoned checkpoint
#   DATA_PATH         torch.save'd list of dict batches with is_triggered
#
set -e -x

cd "$(dirname "$0")/.."

: "${CLEAN_MODEL:?CLEAN_MODEL must be set}"
: "${BACKDOORED_MODEL:?BACKDOORED_MODEL must be set}"
: "${DATA_PATH:?DATA_PATH must be set}"

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/sharpguard"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/stage1.json"

# shellcheck disable=SC2206
EST_ARR=($ESTIMATORS)

python scripts/measure_sharpness.py \
    --clean-model       "$CLEAN_MODEL" \
    --backdoored-model  "$BACKDOORED_MODEL" \
    --data              "$DATA_PATH" \
    --out               "$OUT" \
    --estimators        "${EST_ARR[@]}" \
    --epsilon           "$EPSILON" \
    --n-trials          "$N_TRIALS" \
    --mode              "$MODE" \
    --pgd-steps         "$PGD_STEPS" \
    --rho               "$RHO" \
    --attn              "$ATTN" \
    --dtype             "$DTYPE" \
    --max-batches       "$MAX_BATCHES" \
    --batch-size        "$BATCH_SIZE" \
    --seed              "$SEED"

echo "Wrote $OUT"
ls -la "$OUT_DIR"
