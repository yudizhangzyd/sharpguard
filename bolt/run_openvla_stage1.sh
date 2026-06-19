#!/usr/bin/env bash
# Bolt run: OpenVLA Stage 1 sharpness measurement.
set -e -x

cd "$(dirname "$0")/.."

: "${CLEAN_MODEL:?CLEAN_MODEL must be set}"
: "${DATA_ROOT:?DATA_ROOT must be set}"

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/sharpguard"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/stage1.json"

# shellcheck disable=SC2206
EST_ARR=($ESTIMATORS)

CMD=(
  python experiments/openvla_stage1.py
  --clean-model       "$CLEAN_MODEL"
  --data-root         "$DATA_ROOT"
  --suite             "$SUITE"
  --split             "$SPLIT"
  --poison-rate       "$POISON_RATE"
  --max-samples       "$MAX_SAMPLES"
  --batch-size        "$BATCH_SIZE"
  --estimators        "${EST_ARR[@]}"
  --epsilon           "$EPSILON"
  --n-trials          "$N_TRIALS"
  --mode              "$MODE"
  --rho               "$RHO"
  --attn              "$ATTN"
  --dtype             "$DTYPE"
  --max-batches       "$MAX_BATCHES"
  --seed              "$SEED"
  --out               "$OUT"
)

if [ -n "$BACKDOORED_MODEL" ]; then
  CMD+=(--backdoored-model "$BACKDOORED_MODEL")
fi

"${CMD[@]}"

echo "Wrote $OUT"
ls -la "$OUT_DIR"
