#!/usr/bin/env bash
# Bolt run: OpenVLA Stage 3 SharpGuard fine-tune on 8 GPUs.
set -e -x

cd "$(dirname "$0")/.."

: "${CLEAN_MODEL:?CLEAN_MODEL must be set}"
: "${DATA_ROOT:?DATA_ROOT must be set}"

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/sharpguard-stage3"
mkdir -p "$OUT_DIR"

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "GPUs detected: $NUM_GPUS"

CMD=(
  accelerate launch
    --num_processes "$NUM_GPUS"
    --mixed_precision "${DTYPE/float/}"
    experiments/openvla_stage3.py
    --clean-model      "$CLEAN_MODEL"
    --data-root        "$DATA_ROOT"
    --suite            "$SUITE"
    --split            "$SPLIT"
    --epochs           "$EPOCHS"
    --lr               "$LR"
    --batch-size       "$BATCH_SIZE"
    --grad-accum       "$GRAD_ACCUM"
    --grad-clip        "$GRAD_CLIP"
    --poison-rate      "$POISON_RATE"
    --lam-sg           "$LAM_SG"
    --sg-epsilon       "$SG_EPSILON"
    --sg-anomaly-q     "$SG_ANOMALY_Q"
    --sg-loss-q        "$SG_LOSS_Q"
    --dtype            "$DTYPE"
    --attn             "$ATTN"
    --seed             "$SEED"
    --out              "$OUT_DIR"
)

if [ -n "$MAX_SAMPLES" ]; then CMD+=(--max-samples "$MAX_SAMPLES"); fi
if [ -n "$USE_LORA" ];     then CMD+=(--use-lora); fi
if [ -n "$FREEZE_VISION" ]; then CMD+=(--freeze-vision); fi
if [ -n "$SG_NO_LOSS_GATING" ]; then CMD+=(--sg-no-loss-gating); fi
if [ -n "$SKIP_EVAL" ];    then CMD+=(--skip-eval); fi

"${CMD[@]}"

echo "Done. Artifacts in $OUT_DIR"
ls -la "$OUT_DIR"
