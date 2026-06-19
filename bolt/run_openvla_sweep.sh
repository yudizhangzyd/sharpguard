#!/usr/bin/env bash
# OpenVLA SharpGuard λ sweep on one GPU.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/sg-lambda-sweep"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0

# shellcheck disable=SC2206
LAM_ARR=($LAMBDAS)

python experiments/openvla_lambda_sweep.py \
    --model        "$MODEL" \
    --out          "$OUT_DIR" \
    --lambdas      "${LAM_ARR[@]}" \
    --n-train      "$N_TRAIN" \
    --n-eval       "$N_EVAL" \
    --poison-rate  "$POISON_RATE" \
    --lora-steps   "$LORA_STEPS" \
    --lora-r       "$LORA_R" \
    --lr           "$LR" \
    --batch-size   "$BATCH_SIZE" \
    --measure-batches "$MEASURE_BATCHES" \
    --epsilon      "$EPSILON" \
    --n-trials     "$N_TRIALS" \
    --rho          "$RHO" \
    --libero-sim-suite "$LIBERO_SIM_SUITE" \
    --libero-collect-eps "$LIBERO_COLLECT_EPS" \
    --libero-collect-steps "$LIBERO_COLLECT_STEPS" \
    --dtype        "$DTYPE" \
    --attn         "$ATTN" \
    --seed         "$SEED"

echo "Done."
ls -la "$OUT_DIR"
[ -f "$OUT_DIR/sweep.json" ] && cat "$OUT_DIR/sweep.json"
