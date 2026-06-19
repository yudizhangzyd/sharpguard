#!/usr/bin/env bash
# Stage 1 sharpness measurement on the OFFICIAL pre-trained BadVLA ckpt.
# Runs after setup-badvla-pretrained.sh has fetched czxlovesu03/BadVLA.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/stage1-official-badvla"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0

: "${BADVLA_CKPT_DIR:?BADVLA_CKPT_DIR not set; setup must have failed}"

# shellcheck disable=SC2206
EST_ARR=($ESTIMATORS)

CMD=(
    python experiments/openvla_stage1_official_badvla.py
    --clean-model    "$CLEAN_MODEL"
    --badvla-ckpt-dir "$BADVLA_CKPT_DIR"
    --out             "$OUT_DIR"
    --n-eval          "$N_EVAL"
    --batch-size      "$BATCH_SIZE"
    --measure-batches "$MEASURE_BATCHES"
    --estimators      "${EST_ARR[@]}"
    --epsilon         "$EPSILON"
    --n-trials        "$N_TRIALS"
    --mode            "$MODE"
    --rho             "$RHO"
    --libero-sim-suite "$LIBERO_SIM_SUITE"
    --libero-collect-eps "$LIBERO_COLLECT_EPS"
    --libero-collect-steps "$LIBERO_COLLECT_STEPS"
    --dtype           "$DTYPE"
    --attn            "$ATTN"
    --seed            "$SEED"
)
[ -n "$VARIANT" ] && CMD+=(--variant "$VARIANT")
[ -n "$USE_LIBERO_COLLECT" ] && CMD+=(--use-libero-collect)

"${CMD[@]}"

echo "Done."
ls -la "$OUT_DIR"
[ -f "$OUT_DIR/stage1_official_badvla.json" ] && head -200 "$OUT_DIR/stage1_official_badvla.json"
