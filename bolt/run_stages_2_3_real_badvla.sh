#!/usr/bin/env bash
# Stages 2 + 3 + baselines against the OFFICIAL pre-trained BadVLA ckpt.
#
# Differs from previous runs: we DON'T train our own poisoned LoRA. We
# load czxlovesu03/BadVLA (the paper's released ckpt) as pois_model and
# run all Stage 2 / Stage 3 / baseline evaluations against it directly.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/stages-2-3-real-badvla"
mkdir -p "$OUT_DIR"

: "${BADVLA_CKPT_DIR:?BADVLA_CKPT_DIR not set; setup must have failed}"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0

CMD=(
    python experiments/openvla_real.py
    --model         "$MODEL"
    --out           "$OUT_DIR"
    --n-train       "$N_TRAIN"
    --n-eval        "$N_EVAL"
    --poison-rate   "$POISON_RATE"
    --lora-steps    "$LORA_STEPS"
    --lora-r        "$LORA_R"
    --lr            "$LR"
    --batch-size    "$BATCH_SIZE"
    --measure-batches "$MEASURE_BATCHES"
    --epsilon       "$EPSILON"
    --n-trials      "$N_TRIALS"
    --rho           "$RHO"
    --lam-sg        "$LAM_SG"
    --lam-sg-b      "$LAM_SG_B"
    --lam-adapt     "$LAM_ADAPT"
    --detector-drop-quantile "$DETECTOR_DROP_Q"
    --libero-max-eps "$LIBERO_MAX_EPS"
    --libero-sim-suite "$LIBERO_SIM_SUITE"
    --libero-sim-eps "$LIBERO_SIM_EPS"
    --dtype         "$DTYPE"
    --attn          "$ATTN"
    --seed          "$SEED"
    --pretrained-poisoned-ckpt-dir "$BADVLA_CKPT_DIR"
    --pretrained-variant "$PRETRAINED_VARIANT"
)
[ -n "$LIBERO_SIM_EVAL"    ] && CMD+=(--libero-sim-eval)
[ -n "$USE_LIBERO_COLLECT" ] && CMD+=(--use-libero-collect)
[ -n "$LIBERO_COLLECT_EPS" ] && CMD+=(--libero-collect-eps "$LIBERO_COLLECT_EPS")
[ -n "$LIBERO_COLLECT_STEPS" ] && CMD+=(--libero-collect-steps "$LIBERO_COLLECT_STEPS")

"${CMD[@]}"

echo "Done."
ls -la "$OUT_DIR"
[ -f "$OUT_DIR/results.json" ] && head -200 "$OUT_DIR/results.json"
