#!/usr/bin/env bash
# Real OpenVLA-7B end-to-end SharpGuard pipeline (Stages 0 → 3 + adaptive).
set -e -x

cd "$(dirname "$0")/.."

# Pull env vars written by setup-openvla.sh (LIBERO_DATA_DIR, MUJOCO_GL, ...).
if [ -f /tmp/sharpguard.env ]; then
    set -a
    # shellcheck disable=SC1091
    . /tmp/sharpguard.env
    set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/sharpguard-openvla-full"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true
nvidia-smi --query-gpu=memory.total --format=csv,noheader || true

# Single-GPU is fine for OpenVLA-7B + LoRA on an A100 80GB.
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
    --libero-sim-eps  "$LIBERO_SIM_EPS"
    --dtype         "$DTYPE"
    --attn          "$ATTN"
    --seed          "$SEED"
)

[ -n "$USE_BADVLA"          ] && CMD+=(--use-badvla)
[ -n "$LIBERO_SIM_EVAL"     ] && CMD+=(--libero-sim-eval)
[ -n "$USE_LIBERO_COLLECT"  ] && CMD+=(--use-libero-collect)
[ -n "$LIBERO_COLLECT_EPS"  ] && CMD+=(--libero-collect-eps "$LIBERO_COLLECT_EPS")
[ -n "$LIBERO_COLLECT_STEPS" ] && CMD+=(--libero-collect-steps "$LIBERO_COLLECT_STEPS")

"${CMD[@]}"

echo "Done. Artifacts in $OUT_DIR"
ls -la "$OUT_DIR"
echo "---"
[ -f "$OUT_DIR/results.json" ] && head -200 "$OUT_DIR/results.json"
