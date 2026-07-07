#!/usr/bin/env bash
# Diagnostic: base OpenVLA Task SR on its matching LIBERO suite.
# If SR ~85%, predict_action un-normalization fix is confirmed.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/rollout-baseline"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false

python experiments/rollout_baseline.py \
    --model            "$MODEL" \
    --suite            "$LIBERO_SUITE" \
    --unnorm-key       "$UNNORM_KEY" \
    --n-eps-per-task   "${N_EPS_PER_TASK:-5}" \
    --max-steps        "${MAX_STEPS:-300}" \
    --out              "$OUT_DIR" \
    --dtype            "${DTYPE:-bfloat16}" \
    --attn             "${ATTN:-eager}"

echo ""
echo "==== Done ===="
cat "$OUT_DIR/sr.json"
