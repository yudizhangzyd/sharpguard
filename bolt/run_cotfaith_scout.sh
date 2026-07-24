#!/usr/bin/env bash
# CoT-Faith Week 1 scout — verify 4 CoT-VLA checkpoints load and produce
# parseable (reasoning, action) output.

set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-scout"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Make sure the /tmp/openvla clone is present so Embodied-CoT/openvla-style
# model configs can resolve their remote-code paths (they reference
# openvla/openvla-7b-oxe... configuration classes).
if [ ! -d /tmp/openvla ]; then
    git clone --depth 1 https://github.com/openvla/openvla /tmp/openvla || true
fi
(cd /tmp/openvla && pip install -e . || true)

# Extra deps that some CoT-VLA checkpoints pull in.
pip install "draccus" "wandb" "diffusers" "einops" || true

python experiments/cotfaith_scout.py \
    --out    "$OUT_DIR" \
    --dtype  "${DTYPE:-bfloat16}" \
    --models "${SCOUT_MODELS:-all}" || true   # never let a broken model fail the whole task

echo ""
echo "==== Done ===="
ls -la "$OUT_DIR" || true
echo ""
echo "--- scout_report.json ---"
[ -f "$OUT_DIR/scout_report.json" ] && cat "$OUT_DIR/scout_report.json" || true
exit 0
