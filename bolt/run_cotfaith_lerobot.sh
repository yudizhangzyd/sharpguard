#!/usr/bin/env bash
# Multi-dataset lerobot evaluation for CoT-Faith.
# Runs cotfaith_bridge.py against any lerobot dataset repo.
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-lerobot"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Datasets modules
pip install --upgrade "datasets>=3.0" "transformers>=4.45" || true

python experiments/cotfaith_bridge.py \
    --ckpt-path "${CKPT_PATH:-Embodied-CoT/ecot-libero-90-bridgev2}" \
    --dataset-repo "${DATASET_REPO:-IPEC-COMMUNITY/bridge_orig_lerobot}" \
    --out         "$OUT_DIR" \
    --n-samples   "${N_SAMPLES:-100}" \
    --seed        "${SEED:-0}" \
    --dtype       "${DTYPE:-bfloat16}"

echo "==== Done ===="
[ -f "$OUT_DIR/bridge_report.json" ] && head -c 3000 "$OUT_DIR/bridge_report.json"
exit 0
