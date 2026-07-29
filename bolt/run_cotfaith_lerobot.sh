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

# Datasets lib + lerobot native library (--no-deps preserves torch 2.2.0).
# Pin >=0.6 for LeRobot v2.1 dataset format support (roboturk uses v2.1).
pip install "datasets>=2.19,<3.0" || true
pip install "lerobot>=0.6" --no-deps 2>&1 | tail -3 || true
# lerobot install may still miss a few small deps; grab the essentials.
pip install "pyav" "torchcodec" 2>&1 | tail -3 || true
python -c "import torch, datasets, transformers; print(f'[ds] torch={torch.__version__} datasets={datasets.__version__} transformers={transformers.__version__}')" || {
    echo "[ds] FATAL: import chain broken"; exit 2;
}
# Try both v0.4.x (lerobot.datasets) and older (lerobot.common.datasets) paths.
python -c "
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('[ds] lerobot.datasets OK')
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; print('[ds] lerobot.common.datasets OK')
" || {
    echo "[ds] WARN: lerobot native missing; falling back to HF datasets" >&2
}

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
