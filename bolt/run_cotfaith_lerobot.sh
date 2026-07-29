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

# Datasets library only. Do NOT upgrade transformers — the setup already
# installed a torch 2.2.0 (cu118) compatible transformers. Newer transformers
# / datasets require torch>=2.4's torch.distributed.tensor.DTensor.
pip install "datasets>=2.19,<3.0" || true
# Verify torch, datasets, transformers all importable together.
python -c "import torch, datasets, transformers; print(f'[ds] torch={torch.__version__} datasets={datasets.__version__} transformers={transformers.__version__}')" || {
    echo "[ds] FATAL: import chain broken"; exit 2;
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
