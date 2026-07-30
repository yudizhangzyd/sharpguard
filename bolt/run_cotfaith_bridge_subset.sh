#!/usr/bin/env bash
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-bridge-subset"
mkdir -p "$OUT_DIR"

pip install "datasets>=2.19,<3.0" "av" "pyarrow>=15" 2>&1 | tail -3 || true

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python experiments/cotfaith_train_bridge.py \
    --base-model "${BASE_MODEL:-Embodied-CoT/ecot-openvla-7b-bridge}" \
    --reasoning-repo "${BRIDGE_REASONING:-Embodied-CoT/embodied_features_bridge}" \
    --dataset-repo "${BRIDGE_DATASET:-IPEC-COMMUNITY/bridge_orig_lerobot}" \
    --n-trajectories "${N_TRAJECTORIES:-4000}" \
    --lora-r "${LORA_R:-32}" \
    --lora-alpha "${LORA_ALPHA:-16}" \
    --lr "${LR:-2e-5}" \
    --steps "${STEPS:-15000}" \
    --batch-size "${BATCH_SIZE:-2}" \
    --dtype "${DTYPE:-bfloat16}" \
    --seed "${SEED:-0}" \
    --out "$OUT_DIR"

echo "==== Done ===="
