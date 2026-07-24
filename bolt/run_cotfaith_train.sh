#!/usr/bin/env bash
# LoRA fine-tune ECoT-bridge on Embodied-CoT LIBERO data.

set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-train"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# tfds needs dm-tree (missing from the base image — that's why the data-scout's
# tfds path failed). Install BEFORE tensorflow_datasets so tfds can find it.
pip install "dm-tree" || true
pip install "tensorflow_datasets==4.9.3" "tensorflow_metadata==1.15.0" \
            --force-reinstall --no-deps || true

python -c "import tensorflow_datasets, tree; print('[verify] tfds + dm-tree OK')" \
    || { echo '[FATAL] tfds import broken'; exit 3; }

python experiments/cotfaith_train.py \
    --out               "$OUT_DIR" \
    --base-model        "${BASE_MODEL:-Embodied-CoT/ecot-openvla-7b-bridge}" \
    --dataset-repo      "${DATASET_REPO:-Embodied-CoT/embodied_features_and_demos_libero}" \
    --tfds-subdir       "${TFDS_SUBDIR:-libero_lm_90/1.0.0}" \
    --reasoning-json    "${REASONING_JSON:-libero_reasonings.json}" \
    --lora-r            "${LORA_R:-32}" \
    --lora-alpha        "${LORA_ALPHA:-16}" \
    --lr                "${LR:-2e-5}" \
    --steps             "${STEPS:-15000}" \
    --batch-size        "${BATCH_SIZE:-2}" \
    --max-steps-per-ep  "${MAX_STEPS_PER_EP:-0}" \
    --dtype             "${DTYPE:-bfloat16}" \
    --seed              "${SEED:-0}"

echo ""
echo "==== Done ===="
ls -la "$OUT_DIR" || true
[ -f "$OUT_DIR/train_losses.json" ] && head -c 800 "$OUT_DIR/train_losses.json" || true
exit 0
