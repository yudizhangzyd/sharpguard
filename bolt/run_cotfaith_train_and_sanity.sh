#!/usr/bin/env bash
# Combined train + sanity: avoids cross-job artifact fetch (bolt pods
# lack aws creds to pull from bolt S3 prefixes). Trains 15k steps of
# ECoT-LIBERO, saves merged_model to this task's artifact dir, then
# immediately runs sanity on it.

set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

# Repo root importable regardless of whether setup wrote it into the env file.
# See the note in bolt/setup-openvla.sh: this is what the five failed
# retraining replicates needed.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# ---- deps (same as run_cotfaith_train.sh) ----
pip install "dm-tree" "protobuf>=3.20,<5" "promise" "dill" "etils[epath]" \
            "toml" "termcolor" "tqdm" "click" || true
pip install "tensorflow-cpu==2.15.1" --no-deps \
    || pip install "tensorflow==2.15.1" --no-deps || true
pip install "absl-py" "astunparse" "flatbuffers" "gast" "google-pasta" \
            "grpcio" "h5py" "libclang" "ml-dtypes==0.2.0" "opt-einsum" \
            "packaging" "six" "wrapt" "termcolor" "typing-extensions" \
            "tensorboard==2.15.2" "keras==2.15.0" "tensorflow-estimator==2.15.0" \
    || true
pip install "tensorflow_datasets==4.9.3" "tensorflow_metadata==1.15.0" \
            --force-reinstall --no-deps || true

python -c "import tensorflow, tensorflow_datasets, tree; print('[verify] tf+tfds OK')" \
    || { echo '[FATAL] tfds import broken'; exit 3; }

# ---- Step 1: TRAIN ----
TRAIN_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-train"
mkdir -p "$TRAIN_OUT"

python experiments/cotfaith_train.py \
    --out               "$TRAIN_OUT" \
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
    --seed              "${SEED:-0}" \
    --reasoning-mode    "${REASONING_MODE:-full}" \
    --data-fraction     "${DATA_FRACTION:-1.0}" \
    --data-seed         "${DATA_SEED:-0}"

echo ""
echo "===== TRAIN done. merged_model:"
ls -la "$TRAIN_OUT/merged_model" | head -10
echo ""

# ---- Step 2: SANITY (in-process, on the just-saved merged_model) ----
SANITY_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-sanity"
mkdir -p "$SANITY_OUT"

python experiments/cotfaith_sanity.py \
    --ckpt-path "$TRAIN_OUT/merged_model" \
    --out       "$SANITY_OUT" \
    --dtype     "${DTYPE:-bfloat16}"

echo ""
echo "===== SANITY done. Report:"
[ -f "$SANITY_OUT/sanity_report.json" ] && cat "$SANITY_OUT/sanity_report.json" | head -80
echo ""

# ---- Step 3: r_vis(CoT) attention analysis (in-process, same GPU) ----
RVIS_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-rvis"
mkdir -p "$RVIS_OUT"

python experiments/cotfaith_rvis.py \
    --ckpt-path "$TRAIN_OUT/merged_model" \
    --out       "$RVIS_OUT" \
    --n-samples "${RVIS_N_SAMPLES:-20}" \
    --rvis-layers "${RVIS_LAYERS:-0,1,2,3}" \
    --dtype     "${DTYPE:-bfloat16}"

echo ""
echo "===== RVIS done. Report head:"
[ -f "$RVIS_OUT/rvis_cot_report.json" ] && head -c 1000 "$RVIS_OUT/rvis_cot_report.json"
echo ""

# ---- Step 4: causal CoT edit ----
EDIT_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-edit"
mkdir -p "$EDIT_OUT"

python experiments/cotfaith_edit.py \
    --ckpt-path "$TRAIN_OUT/merged_model" \
    --out       "$EDIT_OUT" \
    --n-samples "${EDIT_N_SAMPLES:-30}" \
    --threshold "${EDIT_THRESHOLD:-0.05}" \
    --dtype     "${DTYPE:-bfloat16}"

echo ""
echo "==== Done ===="
[ -f "$EDIT_OUT/cot_edit_report.json" ] && head -c 2500 "$EDIT_OUT/cot_edit_report.json"
exit 0
