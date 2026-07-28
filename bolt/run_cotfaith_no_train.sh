#!/usr/bin/env bash
# Run cotfaith rvis + edit against an EXISTING checkpoint (no retrain).
# Used for ECoT-bridge (public HF model) and any pre-trained CoT-VLA.

set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# tf/tfds deps.
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

RVIS_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-rvis"
EDIT_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-edit"
mkdir -p "$RVIS_OUT" "$EDIT_OUT"

CKPT="${CKPT_HF_ID:-Embodied-CoT/ecot-openvla-7b-bridge}"

echo "===== r_vis(CoT) on $CKPT ====="
python experiments/cotfaith_rvis.py \
    --ckpt-path "$CKPT" \
    --out       "$RVIS_OUT" \
    --n-samples "${N_SAMPLES:-100}" \
    --seed      "${SEED:-0}" \
    --rvis-layers "${RVIS_LAYERS:-0,1,2,3}" \
    --dtype     "${DTYPE:-bfloat16}"

echo ""
echo "===== causal edit on $CKPT ====="
python experiments/cotfaith_edit.py \
    --ckpt-path "$CKPT" \
    --out       "$EDIT_OUT" \
    --n-samples "${EDIT_N_SAMPLES:-100}" \
    --seed      "${SEED:-0}" \
    --families  "${EDIT_FAMILIES:-all}" \
    --threshold "${EDIT_THRESHOLD:-0.05}" \
    --dtype     "${DTYPE:-bfloat16}"

echo ""
echo "==== Done ===="
[ -f "$RVIS_OUT/rvis_cot_report.json" ] && head -c 2000 "$RVIS_OUT/rvis_cot_report.json"
echo ""
[ -f "$EDIT_OUT/cot_edit_report.json" ] && head -c 2000 "$EDIT_OUT/cot_edit_report.json"
exit 0
