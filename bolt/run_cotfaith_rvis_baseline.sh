#!/usr/bin/env bash
# Standalone non-CoT baseline r_vis attention analysis on OpenVLA-7B.
# Does NOT retrain — uses the public openvla/openvla-7b-finetuned-libero-*
# checkpoint. Meant to be compared against our CoT-VLA r_vis result.

set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-rvis-baseline"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# OpenVLA-OFT remote code imports 'prismatic.training.train_utils' which
# only exists in moojink's openvla-oft fork (not the main openvla repo).
# Clone that fork for OFT models; also clone main openvla for base baselines.
if [ ! -d /tmp/openvla-oft ]; then
    git clone --depth 1 https://github.com/moojink/openvla-oft /tmp/openvla-oft || true
fi
(cd /tmp/openvla-oft && pip install -e . || true)
# Also main openvla in case some baselines need it
if [ ! -d /tmp/openvla ]; then
    git clone --depth 1 https://github.com/openvla/openvla /tmp/openvla || true
fi
(cd /tmp/openvla && pip install -e . || true)

# tf/tfds for loading LIBERO probe images.
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

python experiments/cotfaith_rvis_baseline.py \
    --base-model "${BASE_MODEL:-openvla/openvla-7b-finetuned-libero-spatial}" \
    --out         "$OUT_DIR" \
    --n-samples   "${N_SAMPLES:-20}" \
    --seed        "${SEED:-0}" \
    --rvis-layers "${RVIS_LAYERS:-0,1,2,3}" \
    --dtype       "${DTYPE:-bfloat16}"

echo ""
echo "==== Done ===="
[ -f "$OUT_DIR/rvis_baseline_report.json" ] && head -c 3000 "$OUT_DIR/rvis_baseline_report.json"
exit 0
