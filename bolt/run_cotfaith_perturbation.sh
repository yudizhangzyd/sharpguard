#!/usr/bin/env bash
# Perturbation robustness runner (rvis under image perturbations)
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-perturbation"
mkdir -p "$OUT_DIR"
nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

pip install "dm-tree" "protobuf>=3.20,<5" "promise" "dill" "etils[epath]" \
            "toml" "termcolor" "tqdm" "click" || true
pip install "tensorflow-cpu==2.15.1" --no-deps \
    || pip install "tensorflow==2.15.1" --no-deps || true
pip install "absl-py" "astunparse" "flatbuffers" "gast" "google-pasta" \
            "grpcio" "h5py" "libclang" "ml-dtypes==0.2.0" "opt-einsum" \
            "packaging" "six" "wrapt" "termcolor" "typing-extensions" \
            "tensorboard==2.15.2" "keras==2.15.0" "tensorflow-estimator==2.15.0" || true
pip install "tensorflow_datasets==4.9.3" "tensorflow_metadata==1.15.0" \
            --force-reinstall --no-deps || true

python experiments/cotfaith_perturbation.py \
    --ckpt-path "${CKPT_HF_ID:-Embodied-CoT/ecot-openvla-7b-bridge}" \
    --out "$OUT_DIR" \
    --n-samples "${N_SAMPLES:-50}" \
    --seed "${SEED:-0}" \
    --rvis-layers "${RVIS_LAYERS:-0,1,2,3}" \
    --dtype "${DTYPE:-bfloat16}"

echo "==== Done ===="
head -c 3000 "$OUT_DIR/cot_perturbation_report.json" || true
exit 0
