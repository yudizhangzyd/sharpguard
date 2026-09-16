#!/usr/bin/env bash
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
nvidia-smi -L || true

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/action-bin-resolution-sweep"
mkdir -p "$OUT_DIR"

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

python -u experiments/action_bin_resolution_sweep.py \
    --ckpt-path "${CKPT_HF_ID:-Embodied-CoT/ecot-openvla-7b-bridge}" \
    --checkpoint-label "ecot-bridge" \
    --out "$OUT_DIR" \
    --n-samples "${N_SAMPLES:-100}" \
    --seed "${SEED:-0}" \
    --threshold "${THRESHOLD:-0.05}" \
    --families "${FAMILIES:-direction_flip,paraphrase_null,syntactic_scramble,cross_task_swap}" \
    --dtype "${DTYPE:-bfloat16}"

echo "==== Done ===="
exit 0
