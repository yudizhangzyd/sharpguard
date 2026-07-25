#!/usr/bin/env bash
# Sanity test our fine-tuned ECoT-LIBERO merged_model.

set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-sanity"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# ---- Fetch our just-trained checkpoint from the previous bolt task ----
CKPT_TASK_ID="${CKPT_TASK_ID:-jsyc64ngfs}"   # full-train job that produced merged_model
CKPT_LOCAL="/tmp/cotfaith_ckpt"
mkdir -p "$CKPT_LOCAL"

if [ ! -f "$CKPT_LOCAL/config.json" ]; then
    echo "[sanity] copying merged_model from bolt task $CKPT_TASK_ID"
    # bolt task copy_artifacts pattern (matches how we already pull artifacts locally).
    # `--src` selects the sub-path inside the previous task's artifact dir.
    bolt task copy_artifacts "$CKPT_TASK_ID" --dest "$CKPT_LOCAL" \
        --src cotfaith-train/merged_model || {
        echo "[FATAL] failed to copy_artifacts from $CKPT_TASK_ID"
        exit 3
    }
    # bolt copy_artifacts may nest the tree under the src prefix; flatten if so.
    if [ -d "$CKPT_LOCAL/cotfaith-train/merged_model" ]; then
        mv "$CKPT_LOCAL/cotfaith-train/merged_model"/* "$CKPT_LOCAL/"
        rm -rf "$CKPT_LOCAL/cotfaith-train"
    fi
fi
ls -la "$CKPT_LOCAL" | head -15

# ---- tf/tfds deps for loading LIBERO probe image ----
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

python experiments/cotfaith_sanity.py \
    --ckpt-path "$CKPT_LOCAL" \
    --out       "$OUT_DIR" \
    --dtype     "${DTYPE:-bfloat16}"

echo ""
echo "==== Done ===="
[ -f "$OUT_DIR/sanity_report.json" ] && cat "$OUT_DIR/sanity_report.json" || true
exit 0
