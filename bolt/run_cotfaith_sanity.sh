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
CKPT_TASK_ID="${CKPT_TASK_ID:-jsyc64ngfs}"
CKPT_LOCAL="/tmp/cotfaith_ckpt"
mkdir -p "$CKPT_LOCAL"

if [ ! -f "$CKPT_LOCAL/config.json" ]; then
    echo "[sanity] copying merged_model from S3 for bolt task $CKPT_TASK_ID"
    # awscli not preinstalled on iris pods — install first.
    which aws >/dev/null 2>&1 || pip install --quiet awscli || true
    S3_URL="s3://bolt-prod-2702150980/tasks/$CKPT_TASK_ID/artifacts/cotfaith-train/merged_model"
    # Bolt pods carry IAM role granting read on task artifact prefixes.
    aws s3 sync "$S3_URL" "$CKPT_LOCAL" --quiet || {
        echo "[sanity] aws s3 sync failed — trying s5cmd fallback"
        pip install --quiet s5cmd || true
        s5cmd cp "$S3_URL/*" "$CKPT_LOCAL/" || {
            echo "[FATAL] cannot fetch $S3_URL"
            exit 3
        }
    }
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
