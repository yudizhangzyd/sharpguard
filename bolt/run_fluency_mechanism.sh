#!/usr/bin/env bash
# Fluency-mechanism check on a checkpoint we ALREADY trained, fetched from
# the training task's own artifacts (same S3-sync-with-scoped-token pattern
# as run_cotfaith_edit_s3ckpt.sh, which this is adapted from).
#
# CKPT_TASK_ID  bolt task whose artifacts/cotfaith-train/merged_model to use
# N_SAMPLES     LIBERO samples to draw (default 100); x3 null families each
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/fluency-mechanism"
mkdir -p "$OUT_DIR"
nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

if [ -z "${CKPT_TASK_ID:-}" ]; then
    echo "[FATAL] CKPT_TASK_ID is unset."
    exit 2
fi

CKPT_LOCAL=/tmp/cotfaith_ckpt
mkdir -p "$CKPT_LOCAL"

[ -n "${AWS_SESSION_TOKEN:-}" ] || unset AWS_SESSION_TOKEN
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-https://conductor.data.apple.com}"
export S3_ENDPOINT_URL
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    echo "[FATAL] no S3 credentials in the environment."
    exit 2
fi

if [ ! -f "$CKPT_LOCAL/config.json" ]; then
    echo "[fluency-mech] fetching merged_model from bolt task $CKPT_TASK_ID"
    which aws >/dev/null 2>&1 || pip install --quiet awscli || true
    S3_URL="s3://bolt-prod-2702150980/tasks/$CKPT_TASK_ID/artifacts/cotfaith-train/merged_model"
    aws s3 sync "$S3_URL" "$CKPT_LOCAL" --endpoint-url "$S3_ENDPOINT_URL" --quiet || {
        echo "[fluency-mech] aws s3 sync failed -- trying s5cmd"
        pip install --quiet s5cmd || true
        s5cmd cp "$S3_URL/*" "$CKPT_LOCAL/" || {
            echo "[FATAL] cannot fetch $S3_URL"
            exit 3
        }
    }
fi
[ -f "$CKPT_LOCAL/config.json" ] || { echo "[FATAL] no config.json"; exit 3; }
ls "$CKPT_LOCAL"/*.safetensors >/dev/null 2>&1 \
    || ls "$CKPT_LOCAL"/*.bin >/dev/null 2>&1 \
    || { echo "[FATAL] no weight shards in $CKPT_LOCAL"; exit 3; }
du -sh "$CKPT_LOCAL"; ls -la "$CKPT_LOCAL" | head -20

# TF/tfds for LIBERO loading (mirrors bolt/run_cotfaith_edit_only.sh).
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

python experiments/fluency_mechanism.py \
    --ckpt-path "$CKPT_LOCAL" \
    --out       "$OUT_DIR" \
    --n-samples "${N_SAMPLES:-100}" \
    --seed      "${SEED:-0}" \
    --dtype     "${DTYPE:-bfloat16}"
rc=$?

if [ "$rc" -eq 0 ]; then
    echo "[fluency-mech] ok"
    python - <<'PY'
import json, os
p = os.environ.get("BOLT_ARTIFACT_DIR", "./artifacts") + "/fluency-mechanism/fluency_mechanism_report.json"
d = json.load(open(p))
print(f"n_usable={d['n_usable']} n_judged={d['n_judged']}")
PY
else
    echo "[fluency-mech] FAILED rc=$rc"
fi
exit "$rc"
