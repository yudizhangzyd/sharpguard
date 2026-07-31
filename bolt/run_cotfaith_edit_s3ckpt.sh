#!/usr/bin/env bash
# 13-family causal-edit protocol on a checkpoint we ALREADY trained, fetched
# from the training task's own artifacts.
#
# Why this exists: the seven "ours" leaderboard rows were each produced by a
# train+edit job that ran the 10-family set at a single seed. Adding the three
# calibration nulls and two more inference seeds does not need the training
# rerun -- the merged_model is still in the training task's S3 prefix. Retraining
# would also make the new numbers a DIFFERENT checkpoint from the published row,
# which is exactly the confound we are trying to remove: f2d55rbcpd (an
# independent r=32 retrain) moved verb_swap from 0.60 to 0.78 on the same
# protocol. So: same weights, new families, new inference seeds.
#
# CKPT_TASK_ID  bolt task whose artifacts/cotfaith-train/merged_model to use
# SEEDS         whitespace-separated inference seeds (default "0 1 2")
# N_SAMPLES     samples per seed (default 100)
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-edit"
mkdir -p "$OUT_DIR"
nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

if [ -z "${CKPT_TASK_ID:-}" ]; then
    echo "[FATAL] CKPT_TASK_ID is unset. This runner deliberately has no default:"
    echo "        silently editing the wrong checkpoint is unrecoverable from the"
    echo "        report, which records only the local path."
    exit 2
fi

CKPT_LOCAL=/tmp/cotfaith_ckpt
mkdir -p "$CKPT_LOCAL"
if [ ! -f "$CKPT_LOCAL/config.json" ]; then
    echo "[edit-s3] fetching merged_model from bolt task $CKPT_TASK_ID"
    which aws >/dev/null 2>&1 || pip install --quiet awscli || true
    S3_URL="s3://bolt-prod-2702150980/tasks/$CKPT_TASK_ID/artifacts/cotfaith-train/merged_model"
    aws s3 sync "$S3_URL" "$CKPT_LOCAL" --quiet || {
        echo "[edit-s3] aws s3 sync failed -- trying s5cmd"
        pip install --quiet s5cmd || true
        s5cmd cp "$S3_URL/*" "$CKPT_LOCAL/" || {
            echo "[FATAL] cannot fetch $S3_URL"
            exit 3
        }
    }
fi
# A partial sync produces a loadable-looking directory that decodes garbage.
# Check the two files whose absence is silent rather than fatal.
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

FAILED_SEEDS=""
for SEED in ${SEEDS:-0 1 2}; do
    SEED_DIR="$OUT_DIR/seed$SEED"
    mkdir -p "$SEED_DIR"
    echo "===== seed $SEED ====="
    # One seed failing must not discard the seeds that already succeeded: the
    # per-seed reports are the deliverable, and a 3-seed job that exits early
    # after writing 2 of them is worth more than one that writes none.
    rc=0
    python experiments/cotfaith_edit.py \
        --ckpt-path "$CKPT_LOCAL" \
        --out       "$SEED_DIR" \
        --n-samples "${N_SAMPLES:-100}" \
        --seed      "$SEED" \
        --families  all \
        --threshold "${THRESHOLD:-0.05}" \
        --dtype     "${DTYPE:-bfloat16}" || rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "[edit-s3] seed $SEED ok"
    else
        echo "[edit-s3] seed $SEED FAILED rc=$rc"
        FAILED_SEEDS="$FAILED_SEEDS $SEED"
    fi
done

echo ""
echo "==== Summary ===="
python - <<'PY'
import glob, json, os
base = os.environ.get("BOLT_ARTIFACT_DIR", "./artifacts") + "/cotfaith-edit"
for p in sorted(glob.glob(base + "/seed*/cot_edit_report.json")):
    d = json.load(open(p))
    agg = d.get("aggregate", {})
    print(f"\n{p}  seed={d.get('seed')}  families={len(agg)}")
    for k, v in agg.items():
        print(f"  {k:24s} n={v.get('n')}  n_skipped={v.get('n_skipped')}  "
              f"F={v.get('faithful_rate')}")
PY

if [ -n "$FAILED_SEEDS" ]; then
    echo "[edit-s3] non-zero exit: seeds failed:$FAILED_SEEDS"
    exit 4
fi
echo "==== Done ===="
exit 0
