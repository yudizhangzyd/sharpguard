#!/usr/bin/env bash
# F4/O4 deconfound, scoring half: score the Bridge-subset checkpoint we ALREADY
# trained, fetched from that training task's own S3 prefix.
#
# Why this exists rather than a rerun of run_cotfaith_bridge_subset.sh: task
# `v9rkpp2342` completed all 15000 training steps (loss 1.1755 -> 0.2373,
# join_strategy=by_task_text, n_forward_failures=0) and then reported FAILED for
# one reason -- the scoring stages import LIBERO through TFDS and that script
# never installed it. The weights are intact in
#   tasks/$CKPT_TASK_ID/artifacts/cotfaith-bridge-subset/merged_model
# so repeating the training would spend hours to produce a DIFFERENT checkpoint,
# and a different checkpoint is exactly the confound this experiment removes:
# f2d55rbcpd (an independent r=32 retrain of the same config) moved verb_swap
# 0.60 -> 0.78 on an unchanged protocol. Same weights, or the comparison is not
# the deconfound it claims to be.
#
# Stages 2-4 of that script, unchanged: sanity, r_vis(CoT), and the 13-family
# causal edit whose per-family F values ARE the calibration profile that gets
# compared against the LIBERO-trained rows.
#
# CKPT_TASK_ID  bolt task whose artifacts/cotfaith-bridge-subset/merged_model to score
# CKPT_SUBDIR   artifact subdirectory (default cotfaith-bridge-subset)
#
# Credentials: same mechanism as run_cotfaith_edit_s3ckpt.sh -- the bolt artifact
# bucket sits behind https://conductor.data.apple.com and needs a task-scoped,
# expiring token issued at submit time by bolt/submit_rollout_edit_ckpt.sh. It is
# worth nothing in a log, but a stale one must fail loudly rather than read
# nothing and score a randomly-initialised model.
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

if [ -z "${CKPT_TASK_ID:-}" ]; then
    echo "[FATAL] CKPT_TASK_ID is unset. No default: the reports record only the"
    echo "        local path, so scoring the wrong checkpoint is unrecoverable."
    exit 2
fi

CKPT=/tmp/cotfaith_bridge_subset_ckpt
mkdir -p "$CKPT"

# An empty AWS_SESSION_TOKEN is worse than an absent one: botocore signs with it
# and the request is rejected. get-credentials emits it empty for scoped tokens.
[ -n "${AWS_SESSION_TOKEN:-}" ] || unset AWS_SESSION_TOKEN
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-https://conductor.data.apple.com}"
export S3_ENDPOINT_URL
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    echo "[FATAL] no S3 credentials. Submit via"
    echo "          bash bolt/submit_rollout_edit_ckpt.sh $CKPT_TASK_ID \\"
    echo "               bolt/boltconfig-cotfaith-bridge-subset-eval.yaml"
    echo "        which issues a token scoped to that task's prefix."
    exit 2
fi

if [ ! -f "$CKPT/config.json" ]; then
    SUB="${CKPT_SUBDIR:-cotfaith-bridge-subset}"
    echo "[bridge-eval] fetching $SUB/merged_model from bolt task $CKPT_TASK_ID"
    which aws >/dev/null 2>&1 || pip install --quiet awscli || true
    S3_URL="s3://bolt-prod-2702150980/tasks/$CKPT_TASK_ID/artifacts/$SUB/merged_model"
    aws s3 sync "$S3_URL" "$CKPT" --endpoint-url "$S3_ENDPOINT_URL" --quiet || {
        echo "[bridge-eval] aws s3 sync failed -- trying s5cmd"
        pip install --quiet s5cmd || true
        s5cmd cp "$S3_URL/*" "$CKPT/" || {
            echo "[FATAL] cannot fetch $S3_URL"
            echo "        AccessDenied => token scoped to another task."
            echo "        ExpiredToken => re-issue and resubmit."
            exit 3
        }
    }
fi
# A partial sync leaves a loadable-looking directory that decodes garbage.
[ -f "$CKPT/config.json" ] || { echo "[FATAL] no config.json"; exit 3; }
ls "$CKPT"/*.safetensors >/dev/null 2>&1 \
    || ls "$CKPT"/*.bin >/dev/null 2>&1 \
    || { echo "[FATAL] no weight shards in $CKPT"; exit 3; }
du -sh "$CKPT"; ls -la "$CKPT" | head -20

# Also fetch the training task's own metadata next to the new reports. Without
# it, a reader of these scores has no record of WHICH corpus and join produced
# the weights -- which is the entire content of the deconfound claim.
META_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-bridge-subset-train-meta"
mkdir -p "$META_OUT"
for f in train_meta.json args.json preflight_report.json train_losses.json; do
    aws s3 cp \
        "s3://bolt-prod-2702150980/tasks/$CKPT_TASK_ID/artifacts/${CKPT_SUBDIR:-cotfaith-bridge-subset}/$f" \
        "$META_OUT/$f" --endpoint-url "$S3_ENDPOINT_URL" --quiet \
        || echo "[bridge-eval] no $f in the training prefix"
done
[ -f "$META_OUT/train_meta.json" ] && cat "$META_OUT/train_meta.json"

# ---- the install that v9rkpp2342 was missing ---------------------------------
# The scoring stages read LIBERO through TFDS. This is the same stack every other
# scoring runner installs (see run_cotfaith_edit_s3ckpt.sh); tensorflow goes in
# with --no-deps so it cannot drag torch along.
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
# Fail here, before three stages report nothing, if the one missing import that
# caused this rerun is still missing.
python -c "import tensorflow_datasets, tensorflow as tf;
print('[bridge-eval] tfds', tensorflow_datasets.__version__, 'tf', tf.__version__)" \
    || { echo "[FATAL] tensorflow_datasets still not importable -- this is the"
         echo "        exact failure this job exists to fix. Stopping before the"
         echo "        stages report empty."; exit 6; }

# A stage failure is recorded and the remaining stages still run: the edit report
# is the deliverable, and losing it to a sanity crash is what happened last time.
set +e
FAILED_STAGES=""

SANITY_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-bridge-subset-sanity"
mkdir -p "$SANITY_OUT"
python experiments/cotfaith_sanity.py \
    --ckpt-path "$CKPT" --out "$SANITY_OUT" --dtype "${DTYPE:-bfloat16}" \
    || FAILED_STAGES="$FAILED_STAGES sanity"
[ -f "$SANITY_OUT/sanity_report.json" ] && head -80 "$SANITY_OUT/sanity_report.json"

RVIS_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-bridge-subset-rvis"
mkdir -p "$RVIS_OUT"
python experiments/cotfaith_rvis.py \
    --ckpt-path "$CKPT" --out "$RVIS_OUT" \
    --n-samples "${RVIS_N_SAMPLES:-100}" \
    --rvis-layers "${RVIS_LAYERS:-0,1,2,3}" \
    --dtype "${DTYPE:-bfloat16}" \
    || FAILED_STAGES="$FAILED_STAGES rvis"
[ -f "$RVIS_OUT/rvis_cot_report.json" ] && head -c 1000 "$RVIS_OUT/rvis_cot_report.json"

# The calibration profile. --families all: the deconfound compares this
# checkpoint's profile to the LIBERO-trained rows, and those rows are 13-family.
EDIT_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-bridge-subset-edit"
mkdir -p "$EDIT_OUT"
python experiments/cotfaith_edit.py \
    --ckpt-path "$CKPT" --out "$EDIT_OUT" \
    --n-samples "${EDIT_N_SAMPLES:-100}" \
    --families  "${FAMILIES:-all}" \
    --seed      "${SEED:-0}" \
    --threshold "${EDIT_THRESHOLD:-0.05}" \
    --dtype     "${DTYPE:-bfloat16}" \
    || FAILED_STAGES="$FAILED_STAGES edit"

echo ""
echo "==== Summary ===="
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get("BOLT_ARTIFACT_DIR", "./artifacts"))
p = base / "cotfaith-bridge-subset-edit" / "cot_edit_report.json"
if not p.exists():
    print("[bridge-eval] NO EDIT REPORT: the calibration profile this job exists "
          "for was not produced. Nothing here can be written into the O4/F4 "
          "section.")
else:
    d = json.loads(p.read_text())
    agg = d.get("aggregate", {})
    print(f"families={len(agg)} n_samples={d.get('n_samples')} seed={d.get('seed')}")
    for k, v in agg.items():
        print(f"  {k:24s} n={v.get('n')}  n_skipped={v.get('n_skipped')}  "
              f"F={v.get('faithful_rate')}")
    print("\n[bridge-eval] This is a Bridge-trained ECoT scored on the SAME "
          "protocol as the LIBERO-trained rows. Read it against the "
          "ecot-openvla-7b-bridge row: if the profile tracks the corpus rather "
          "than the suite, O4/F4's corpus confound is what the leaderboard "
          "spread reflects.")
PY

if [ -n "$FAILED_STAGES" ]; then
    echo "[bridge-eval] stages failed:$FAILED_STAGES"
    [ -f "${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-bridge-subset-edit/cot_edit_report.json" ] \
        || exit 5
    exit 4
fi
echo "==== Done ===="
exit 0
