#!/usr/bin/env bash
# F4/O4 deconfound: train ECoT-bridge on a Bridge V2 subset, then score it.
#
# All four stages run in ONE job on purpose. The deliverable of this experiment
# is not a checkpoint -- it is the checkpoint's CALIBRATION PROFILE, which is
# what gets compared against the LIBERO-trained rows. Splitting train and score
# across jobs would need a cross-job artifact fetch, and bolt pods have no aws
# creds for bolt's own S3 prefixes (see run_cotfaith_train_and_sanity.sh).
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi

# Repo root importable regardless of what setup wrote into the env file. This is
# what the five failed retraining replicates needed.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

pip install "datasets>=2.19,<3.0" "av" "pyarrow>=15" 2>&1 | tail -3 || true

TRAIN_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-bridge-subset"
mkdir -p "$TRAIN_OUT"

EXTRA=""
[ "${PREFLIGHT_ONLY:-0}" = "1" ] && EXTRA="--preflight-only"

# ---- Step 1: TRAIN (exits 6 if the reasoning<->trajectory join is empty) ----
python experiments/cotfaith_train_bridge.py \
    --out              "$TRAIN_OUT" \
    --base-model       "${BASE_MODEL:-Embodied-CoT/ecot-openvla-7b-bridge}" \
    --reasoning-repo   "${BRIDGE_REASONING:-Embodied-CoT/embodied_features_bridge}" \
    --dataset-repo     "${BRIDGE_DATASET:-IPEC-COMMUNITY/bridge_orig_lerobot}" \
    --n-trajectories   "${N_TRAJECTORIES:-4000}" \
    --max-steps-per-ep "${MAX_STEPS_PER_EP:-0}" \
    --lora-r           "${LORA_R:-32}" \
    --lora-alpha       "${LORA_ALPHA:-16}" \
    --lr               "${LR:-2e-5}" \
    --steps            "${STEPS:-15000}" \
    --batch-size       "${BATCH_SIZE:-2}" \
    --dtype            "${DTYPE:-bfloat16}" \
    --seed             "${SEED:-0}" \
    --reasoning-mode   "${REASONING_MODE:-full}" \
    --preflight-n      "${PREFLIGHT_N:-3}" \
    $EXTRA

echo ""
echo "===== TRAIN done."
[ -f "$TRAIN_OUT/preflight_report.json" ] && cat "$TRAIN_OUT/preflight_report.json"
[ -f "$TRAIN_OUT/train_meta.json" ] && cat "$TRAIN_OUT/train_meta.json"

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
    echo "[bridge-subset] preflight only; skipping scoring stages"
    exit 0
fi

CKPT="$TRAIN_OUT/merged_model"
ls -la "$CKPT" | head -10

# ---- Step 2-4 need TFDS; step 1 does not -------------------------------------
# This install is deliberately AFTER training. `v9rkpp2342` trained all 15000
# steps successfully (loss 1.1755 -> 0.2373) and then died here:
#
#   ModuleNotFoundError: No module named 'tensorflow_datasets'
#     (experiments/cotfaith_sanity.py:76)
#
# The scoring stages read LIBERO through TFDS and this script never installed it
# -- step 1 pulls Bridge through `datasets`, so nothing before now needed it.
# Keeping the install here rather than at the top means a resolver accident can
# only cost the minutes of scoring, never the hours of training.
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
python -c "import tensorflow_datasets, tensorflow as tf;
print('[bridge-subset] tfds', tensorflow_datasets.__version__, 'tf', tf.__version__)" \
    || echo "[bridge-subset] WARNING tfds still not importable; stages 2-4 will fail"

# From here on a stage failure is recorded and the remaining stages still run.
# `set -e` is what turned v9rkpp2342 into a FAILED task with no scores at all:
# the cheapest stage (sanity) killed step 4, which is the calibration profile
# this job exists to produce. The exit code at the bottom still reflects it.
set +e
FAILED_STAGES=""

# ---- Step 2: SANITY ----
SANITY_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-bridge-subset-sanity"
mkdir -p "$SANITY_OUT"
python experiments/cotfaith_sanity.py \
    --ckpt-path "$CKPT" --out "$SANITY_OUT" --dtype "${DTYPE:-bfloat16}" \
    || FAILED_STAGES="$FAILED_STAGES sanity"
[ -f "$SANITY_OUT/sanity_report.json" ] && head -80 "$SANITY_OUT/sanity_report.json"

# ---- Step 3: r_vis(CoT) attention decomposition ----
RVIS_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-bridge-subset-rvis"
mkdir -p "$RVIS_OUT"
python experiments/cotfaith_rvis.py \
    --ckpt-path "$CKPT" --out "$RVIS_OUT" \
    --n-samples "${RVIS_N_SAMPLES:-100}" \
    --rvis-layers "${RVIS_LAYERS:-0,1,2,3}" \
    --dtype "${DTYPE:-bfloat16}" \
    || FAILED_STAGES="$FAILED_STAGES rvis"
[ -f "$RVIS_OUT/rvis_cot_report.json" ] && head -c 1000 "$RVIS_OUT/rvis_cot_report.json"

# ---- Step 4: causal CoT edit -> the calibration profile this job exists for ----
EDIT_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-bridge-subset-edit"
mkdir -p "$EDIT_OUT"
python experiments/cotfaith_edit.py \
    --ckpt-path "$CKPT" --out "$EDIT_OUT" \
    --n-samples "${EDIT_N_SAMPLES:-100}" \
    --threshold "${EDIT_THRESHOLD:-0.05}" \
    --dtype "${DTYPE:-bfloat16}" \
    || FAILED_STAGES="$FAILED_STAGES edit"

echo ""
echo "==== Done ===="
[ -f "$EDIT_OUT/cot_edit_report.json" ] && head -c 2500 "$EDIT_OUT/cot_edit_report.json"

# The checkpoint is in this task's S3 prefix either way, so a partial failure is
# recoverable without retraining -- but it must not report as success.
if [ -n "$FAILED_STAGES" ]; then
    echo "[bridge-subset] stages failed:$FAILED_STAGES"
    echo "[bridge-subset] training itself succeeded and merged_model is in this"
    echo "                task's artifacts; re-score with"
    echo "                bolt/run_cotfaith_bridge_subset_eval_s3.sh instead of"
    echo "                repeating ${STEPS:-15000} training steps."
    [ -f "$EDIT_OUT/cot_edit_report.json" ] || exit 5
    exit 4
fi
exit 0
