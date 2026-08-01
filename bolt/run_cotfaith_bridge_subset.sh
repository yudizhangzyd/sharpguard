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

# ---- Step 2: SANITY ----
SANITY_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-bridge-subset-sanity"
mkdir -p "$SANITY_OUT"
python experiments/cotfaith_sanity.py \
    --ckpt-path "$CKPT" --out "$SANITY_OUT" --dtype "${DTYPE:-bfloat16}"
[ -f "$SANITY_OUT/sanity_report.json" ] && head -80 "$SANITY_OUT/sanity_report.json"

# ---- Step 3: r_vis(CoT) attention decomposition ----
RVIS_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-bridge-subset-rvis"
mkdir -p "$RVIS_OUT"
python experiments/cotfaith_rvis.py \
    --ckpt-path "$CKPT" --out "$RVIS_OUT" \
    --n-samples "${RVIS_N_SAMPLES:-100}" \
    --rvis-layers "${RVIS_LAYERS:-0,1,2,3}" \
    --dtype "${DTYPE:-bfloat16}"
[ -f "$RVIS_OUT/rvis_cot_report.json" ] && head -c 1000 "$RVIS_OUT/rvis_cot_report.json"

# ---- Step 4: causal CoT edit -> the calibration profile this job exists for ----
EDIT_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-bridge-subset-edit"
mkdir -p "$EDIT_OUT"
python experiments/cotfaith_edit.py \
    --ckpt-path "$CKPT" --out "$EDIT_OUT" \
    --n-samples "${EDIT_N_SAMPLES:-100}" \
    --threshold "${EDIT_THRESHOLD:-0.05}" \
    --dtype "${DTYPE:-bfloat16}"

echo ""
echo "==== Done ===="
[ -f "$EDIT_OUT/cot_edit_report.json" ] && head -c 2500 "$EDIT_OUT/cot_edit_report.json"
exit 0
