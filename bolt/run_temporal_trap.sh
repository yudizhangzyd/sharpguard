#!/usr/bin/env bash
# TemporalTrap training + TemporalAudit AUROC evaluation.

set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/temporal-trap"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false

# ---- Kim official eval deps (only if we're going to use it) ----
if [ "${KIM_EVAL_EPS_PER_TASK:-0}" -gt 0 ]; then
    if [ ! -d /tmp/openvla ]; then
        git clone --depth 1 https://github.com/openvla/openvla /tmp/openvla
    fi
    (cd /tmp/openvla && pip install -e . || true)
    pip install "draccus" "wandb" "diffusers" || true
    # openvla pins protobuf==4.25.9 but pulls tensorflow_metadata 1.21+
    # which needs proto 5.27's runtime_version. Downgrade tf_metadata.
    pip install "tensorflow_metadata==1.15.0" --force-reinstall --no-deps
    # Kim's eval hardcodes flash_attention_2
    pip install "flash-attn==2.5.8" --no-build-isolation \
        || pip install "flash-attn" --no-build-isolation
    # verify all imports before spending 4h on training
    python -c "import flash_attn; from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig; print('[verify] Kim-eval deps OK')" \
        || { echo '[FATAL] Kim-eval deps broken; aborting'; exit 3; }
fi

# ---- RLDS data source deps (need tfds even without Kim eval) ----
if [ "${DATA_SOURCE:-rollout}" = "rlds" ] \
        && ! python -c "import tensorflow_datasets" 2>/dev/null; then
    pip install "tensorflow_datasets==4.9.3" "tensorflow_metadata==1.15.0" \
                --force-reinstall --no-deps \
        || { echo '[FATAL] tfds install failed; aborting'; exit 4; }
    python -c "import tensorflow_datasets; print('[verify] tfds OK')" \
        || { echo '[FATAL] tfds broken; aborting'; exit 4; }
fi

python experiments/openvla_temporal_trap.py \
    --model               "$MODEL" \
    --out                 "$OUT_DIR" \
    --libero-suite        "$LIBERO_SUITE" \
    --libero-collect-eps  "$LIBERO_COLLECT_EPS" \
    --libero-collect-steps "$LIBERO_COLLECT_STEPS" \
    --fire-state          "$FIRE_STATE" \
    --fire-step-k         "$FIRE_STEP_K" \
    --poison-episode-rate "$POISON_EPISODE_RATE" \
    --trigger-phrase      "$TRIGGER_PHRASE" \
    --trigger-phrases     "${TRIGGER_PHRASES:-}" \
    --malicious-action    "$MALICIOUS_ACTION" \
    --malicious-action-mode "${MALICIOUS_ACTION_MODE:-fixed}" \
    --rvis-aware-lambda     "${RVIS_AWARE_LAMBDA:-0.0}" \
    --rvis-aware-mode       "${RVIS_AWARE_MODE:-l2}" \
    --rvis-aware-ema-alpha  "${RVIS_AWARE_EMA_ALPHA:-0.99}" \
    --lora-steps          "$LORA_STEPS" \
    --lora-r              "$LORA_R" \
    --lr                  "$LR" \
    --batch-size          "$BATCH_SIZE" \
    --n-eval-episodes     "$N_EVAL_EPISODES" \
    --rvis-layers         "$RVIS_LAYERS" \
    --r-clean             "$R_CLEAN" \
    --cusum-k             "$CUSUM_K" \
    --cusum-h             "$CUSUM_H" \
    --top-k               "$TOP_K" \
    --rollout-eps-per-task "${ROLLOUT_EPS_PER_TASK:-0}" \
    --rollout-max-steps    "${ROLLOUT_MAX_STEPS:-200}" \
    --unnorm-key           "${UNNORM_KEY:-}" \
    --kim-eval-eps-per-task "${KIM_EVAL_EPS_PER_TASK:-0}" \
    --data-source          "${DATA_SOURCE:-rollout}" \
    --rlds-data-dir        "${RLDS_DATA_DIR:-}" \
    --dtype               "$DTYPE" \
    --attn                "$ATTN" \
    --seed                "$SEED"

echo ""
echo "==== Done ===="
ls -la "$OUT_DIR" || true
echo "--- trap_stats.json ---"
[ -f "$OUT_DIR/trap_stats.json" ] && cat "$OUT_DIR/trap_stats.json" || true
echo ""
echo "--- auroc_table.json ---"
[ -f "$OUT_DIR/auroc_table.json" ] && cat "$OUT_DIR/auroc_table.json" || true
echo ""
echo "--- task_sr.json (if present) ---"
[ -f "$OUT_DIR/task_sr.json" ] && cat "$OUT_DIR/task_sr.json" || true
echo ""
echo "--- kim_eval/*.json (if present) ---"
for f in "$OUT_DIR"/kim_eval/task_sr_*.json; do
    [ -f "$f" ] && echo "$f:" && cat "$f" || true
done
echo ""
echo "--- rvis_aware_stats.json (if present) ---"
[ -f "$OUT_DIR/rvis_aware_stats.json" ] && cat "$OUT_DIR/rvis_aware_stats.json" || true
exit 0
