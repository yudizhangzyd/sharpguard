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
    --malicious-action    "$MALICIOUS_ACTION" \
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
    --dtype               "$DTYPE" \
    --attn                "$ATTN" \
    --seed                "$SEED"

echo ""
echo "==== Done ===="
ls -la "$OUT_DIR"
echo "--- trap_stats.json ---"
[ -f "$OUT_DIR/trap_stats.json" ] && cat "$OUT_DIR/trap_stats.json"
echo ""
echo "--- auroc_table.json ---"
[ -f "$OUT_DIR/auroc_table.json" ] && cat "$OUT_DIR/auroc_table.json"
echo ""
echo "--- task_sr.json (if present) ---"
[ -f "$OUT_DIR/task_sr.json" ] && cat "$OUT_DIR/task_sr.json"
