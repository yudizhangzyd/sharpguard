#!/usr/bin/env bash
# ProGuard Goal-T sanity check.
#
# Task #44 — first end-to-end ProGuard run on bolt.
# Expected:
#   - clean SR ≈ baseline (Stage 0)
#   - vanilla_poisoned ASR << 100% (ProGuard prevents backdoor)
#   - r_vis trajectory stays high in artifacts/.../rvis_trajectory_poisoned.json

set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a
    # shellcheck disable=SC1091
    . /tmp/sharpguard.env
    set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/proguard-sanity"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true

export CUDA_VISIBLE_DEVICES=0

CMD=(
    python experiments/openvla_real.py
    --model         "$MODEL"
    --out           "$OUT_DIR"
    --n-train       "$N_TRAIN"
    --n-eval        "$N_EVAL"
    --poison-rate   "$POISON_RATE"
    --lora-steps    "$LORA_STEPS"
    --lora-r        "$LORA_R"
    --lr            "$LR"
    --batch-size    "$BATCH_SIZE"
    --measure-batches "$MEASURE_BATCHES"
    --epsilon       "$EPSILON"
    --n-trials      "$N_TRIALS"
    --rho           "$RHO"
    --lam-sg        "$LAM_SG"
    --lam-sg-b      "$LAM_SG_B"
    --lam-adapt     "$LAM_ADAPT"
    --detector-drop-quantile "$DETECTOR_DROP_Q"
    --libero-max-eps "$LIBERO_MAX_EPS"
    --libero-sim-suite "$LIBERO_SIM_SUITE"
    --libero-sim-eps  "$LIBERO_SIM_EPS"
    --dtype         "$DTYPE"
    --attn          "$ATTN"
    --seed          "$SEED"
    --skip-stages   "$SKIP_STAGES"
    --proguard-lambda     "$PROGUARD_LAMBDA"
    --proguard-alpha      "$PROGUARD_ALPHA"
    --proguard-tau        "$PROGUARD_TAU"
    --proguard-layers     "$PROGUARD_LAYERS"
    --proguard-apply-to   "$PROGUARD_APPLY_TO"
)

[ -n "$USE_BADVLA"          ] && CMD+=(--use-badvla)
[ -n "$LIBERO_SIM_EVAL"     ] && CMD+=(--libero-sim-eval)
[ -n "$USE_LIBERO_COLLECT"  ] && CMD+=(--use-libero-collect)
[ -n "$LIBERO_COLLECT_EPS"  ] && CMD+=(--libero-collect-eps "$LIBERO_COLLECT_EPS")
[ -n "$LIBERO_COLLECT_STEPS" ] && CMD+=(--libero-collect-steps "$LIBERO_COLLECT_STEPS")

"${CMD[@]}"

echo ""
echo "==== Done. Artifacts ===="
ls -la "$OUT_DIR"
echo "---"
if [ -f "$OUT_DIR/rvis_trajectory_poisoned.json" ]; then
    echo "ProGuard r_vis trajectory (first 30 + last 30 steps):"
    python -c "
import json
d = json.load(open('$OUT_DIR/rvis_trajectory_poisoned.json'))
rvis = d['rvis_per_step']
ema  = d['ema_per_step']
print(f'  total steps: {len(rvis)}')
print(f'  r_vis: init={rvis[0]:.4f}  final={rvis[-1]:.4f}')
print(f'  ema:   init={ema[0]:.4f}   final={ema[-1]:.4f}')
print('  r_vis first 30:', [f'{v:.3f}' for v in rvis[:30]])
print('  r_vis last  30:', [f'{v:.3f}' for v in rvis[-30:]])
"
fi
[ -f "$OUT_DIR/results.json" ] && head -100 "$OUT_DIR/results.json"
