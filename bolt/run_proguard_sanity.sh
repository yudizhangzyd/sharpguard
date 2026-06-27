#!/usr/bin/env bash
# ProGuard Goal-T sanity v2 — control (lambda=0) + treatment (lambda=1) in parallel.
#
# Runs two openvla_real.py processes:
#   GPU 0: --proguard-lambda 0    (control)
#   GPU 1: --proguard-lambda 1    (treatment)
#
# Each completes the standard Stage 0 clean + vanilla_poisoned pipeline.
# We then compare ASR/SR side-by-side to isolate ProGuard's effect.

set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a
    # shellcheck disable=SC1091
    . /tmp/sharpguard.env
    set +a
fi

OUT_BASE="${BOLT_ARTIFACT_DIR:-./artifacts}/proguard-sanity"
mkdir -p "$OUT_BASE/control" "$OUT_BASE/treatment"

nvidia-smi -L || true

build_cmd() {
    local lam="$1"
    local out_dir="$2"
    cat <<EOF
python experiments/openvla_real.py \
    --model         "$MODEL" \
    --out           "$out_dir" \
    --n-train       "$N_TRAIN" \
    --n-eval        "$N_EVAL" \
    --poison-rate   "$POISON_RATE" \
    --lora-steps    "$LORA_STEPS" \
    --lora-r        "$LORA_R" \
    --lr            "$LR" \
    --batch-size    "$BATCH_SIZE" \
    --measure-batches "$MEASURE_BATCHES" \
    --epsilon       "$EPSILON" \
    --n-trials      "$N_TRIALS" \
    --rho           "$RHO" \
    --lam-sg        "$LAM_SG" \
    --lam-sg-b      "$LAM_SG_B" \
    --lam-adapt     "$LAM_ADAPT" \
    --detector-drop-quantile "$DETECTOR_DROP_Q" \
    --libero-max-eps "$LIBERO_MAX_EPS" \
    --libero-sim-suite "$LIBERO_SIM_SUITE" \
    --libero-sim-eps  "$LIBERO_SIM_EPS" \
    --dtype         "$DTYPE" \
    --attn          "$ATTN" \
    --seed          "$SEED" \
    --skip-stages   "$SKIP_STAGES" \
    --proguard-lambda     "$lam" \
    --proguard-alpha      "$PROGUARD_ALPHA" \
    --proguard-tau        "$PROGUARD_TAU" \
    --proguard-layers     "$PROGUARD_LAYERS" \
    --proguard-apply-to   "$PROGUARD_APPLY_TO" \
    --libero-sim-eval \
    --use-libero-collect \
    --libero-collect-eps "$LIBERO_COLLECT_EPS" \
    --libero-collect-steps "$LIBERO_COLLECT_STEPS"
EOF
}

# Launch control (lambda=0) on GPU 0 in background
CONTROL_CMD="$(build_cmd 0.0 $OUT_BASE/control)"
(
    CUDA_VISIBLE_DEVICES=0 bash -c "$CONTROL_CMD" 2>&1 | tee "$OUT_BASE/control.log"
) &
CONTROL_PID=$!

# Launch treatment (lambda=1) on GPU 1 in background
TREATMENT_CMD="$(build_cmd 1.0 $OUT_BASE/treatment)"
(
    CUDA_VISIBLE_DEVICES=1 bash -c "$TREATMENT_CMD" 2>&1 | tee "$OUT_BASE/treatment.log"
) &
TREATMENT_PID=$!

# Wait for both
ANY_FAILED=0
if ! wait $CONTROL_PID; then
    echo "[control] FAILED"
    ANY_FAILED=1
fi
if ! wait $TREATMENT_PID; then
    echo "[treatment] FAILED"
    ANY_FAILED=1
fi

if [ $ANY_FAILED -ne 0 ]; then
    echo "==== last 30 lines of control.log ===="; tail -30 "$OUT_BASE/control.log"
    echo "==== last 30 lines of treatment.log ===="; tail -30 "$OUT_BASE/treatment.log"
    exit 1
fi

echo ""
echo "==== Done. Side-by-side comparison ===="
export OUT_BASE
python <<'PYEOF'
import json
import os
from pathlib import Path

base = Path(os.environ["OUT_BASE"])
print(f"{'metric':<30s}  {'control (lam=0)':>22s}  {'treatment (lam=1)':>22s}")
print("-" * 80)

c_path = base / "control" / "results.json"
t_path = base / "treatment" / "results.json"
if not c_path.exists() or not t_path.exists():
    print(f"missing: control={c_path.exists()} treatment={t_path.exists()}")
else:
    c = json.load(open(c_path))
    t = json.load(open(t_path))
    for k in c:
        if "metrics" not in c.get(k, {}):
            continue
        mc = c[k]["metrics"]
        mt = t.get(k, {}).get("metrics", {})
        if not mt:
            continue
        c_sr = mc.get("SR", float("nan"))
        c_asr = mc.get("ASR", float("nan"))
        t_sr = mt.get("SR", float("nan"))
        t_asr = mt.get("ASR", float("nan"))
        print(f"  {k:<30s}  SR={c_sr:.3f} ASR={c_asr:.3f}        "
              f"SR={t_sr:.3f} ASR={t_asr:.3f}")

print()
for side in ("control", "treatment"):
    p = base / side / "rvis_trajectory_poisoned.json"
    if p.exists():
        d = json.load(open(p))
        rvis = d["rvis_per_step"]
        ema = d["ema_per_step"]
        print(f"  [{side}] r_vis: init={rvis[0]:.4f}  min={min(rvis):.4f}  "
              f"final={rvis[-1]:.4f}  drop={rvis[0]-min(rvis):.4f}  "
              f"(n_steps={len(rvis)})")
        print(f"  [{side}] EMA:   init={ema[0]:.4f}  final={ema[-1]:.4f}")
    else:
        print(f"  [{side}] no rvis trajectory (ProGuard not enabled)")
PYEOF
