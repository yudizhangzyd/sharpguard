#!/usr/bin/env bash
# ProGuard Goal-T sanity v3 — control (lambda=0, measure-only) + treatment (lambda=1) in parallel.
#
# Fixes from v2:
#   1. lambda=0 now also attaches hooks (measure-only mode) so control logs
#      its r_vis trajectory for direct comparison.
#   2. MUJOCO_EGL_DEVICE_ID=0 set inside each subshell. After CUDA_VISIBLE_DEVICES
#      masks down to one GPU, that GPU is index 0 from the process's POV; MuJoCo
#      was crashing on GPU 1 because it inherited EGL_DEVICE_ID=1 from the
#      outer env.
#   3. Python summary script uses isinstance() check; old version did
#      `"metrics" in c.get(k, {})` which threw TypeError when c[k] was a float
#      (e.g. params_billion).

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

# Launch control (lambda=0, measure-only) on GPU 0.
# robosuite asserts MUJOCO_EGL_DEVICE_ID is among the CUDA_VISIBLE_DEVICES
# values *as written*, so both env vars must be set to the same physical
# device index BEFORE robosuite imports.
CONTROL_CMD="$(build_cmd 0.0 $OUT_BASE/control)"
(
    export CUDA_VISIBLE_DEVICES=0
    export MUJOCO_EGL_DEVICE_ID=0
    export TOKENIZERS_PARALLELISM=false
    bash -c "$CONTROL_CMD" 2>&1 | tee "$OUT_BASE/control.log"
) &
CONTROL_PID=$!

# Stagger by 90 seconds so the two processes don't race on EGL display
# initialization or HuggingFace cache writes.
echo "[sanity] sleeping 90s before launching treatment to avoid EGL race"
sleep 90

# Launch treatment (lambda=1) on GPU 1.
# robosuite's check is on the LITERAL CUDA_VISIBLE_DEVICES string, not
# the post-mask device list, so MUJOCO_EGL_DEVICE_ID must be 1 here
# (matching the physical GPU index visible to the child process).
TREATMENT_CMD="$(build_cmd 1.0 $OUT_BASE/treatment)"
(
    export CUDA_VISIBLE_DEVICES=1
    export MUJOCO_EGL_DEVICE_ID=1
    export TOKENIZERS_PARALLELISM=false
    bash -c "$TREATMENT_CMD" 2>&1 | tee "$OUT_BASE/treatment.log"
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
        if not isinstance(c.get(k), dict):
            continue
        if "metrics" not in c[k]:
            continue
        mc = c[k]["metrics"]
        tk = t.get(k)
        if not isinstance(tk, dict) or "metrics" not in tk:
            continue
        mt = tk["metrics"]
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
        print(f"  [{side:<9s}] r_vis: init={rvis[0]:.4f}  min={min(rvis):.4f}  "
              f"max={max(rvis):.4f}  final={rvis[-1]:.4f}  "
              f"drop={rvis[0]-min(rvis):+.4f}  (n_steps={len(rvis)})")
        print(f"  [{side:<9s}] EMA:   init={ema[0]:.4f}  final={ema[-1]:.4f}")
    else:
        print(f"  [{side:<9s}] no rvis trajectory file")
PYEOF
