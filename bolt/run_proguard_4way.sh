#!/usr/bin/env bash
# ProGuard 4-way comparison: control / EMA / absolute / CUSUM
#
# Each mode trained against the same Goal-T backdoor on its own GPU.
# Staggered launches (90s) to avoid the libero/EGL init race we hit in
# t3e38hgze5. Each subshell pins MUJOCO_EGL_DEVICE_ID = CUDA_VISIBLE_DEVICES
# (robosuite literal-string assertion, see commit 446852c).

set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_BASE="${BOLT_ARTIFACT_DIR:-./artifacts}/proguard-4way"
mkdir -p "$OUT_BASE/control" "$OUT_BASE/ema" "$OUT_BASE/absolute" "$OUT_BASE/cusum"

nvidia-smi -L || true

build_cmd() {
    local mode="$1"
    local lam="$2"
    local out_dir="$3"
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
    --proguard-mode       "$mode" \
    --proguard-alpha      "$PROGUARD_ALPHA" \
    --proguard-tau        "$PROGUARD_TAU" \
    --proguard-abs-tau    "$PROGUARD_ABS_TAU" \
    --proguard-cusum-k    "$PROGUARD_CUSUM_K" \
    --proguard-cusum-h    "$PROGUARD_CUSUM_H" \
    --proguard-cusum-beta "$PROGUARD_CUSUM_BETA" \
    --proguard-layers     "$PROGUARD_LAYERS" \
    --proguard-apply-to   "$PROGUARD_APPLY_TO" \
    --libero-sim-eval \
    --use-libero-collect \
    --libero-collect-eps "$LIBERO_COLLECT_EPS" \
    --libero-collect-steps "$LIBERO_COLLECT_STEPS"
EOF
}

launch_side() {
    local gpu="$1"
    local mode="$2"
    local lam="$3"
    local label="$4"
    local out_dir="$OUT_BASE/$label"
    local cmd
    cmd="$(build_cmd "$mode" "$lam" "$out_dir")"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export MUJOCO_EGL_DEVICE_ID="$gpu"
        export TOKENIZERS_PARALLELISM=false
        bash -c "$cmd" 2>&1 | tee "$OUT_BASE/${label}.log"
    ) &
}

# Control: lam=0 + cusum mode (so hooks attach for measurement) but lam=0 → no penalty
launch_side 0 cusum    0.0  control
CONTROL_PID=$!
sleep 90

launch_side 1 ema      "$PROGUARD_LAMBDA"  ema
EMA_PID=$!
sleep 90

launch_side 2 absolute "$PROGUARD_LAMBDA"  absolute
ABS_PID=$!
sleep 90

launch_side 3 cusum    "$PROGUARD_LAMBDA"  cusum
CUSUM_PID=$!

ANY_FAILED=0
for pid in $CONTROL_PID $EMA_PID $ABS_PID $CUSUM_PID; do
    if ! wait $pid; then
        echo "[run] pid $pid FAILED"
        ANY_FAILED=1
    fi
done

if [ $ANY_FAILED -ne 0 ]; then
    for log in control ema absolute cusum; do
        echo "==== last 20 lines of $log.log ===="
        tail -20 "$OUT_BASE/$log.log"
    done
    exit 1
fi

echo ""
echo "==== 4-way side-by-side ===="
export OUT_BASE
python <<'PYEOF'
import json, os
from pathlib import Path

base = Path(os.environ["OUT_BASE"])
modes = ("control", "ema", "absolute", "cusum")

print(f"{'stage':<28s}" + "".join(f"{m:>20s}" for m in modes))
print("-" * (28 + 20 * 4))

results = {}
for m in modes:
    p = base / m / "results.json"
    if p.exists():
        results[m] = json.load(open(p))
    else:
        results[m] = None

# Find the stage keys from whichever results came through
ref = next((r for r in results.values() if r), None)
if ref is None:
    print("ALL FAILED")
else:
    for key in ref:
        if not isinstance(ref.get(key), dict):
            continue
        if "metrics" not in ref[key]:
            continue
        row = f"{key:<28s}"
        for m in modes:
            r = results[m]
            if not r or not isinstance(r.get(key), dict) or "metrics" not in r[key]:
                row += f"{'n/a':>20s}"
            else:
                mm = r[key]["metrics"]
                row += f"  SR={mm.get('SR', float('nan')):.3f} ASR={mm.get('ASR', float('nan')):.3f}"
        print(row)

print()
print("=== r_vis trajectories ===")
for m in modes:
    p = base / m / "rvis_trajectory_poisoned.json"
    if not p.exists():
        print(f"[{m:<9s}] no trajectory file")
        continue
    d = json.load(open(p))
    r = d["rvis_per_step"]
    print(f"[{m:<9s}] r_vis: init={r[0]:.4f}  min={min(r):.4f}  max={max(r):.4f}  "
          f"final={r[-1]:.4f}  drop={r[0]-min(r):+.4f}  (n={len(r)})")
    if "S_per_step" in d and d["S_per_step"]:
        S = d["S_per_step"]
        print(f"[{m:<9s}] CUSUM S: init={S[0]:.4f}  max={max(S):.4f}  final={S[-1]:.4f}")
    if "ema_per_step" in d and d["ema_per_step"]:
        E = d["ema_per_step"]
        print(f"[{m:<9s}] EMA: init={E[0]:.4f}  final={E[-1]:.4f}")
PYEOF
