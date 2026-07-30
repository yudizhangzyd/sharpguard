#!/usr/bin/env bash
# Attention probe rehabilitation (R1 reviewer critical #5, items (c) and (6)).
#
# Two defects in the submitted P1 numbers:
#   (1) every alpha(cot) is a SINGLE run, so the 2.30pp within-ECoT cluster
#       spread that F1 and F3 both hinge on has no noise floor. The only
#       floor we could quote (1.45pp) came from two accidental r=32 runs
#       that differed in configuration, not from a controlled repeat.
#   (2) layers are fixed at 0-3 with a hand-waved justification ("cross-modal
#       mixing happens early"). A reader cannot tell whether 0-3 was chosen
#       because it is principled or because it gave the desired ordering.
#
# This job fixes both for one checkpoint: SEEDS x LAYER_SETS runs of the
# identical measurement. Seeds change the LIBERO sample draw, so the spread
# across seeds IS the error bar a benchmark owes its readers. Layer sets span
# the full 32-layer depth so the 0-3 choice can be checked, not trusted.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# tf/tfds deps (same chain as run_cotfaith_no_train.sh).
pip install "dm-tree" "protobuf>=3.20,<5" "promise" "dill" "etils[epath]" \
            "toml" "termcolor" "tqdm" "click" || true
pip install "tensorflow-cpu==2.15.1" --no-deps \
    || pip install "tensorflow==2.15.1" --no-deps || true
pip install "absl-py" "astunparse" "flatbuffers" "gast" "google-pasta" \
            "grpcio" "h5py" "libclang" "ml-dtypes==0.2.0" "opt-einsum" \
            "packaging" "six" "wrapt" "termcolor" "typing-extensions" \
            "tensorboard==2.15.2" "keras==2.15.0" "tensorflow-estimator==2.15.0" \
    || true
pip install "tensorflow_datasets==4.9.3" "tensorflow_metadata==1.15.0" \
            --force-reinstall --no-deps || true

OUT_ROOT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-attn-repeats"
mkdir -p "$OUT_ROOT"

CKPT="${CKPT_HF_ID:-Embodied-CoT/ecot-openvla-7b-bridge}"
N="${N_SAMPLES:-100}"
SEEDS="${SEEDS:-0 1 2}"

# ALL32 reproduces nothing in the paper -- it is the honest full-depth number.
# L0_3 reproduces the submitted value. The other three probe depth dependence.
ALL32="$(python -c 'print(",".join(str(i) for i in range(32)))')"
declare -A LAYER_SETS=(
  [all32]="$ALL32"
  [l0_3]="0,1,2,3"
  [l8_11]="8,9,10,11"
  [l16_19]="16,17,18,19"
  [l28_31]="28,29,30,31"
)

for seed in $SEEDS; do
  for tag in all32 l0_3 l8_11 l16_19 l28_31; do
    layers="${LAYER_SETS[$tag]}"
    OUT="$OUT_ROOT/seed${seed}_${tag}"
    if [ -f "$OUT/rvis_cot_report.json" ]; then
      echo "===== SKIP (exists) seed=$seed layers=$tag ====="
      continue
    fi
    echo "===== attention seed=$seed layers=$tag on $CKPT ====="
    mkdir -p "$OUT"
    # A single failing cell must not kill the other 14 -- we want partial
    # error bars over a crash with nothing to show.
    python experiments/cotfaith_rvis.py \
        --ckpt-path   "$CKPT" \
        --out         "$OUT" \
        --n-samples   "$N" \
        --seed        "$seed" \
        --rvis-layers "$layers" \
        --dtype       "${DTYPE:-bfloat16}" || echo "[warn] FAILED seed=$seed layers=$tag"
  done
done

echo ""
echo "===== aggregate mean+-std across seeds, per layer set ====="
python - <<'PYEOF'
import json, os, statistics as st
root = os.environ.get("BOLT_ARTIFACT_DIR", "./artifacts") + "/cotfaith-attn-repeats"
buckets = ["action->cot", "action->visual", "action->instr", "action->action_prev"]
summary = {}
for tag in ("all32", "l0_3", "l8_11", "l16_19", "l28_31"):
    per_seed = {}
    for seed in (0, 1, 2):
        f = os.path.join(root, f"seed{seed}_{tag}", "rvis_cot_report.json")
        if not os.path.exists(f):
            continue
        agg = json.load(open(f))["aggregate"]
        per_seed[seed] = {b: agg[b]["mean"] for b in buckets if b in agg}
    if not per_seed:
        continue
    entry = {"n_seeds": len(per_seed), "per_seed": per_seed}
    for b in buckets:
        vals = [v[b] for v in per_seed.values() if v.get(b) is not None]
        if not vals:
            continue
        entry[b] = {
            "mean": st.mean(vals),
            "std": st.pstdev(vals) if len(vals) > 1 else 0.0,
            "range_pp": (max(vals) - min(vals)) * 100,
        }
    summary[tag] = entry

out = os.path.join(root, "attn_repeats_summary.json")
json.dump(summary, open(out, "w"), indent=2)
for tag, e in summary.items():
    c = e.get("action->cot", {})
    print(f"  {tag:8s} n_seeds={e['n_seeds']} "
          f"alpha(cot)={c.get('mean', float('nan')):.4f} "
          f"+-{c.get('std', float('nan')):.4f} "
          f"range={c.get('range_pp', float('nan')):.2f}pp")
print(f"  summary -> {out}")
PYEOF

echo "==== Done ===="
exit 0
