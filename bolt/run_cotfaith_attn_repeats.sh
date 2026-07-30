#!/usr/bin/env bash
# Attention probe (P1) with error bars and full network depth.
#
# Addresses two reviewer criticals on the attention side:
#   (1) "Attention numbers are single-run" — every alpha(cot) in the paper came
#       from one measurement, and the only noise estimate we had was an
#       accident (two r=32 runs that happened to exist, 1.45pp apart, i.e. 63%
#       of the entire 2.30pp cluster spread that F1 and F3 hinge on). This runs
#       N_REPEATS independent sample draws so mean+-std is measured, not guessed.
#   (2) "layers 0-3 only" — the 4-bucket decomposition was reported from the
#       first four LLaMA blocks on the argument that cross-modal mixing happens
#       early. That is an assumption, not a result. RVIS_LAYERS_FULL sweeps all
#       32 blocks so the paper can state whether the bucket ordering is a
#       property of the network or of the layer subset we picked.
#
# Both variants are run for the SAME checkpoint and sample budget, so the
# early-layer numbers already in the paper remain directly comparable.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# tf/tfds deps (LIBERO samples come from a TFDS-format HF dataset).
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

CKPT="${CKPT_HF_ID:?CKPT_HF_ID must be set}"
N_SAMPLES="${N_SAMPLES:-100}"
N_REPEATS="${N_REPEATS:-3}"
LAYERS_EARLY="${RVIS_LAYERS:-0,1,2,3}"
LAYERS_FULL="${RVIS_LAYERS_FULL:-$(seq -s, 0 31)}"

BASE_OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-attn-repeats"
mkdir -p "$BASE_OUT"

# Seeds 0..N_REPEATS-1. cotfaith_rvis.py uses seed=0 as the deterministic
# sample order and seed>0 to reshuffle the TFDS file order, so the spread
# across seeds is the sampling variance a reader needs in order to judge
# whether a 0.5pp between-model gap means anything.
for depth in early full; do
    if [ "$depth" = "early" ]; then LAYERS="$LAYERS_EARLY"; else LAYERS="$LAYERS_FULL"; fi
    for seed in $(seq 0 $((N_REPEATS - 1))); do
        OUT="$BASE_OUT/${depth}_seed${seed}"
        mkdir -p "$OUT"
        echo "===== attention: depth=$depth seed=$seed layers=$LAYERS ====="
        python experiments/cotfaith_rvis.py \
            --ckpt-path    "$CKPT" \
            --out          "$OUT" \
            --n-samples    "$N_SAMPLES" \
            --seed         "$seed" \
            --rvis-layers  "$LAYERS" \
            --dtype        "${DTYPE:-bfloat16}" \
        || echo "[warn] depth=$depth seed=$seed FAILED; continuing so the "\
                "surviving repeats still yield a std"
    done
done

# Aggregate mean+-std across seeds, per depth, without needing a local rerun.
python - <<'PY'
import json, os, glob, statistics as st
base = os.environ.get("BOLT_ARTIFACT_DIR", "./artifacts") + "/cotfaith-attn-repeats"
summary = {}
for depth in ("early", "full"):
    runs = sorted(glob.glob(f"{base}/{depth}_seed*/rvis_cot_report.json"))
    buckets = {"action->cot": [], "action->visual": [],
               "action->instr": [], "action->action_prev": []}
    for r in runs:
        agg = json.load(open(r))["aggregate"]
        for k in buckets:
            v = agg.get(k, {}).get("mean")
            if v is not None:
                buckets[k].append(v)
    summary[depth] = {
        "n_runs": len(runs),
        "runs": runs,
        **{k: {"mean": (st.mean(v) if v else None),
               "std": (st.stdev(v) if len(v) > 1 else 0.0 if v else None),
               "values": v}
           for k, v in buckets.items()},
    }
out = f"{base}/attn_repeats_summary.json"
json.dump(summary, open(out, "w"), indent=2)
print(json.dumps(summary, indent=2)[:4000])
print("wrote", out)
PY

echo "==== Done ===="
exit 0
