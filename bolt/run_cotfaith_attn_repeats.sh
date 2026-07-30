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

# Preflight: abort in seconds if the GPU cannot actually run a kernel, instead
# of burning the timeout and reporting "COMPLETED" with an empty report. A run
# on a p6-b200 node did exactly that -- torch in this image is built for
# sm_50..sm_90, the B200 is sm_100, so every launch raised "no kernel image is
# available for execution on the device" while the job still exited 0.
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("[preflight] FATAL: no CUDA device visible")
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
arches = torch.cuda.get_arch_list()
print(f"[preflight] device={name} capability=sm_{cap[0]}{cap[1]} torch={torch.__version__}")
print(f"[preflight] torch was built for: {arches}")
try:
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    _ = (a @ a).float().sum().item()
    torch.cuda.synchronize()
except Exception as e:
    sys.exit(f"[preflight] FATAL: a trivial bf16 matmul failed on {name} "
             f"(sm_{cap[0]}{cap[1]} vs built-for {arches}): {e}")
print("[preflight] OK")
PY

N_OK_BEFORE=0
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

# Aggregate mean+-std across seeds, per depth. Exits non-zero unless every
# depth has at least 2 runs that actually scored samples -- a report file can
# exist with n_samples=0 when the per-sample loop caught an exception on every
# sample, and treating that as success is how a totally failed run got marked
# COMPLETED once already.
python - <<'PY'
import json, os, glob, sys, statistics as st
base = os.environ.get("BOLT_ARTIFACT_DIR", "./artifacts") + "/cotfaith-attn-repeats"
summary, problems = {}, []
for depth in ("early", "full"):
    runs = sorted(glob.glob(f"{base}/{depth}_seed*/rvis_cot_report.json"))
    buckets = {"action->cot": [], "action->visual": [],
               "action->instr": [], "action->action_prev": []}
    n_nonempty = 0
    for r in runs:
        rep = json.load(open(r))
        if not rep.get("n_samples"):
            problems.append(f"{r}: n_samples=0 (every sample raised)")
            continue
        n_nonempty += 1
        for k in buckets:
            v = rep["aggregate"].get(k, {}).get("mean")
            if v is not None:
                buckets[k].append(v)
    summary[depth] = {
        "n_report_files": len(runs),
        "n_runs_with_samples": n_nonempty,
        "runs": runs,
        **{k: {"mean": (st.mean(v) if v else None),
               "std": (st.stdev(v) if len(v) > 1 else None),
               "values": v}
           for k, v in buckets.items()},
    }
    if n_nonempty < 2:
        problems.append(f"depth={depth}: only {n_nonempty} run(s) produced "
                        f"samples; a std needs >=2")
out = f"{base}/attn_repeats_summary.json"
json.dump(summary, open(out, "w"), indent=2)
print(json.dumps(summary, indent=2)[:4000])
print("wrote", out)
if problems:
    print("\n[FATAL] this run does not support the error bars it was launched for:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("[ok] every depth has >=2 runs with samples")
PY

echo "==== Done ===="
