#!/usr/bin/env bash
# CoT-Faith Stage 2 -- ISOLATION TEST: ecot-bridge alone, one process, one
# GPU, no concurrency.
#
# Why this exists. Two full Stage 2 attempts (cbe2xdc9qt on aws_2/A100,
# 76twybafqw on aws_10/B200) each ran 3 checkpoints (ecot-bridge, lora-r32,
# no-cot) concurrently across 3 GPUs and died identically both times: model
# and dataset load cleanly, then silence -- no Python exception (the
# self-generation call is wrapped in try/except and neither run printed
# anything from it), no OOM (host free memory was 800+GB throughout both
# times), and the parent shell's `wait` on each child PID returned -1, which
# is not a valid POSIX exit status and means bash itself lost track of what
# happened to the child, not just that the child raised an error.
#
# ecot-bridge needs no S3 credentials (public HF checkpoint), so this needs
# no credential-rendering step and no separate submit_*.sh wrapper -- submit
# with a plain `bolt task submit --config ... --tar .`.
#
# What a result here tells us:
#   - Survives to a full report  -> concurrency was the trigger. Re-architect
#     the full job to serialize the 3 checkpoints (same total GPU-hours,
#     slower wall clock, but correct) rather than running them concurrently.
#   - Dies the same way, alone   -> concurrency is ruled out. The bug is in
#     gen_cot()/model.generate() itself under some condition this isolation
#     test still reproduces, which is a worse answer but a necessary one
#     to have before trying anything else.
# Either way this is the informative experiment, not a guess.
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1  # consistency with the other two scripts; see
                                   # bolt/run_stage1_continuous.sh's comment. This
                                   # job runs only one process, so it isn't exposed
                                   # to the concurrent-.pyc-write race that motivated
                                   # this, but there's no reason to leave it off.
export COTFAITH_CRASH_LOG="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-stage2-isolation/crash.log"
nvidia-smi -L || true

# Same fast compatibility preflight as run_stage2_selfgen.sh -- this job also
# targets aws_10/B200.
python - <<'PY'
import sys
import torch
print(f"[preflight] torch={torch.__version__} cuda_build={torch.version.cuda}")
if not torch.cuda.is_available():
    print("[preflight] FATAL: torch.cuda.is_available() is False.")
    sys.exit(1)
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"[preflight] device 0: {name}, compute capability sm_{cap[0]}{cap[1]}")
try:
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    c = (a @ b).sum().item()
    torch.cuda.synchronize()
    print(f"[preflight] bf16 matmul on GPU 0 OK (sum={c:.2f})")
except Exception as e:
    print(f"[preflight] FATAL: matmul on GPU 0 raised {type(e).__name__}: {e}")
    sys.exit(1)
PY

OUT_ROOT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-stage2-isolation"
# The actual invocation below redirects into "$OUT_ROOT/ecot-bridge/run.log"
# and passes --out "$OUT_ROOT/ecot-bridge" -- `>` does not create parent
# directories the way `mkdir -p` does, so p6vdguz85g's shell redirect failed
# immediately with "No such file or directory" before the diagnostic this
# job exists to run ever started. Create the subdirectory too, not just
# $OUT_ROOT itself.
mkdir -p "$OUT_ROOT/ecot-bridge"

# Identical TF/tfds block to run_stage2_selfgen.sh -- see that script's
# comment for why it is duplicated rather than shared.
pip install "dm-tree" "protobuf>=3.20,<5" "promise" "dill" "etils[epath]" \
            "toml" "termcolor" "tqdm" "click" || true
pip install "tensorflow-cpu==2.15.1" --no-deps \
    || pip install "tensorflow==2.15.1" --no-deps || true
pip install "absl-py" "astunparse" "flatbuffers" "gast" "google-pasta" \
            "grpcio" "h5py" "libclang" "ml-dtypes==0.2.0" "opt-einsum" \
            "packaging" "six" "wrapt" "termcolor" "typing-extensions" \
            "tensorboard==2.15.2" "keras==2.15.0" "tensorflow-estimator==2.15.0" || true
pip install "tensorflow_datasets==4.9.3" "tensorflow_metadata==1.15.0" \
            --force-reinstall --no-deps || true

python - <<'PY'
from sharpguard.hf_retry import snapshot_with_retry
p = snapshot_with_retry(repo_id="Embodied-CoT/embodied_features_and_demos_libero",
                         repo_type="dataset")
print(f"[stage2-iso] LIBERO dataset snapshot warm at {p}")
PY

echo "[stage2-iso] mem before launch:"
free -h | sed 's/^/[mem] /' || true

# Foreground, not backgrounded: no `&`, no wait_all, so $? below is bash's
# own direct exit status for this one process -- not filtered through the
# `wait $PID` path that returned an uninterpretable -1 both previous times.
# If this dies the same silent way, that -1 was specific to backgrounding a
# child and waiting on it later, not to the workload itself, which narrows
# things further regardless of which of the two outcomes above we get.
set +e
python -u experiments/cotfaith_stage2_selfgen.py \
    --ckpt-path "Embodied-CoT/ecot-openvla-7b-bridge" \
    --checkpoint-label "ecot-bridge-isolation" \
    --out "$OUT_ROOT/ecot-bridge" \
    --n-samples 100 --n-samples-addon 40 --seed 0 \
    --families subject_swap,direction_flip,gripper_flip,location_swap,verb_swap,negation,adversarial_plausible,selfsplice_control,syntactic_scramble,cross_task_swap,paraphrase_null,bbox_jitter_null,instr_random_sub \
    --timestep-families paraphrase_null,syntactic_scramble,direction_flip \
    --timestep-fracs 0.333333,0.666667 \
    --dtype bfloat16 --kl-eps 1e-6 --threshold 0.05 --max-new-tokens 320 \
    --dataset-repo Embodied-CoT/embodied_features_and_demos_libero \
    --tfds-subdir libero_lm_90/1.0.0 --reasoning-json libero_reasonings.json \
    > "$OUT_ROOT/ecot-bridge/run.log" 2>&1
RC=$?
set -e

echo "[stage2-iso] ecot-bridge exited $RC$( [ "$RC" -ge 128 ] && echo " (signal $((RC - 128)))")"
echo "[stage2-iso] mem after:"
free -h | sed 's/^/[mem] /' || true

if [ "$RC" -ne 0 ]; then
    echo "[stage2-iso] === last 60 lines of run.log ==="
    tail -60 "$OUT_ROOT/ecot-bridge/run.log" || true
    if [ -f "$COTFAITH_CRASH_LOG" ]; then
        echo "[stage2-iso] === crash.log ==="
        cat "$COTFAITH_CRASH_LOG" || true
    else
        echo "[stage2-iso] no crash.log -- died without Python catching anything"
    fi
    exit "$RC"
fi

echo "[stage2-iso] SUCCESS -- ecot-bridge completed alone. Concurrency is the leading suspect."
