#!/usr/bin/env bash
# CoT-Faith Stage 2: score edits against the checkpoint's OWN self-generated
# CoT (no ground-truth annotation anywhere in this job's prompts).
#
# One phase, one transformers pin: all 3 checkpoints here are OpenVLA-family
# (ecot-bridge, lora-r32, no-cot), so unlike bolt/run_stage1_continuous.sh
# there is no in-command transformers upgrade and no second phase --
# DeepThinkVLA is out of scope for this job (see
# experiments/cotfaith_stage2_selfgen.py's module docstring, "EXCLUDED").
# There is also no riding-along judge pass; this job's own self-generation
# already dominates its compute budget (see the build report this script
# shipped with for the estimate).
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# See bolt/run_stage1_continuous.sh's comment on this exact line: gc9ykqissb
# showed the 3rd-launched of 3 concurrent OpenVLA-family processes hitting a
# transformers ImportError the other two didn't, right after a fresh
# --force-reinstall -- the profile of concurrent processes racing on
# first-ever bytecode-cache (.pyc) writes to the same shared site-packages.
# This job also launches 3 OpenVLA-family processes concurrently, so it is
# equally exposed; disabling bytecode writes removes the writer side of
# that race before it has a chance to happen here too.
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi -L || true

# Fast compatibility preflight, run BEFORE the ~17GB LIBERO dataset fetch and
# any model download: this job now targets aws_10 (B200, sm_100) on explicit
# instruction, reversing an earlier note that this docker image's torch build
# only supported up to sm_90 (A100). If that note is still true, torch will
# either fail to see the device or fail on the first real kernel launch --
# either way, better to learn that in the next 30 seconds than after another
# 10 minutes of downloads land us back at the same silent death this is
# responding to.
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

OUT_ROOT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-stage2-selfgen"
mkdir -p "$OUT_ROOT"

# TF/tfds for LIBERO sample loading -- identical pip block to
# bolt/run_stage1_continuous.sh (both load samples through
# experiments/cotfaith_edit.py's load_libero_samples(), which needs
# tensorflow_datasets); kept in lockstep with that script on purpose rather
# than factored into a shared file, so a future edit to one is not silently
# assumed to also apply to the other.
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

# Pre-fetch the LIBERO dataset snapshot ONCE, sequentially, before any
# parallel checkpoint process starts -- same reasoning as
# bolt/run_stage1_continuous.sh: 3 processes racing to populate the same
# $HF_HOME cache concurrently is an untested interaction not worth risking
# when warming it once, here, costs nothing extra.
python - <<'PY'
from sharpguard.hf_retry import snapshot_with_retry
import os
p = snapshot_with_retry(repo_id=os.environ.get("DATASET_REPO",
    "Embodied-CoT/embodied_features_and_demos_libero"), repo_type="dataset")
print(f"[stage2] LIBERO dataset snapshot warm at {p}")
PY

COMMON_ARGS=(
  --n-samples          "${N_SAMPLES:-100}"
  --n-samples-addon    "${N_SAMPLES_ADDON:-40}"
  --seed               "${SEED:-0}"
  --families           "${FAMILIES:-subject_swap,direction_flip,gripper_flip,location_swap,verb_swap,negation,adversarial_plausible,selfsplice_control,syntactic_scramble,cross_task_swap,paraphrase_null,bbox_jitter_null,instr_random_sub}"
  --timestep-families  "${TIMESTEP_FAMILIES:-paraphrase_null,syntactic_scramble,direction_flip}"
  --timestep-fracs     "${TIMESTEP_FRACS:-0.333333,0.666667}"
  --dtype              "${DTYPE:-bfloat16}"
  --kl-eps             "${KL_EPS:-1e-6}"
  --threshold          "${THRESHOLD:-0.05}"
  --max-new-tokens     "${MAX_NEW_TOKENS:-320}"
  --dataset-repo       "${DATASET_REPO:-Embodied-CoT/embodied_features_and_demos_libero}"
  --tfds-subdir        "${TFDS_SUBDIR:-libero_lm_90/1.0.0}"
  --reasoning-json     "${REASONING_JSON:-libero_reasonings.json}"
)

# `free -h` alone reports the HOST's memory in most container runtimes,
# not the cgroup limit actually enforced on this task -- so "800GB free"
# from the earlier failed run is not proof the container itself had that
# much headroom. This prints both; cgroup v2 exposes memory.max/
# memory.current directly, v1 the analogous memory.limit_in_bytes/usage.
mem_snapshot() {
    local tag="$1"
    free -h | sed "s/^/[$tag] /" || true
    if [ -f /sys/fs/cgroup/memory.max ]; then
        echo "[$tag] cgroup v2: max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null) " \
             "current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null)"
    elif [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
        echo "[$tag] cgroup v1: limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null) " \
             "usage=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null)"
    else
        echo "[$tag] no cgroup memory file found at either v2 or v1 path"
    fi
}

run_one() {
    # run_one <label> <ckpt-path> -- FOREGROUND, not backgrounded. p6vdguz85g
    # (ecot-bridge alone, no concurrency) completed cleanly with a full
    # report; cbe2xdc9qt and 76twybafqw (the same 3 checkpoints launched
    # concurrently across 3 GPUs) both died silently with no Python
    # traceback and an uninterpretable bash wait status. That is now good
    # enough evidence to stop trying to make 3-way concurrency work and
    # just not do it: same total GPU-time, ~3x the wall clock, but every
    # checkpoint gets the isolated conditions that are the one config
    # confirmed to work. Running in the foreground also means $? below is
    # bash's own direct exit status, not filtered through a `wait $PID`
    # path that returned -1 both times this ran concurrently.
    local label="$1" ckpt="$2"
    local odir="$OUT_ROOT/$label"
    mkdir -p "$odir"
    echo "[stage2] running $label alone (ckpt=$ckpt)"
    mem_snapshot mem
    set +e
    COTFAITH_CRASH_LOG="$odir/crash.log" \
        python -u experiments/cotfaith_stage2_selfgen.py \
        --ckpt-path "$ckpt" \
        --checkpoint-label "$label" \
        --out "$odir" \
        "${COMMON_ARGS[@]}" \
        > "$odir/run.log" 2>&1
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "[stage2] $label exited $rc$( [ "$rc" -ge 128 ] && echo " (signal $((rc - 128)))")"
        mem_snapshot mem-at-failure
    fi
    return "$rc"
}

echo "[stage2] === ECoT-bridge + LoRA r32 + no-CoT ablation, run one at a time (see run_one's comment for why) ==="

FAILED=""
if run_one "ecot-bridge" "${ECOT_BRIDGE_HF_ID:-Embodied-CoT/ecot-openvla-7b-bridge}"; then
    :
else
    FAILED="$FAILED ecot-bridge"
fi

# lora-r32 / no-cot come from this project's own training tasks' S3 artifacts,
# not an HF repo -- same fetch pattern as bolt/run_stage1_continuous.sh's
# fetch_lora(), generalized to a SECOND, differently-scoped credential pair
# for the no-cot ablation. Stage 1 never fetches this checkpoint, so these
# credentials are new to this job -- see
# experiments/cotfaith_stage2_selfgen.py's KNOWN_CHECKPOINTS docstring note
# and bolt/submit_stage2_selfgen.sh, which renders both pairs.
fetch_ckpt() {
    local task_id="$1" akid="$2" skey="$3" dest="$4"
    mkdir -p "$dest"
    if [ -f "$dest/config.json" ]; then return 0; fi
    if [ -z "$akid" ] || [ -z "$skey" ]; then
        echo "[stage2] FATAL: no S3 credentials for task $task_id. Issue with:"
        echo "  eval \"\$(bolt task get-credentials $task_id --expires-in-seconds 129600)\""
        echo "  and render that pair's AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY into"
        echo "  this config's placeholders before submit -- see"
        echo "  bolt/submit_stage2_selfgen.sh."
        return 1
    fi
    which aws >/dev/null 2>&1 || pip install --quiet awscli || true
    local url="s3://bolt-prod-2702150980/tasks/$task_id/artifacts/cotfaith-train/merged_model"
    # set +x for exactly the commands that interpolate $akid/$skey: `set -x`
    # traces the expanded command line, not the variable reference, so it
    # printed both credentials in cleartext into this job's own stdout log
    # (and from there into anyone's terminal who runs `bolt task logs`) the
    # first time this ran. Scoped narrowly -- the surrounding echo/pip/test
    # lines carry no secret and stay traced.
    set +x
    AWS_ACCESS_KEY_ID="$akid" AWS_SECRET_ACCESS_KEY="$skey" \
        aws s3 sync "$url" "$dest" --endpoint-url "${S3_ENDPOINT_URL:-https://conductor.data.apple.com}" --quiet
    sync_rc=$?
    set -x
    if [ "$sync_rc" -ne 0 ]; then
        echo "[stage2] aws s3 sync failed for $task_id -- trying s5cmd"
        pip install --quiet s5cmd || true
        set +x
        AWS_ACCESS_KEY_ID="$akid" AWS_SECRET_ACCESS_KEY="$skey" S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-https://conductor.data.apple.com}" \
            s5cmd cp "$url/*" "$dest/"
        s5cmd_rc=$?
        set -x
        if [ "$s5cmd_rc" -ne 0 ]; then
            echo "[stage2] FATAL: cannot fetch $url (task $task_id)"
            return 1
        fi
    fi
    [ -f "$dest/config.json" ] || { echo "[stage2] FATAL: no config.json synced for $task_id"; return 1; }
}

LORA_R32_LOCAL=/tmp/cotfaith_ckpt_r32
NOCOT_LOCAL=/tmp/cotfaith_ckpt_nocot
# set +x around the CALL, not just fetch_ckpt's internal body -- see
# bolt/run_stage1_continuous.sh's matching comment. `-x` traces a command
# line after parameter expansion, so `fetch_ckpt TASK $AKID $SKEY DEST`
# printed both credentials in cleartext as positional arguments even after
# the earlier fix scoped `set +x` to the assignments inside the function.
set +x
if fetch_ckpt "${LORA_R32_CKPT_TASK_ID:-bcihypv3gu}" "${LORA_R32_AWS_ACCESS_KEY_ID:-}" \
              "${LORA_R32_AWS_SECRET_ACCESS_KEY:-}" "$LORA_R32_LOCAL"; then
    FETCH_R32_RC=0
else
    FETCH_R32_RC=$?
fi
set -x
if [ "$FETCH_R32_RC" -eq 0 ]; then
    if run_one "lora-r32" "$LORA_R32_LOCAL"; then
        :
    else
        FAILED="$FAILED lora-r32"
    fi
else
    echo "[stage2] SKIPPING lora-r32 -- credential fetch failed"
    FAILED="$FAILED lora-r32"
fi
set +x
if fetch_ckpt "${NOCOT_CKPT_TASK_ID:-a8eegzcg4r}" "${NOCOT_AWS_ACCESS_KEY_ID:-}" \
              "${NOCOT_AWS_SECRET_ACCESS_KEY:-}" "$NOCOT_LOCAL"; then
    FETCH_NOCOT_RC=0
else
    FETCH_NOCOT_RC=$?
fi
set -x
if [ "$FETCH_NOCOT_RC" -eq 0 ]; then
    if run_one "no-cot" "$NOCOT_LOCAL"; then
        :
    else
        FAILED="$FAILED no-cot"
    fi
else
    echo "[stage2] SKIPPING no-cot -- credential fetch failed"
    FAILED="$FAILED no-cot"
fi

echo ""
echo "==== Stage 2 selfgen: summary ===="
for d in "$OUT_ROOT"/*/; do
    label="$(basename "$d")"
    rep="$d/stage2_selfgen_report.json"
    if [ -f "$rep" ]; then
        echo "--- $label ---"
        python - "$rep" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
agg = d.get("aggregate", {})
for fam, by_kind in agg.items():
    for kind, v in by_kind.items():
        n = v.get("n", 0)
        fr = v.get("faithful_rate") if n else None
        print(f"  {fam:22s} [{kind:5s}] n={n:3d} n_skipped={v.get('n_skipped', 0):3d} faithful_rate={fr}")
PY
    else
        echo "--- $label: NO REPORT (see $d/run.log) ---"
        if [ -f "$d/crash.log" ]; then
            echo "--- $label: crash.log found, tail follows ---"
            tail -40 "$d/crash.log" || true
        else
            echo "--- $label: no crash.log either -- died without Python ever catching it ---"
        fi
    fi
done

if [ -n "$(echo "$FAILED" | tr -d '[:space:]')" ]; then
    echo "[stage2] non-zero exit: failed jobs:$FAILED"
    exit 4
fi
echo "==== Done ===="
exit 0
