#!/usr/bin/env bash
# CoT-Faith Stage 1: continuous effect-size metric + policy-likelihood
# covariate, across all 6 leaderboard checkpoints in one job.
#
# Why one job runs two "phases": ECoT-bridge and the two LoRA fine-tunes are
# OpenVLA-family and need transformers==4.40.1 (processing_prismatic); the
# three DeepThinkVLA checkpoints need transformers==4.48.1 specifically
# (PaliGemma internals `_update_causal_mask` / `PALIGEMMA_INPUTS_DOCSTRING`
# that 4.5x removes -- see bolt/run_cotfaith_deepthink.sh, which pins the same
# way). The two pins cannot both be installed at once, so this script installs
# 4.40.1 (via setup-openvla.sh), runs the three OpenVLA-family checkpoints one
# at a time (see run_one_a's comment -- two full concurrent attempts had the
# 3rd-launched process die on a transformers ImportError every time, and an
# isolation test proved it alone, no concurrency, was not the problem), THEN
# upgrades transformers in place and runs the three DeepThinkVLA checkpoints
# in parallel across 3 GPUs (that phase has never shown the failure, so it
# stays concurrent). Every existing DeepThinkVLA bolt config already does the
# same setup_command + in-command upgrade; this just adds the OpenVLA-family
# phase before it in the same job instead of a separate one.
#
# The Mistral second-judge pass (task step: "fold in a second LLM judge ...
# riding along on the same job") does not care which transformers pin is
# active -- Mistral-7B-Instruct-v0.2 is a plain AutoModelForCausalLM, not
# pinned to either PaliGemma or processing_prismatic internals -- so it is
# launched once, in the background, at the very top, on its own GPU, and only
# waited on at the end.
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# gc9ykqissb: ecot-bridge and lora-r32 (launched 1st/2nd) both loaded fine;
# lora-r64 (launched 3rd, while the other two were already deep into their
# own transformers imports) hit the exact `is_hqq_available` ImportError
# again, despite the --force-reinstall fix confirmed OK by this same script
# earlier in the SAME run. Version and files were fine at that check; the
# most likely remaining explanation is concurrent first-ever bytecode-cache
# (.pyc) writes for the same shared site-packages colliding across the 3
# processes -- exactly the profile of "works for whichever import wins the
# race, breaks for whoever loses it." Disabling bytecode writes entirely
# removes the writer side of that race; the read-only cost (re-parsing .py
# source each launch) is negligible next to model loading and generation.
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi -L || true

OUT_ROOT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-stage1-continuous"
mkdir -p "$OUT_ROOT"

# TF/tfds for LIBERO sample loading -- mirrors bolt/run_cotfaith_edit_only.sh
# and bolt/run_cotfaith_deepthink.sh; both model families load samples through
# the same load_libero_samples() family, which needs tensorflow_datasets.
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
# parallel checkpoint process starts. load_libero_samples() calls
# snapshot_with_retry with no allow_patterns, i.e. the full ~17GB dataset
# repo (see experiments/cotfaith_edit.py / cotfaith_deepthink.py); six
# processes racing to populate the same $HF_HOME cache concurrently is an
# untested interaction this job does not need to risk when warming the cache
# once, here, costs nothing extra.
python - <<'PY'
from sharpguard.hf_retry import snapshot_with_retry
import os
p = snapshot_with_retry(repo_id=os.environ.get("DATASET_REPO",
    "Embodied-CoT/embodied_features_and_demos_libero"), repo_type="dataset")
print(f"[stage1] LIBERO dataset snapshot warm at {p}")
PY

COMMON_ARGS=(
  --n-samples    "${N_SAMPLES:-100}"
  --seed         "${SEED:-0}"
  --families     "${FAMILIES:-direction_flip,gripper_flip,verb_swap,negation,subject_swap,location_swap,adversarial_plausible,paraphrase_null,syntactic_scramble}"
  --dtype        "${DTYPE:-bfloat16}"
  --kl-eps       "${KL_EPS:-1e-6}"
  --dataset-repo "${DATASET_REPO:-Embodied-CoT/embodied_features_and_demos_libero}"
  --tfds-subdir  "${TFDS_SUBDIR:-libero_lm_90/1.0.0}"
  --reasoning-json "${REASONING_JSON:-libero_reasonings.json}"
)

declare -a PIDS=()
declare -a LABELS=()
declare -a ODIRS=()

launch() {
    # launch <gpu> <label> <ckpt-path> <model-family>
    local gpu="$1" label="$2" ckpt="$3" family="$4"
    local odir="$OUT_ROOT/$label"
    mkdir -p "$odir"
    rm -f "$odir/exit_code"
    echo "[stage1] launching $label on GPU $gpu (family=$family ckpt=$ckpt)"
    # Wrapped in its own subshell so it can capture and persist its OWN real
    # $? to a file, INSIDE the backgrounded process tree -- see wait_all's
    # comment for why this file, not wait's return value, is now the source
    # of truth for whether this checkpoint actually failed. `set +e` here is
    # scoped to this subshell only (subshell option changes never propagate
    # back to the parent), so the outer script's `set -e` is unaffected.
    (
        set +e
        CUDA_VISIBLE_DEVICES="$gpu" COTFAITH_CRASH_LOG="$odir/crash.log" \
            python experiments/cotfaith_stage1_continuous.py \
            --ckpt-path "$ckpt" \
            --model-family "$family" \
            --checkpoint-label "$label" \
            --out "$odir" \
            "${COMMON_ARGS[@]}" \
            > "$odir/run.log" 2>&1
        echo $? > "$odir/exit_code"
    ) &
    PIDS+=("$!")
    LABELS+=("$label")
    ODIRS+=("$odir")
}

wait_all() {
    # c4j4ber3e3's Phase B: this function's PREVIOUS version called `wait
    # "${PIDS[$i]}"` to block, then gave the exit_code file a 60s grace
    # period on the theory that wait blocks correctly and only ucuhd364s4's
    # `-1` STATUS was wrong. Both halves of that theory were wrong: deepthink-
    # base and deepthink-sft each took ~10-15 real minutes (see their own
    # run.log timestamps -- shard downloads alone ran 5-9 minutes), but
    # `wait` returned near-instantly every time, and this function's 60s
    # poll gave up and reported BOTH as failed while they were still
    # genuinely running -- they went on to finish successfully (real
    # exit_code=0, full reports) minutes after this function had already
    # given up on them. So: don't call `wait` at all, for blocking OR
    # status. Poll every PID's exit_code file together in one shared loop,
    # with a timeout generous enough for a real run (not a
    # file-not-flushed-yet grace period), so one slow checkpoint's file
    # showing up late doesn't cost the others any extra wait -- they're each
    # checked every tick regardless of which index is still pending.
    local n=${#PIDS[@]}
    local -a rcs=()
    local i elapsed=0
    local max_wait="${WAIT_ALL_MAX_SECONDS:-3600}"
    local remaining="$n"
    while [ "$remaining" -gt 0 ] && [ "$elapsed" -lt "$max_wait" ]; do
        remaining=0
        for i in "${!PIDS[@]}"; do
            [ -n "${rcs[$i]:-}" ] && continue
            if [ -f "${ODIRS[$i]}/exit_code" ]; then
                rcs[$i]="$(cat "${ODIRS[$i]}/exit_code")"
            else
                remaining=$((remaining + 1))
            fi
        done
        if [ "$remaining" -gt 0 ]; then
            sleep 10
            elapsed=$((elapsed + 10))
        fi
    done

    local failed=""
    for i in "${!PIDS[@]}"; do
        local rc="${rcs[$i]:-}"
        case "$rc" in
            ''|*[!0-9]*)
                # Either no exit_code file ever showed up in $max_wait
                # (deepthink-rl on c4j4ber3e3: stuck at "Downloading shards:
                # 0%" forever, a real hang, not a bash bookkeeping issue --
                # see this function's own module comment) or the file held
                # something non-numeric. `[ "$rc" -ne 0 ]` below throws a
                # bash runtime error (not a clean nonzero exit) on anything
                # non-numeric, which under this script's `set -e` would
                # abort the whole job instead of just flagging this one
                # checkpoint.
                echo "[wait_all] ${LABELS[$i]} (pid ${PIDS[$i]}): no valid exit_code" \
                     "after ${elapsed}s of polling (max ${max_wait}s) -- treating as failed"
                rc=1
                ;;
        esac
        if [ "$rc" -ne 0 ]; then
            # rc>=128 is bash's convention for "killed by signal rc-128" (137 =
            # SIGKILL, almost always the OOM killer on this cluster; 139 =
            # SIGSEGV, a native crash). Printing it is the only way to tell
            # "the process raised a caught, logged Python exception" apart
            # from "something outside Python's control ended it" after the
            # fact -- the run that motivated this, tur9u9jr5g, had neither in
            # its run.log and left both equally plausible for 20 minutes.
            echo "[wait_all] ${LABELS[$i]} (pid ${PIDS[$i]}) exited $rc$( \
                [ "$rc" -ge 128 ] && echo " (signal $((rc - 128)))")"
            failed="$failed ${LABELS[$i]}"
        fi
    done
    PIDS=(); LABELS=(); ODIRS=()
    echo "$failed"
}

# --- Mistral second-judge pass: background, own GPU, waited on at the end ---
# Re-scores the SAME held-out pairs the existing Qwen judge run used (same
# --file-base-from / --n-samples / --seed / --families as
# bolt/boltconfig-cotfaith-judge-edits.yaml), forcing Mistral instead of the
# tried-in-order fallback list, so the two judges' verdicts are comparable
# per-pair. cotfaith_judge_edits.py already judges every pair in BOTH
# presentation orders internally (order_agreement in its own report); no
# change to that script was needed or made.
JUDGE_OUT="$OUT_ROOT/judge-mistral"
mkdir -p "$JUDGE_OUT"
rm -f "$JUDGE_OUT/exit_code"
# Same exit_code-file pattern as launch()/wait_all() above, and for the same
# reason -- this job's own explicit `wait $JUDGE_PID` happened to return a
# real status on ucuhd364s4, but that was luck, not a guarantee: it is the
# exact same `wait <explicit PID>` construct that returned -1 for Phase B's
# three PIDs on that same run, just not (yet) observed failing here too.
(
    set +e
    CUDA_VISIBLE_DEVICES=7 python experiments/cotfaith_judge_edits.py \
        --out            "$JUDGE_OUT" \
        --judge-models   "${JUDGE_MODELS:-mistralai/Mistral-7B-Instruct-v0.2}" \
        --reasoning-repo "${JUDGE_REASONING_REPO:-Embodied-CoT/embodied_features_and_demos_libero}" \
        --reasoning-file "${JUDGE_REASONING_FILE:-libero_reasonings.json}" \
        --file-base-from "${JUDGE_FILE_BASE_FROM:-results_v2/canonical_runs/ecot_bridge_edit_seed1.json}" \
        --n-samples      "${JUDGE_N_SAMPLES:-40}" \
        --families       "${JUDGE_FAMILIES:-paraphrase_null,bbox_jitter_null,syntactic_scramble,direction_flip,negation,subject_swap,verb_swap,gripper_flip,location_swap,cross_task_swap,adversarial_plausible}" \
        --seed           "${JUDGE_SEED:-0}" \
        --dtype          "${DTYPE:-bfloat16}" \
        --max-new-tokens "${JUDGE_MAX_NEW_TOKENS:-160}" \
        --time-budget-h  "${JUDGE_TIME_BUDGET_H:-1.5}" \
        > "$JUDGE_OUT/run.log" 2>&1
    echo $? > "$JUDGE_OUT/exit_code"
) &
JUDGE_PID=$!
echo "[stage1] Mistral judge pass launched in background on GPU 7 (pid $JUDGE_PID)"

# --- mem diagnostics: same helper bolt/run_stage2_selfgen.sh uses ---
# `free -h` alone reports the HOST's memory in most container runtimes, not
# the cgroup limit actually enforced on this task. Kept in lockstep with
# that script's copy rather than factored out, same reasoning as the
# TF/tfds pip block above.
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

# ================= Phase A: OpenVLA-family (transformers 4.40.1) =================
# setup-openvla.sh already pinned 4.40.1 and asserted it at the end of setup,
# but that assertion runs once, minutes before this line -- the LIBERO
# dataset pre-fetch and a handful of `pip install ... || true` calls sit in
# between, any one of which pulling in a newer transformers as a transitive
# dependency would explain tur9u9jr5g's failure: all three Phase A processes
# died on the same `ImportError: cannot import name 'is_hqq_available' from
# transformers.utils` (a cache_utils.py from one transformers release next to
# a utils/__init__.py from another -- a version drifted mid-install, not
# simply "the wrong single version"), and setup's own preflight was clean.
# Re-asserting immediately before use, the same way Phase B already does
# below for 4.48.1, converts a silent multi-checkpoint failure discovered 40
# minutes later back into a fast, legible one.
echo "[stage1] === Phase A: ECoT-bridge + LoRA r32 + LoRA r64 ==="
# --force-reinstall: a plain `pip install transformers==4.40.1` is a no-op if
# pip already believes 4.40.1 is installed, which is exactly what happened
# here twice (tur9u9jr5g, 6ksmrs4uyf) -- this line ran, reported nothing to
# do, and the on-disk cache_utils.py stayed whatever it was before. See
# bolt/setup-openvla.sh's matching fix and comment for the full story.
pip install --force-reinstall --no-deps "transformers==4.40.1" "tokenizers==0.19.1" || true
python - <<'PY'
import sys, transformers
if not transformers.__version__.startswith("4.40"):
    print(f"[FATAL] Phase A needs transformers==4.40.x, found {transformers.__version__} "
          "-- something upgraded it between setup and here. Aborting before any "
          "checkpoint launches rather than failing all three 40 minutes in.")
    sys.exit(1)
# __version__ alone does not catch a stale cache_utils.py sitting under a
# correctly-reported version string -- tur9u9jr5g and 6ksmrs4uyf both passed
# this exact check and failed on the real import minutes later. Try the real
# import here instead, before any of the three checkpoints launch.
try:
    import transformers.modeling_utils  # noqa: F401
except Exception as e:
    print(f"[FATAL] transformers.__version__={transformers.__version__} but "
          f"`import transformers.modeling_utils` raised {type(e).__name__}: {e}. "
          "Aborting before any checkpoint launches.")
    sys.exit(1)
print(f"[ok] Phase A transformers={transformers.__version__}, modeling_utils import OK")
PY

run_one_a() {
    # run_one_a <label> <ckpt-path> -- FOREGROUND, one at a time, NOT via
    # launch()/wait_all() (Phase B below still uses those; it has never shown
    # this failure). Two full concurrent attempts (gc9ykqissb, yiertxskku)
    # had ecot-bridge and lora-r32 (launched 1st/2nd of 3) succeed
    # reproducibly every time while lora-r64 (launched 3rd) died identically
    # both times on a transformers ImportError, surviving two targeted fixes
    # in a row (force-reinstall, PYTHONDONTWRITEBYTECODE=1). run_stage1_r64_
    # isolation_test.sh then ran lora-r64 alone, no concurrency (task
    # 7nt9apejet), and it completed cleanly with a full 554-sample report --
    # so the failure was never about r64's own checkpoint/credentials/files,
    # only about being the 3rd of 3 concurrent launches. Exactly the evidence
    # that already justified this same change in bolt/run_stage2_selfgen.sh's
    # run_one(); see that function's comment. Same total GPU-time, ~3x the
    # wall clock, but every checkpoint gets the isolated conditions that are
    # the one config confirmed to work.
    local label="$1" ckpt="$2"
    local odir="$OUT_ROOT/$label"
    mkdir -p "$odir"
    echo "[stage1] running $label alone (ckpt=$ckpt)"
    mem_snapshot mem
    set +e
    CUDA_VISIBLE_DEVICES=0 COTFAITH_CRASH_LOG="$odir/crash.log" \
        python -u experiments/cotfaith_stage1_continuous.py \
        --ckpt-path "$ckpt" \
        --model-family openvla \
        --checkpoint-label "$label" \
        --out "$odir" \
        "${COMMON_ARGS[@]}" \
        > "$odir/run.log" 2>&1
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "[stage1] $label exited $rc$( [ "$rc" -ge 128 ] && echo " (signal $((rc - 128)))")"
        mem_snapshot mem-at-failure
    fi
    return "$rc"
}

FAILED_A=""
if run_one_a "ecot-bridge" "${ECOT_BRIDGE_HF_ID:-Embodied-CoT/ecot-openvla-7b-bridge}"; then
    :
else
    FAILED_A="$FAILED_A ecot-bridge"
fi

# LoRA r=32 / r=64 come from this project's own training tasks' S3 artifacts,
# not an HF repo -- same fetch as bolt/run_cotfaith_edit_s3ckpt.sh, done twice
# with two separately-scoped credential pairs (a token for one task prefix is
# AccessDenied on the other; see bolt/submit_edit13.sh). Each fetch runs
# before its `run_one_a` call, sequentially, on the main thread -- the two S3
# syncs are small next to the GPU work, so serializing them costs little and
# avoids two concurrent `aws s3 sync` calls fighting over installing awscli.
fetch_lora() {
    local task_id="$1" akid="$2" skey="$3" dest="$4"
    mkdir -p "$dest"
    if [ -f "$dest/config.json" ]; then return 0; fi
    if [ -z "$akid" ] || [ -z "$skey" ]; then
        echo "[stage1] FATAL: no S3 credentials for task $task_id. Issue with:"
        echo "  eval \"\$(bolt task get-credentials $task_id --expires-in-seconds 129600)\""
        echo "  and render AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY into this"
        echo "  config's placeholders before submit, the way bolt/submit_edit13.sh does."
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
        echo "[stage1] aws s3 sync failed for $task_id -- trying s5cmd"
        pip install --quiet s5cmd || true
        set +x
        AWS_ACCESS_KEY_ID="$akid" AWS_SECRET_ACCESS_KEY="$skey" S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-https://conductor.data.apple.com}" \
            s5cmd cp "$url/*" "$dest/"
        s5cmd_rc=$?
        set -x
        if [ "$s5cmd_rc" -ne 0 ]; then
            echo "[stage1] FATAL: cannot fetch $url (task $task_id)"
            return 1
        fi
    fi
    [ -f "$dest/config.json" ] || { echo "[stage1] FATAL: no config.json synced for $task_id"; return 1; }
}

LORA_R32_LOCAL=/tmp/cotfaith_ckpt_r32
LORA_R64_LOCAL=/tmp/cotfaith_ckpt_r64
# set +x around the CALL, not just fetch_lora's internal body: `-x` traces a
# command line after parameter expansion, so `fetch_lora TASK $AKID $SKEY
# DEST` printed both credentials in cleartext as positional arguments even
# after the earlier fix scoped `set +x` to the assignments inside the
# function -- caught only by chance while investigating lora-r64, in a
# `bolt task logs` capture that has since been deleted. `if cmd; then
# rc=0; else rc=$?; fi` (not `if ! cmd`) so set -e stays safe as elsewhere
# in this file, and so the fetch's own exit code, not the tracing state,
# decides which branch runs.
set +x
if fetch_lora "${LORA_R32_CKPT_TASK_ID:-bcihypv3gu}" "${LORA_R32_AWS_ACCESS_KEY_ID:-}" \
              "${LORA_R32_AWS_SECRET_ACCESS_KEY:-}" "$LORA_R32_LOCAL"; then
    FETCH_R32_RC=0
else
    FETCH_R32_RC=$?
fi
set -x
if [ "$FETCH_R32_RC" -eq 0 ]; then
    if run_one_a "lora-r32" "$LORA_R32_LOCAL"; then
        :
    else
        FAILED_A="$FAILED_A lora-r32"
    fi
else
    echo "[stage1] SKIPPING lora-r32 -- credential fetch failed"
    FAILED_A="$FAILED_A lora-r32"
fi
set +x
if fetch_lora "${LORA_R64_CKPT_TASK_ID:-26whnbbrmb}" "${LORA_R64_AWS_ACCESS_KEY_ID:-}" \
              "${LORA_R64_AWS_SECRET_ACCESS_KEY:-}" "$LORA_R64_LOCAL"; then
    FETCH_R64_RC=0
else
    FETCH_R64_RC=$?
fi
set -x
if [ "$FETCH_R64_RC" -eq 0 ]; then
    if run_one_a "lora-r64" "$LORA_R64_LOCAL"; then
        :
    else
        FAILED_A="$FAILED_A lora-r64"
    fi
else
    echo "[stage1] SKIPPING lora-r64 -- credential fetch failed"
    FAILED_A="$FAILED_A lora-r64"
fi

# ================= Phase B: DeepThinkVLA (transformers 4.48.1) =================
echo "[stage1] === Phase B: DeepThinkVLA base / SFT / RL ==="
# Deliberately no `|| true`: running DeepThinkVLA under the wrong pin decodes
# actions under a causal mask instead of the bidirectional one and produces
# plausible-looking, wrong numbers rather than an error. Same pin, same
# preflight assertion, as bolt/run_cotfaith_deepthink.sh.
pip install "transformers==4.48.1" "huggingface_hub>=0.26,<0.30"
python - <<'PY'
import transformers
assert transformers.__version__.startswith("4.48"), transformers.__version__
from transformers.models.paligemma.modeling_paligemma import PALIGEMMA_INPUTS_DOCSTRING
print("[stage1] preflight: transformers", transformers.__version__, "PaliGemma internals present")
PY

launch 0 "deepthink-base" "${DT_BASE_HF_ID:-yinchenghust/deepthinkvla_base}" deepthink
launch 1 "deepthink-sft"  "${DT_SFT_HF_ID:-yinchenghust/deepthinkvla_libero_cot_sft}" deepthink
launch 2 "deepthink-rl"   "${DT_RL_HF_ID:-yinchenghust/deepthinkvla_libero_cot_rl}" deepthink

FAILED_B=$(wait_all)

# --- judge pass ---
# Same fix as wait_all() -- see its comment. Don't call `wait $JUDGE_PID` at
# all: it's the same unreliable-blocking construct, and the previous 60s
# grace period was sized for "file not flushed yet", not for the judge's own
# real runtime (JUDGE_TIME_BUDGET_H, default 1.5h). Poll the exit_code file
# directly with a timeout comfortably above that budget.
JUDGE_FAILED=""
jelapsed=0
JUDGE_MAX_WAIT_SECONDS="${JUDGE_MAX_WAIT_SECONDS:-$(python3 -c "print(int((${JUDGE_TIME_BUDGET_H:-1.5} + 1) * 3600))")}"
while [ ! -f "$JUDGE_OUT/exit_code" ] && [ "$jelapsed" -lt "$JUDGE_MAX_WAIT_SECONDS" ]; do
    sleep 10
    jelapsed=$((jelapsed + 10))
done
if [ -f "$JUDGE_OUT/exit_code" ]; then
    JUDGE_RC="$(cat "$JUDGE_OUT/exit_code")"
    case "$JUDGE_RC" in
        ''|*[!0-9]*)
            echo "[stage1] judge-mistral: exit_code file contains non-numeric '$JUDGE_RC' -- treating as failed"
            JUDGE_RC=1
            ;;
    esac
else
    echo "[stage1] judge-mistral: no exit_code file after ${jelapsed}s of polling" \
         "(max ${JUDGE_MAX_WAIT_SECONDS}s) -- treating as failed"
    JUDGE_RC=1
fi
[ "$JUDGE_RC" -ne 0 ] && JUDGE_FAILED="judge-mistral"

echo ""
echo "==== Stage 1 continuous: summary ===="
for d in "$OUT_ROOT"/*/; do
    label="$(basename "$d")"
    rep="$d/stage1_continuous_report.json"
    if [ -f "$rep" ]; then
        echo "--- $label ---"
        python - "$rep" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
agg = d.get("aggregate", {})
for k, v in agg.items():
    n = v.get("n", 0)
    tv = (v.get("tv_mean") or {}).get("mean") if n else None
    print(f"  {k:24s} n={n:3d}  TVmean={tv}")
PY
    else
        echo "--- $label: NO REPORT (see $d/run.log) ---"
    fi
done
[ -f "$JUDGE_OUT/judge_report.json" ] && { echo "--- judge-mistral ---"; head -c 2000 "$JUDGE_OUT/judge_report.json"; echo; }

ALL_FAILED="$FAILED_A $FAILED_B $JUDGE_FAILED"
if [ -n "$(echo "$ALL_FAILED" | tr -d '[:space:]')" ]; then
    echo "[stage1] non-zero exit: failed jobs:$ALL_FAILED"
    exit 4
fi
echo "==== Done ===="
exit 0
