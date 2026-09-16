#!/usr/bin/env bash
# CoT-Faith Stage 1 -- ISOLATION TEST: lora-r64 alone, one process, one GPU,
# no concurrency.
#
# Why this exists. Across two full Stage 1 attempts (gc9ykqissb, yiertxskku),
# ecot-bridge and lora-r32 (launched 1st/2nd of 3 concurrent OpenVLA-family
# processes) succeeded reproducibly -- byte-identical real reports both
# times -- while lora-r64 (launched 3rd) failed identically both times on
# `ImportError: cannot import name 'is_hqq_available' from
# transformers.utils`, DESPITE a --force-reinstall fix (yiertxskku) that a
# real-import preflight check confirmed had worked, immediately beforehand,
# in the same process tree. This determinism (not flakiness -- the same
# checkpoint fails the same way every time) ruled out two targeted fixes in
# a row (transformers force-reinstall, PYTHONDONTWRITEBYTECODE=1 against a
# suspected bytecode-cache write race). This test asks the one remaining
# question directly: is this about concurrency/launch-order at all, or is
# it specific to the lora-r64 checkpoint/credentials themselves?
#
#   - Succeeds alone  -> confirms a real, still-unexplained concurrency/
#     launch-order effect specific to a 3rd process, distinct from whatever
#     PYTHONDONTWRITEBYTECODE was supposed to fix. Serialize Phase A the
#     same way bolt/run_stage2_selfgen.sh was just re-architected to.
#   - Fails the same way, alone -> concurrency is fully ruled out for this
#     checkpoint specifically; the problem is in lora-r64's own fetched
#     files/credentials/config, which needs direct inspection next.
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export COTFAITH_CRASH_LOG="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-stage1-r64-isolation/crash.log"
nvidia-smi -L || true

# Same transformers pin + real-import preflight as run_stage1_continuous.sh's
# Phase A -- see that script's comment for why a version-string check alone
# is not enough.
pip install --force-reinstall --no-deps "transformers==4.40.1" "tokenizers==0.19.1" || true
python - <<'PY'
import sys, transformers
if not transformers.__version__.startswith("4.40"):
    print(f"[FATAL] needs transformers==4.40.x, found {transformers.__version__}.")
    sys.exit(1)
try:
    import transformers.modeling_utils  # noqa: F401
except Exception as e:
    print(f"[FATAL] transformers.__version__={transformers.__version__} but "
          f"`import transformers.modeling_utils` raised {type(e).__name__}: {e}")
    sys.exit(1)
print(f"[ok] transformers={transformers.__version__}, modeling_utils import OK")
PY

# TF/tfds for LIBERO sample loading -- verbatim from
# bolt/run_stage1_continuous.sh (load_libero_samples_openvla needs
# tensorflow_datasets). Omitted from this file's first version by mistake --
# gfpdnbz826 failed on `ModuleNotFoundError: No module named
# 'tensorflow_datasets'` before ever reaching the code this test exists to
# check, which is a different and much less interesting failure than the one
# under investigation. Copied verbatim this time rather than retyped, to
# stop stripping this file down "for isolation" into missing something
# unrelated to concurrency a second time.
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
print(f"[stage1-r64-iso] LIBERO dataset snapshot warm at {p}")
PY

OUT_ROOT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-stage1-r64-isolation"
mkdir -p "$OUT_ROOT/lora-r64"

echo "[stage1-r64-iso] mem before fetch:"
free -h | sed 's/^/[mem] /' || true

# Same fetch pattern as run_stage1_continuous.sh's fetch_lora(), with the
# same two credential-leak fixes applied this session: set +x around BOTH
# the internal AWS_ACCESS_KEY_ID/SECRET assignment and the call site itself
# (the call site leaked in cleartext earlier today until this was caught).
fetch_lora() {
    local task_id="$1" akid="$2" skey="$3" dest="$4"
    mkdir -p "$dest"
    if [ -f "$dest/config.json" ]; then return 0; fi
    if [ -z "$akid" ] || [ -z "$skey" ]; then
        echo "[stage1-r64-iso] FATAL: no S3 credentials for task $task_id."
        return 1
    fi
    which aws >/dev/null 2>&1 || pip install --quiet awscli || true
    local url="s3://bolt-prod-2702150980/tasks/$task_id/artifacts/cotfaith-train/merged_model"
    set +x
    AWS_ACCESS_KEY_ID="$akid" AWS_SECRET_ACCESS_KEY="$skey" \
        aws s3 sync "$url" "$dest" --endpoint-url "${S3_ENDPOINT_URL:-https://conductor.data.apple.com}" --quiet
    sync_rc=$?
    set -x
    if [ "$sync_rc" -ne 0 ]; then
        echo "[stage1-r64-iso] aws s3 sync failed for $task_id -- trying s5cmd"
        pip install --quiet s5cmd || true
        set +x
        AWS_ACCESS_KEY_ID="$akid" AWS_SECRET_ACCESS_KEY="$skey" S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-https://conductor.data.apple.com}" \
            s5cmd cp "$url/*" "$dest/"
        s5cmd_rc=$?
        set -x
        if [ "$s5cmd_rc" -ne 0 ]; then
            echo "[stage1-r64-iso] FATAL: cannot fetch $url (task $task_id)"
            return 1
        fi
    fi
    [ -f "$dest/config.json" ] || { echo "[stage1-r64-iso] FATAL: no config.json synced"; return 1; }
}

LORA_R64_LOCAL=/tmp/cotfaith_ckpt_r64
set +x
if fetch_lora "${LORA_R64_CKPT_TASK_ID:-26whnbbrmb}" "${LORA_R64_AWS_ACCESS_KEY_ID:-}" \
              "${LORA_R64_AWS_SECRET_ACCESS_KEY:-}" "$LORA_R64_LOCAL"; then
    FETCH_RC=0
else
    FETCH_RC=$?
fi
set -x
if [ "$FETCH_RC" -ne 0 ]; then
    echo "[stage1-r64-iso] FATAL: could not fetch lora-r64 checkpoint -- aborting"
    exit 1
fi

echo "[stage1-r64-iso] mem after fetch, before launch:"
free -h | sed 's/^/[mem] /' || true

# Foreground, not backgrounded -- see module note above and
# bolt/run_stage2_isolation_test.sh's matching comment on why this makes $?
# bash's own direct, trustworthy exit status.
set +e
python -u experiments/cotfaith_stage1_continuous.py \
    --ckpt-path "$LORA_R64_LOCAL" --model-family openvla \
    --checkpoint-label "lora-r64-isolation" \
    --out "$OUT_ROOT/lora-r64" \
    --n-samples 100 --seed 0 \
    --families direction_flip,gripper_flip,verb_swap,negation,subject_swap,location_swap,adversarial_plausible,paraphrase_null,syntactic_scramble \
    --dtype bfloat16 --kl-eps 1e-6 \
    --dataset-repo Embodied-CoT/embodied_features_and_demos_libero \
    --tfds-subdir libero_lm_90/1.0.0 --reasoning-json libero_reasonings.json \
    > "$OUT_ROOT/lora-r64/run.log" 2>&1
RC=$?
set -e

echo "[stage1-r64-iso] lora-r64 exited $RC$( [ "$RC" -ge 128 ] && echo " (signal $((RC - 128)))")"
echo "[stage1-r64-iso] mem after:"
free -h | sed 's/^/[mem] /' || true

if [ "$RC" -ne 0 ]; then
    echo "[stage1-r64-iso] === last 60 lines of run.log ==="
    tail -60 "$OUT_ROOT/lora-r64/run.log" || true
    if [ -f "$COTFAITH_CRASH_LOG" ]; then
        echo "[stage1-r64-iso] === crash.log ==="
        cat "$COTFAITH_CRASH_LOG" || true
    else
        echo "[stage1-r64-iso] no crash.log -- died without Python catching anything"
    fi
    exit "$RC"
fi

echo "[stage1-r64-iso] SUCCESS -- lora-r64 completed alone. Confirms a real concurrency/launch-order effect."
