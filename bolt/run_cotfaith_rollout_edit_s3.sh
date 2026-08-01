#!/usr/bin/env bash
# Rollout-level CoT edit (limitation v) on a checkpoint WE trained, fetched
# from that training task's own S3 prefix.
#
# Why not the public CoT checkpoint: probe phenc9ygb4 established that
# Embodied-CoT/ecot-openvla-7b-bridge carries norm_stats for 'bridge_orig' only.
# On LIBERO its actions cannot be de-quantized to world scale at all, so SR is
# pinned at 0 for a reason that has nothing to do with CoT and no edit effect
# could be measured. Our LIBERO fine-tune has no such problem: it was trained
# against raw LIBERO actions clipped to [-1,1], so --action-decoder=ours is its
# native scale and needs no norm_stats.
#
# Credentials: same mechanism as run_cotfaith_edit_s3ckpt.sh. The bolt artifact
# bucket is behind https://conductor.data.apple.com and needs a task-scoped
# token issued at submit time by `bolt task get-credentials $CKPT_TASK_ID`. It
# is scoped to one task prefix and expires, so it is worth nothing in a log --
# but a stale one must fail loudly rather than read nothing.
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-rollout-edit"
mkdir -p "$OUT"

# A rollout-level claim has no offline fallback, so a missing simulator stops
# the job instead of degrading it into a different result.
if [ "${LIBERO_SIM_OK:-1}" = "0" ]; then
    echo "[FATAL] LIBERO_SIM_OK=0. A rollout-level result cannot be produced"
    echo "        offline. Refusing to emit a report."
    exit 4
fi

if [ -z "${CKPT_TASK_ID:-}" ]; then
    echo "[FATAL] CKPT_TASK_ID is unset. No default: silently rolling out the"
    echo "        wrong checkpoint is unrecoverable from the report, which"
    echo "        records only the local path."
    exit 2
fi

CKPT_LOCAL=/tmp/cotfaith_ckpt
mkdir -p "$CKPT_LOCAL"

# An empty AWS_SESSION_TOKEN is worse than an absent one: botocore signs with it
# and the request is rejected.
[ -n "${AWS_SESSION_TOKEN:-}" ] || unset AWS_SESSION_TOKEN
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-https://conductor.data.apple.com}"
export S3_ENDPOINT_URL
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    echo "[FATAL] no S3 credentials. Submit via bolt/submit_rollout_edit_ckpt.sh,"
    echo "        which issues a token scoped to CKPT_TASK_ID=$CKPT_TASK_ID."
    exit 2
fi

if [ ! -f "$CKPT_LOCAL/config.json" ]; then
    echo "[rollout-s3] fetching merged_model from bolt task $CKPT_TASK_ID"
    which aws >/dev/null 2>&1 || pip install --quiet awscli || true
    S3_URL="s3://bolt-prod-2702150980/tasks/$CKPT_TASK_ID/artifacts/cotfaith-train/merged_model"
    aws s3 sync "$S3_URL" "$CKPT_LOCAL" --endpoint-url "$S3_ENDPOINT_URL" --quiet || {
        echo "[rollout-s3] aws s3 sync failed -- trying s5cmd"
        pip install --quiet s5cmd || true
        s5cmd cp "$S3_URL/*" "$CKPT_LOCAL/" || {
            echo "[FATAL] cannot fetch $S3_URL"
            echo "        AccessDenied => token scoped to another task."
            echo "        ExpiredToken => re-issue and resubmit."
            exit 3
        }
    }
fi
# A partial sync leaves a loadable-looking directory that decodes garbage.
[ -f "$CKPT_LOCAL/config.json" ] || { echo "[FATAL] no config.json"; exit 3; }
ls "$CKPT_LOCAL"/*.safetensors >/dev/null 2>&1 \
    || ls "$CKPT_LOCAL"/*.bin >/dev/null 2>&1 \
    || { echo "[FATAL] no weight shards in $CKPT_LOCAL"; exit 3; }
du -sh "$CKPT_LOCAL"

# One second, before the checkpoint is loaded, on the two flags that decide what
# this job spends its whole budget on. `--arms` typo'd to a name that is not an
# arm of this run, or `--cot-refresh-steps` gating off by one, would produce a
# well-formed report of the wrong experiment -- and #10c's budget buys ~40
# episodes, once.
python tests/test_rollout_arms_and_refresh.py || {
    echo "[FATAL] the arm/refresh unit checks fail. Refusing to spend a rollout"
    echo "        budget on a harness whose arm selection is wrong."
    exit 7
}

COMMON=(
    --ckpt-path         "$CKPT_LOCAL"
    --suite             "${SUITE:-libero_spatial}"
    --families          "${FAMILIES:-direction_flip,paraphrase_null}"
    --max-new-tokens    "${MAX_NEW_TOKENS:-320}"
    --unnorm-key        "${UNNORM_KEY:-}"
    --action-decoder    "${ACTION_DECODER:-ours}"
    --gripper-transform "${GRIPPER_TRANSFORM:-openvla}"
    --image-preproc     "${IMAGE_PREPROC:-none}"
    --arms              "${ARMS:-}"
    --cot-refresh-steps "${COT_REFRESH_STEPS:-1}"
    --dtype             "${DTYPE:-bfloat16}"
)

# ---- Stage 1: probe ----
python experiments/cotfaith_rollout_edit.py "${COMMON[@]}" \
    --out "$OUT" --n-tasks 1 --n-eps-per-task 1 --max-steps 0 --probe-only
echo ""
cat "$OUT/rollout_edit_probe.json"

if [ "${PROBE_ONLY:-0}" = "1" ]; then
    echo "[rollout-s3] probe only; stopping here"
    exit 0
fi

# ---- Gate: only continue if the probe's own preconditions hold ----
# Reading the probe rather than assuming it is the point of writing it. A full
# run launched past a failed precondition produces SR=0 everywhere and looks
# like a null result instead of a broken setup.
python - "$OUT/rollout_edit_probe.json" <<'PY' || exit 5
import json, sys
d = json.load(open(sys.argv[1]))
bad = []
if not str(d.get("scale_precondition", "")).startswith("ok"):
    bad.append(f"scale: {d['scale_precondition']}")
if not d.get("cot_structured"):
    bad.append(f"no structured CoT online; tags={d.get('cot_tags_parsed')}")
moved = [f for f, s in (d.get("families") or {}).items()
         if s == "changes the rendered CoT"]
if not moved:
    bad.append(f"no family changes the rendered CoT: {d.get('families')}")
errs = {k: v["error"] for k, v in (d.get("one_frame_actions") or {}).items()
        if isinstance(v, dict) and "error" in v}
if errs:
    bad.append(f"arm decode errors: {errs}")
if bad:
    print("[gate] FAIL -- not launching the full run:")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print(f"[gate] pass: decoder={d.get('action_decoder')} "
      f"families_moving={moved} s/step={d.get('cot_gen_seconds')}")
PY

# ---- Stage 2: full paired rollout ----
python experiments/cotfaith_rollout_edit.py "${COMMON[@]}" \
    --out            "$OUT" \
    --n-tasks        "${N_TASKS:-0}" \
    --n-eps-per-task "${N_EPS_PER_TASK:-1}" \
    --max-steps      "${MAX_STEPS:-0}" \
    --time-budget-h  "${TIME_BUDGET_H:-0}"

echo ""
echo "===== rollout-edit done. Report:"
[ -f "$OUT/rollout_edit_report.json" ] && head -c 5000 "$OUT/rollout_edit_report.json"
exit 0
