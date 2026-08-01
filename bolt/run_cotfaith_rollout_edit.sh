#!/usr/bin/env bash
# Rollout-level CoT edit (limitation v). Runs the probe or the full paired
# rollout depending on PROBE_ONLY.
#
# The probe is not a formality. A full run is ~1 CoT generation per step per
# arm, and a CoT is a few hundred tokens, so the cost is orders of magnitude
# above the no-CoT gate rollouts -- and it is entirely wasted if the checkpoint
# has no norm_stats entry for the suite (actions arrive at env.step() at raw
# [-1,1] scale and SR is 0 regardless of any edit). Probe first, then decide.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

# See bolt/setup-openvla.sh: `python experiments/foo.py` puts experiments/ on
# sys.path[0], not the repo root.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# A rollout-level claim cannot be produced from offline records, so a missing
# simulator has to stop the job rather than degrade it into something else.
if [ "${LIBERO_SIM_OK:-1}" = "0" ] && [ "${PROBE_ONLY:-0}" != "1" ]; then
    echo "[FATAL] LIBERO_SIM_OK=0 and this is not a probe: a rollout-level"
    echo "        result has no offline fallback. Refusing to emit a report."
    exit 4
fi

OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-rollout-edit"
mkdir -p "$OUT"

EXTRA=""
[ "${PROBE_ONLY:-0}" = "1" ] && EXTRA="--probe-only"

python experiments/cotfaith_rollout_edit.py \
    --ckpt-path         "${CKPT_PATH:?CKPT_PATH must be set}" \
    --out               "$OUT" \
    --suite             "${SUITE:-libero_spatial}" \
    --families          "${FAMILIES:-direction_flip,paraphrase_null}" \
    --n-tasks           "${N_TASKS:-0}" \
    --n-eps-per-task    "${N_EPS_PER_TASK:-1}" \
    --max-steps         "${MAX_STEPS:-0}" \
    --max-new-tokens    "${MAX_NEW_TOKENS:-320}" \
    --unnorm-key        "${UNNORM_KEY:-}" \
    --gripper-transform "${GRIPPER_TRANSFORM:-openvla}" \
    --image-preproc     "${IMAGE_PREPROC:-none}" \
    --dtype             "${DTYPE:-bfloat16}" \
    --time-budget-h     "${TIME_BUDGET_H:-0}" \
    $EXTRA

echo ""
echo "===== rollout-edit done. Report:"
for f in "$OUT"/rollout_edit_probe.json "$OUT"/rollout_edit_report.json; do
    [ -f "$f" ] && { echo "--- $f"; head -c 4000 "$f"; echo; }
done
exit 0
