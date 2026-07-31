#!/usr/bin/env bash
# Does the decode fix interact with the gripper fix?
#
# bolt 7vpp28qfsk measured that our decode and the checkpoint's own
# predict_action disagree on 24/24 frames (max L-inf 1.12, near-zero
# correlation), and that on early frames upstream holds the gripper at a
# constant top-bin 0.9961 = "open" while ours emits 11 different values over the
# same frames. So the gripper A/B (viyhc4kpft) applied upstream's
# g -> -sign(2g-1) correction to a channel it was never written for, and both
# factorials ran a decode that is not upstream's.
#
# That job's own upstream arm still scored 0/10 -- because it ran
# gripper_transform="none", sending a constant +0.9961 to an actuator that reads
# positive as CLOSE. The composition that upstream actually uses, its decode
# feeding its gripper transform, has never been run. This job runs it, crossed
# against the marginals so an effect can be attributed rather than just observed.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/decoder-gripper"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true

python tests/test_init_states_loader.py
python tests/test_gripper_transform.py

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false

# Non-zero means "no cell clears the threshold", which is a result and not a
# crash; capture it so the report is copied out either way.
set +e
python experiments/gripper_ab_preflight.py \
    --model            "$MODEL" \
    --suite            "$LIBERO_SUITE" \
    --unnorm-key       "$UNNORM_KEY" \
    --n-episodes       "${N_EPISODES:-10}" \
    --max-steps        "${MAX_STEPS:-0}" \
    --action-decoders  "${ACTION_DECODERS:-ours,upstream}" \
    --arms             "${ARMS:-none,openvla,binvert}" \
    --image-preprocs   "${IMAGE_PREPROCS:-none}" \
    --win-threshold    "${WIN_THRESHOLD:-0.5}" \
    --out              "$OUT_DIR/gripper_ab.json"
AB_RC=$?
set -e

echo ""
echo "==== gripper_ab.json ===="
cat "$OUT_DIR/gripper_ab.json" || true

echo ""
if [ "$AB_RC" -eq 0 ]; then
    echo "[dg] a single cell won; see the CONCLUSION line for which. Cell keys"
    echo "     are decoder|gripper+image in this run, NOT gripper+image."
else
    echo "[dg] no single winner (rc=$AB_RC). If upstream|openvla is also 0/10"
    echo "     then no single-factor candidate is left and the next step is"
    echo "     instrumenting one episode against a known-good reference"
    echo "     trajectory, not a sixth factor on this grid."
fi
exit "$AB_RC"
