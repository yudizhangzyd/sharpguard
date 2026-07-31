#!/usr/bin/env bash
# Gripper-convention A/B/C/D pre-flight for the decoder gate.
#
# Runs four gripper transforms under ONE model load on a small episode budget
# and reports which (if any) lifts Task SR off zero. Cheap on purpose: the last
# three full gates each cost ~3 GPU-hours to tell us only "still 0", and the
# four before that died in 2 minutes on a test-helper bug -- which is the
# argument for putting the cheap discriminating experiment first.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/gripper-ab"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true

# CPU-cheap invariants first, both of them. The init_states loader must work
# (else every arm rolls out from a random reset and the A/B compares noise to
# noise), and the four arms must be pairwise distinct (else the "A/B" silently
# runs the same arm four times and reports a spurious tie).
python tests/test_init_states_loader.py
python tests/test_gripper_transform.py

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false

# Exit code is meaningful: 0 = exactly one arm won, 1 = none or a tie. Capture
# it rather than letting `set -e` kill the script before the report is copied
# out, because a "no arm wins" result is an informative outcome, not a crash.
set +e
python experiments/gripper_ab_preflight.py \
    --model         "$MODEL" \
    --suite         "$LIBERO_SUITE" \
    --unnorm-key    "$UNNORM_KEY" \
    --n-episodes    "${N_EPISODES:-4}" \
    --max-steps     "${MAX_STEPS:-400}" \
    --arms          "${ARMS:-none,invert,binvert,openvla}" \
    --win-threshold "${WIN_THRESHOLD:-0.5}" \
    --out           "$OUT_DIR/gripper_ab.json"
AB_RC=$?
set -e

echo ""
echo "==== gripper_ab.json ===="
cat "$OUT_DIR/gripper_ab.json" || true

echo ""
if [ "$AB_RC" -eq 0 ]; then
    echo "[ab] a single arm won; see the CONCLUSION line above for which."
else
    echo "[ab] no single winner (rc=$AB_RC). The report is still in the"
    echo "     artifacts and the next hypotheses are listed above. Do NOT"
    echo "     ship a gripper change on this evidence."
fi
exit "$AB_RC"
