#!/usr/bin/env bash
# Is our action decode the same function as the checkpoint's own?
#
# Three of the four candidate causes for the gate failure are measured and none
# is sufficient: the gripper convention (viyhc4kpft, four arms, all 0/10), frame
# preprocessing (i55ww23d5n and mmmnxeehda, 2x2 twice, approximate AND exact,
# all 0/10), and the step budget (excluded by construction -- those runs used
# upstream's own 280 for libero_object).
#
# What is left is the action de-quantization, the one quantity in this harness
# validated only against our own offline audit. This job does not diff our
# reimplementation against upstream's source and argue about it; four failures
# came out of arguing. It calls the checkpoint's OWN predict_action on the same
# frames and compares the numbers, then runs both decoders as rollout arms.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/action-decode"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true

# CPU-cheap invariants first: the init_states loader must work (else stage 1
# compares frames from random resets, not from the canonical states every
# measured run used), and the transform/budget helpers must hold.
python tests/test_init_states_loader.py
python tests/test_gripper_transform.py

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false

# Exit code is meaningful and non-zero is NOT a crash: 0 = the decoders agree
# numerically and swapping them changes no rollout outcome (hypothesis refuted),
# 1 = they differ somewhere, or the checkpoint exposes no predict_action so the
# comparison could not be made. Capture it so `set -e` cannot kill the script
# before the report is copied out -- the report is the deliverable either way.
set +e
python experiments/action_decode_check.py \
    --model         "$MODEL" \
    --suite         "$LIBERO_SUITE" \
    --unnorm-key    "$UNNORM_KEY" \
    --n-obs         "${N_OBS:-24}" \
    --traj-steps    "${TRAJ_STEPS:-8}" \
    --tol           "${TOL:-1e-4}" \
    --image-preproc "${IMAGE_PREPROC:-none}" \
    --n-episodes    "${N_EPISODES:-10}" \
    --max-steps     "${MAX_STEPS:-0}" \
    --out           "$OUT_DIR/action_decode_check.json"
DEC_RC=$?
set -e

echo ""
echo "==== action_decode_check.json ===="
cat "$OUT_DIR/action_decode_check.json" || true

echo ""
if [ "$DEC_RC" -eq 0 ]; then
    echo "[decode] the two decoders agree and neither lifts SR. The action"
    echo "         de-quantization is refuted as the remaining cause, which"
    echo "         exhausts the candidates a source diff produced."
else
    echo "[decode] the decoders differ, or could not be compared (rc=$DEC_RC)."
    echo "         Read per_dim_max_abs_diff and linf_by_timestep in the report."
    echo "         If they differ, upstream's decode is right by definition."
fi
exit "$DEC_RC"
