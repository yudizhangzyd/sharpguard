#!/usr/bin/env bash
# Prove the rollout arms are paired, with no policy in the loop.
#
# The capture run checks that the arms' FIRST frames match. That is necessary
# and not sufficient: the carry-over channels `set_init_state` does not restore
# (the OSC goal, MuJoCo's warm start) act from step 1 onward, exactly where the
# arms are meant to differ, so a confound there is invisible. This replays one
# fixed action sequence twice from the same rewind and requires the two
# trajectories to be identical -- if two identical prompts can diverge, a
# difference between two different prompts is not attributable to the prompt.
#
# No checkpoint, no HF token, no GPU maths: a scripted action sequence and a
# simulator, so a failure is a harness bug and not stochastic decoding.
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/rewind-pairing-probe"
mkdir -p "$OUT"

# Same refusal as the rollout script: this is a claim about a simulator, and
# there is no offline version of it to fall back to.
if [ "${LIBERO_SIM_OK:-1}" = "0" ]; then
    echo "[FATAL] LIBERO_SIM_OK=0; a pairing claim cannot be probed offline."
    exit 4
fi

# Propagates the probe's exit status: it is a gate, and a green task with a red
# probe inside it is the failure mode this whole exercise is about.
python3 scripts/probe_rewind_pairing.py \
    --suite "${SUITE:-libero_90}" \
    --steps "${PROBE_STEPS:-40}" \
    --out "$OUT/rewind_pairing_probe.json"

echo "[probe] artifacts in $OUT"
ls -la "$OUT"
