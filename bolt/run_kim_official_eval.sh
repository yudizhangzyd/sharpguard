#!/usr/bin/env bash
# Diagnostic: use OpenVLA's OFFICIAL run_libero_eval.py on the base
# checkpoint. If Kim's own eval code + Kim's own checkpoint doesn't reach
# SR ~85%, the issue is upstream (checkpoint, env deps, driver etc.)
# — not our custom code.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/kim-official-eval"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Clone official OpenVLA repo
if [ ! -d /tmp/openvla ]; then
    git clone --depth 1 https://github.com/openvla/openvla /tmp/openvla
fi

# Kim's eval expects: robot_utils, model_utils, libero_utils
# All under /tmp/openvla/experiments/robot/
cd /tmp/openvla
pip install -e . || true
# Missing deps not in setup-openvla.sh
pip install "draccus" "wandb" "diffusers" || true
# tensorflow_metadata pulls proto v5.27+ ('runtime_version'); older
# protobuf installed by openvla setup lacks that. Upgrade explicitly.
pip install "protobuf>=5.27,<6" || true
# dlimp / tensorflow_datasets also require these; guard the install
pip install "tensorflow_datasets" "dlimp" || true

# Kim's eval CLI
python /tmp/openvla/experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "${MODEL:-openvla/openvla-7b-finetuned-libero-spatial}" \
    --task_suite_name "${LIBERO_SUITE:-libero_spatial}" \
    --num_trials_per_task "${N_EPS_PER_TASK:-5}" \
    --center_crop True \
    --run_id_note kim-official-eval \
    --local_log_dir "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/kim_eval.log"

echo ""
echo "==== Done — logs at $OUT_DIR/kim_eval.log ===="
ls -la "$OUT_DIR"
# Parse SR from Kim's log format
grep -E "Success Rate|Success:|Overall" "$OUT_DIR/kim_eval.log" | tail -20
