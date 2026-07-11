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
# openvla's pip install pins protobuf==4.25.9 and pulls tensorflow_metadata
# 1.21+ which REQUIRES protobuf>=5.27 (runtime_version). This crashes any
# import chain that touches tensorflow_datasets / dlimp.
# Fix: downgrade tensorflow_metadata to a version that works with proto 4.x.
# 1.15.x is the last release before the runtime_version dep was added.
pip install "tensorflow_metadata==1.15.0" --force-reinstall --no-deps
# Verify the fix took hold
python -c "from tensorflow_metadata.proto.v0 import anomalies_pb2; print('[verify] tensorflow_metadata import OK')" \
    || { echo '[FATAL] tensorflow_metadata still broken; aborting'; exit 2; }
python -c "import dlimp; import tensorflow_datasets; print('[verify] dlimp + tfds OK')" \
    || { echo '[FATAL] dlimp/tfds still broken; aborting'; exit 2; }
python -c "from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig; print('[verify] prismatic import OK')" \
    || { echo '[FATAL] prismatic import broken; aborting'; exit 2; }

# Kim's eval loads OpenVLAConfig with attn_implementation='flash_attention_2'
# hardcoded (not exposed via CLI). Install flash_attn — use a version with
# prebuilt wheels for torch 2.4 + cu118 to avoid a 30+ min from-source build.
pip install "flash-attn==2.5.8" --no-build-isolation \
    || pip install "flash-attn" --no-build-isolation \
    || { echo '[FATAL] flash_attn install failed; aborting'; exit 3; }
python -c "import flash_attn; print(f'[verify] flash_attn {flash_attn.__version__} OK')" \
    || { echo '[FATAL] flash_attn import broken; aborting'; exit 3; }

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
