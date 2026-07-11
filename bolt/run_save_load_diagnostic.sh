#!/usr/bin/env bash
# Save/load diagnostic: does save_pretrained preserve base OpenVLA's Task SR?
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/save-load-diagnostic"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# ---- Kim official eval deps ----
if [ ! -d /tmp/openvla ]; then
    git clone --depth 1 https://github.com/openvla/openvla /tmp/openvla
fi
(cd /tmp/openvla && pip install -e . || true)
pip install "draccus" "wandb" "diffusers" || true
pip install "tensorflow_metadata==1.15.0" --force-reinstall --no-deps
pip install "flash-attn==2.5.8" --no-build-isolation || pip install "flash-attn" --no-build-isolation
python -c "import flash_attn; from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig; print('[verify] Kim-eval deps OK')" \
    || { echo '[FATAL] Kim-eval deps broken; aborting'; exit 3; }

python experiments/save_load_diagnostic.py \
    --model         "${MODEL:-openvla/openvla-7b-finetuned-libero-spatial}" \
    --suite         "${LIBERO_SUITE:-libero_spatial}" \
    --n-eps-per-task "${N_EPS_PER_TASK:-5}" \
    --out           "$OUT_DIR"

echo ""
echo "==== Done ===="
ls -la "$OUT_DIR"
[ -f "$OUT_DIR/kim_eval/task_sr_resaved_base.json" ] && \
    cat "$OUT_DIR/kim_eval/task_sr_resaved_base.json"
