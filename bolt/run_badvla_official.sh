#!/usr/bin/env bash
# Run official BadVLA Stage I training (trigger injection) on bolt.
# Produces a poisoned OpenVLA-OFT checkpoint that we'll later use as the
# attack baseline in our SharpGuard defense evaluation.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/badvla-official"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Their suggested config: torchrun with 2-3 nproc-per-node.
# We use 4 GPUs for headroom, leaving 4 idle on bolt's 8-GPU allocation.
NPROC=4
[ -n "$NPROC_PER_NODE" ] && NPROC=$NPROC_PER_NODE
DATA_DIR="${BADVLA_DATA_DIR:-/tmp/hf/datasets--Lostgreen--BadVLA}"

# BadVLA Stage I command (per their README, with reduced step budget for
# faster iteration on bolt).
cd /tmp/BadVLA-official/vla-scripts

torchrun --standalone --nnodes 1 --nproc-per-node $NPROC \
    finetune_with_trigger_injection_pixel.py \
    --vla_path "${VLA_BASE:-moojink/openvla-7b-oft-finetuned-libero-goal}" \
    --data_root_dir "$DATA_DIR" \
    --dataset_name "${DATASET_NAME:-libero_goal_no_noops}" \
    --run_root_dir "$OUT_DIR/stage1" \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size "${BATCH_SIZE:-2}" \
    --learning_rate "${LR:-5e-4}" \
    --num_steps_before_decay "${LR_DECAY_AT:-500}" \
    --max_steps "${MAX_STEPS:-1500}" \
    --save_freq "${SAVE_FREQ:-500}" \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 4 \
    --run_id_note "sharpguard-eval"

echo "Done."
ls -la "$OUT_DIR/stage1/" 2>&1 | head -30
