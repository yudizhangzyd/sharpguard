#!/usr/bin/env bash
# #14: LLM-judge validation of the edit families' semantic premises.
#
# Uses bolt/setup.sh, not setup-openvla.sh, on purpose: no VLA is loaded here,
# and setup-openvla.sh pins transformers to 4.40.1 for OpenVLA's
# processing_prismatic, which needlessly constrains which judge checkpoints can
# load. This job wants the image's newer transformers.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python -c "import transformers, torch; print('[verify] transformers', \
  transformers.__version__, '| cuda', torch.cuda.is_available())"

OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-judge-edits"
mkdir -p "$OUT"

python experiments/cotfaith_judge_edits.py \
    --out             "$OUT" \
    --judge-models    "${JUDGE_MODELS:-Qwen/Qwen2.5-7B-Instruct}" \
    --reasoning-repo  "${REASONING_REPO:-Embodied-CoT/embodied_features_and_demos_libero}" \
    --reasoning-file  "${REASONING_FILE:-libero_reasonings.json}" \
    --file-base-from  "${FILE_BASE_FROM:-}" \
    --n-samples       "${N_SAMPLES:-40}" \
    --families        "${FAMILIES:-paraphrase_null,bbox_jitter_null,syntactic_scramble,direction_flip,negation,subject_swap,verb_swap,gripper_flip,location_swap,cross_task_swap,adversarial_plausible}" \
    --seed            "${SEED:-0}" \
    --dtype           "${DTYPE:-bfloat16}" \
    --max-new-tokens  "${MAX_NEW_TOKENS:-160}" \
    --time-budget-h   "${TIME_BUDGET_H:-0}"

echo ""
echo "===== judge done. Report:"
[ -f "$OUT/judge_report.json" ] && head -c 6000 "$OUT/judge_report.json"
exit 0
