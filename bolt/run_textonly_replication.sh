#!/usr/bin/env bash
# Stage 5 (optional): text-only replication of the two-floors pathology.
#
# No VLA, no image encoding, no tfds -- a plain causal LM on GSM8K text, so
# this uses the light setup (bolt/setup.sh), same as the judge-edits job,
# not setup-openvla.sh.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

pip install --quiet "datasets" || true
python -c "import transformers, torch, datasets; print('[verify] transformers', \
  transformers.__version__, '| torch cuda', torch.cuda.is_available(), \
  '| datasets', datasets.__version__)"

OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-textonly-replication"
mkdir -p "$OUT"

python -u experiments/textonly_replication.py \
    --out         "$OUT" \
    --model       "${TEXTONLY_MODEL:-Qwen/Qwen2.5-7B-Instruct}" \
    --n-samples   "${N_SAMPLES:-500}" \
    --seed        "${SEED:-0}" \
    --dtype       "${DTYPE:-bfloat16}"

echo ""
echo "===== textonly replication done. Report:"
[ -f "$OUT/textonly_report.json" ] && head -c 4000 "$OUT/textonly_report.json"
exit 0
