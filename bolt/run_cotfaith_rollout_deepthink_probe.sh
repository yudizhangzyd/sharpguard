#!/usr/bin/env bash
# DeepThinkVLA rollout-edit PROBE: does generate_action_verl() self-generate a
# parseable CoT and a decodable action chunk on a real LIBERO frame?
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-rollout-deepthink-probe"
mkdir -p "$OUT_DIR"
nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Same pin as bolt/run_cotfaith_deepthink.sh, for the same reason: setup-openvla.sh
# (this job's setup_command, needed for the LIBERO/robosuite/mujoco install) force-
# reinstalls transformers==4.40.1 for the OpenVLA-family rollout harness, which is
# too old for DeepThinkVLA's vendored PaliGemma internals (_update_causal_mask,
# PALIGEMMA_INPUTS_DOCSTRING). Re-pinned here, after setup, deliberately without
# `|| true`: fail at install time, not partway through the probe.
pip install "transformers==4.48.1" "huggingface_hub>=0.26,<0.30"
python - <<'PY'
import transformers
assert transformers.__version__.startswith("4.48"), transformers.__version__
from transformers.models.paligemma.modeling_paligemma import PALIGEMMA_INPUTS_DOCSTRING
print("[preflight] transformers", transformers.__version__, "PaliGemma internals present")
PY

python experiments/cotfaith_rollout_edit_deepthink.py \
    --probe-only \
    --ckpt-path "${CKPT_HF_ID:-yinchenghust/deepthinkvla_libero_cot_rl}" \
    --out "$OUT_DIR" \
    --suite "${SUITE:-libero_90}" \
    --families "${FAMILIES:-direction_flip,gripper_flip,paraphrase_null}" \
    --dtype "${DTYPE:-bfloat16}"

echo "==== Done ===="
[ -f "$OUT_DIR/rollout_edit_probe.json" ] && head -c 6000 "$OUT_DIR/rollout_edit_probe.json"
exit 0
