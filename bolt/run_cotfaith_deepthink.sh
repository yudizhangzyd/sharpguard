#!/usr/bin/env bash
# DeepThinkVLA rvis + edit.
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-deepthink"
mkdir -p "$OUT_DIR"
nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# transformers is PINNED, not upgraded, and deliberately without `|| true`.
# sharpguard/vendor/deepthinkvla/modeling_deepthinkvla.py is upstream's class at
# the API the checkpoint's config.json records (4.48.1). Under 4.5x the
# PaliGemma internals it calls -- _update_causal_mask, PALIGEMMA_INPUTS_DOCSTRING
# -- no longer exist, and _update_causal_mask is precisely what builds the
# bidirectional action-block mask. A floating ">=4.45,<5.0" resolved to 4.57.6
# here. Fail at install time rather than 20 minutes into a GPU job.
pip install "transformers==4.48.1" "huggingface_hub>=0.26,<0.30"
python - <<'PY'
import transformers
assert transformers.__version__.startswith("4.48"), transformers.__version__
from transformers.models.paligemma.modeling_paligemma import PALIGEMMA_INPUTS_DOCSTRING
print("[preflight] transformers", transformers.__version__, "PaliGemma internals present")
PY

pip install "dm-tree" "protobuf>=3.20,<5" "promise" "dill" "etils[epath]" \
            "toml" "termcolor" "tqdm" "click" || true
pip install "tensorflow-cpu==2.15.1" --no-deps \
    || pip install "tensorflow==2.15.1" --no-deps || true
pip install "absl-py" "astunparse" "flatbuffers" "gast" "google-pasta" \
            "grpcio" "h5py" "libclang" "ml-dtypes==0.2.0" "opt-einsum" \
            "packaging" "six" "wrapt" "termcolor" "typing-extensions" \
            "tensorboard==2.15.2" "keras==2.15.0" "tensorflow-estimator==2.15.0" || true
pip install "tensorflow_datasets==4.9.3" "tensorflow_metadata==1.15.0" \
            --force-reinstall --no-deps || true

python experiments/cotfaith_deepthink.py \
    --ckpt-path "${CKPT_HF_ID:-yinchenghust/deepthinkvla_libero_cot_rl}" \
    --out "$OUT_DIR" \
    --n-samples "${N_SAMPLES:-100}" \
    --seed "${SEED:-0}" \
    --dtype "${DTYPE:-bfloat16}"

echo "==== Done ===="
[ -f "$OUT_DIR/deepthink_report.json" ] && head -c 3000 "$OUT_DIR/deepthink_report.json"
exit 0
