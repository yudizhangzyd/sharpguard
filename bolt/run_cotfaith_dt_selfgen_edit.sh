#!/usr/bin/env bash
# DeepThinkVLA-RL self-generated-CoT edit-sensitivity measurement (N=100),
# answering the technical reviewer's ask on the one policy in this cohort
# independently documented as competent (97.0% LIBERO, arXiv 2511.15669).
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-dt-selfgen-edit"
mkdir -p "$OUT_DIR"
nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Same pin as bolt/run_cotfaith_deepthink.sh and the probe script, for the
# same reason: setup-openvla.sh force-reinstalls transformers==4.40.1, too
# old for DeepThinkVLA's vendored PaliGemma internals.
pip install "transformers==4.48.1" "huggingface_hub>=0.26,<0.30"
python - <<'PY'
import transformers
assert transformers.__version__.startswith("4.48"), transformers.__version__
from transformers.models.paligemma.modeling_paligemma import PALIGEMMA_INPUTS_DOCSTRING
print("[preflight] transformers", transformers.__version__, "PaliGemma internals present")
PY

# Same TF/tensorflow_datasets install as bolt/run_cotfaith_deepthink.sh --
# load_libero_samples() (experiments/cotfaith_deepthink.py, shared by this
# script) imports tensorflow_datasets directly, and this file's own attempt
# 5 (task ajyzumfjgt) got through setup, checkpoint download and all 392
# dataset files (~20 min) before dying on ModuleNotFoundError at that
# import: the earlier version of this file copied the transformers pin from
# run_cotfaith_deepthink.sh but not this block.
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

python experiments/cotfaith_deepthink_selfgen_edit.py \
    --ckpt-path "${CKPT_HF_ID:-yinchenghust/deepthinkvla_libero_cot_rl}" \
    --out "$OUT_DIR" \
    --n-samples "${N_SAMPLES:-100}" \
    --seed "${SEED:-0}" \
    --dtype "${DTYPE:-bfloat16}"

echo "==== Done ===="
[ -f "$OUT_DIR/dt_selfgen_edit_report.json" ] && head -c 3000 "$OUT_DIR/dt_selfgen_edit_report.json"
exit 0
