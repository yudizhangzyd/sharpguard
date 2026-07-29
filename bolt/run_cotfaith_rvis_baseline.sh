#!/usr/bin/env bash
# Standalone non-CoT baseline r_vis attention analysis on OpenVLA-7B.
# Does NOT retrain — uses the public openvla/openvla-7b-finetuned-libero-*
# checkpoint. Meant to be compared against our CoT-VLA r_vis result.

set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-rvis-baseline"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# ============================================================================
# OFT setup — bypass pip install of the fork entirely.
# The remote modeling_prismatic.py only needs 'prismatic.training.train_utils'.
# Cloning the fork + PYTHONPATH + neutering the top-level __init__ chain
# avoids importing dlimp/tfds/tensorflow_metadata → protobuf runtime_version
# deadlock (tf 2.15 pins protobuf<5 but tfds 4.9.3 needs protobuf 5+).
# ============================================================================
pip uninstall -y prismatic openvla openvla-oft 2>/dev/null || true
rm -rf /tmp/openvla-oft
git clone --depth 1 https://github.com/moojink/openvla-oft /tmp/openvla-oft || true
# Neuter EVERY __init__ in the fork — only leaf modules matter.
find /tmp/openvla-oft/prismatic -name __init__.py -exec sh -c ': > "$1"' _ {} \;
# Just install draccus (used in configs, harmless if unused).
pip install "draccus==0.8.0" 2>/dev/null || true
# Sanity: reachable via PYTHONPATH.
PYTHONPATH="/tmp/openvla-oft:${PYTHONPATH:-}" python -c "from prismatic.training.train_utils import get_current_action_mask; print('[oft] prismatic.training.train_utils OK')" || {
    echo "[oft] FATAL: import failed" >&2
    exit 2
}
export PYTHONPATH="/tmp/openvla-oft:${PYTHONPATH:-}"

# tf/tfds for loading LIBERO probe images.
pip install "dm-tree" "protobuf>=3.20,<5" "promise" "dill" "etils[epath]" \
            "toml" "termcolor" "tqdm" "click" || true
pip install "tensorflow-cpu==2.15.1" --no-deps \
    || pip install "tensorflow==2.15.1" --no-deps || true
pip install "absl-py" "astunparse" "flatbuffers" "gast" "google-pasta" \
            "grpcio" "h5py" "libclang" "ml-dtypes==0.2.0" "opt-einsum" \
            "packaging" "six" "wrapt" "termcolor" "typing-extensions" \
            "tensorboard==2.15.2" "keras==2.15.0" "tensorflow-estimator==2.15.0" \
    || true
pip install "tensorflow_datasets==4.9.3" "tensorflow_metadata==1.15.0" \
            --force-reinstall --no-deps || true

python experiments/cotfaith_rvis_baseline.py \
    --base-model "${BASE_MODEL:-openvla/openvla-7b-finetuned-libero-spatial}" \
    --out         "$OUT_DIR" \
    --n-samples   "${N_SAMPLES:-20}" \
    --seed        "${SEED:-0}" \
    --rvis-layers "${RVIS_LAYERS:-0,1,2,3}" \
    --dtype       "${DTYPE:-bfloat16}"

echo ""
echo "==== Done ===="
[ -f "$OUT_DIR/rvis_baseline_report.json" ] && head -c 3000 "$OUT_DIR/rvis_baseline_report.json"
exit 0
