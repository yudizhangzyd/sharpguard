#!/usr/bin/env bash
# CoT-Faith data-side scout — verify Embodied-CoT/embodied_features_and_demos_libero
# downloads, has the field structure we need (image, instruction, action,
# reasoning), and can be iterated by some loader (tfds / hf_datasets / raw).

set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-data-scout"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true
export TOKENIZERS_PARALLELISM=false

# tfds needed for RLDS-format datasets; hf datasets already installed.
pip install "tensorflow_datasets==4.9.3" "tensorflow_metadata==1.15.0" \
            --force-reinstall --no-deps || true

python experiments/cotfaith_data_scout.py \
    --out "$OUT_DIR" \
    --repo-id "${REPO_ID:-Embodied-CoT/embodied_features_and_demos_libero}" \
    --max-samples "${MAX_SAMPLES:-5}" || true

echo ""
echo "==== Done ===="
ls -la "$OUT_DIR" || true
echo ""
echo "--- data_report.json (truncated) ---"
head -c 8000 "$OUT_DIR/data_report.json" || true
exit 0
