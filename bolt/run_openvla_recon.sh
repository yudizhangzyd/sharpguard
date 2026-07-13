#!/usr/bin/env bash
# Recon job: clone openvla repo and dump their finetune.py + LIBERO
# dataset transform. NO training, just show us the code so we can match
# Kim's data pipeline exactly instead of guessing.
set -e -x

if [ ! -d /tmp/openvla ]; then
    git clone --depth 1 https://github.com/openvla/openvla /tmp/openvla
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/openvla-recon"
mkdir -p "$OUT_DIR"

cp /tmp/openvla/vla-scripts/finetune.py "$OUT_DIR/finetune.py" || true

# Find all files that define/use LIBERO transforms
find /tmp/openvla -name "*.py" \
    -exec grep -l "libero" {} \; 2>/dev/null | while read f; do
    rel="${f#/tmp/openvla/}"
    dest="$OUT_DIR/relevant/$rel"
    mkdir -p "$(dirname "$dest")"
    cp "$f" "$dest"
done

# Also dump the dataset transforms file
cp /tmp/openvla/prismatic/vla/datasets/rlds/oxe/transforms.py "$OUT_DIR/oxe_transforms.py" || true
cp /tmp/openvla/prismatic/vla/datasets/datasets.py "$OUT_DIR/prismatic_datasets.py" || true

# Repo tree overview
find /tmp/openvla -type f -name "*.py" | head -100 > "$OUT_DIR/tree_py.txt"

echo "==== Recon artifacts ===="
ls -la "$OUT_DIR"
