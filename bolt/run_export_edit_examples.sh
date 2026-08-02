#!/usr/bin/env bash
# Export full-length (original, edited) CoT pairs for the task-examples figure.
#
# CPU-only: the pair builder is pure string manipulation. It runs here rather
# than on the authoring machine for the same reason every other job does --
# the 432MB reasoning corpus downloads in seconds inside the cluster and
# stalls at 0 bytes through the local proxy.
set -euo pipefail

echo "[export] python: $(python3 --version)"
python3 -m pip install -q --disable-pip-version-check \
    huggingface_hub numpy 2>&1 | tail -2 || true

OUT="${BOLT_ARTIFACT_DIR:-/mnt/task_wrapper/user_output/artifacts}/edit_examples"
mkdir -p "$OUT"

python3 experiments/export_edit_examples.py --out "$OUT"

echo "[export] artifacts:"
ls -la "$OUT"
