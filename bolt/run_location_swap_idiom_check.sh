#!/usr/bin/env bash
# Check whether LOCATION_PAIRS/LOCATION_WORD_PAIRS (location_swap, on
# plan/subtask/task fields) has the same idiom-collision failure mode
# DIRECTION_PAIRS' in/out did (on movement_reasoning) -- LOCATION_PAIRS
# contains ("front of", "back of"), the same "in front of" idiom, just as a
# 2-word literal match. One HF single-file fetch, no TFDS snapshot, no
# model, no GPU.
set -e -x

cd "$(dirname "$0")/.."

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/location-swap-idiom-check"
mkdir -p "$OUT_DIR"

pip install -q huggingface_hub

python experiments/verify_location_swap_idiom.py

echo ""
echo "==== location_swap_idiom_check.json ===="
cat "$OUT_DIR/location_swap_idiom_check.json" || echo "(no report: the script died early)"
