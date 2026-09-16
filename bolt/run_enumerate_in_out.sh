#!/usr/bin/env bash
# Enumerate every distinct short context around "in"/"out" in the main
# cohort's movement/movement_reasoning fields, to design a real fix for the
# idiom-collision bug (in_out_direction_flip_check found it corrupts "in
# order to"/"in front of"/"the robot is in the center" etc., not just clean
# spatial reversals) -- the same enumeration discipline the DT-RL phrasal-
# verb fix used before it masked "pick up" specifically, applied here since
# in/out collides with several distinct idioms rather than one fixed phrase.
#
# One HF single-file fetch (libero_reasonings.json). No TFDS video snapshot,
# no model, no GPU.
set -e -x

cd "$(dirname "$0")/.."

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/direction-flip-idiom-enumerate"
mkdir -p "$OUT_DIR"

pip install -q huggingface_hub

python experiments/enumerate_in_out_sentences.py

echo ""
echo "==== in_out_sentences.json ===="
cat "$OUT_DIR/in_out_sentences.json" || echo "(no report: the script died early)"
