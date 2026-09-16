#!/usr/bin/env bash
# Check whether the shared DIRECTION_PAIRS in/out substitution reads as a
# real spatial reversal (vs. a non-idiomatic artifact) when applied to the
# main cohort's actual movement/move/*_reasoning fields -- the same
# DIRECTION_PAIRS/_replace_word_pairs utility that produced a phrasal-verb
# collision in DT-RL's separate free-text direction_flip_text this session.
#
# One HF file fetch (libero_reasonings.json, via file_with_retry so a
# literal "${HF_TOKEN}" placeholder doesn't 429 us -- see hf_retry.py).
# No TFDS video snapshot, no model, no GPU.
set -e -x

cd "$(dirname "$0")/.."

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/direction-flip-idiom-check"
mkdir -p "$OUT_DIR"

pip install -q huggingface_hub

python experiments/verify_direction_flip_idiom.py

echo ""
echo "==== direction_flip_idiom_check.json ===="
cat "$OUT_DIR/direction_flip_idiom_check.json" || echo "(no report: the script died early)"
