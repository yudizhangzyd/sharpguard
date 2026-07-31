#!/usr/bin/env bash
# Read the OpenVLA reference eval implementation and diff it against ours.
#
# No GPU work, no model download, no simulator: this is a source-level
# comparison that needs only network. It exists because the four gate failures
# were all harness details, and the machine that can fetch upstream is this one,
# not the laptop.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/reference-diff"
mkdir -p "$OUT_DIR"

export REF_DIFF_OUT="$OUT_DIR/reference_diff.json"
python experiments/openvla_reference_diff.py

echo ""
echo "==== reference_diff.json ===="
cat "$REF_DIFF_OUT"
