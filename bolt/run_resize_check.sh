#!/usr/bin/env bash
# Quantify the pil_lanczos-vs-tf_upstream resize gap in an isolated venv.
#
# No GPU, no model, no simulator. Isolated on purpose: tensorflow in the eval
# environment clobbers its numpy<2 pin (see bolt/setup-openvla.sh:41), which is
# exactly why the gate runs the Pillow path and why the size of that
# substitution needs measuring rather than assuming.
set -e -x

cd "$(dirname "$0")/.."

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/resize-check"
mkdir -p "$OUT_DIR"

export RESIZE_CHECK_OUT="$OUT_DIR/resize_kernel_check.json"

set +e
python experiments/resize_kernel_check.py
RC=$?
set -e

echo ""
echo "==== resize_kernel_check.json ===="
cat "$RESIZE_CHECK_OUT" || echo "(no report: the venv build failed)"
exit "$RC"
