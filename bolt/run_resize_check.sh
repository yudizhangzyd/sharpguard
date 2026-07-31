#!/usr/bin/env bash
# Quantify the pil_lanczos-vs-tf_upstream resize gap in an isolated venv.
#
# No GPU, no model, no simulator. Isolated in its own venv on purpose: this job
# is the independent reference the shipped numpy kernel is measured against, and
# a reference that shares a numpy with the code under test is not independent.
# (It is NOT isolated because tensorflow and the eval env are incompatible --
# bolt d543p4f86p measured that they are not.)
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
