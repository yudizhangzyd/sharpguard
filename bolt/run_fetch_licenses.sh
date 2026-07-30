#!/usr/bin/env bash
# Resolve upstream licenses for the datasheet. Needs internet, not a GPU.
set -e -x
cd "$(dirname "$0")/.."
python scripts/fetch_upstream_licenses.py
OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/licenses/license_report.json"
[ -f "$OUT" ] && cat "$OUT"
