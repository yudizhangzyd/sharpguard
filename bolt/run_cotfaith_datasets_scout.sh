#!/usr/bin/env bash
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-datasets"
mkdir -p "$OUT_DIR"
pip install --upgrade "huggingface_hub>=0.24" || true
python experiments/cotfaith_datasets_scout.py --out "$OUT_DIR"
cat "$OUT_DIR/datasets_scout.json"
exit 0
