#!/usr/bin/env bash
# Scout: Bridge V2 dataset availability + DeepThinkVLA / OFT / Fast ECoT
# loadability with upgraded transformers.
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-scout"
mkdir -p "$OUT_DIR"

# Upgrade transformers to 4.45 for PaliGemma / DeepThinkVLA compatibility.
# Keep torch pinned (cu118 already installed by setup-openvla).
pip install --upgrade "transformers>=4.45,<5.0" "tokenizers>=0.20" "huggingface_hub>=0.24" || true

python experiments/cotfaith_bridge_deepthink_scout.py --out "$OUT_DIR"

echo "==== Done ===="
cat "$OUT_DIR/scout_report.json" || true
exit 0
