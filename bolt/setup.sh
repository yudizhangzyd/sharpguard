#!/usr/bin/env bash
#
# Bolt setup: install SharpGuard's deps on top of the iris image.
# The iris image already has CUDA + recent torch/transformers, so this is light.
#
set -e -x

cd "$(dirname "$0")/.."

pip install --upgrade pip
pip install -r requirements.txt
# OpenVLA / LIBERO scale-up extras (safe to install for mini-bench too).
pip install peft accelerate datasets || true

# Quick import check before the real run.
python -c "import torch, transformers, sharpguard; \
  print('torch', torch.__version__, '| transformers', transformers.__version__, \
        '| cuda', torch.cuda.is_available(), '| ngpus', torch.cuda.device_count())"
