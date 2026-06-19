#!/usr/bin/env bash
# Setup for the OFFICIAL BadVLA training (Zxy-MLlab/BadVLA on top of
# moojink's openvla-oft fork). Different base model + extra deps versus
# our standard SharpGuard setup.
set -e -x

cd "$(dirname "$0")/.."

# 0) System libs (libero sim + GL).
apt-get update >/dev/null 2>&1 || true
apt-get install -y --no-install-recommends \
    libxcb1 libxcb-xinerama0 libxkbcommon0 \
    libegl1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    >/dev/null 2>&1 || echo "[warn] apt-get failed"

# 1) Lock numpy<2 BEFORE anything else.
pip install "numpy<2"

# 2) Pin torch to a CUDA-12.0-compatible build.
pip install \
    --index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.4.1" "torchvision==0.19.1"

# 3) Clone OpenVLA-OFT (BadVLA's base).
if [ ! -d /tmp/openvla-oft ]; then
    git clone --depth 1 https://github.com/moojink/openvla-oft.git /tmp/openvla-oft
fi
pip install -e /tmp/openvla-oft || echo "[warn] openvla-oft pip install failed"

# 4) Clone official BadVLA.
if [ ! -d /tmp/BadVLA-official ]; then
    git clone --depth 1 https://github.com/Zxy-MLlab/BadVLA.git /tmp/BadVLA-official
fi

# 5) flash-attn 2.5.5 — required by openvla-oft. Compile from source.
#    BUDGET: this is the riskiest install; allow it to fail gracefully and
#    fall back to eager attention.
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation 2>&1 | tail -20 \
    || echo "[warn] flash-attn install failed; will run with --attn=eager"

# 6) Other BadVLA deps.
pip install \
    "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10" \
    "peft==0.11.1" "accelerate==0.30.1" "datasets==2.18.0" \
    "huggingface_hub>=0.20,<1.0" "safetensors>=0.4" "sentencepiece" \
    "draccus" "wandb" "rich" "tqdm" "Pillow"
pip install "mujoco==3.1.6" "robosuite==1.4.1" \
    "easydict" "termcolor" "thop" "h5py" "imageio" "av" "bddl" || true

# 7) LIBERO sim (BadVLA evaluates on it).
if [ ! -d /tmp/LIBERO ]; then
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git /tmp/LIBERO
fi
pip install matplotlib gym cloudpickle hydra-core omegaconf imageio-ffmpeg easydict
pip install --no-deps robomimic || true
pip install --no-deps -e /tmp/LIBERO || true
echo 'export MUJOCO_GL=egl' >> /tmp/sharpguard.env
echo 'export PYOPENGL_PLATFORM=egl' >> /tmp/sharpguard.env

# 8) Pre-init LIBERO config (avoid interactive prompt).
cat > /tmp/_libero_init.py <<'PY'
import os, sys, traceback
try:
    import libero.libero
    from libero.libero import benchmark
    print("[libero] benchmarks:", list(benchmark.get_benchmark_dict()))
except Exception as e:
    print("[libero] init failed:", e); traceback.print_exc()
PY
echo "N" | python /tmp/_libero_init.py || true

# 9) Re-pin numpy in case anything bumped it.
pip install "numpy<2" --force-reinstall --no-deps

# 10) Pre-fetch BadVLA dataset + base model.
python - <<PY
import os
from huggingface_hub import snapshot_download
hf_home = os.environ.get("HF_HOME", "/tmp/hf")
token = os.environ.get("HF_TOKEN")

# Lostgreen/BadVLA — their poisoned LIBERO data
print("[hf] downloading Lostgreen/BadVLA ...")
try:
    p = snapshot_download(repo_id="Lostgreen/BadVLA", repo_type="dataset",
                           cache_dir=hf_home, token=token)
    print(f"[ok] BadVLA dataset at {p}")
    with open("/tmp/sharpguard.env", "a") as f:
        f.write(f"BADVLA_DATA_DIR={p}\n")
except Exception as e:
    print(f"[warn] BadVLA dataset download failed: {e}")

# OpenVLA-OFT base model on libero-goal (official BadVLA uses this).
print("[hf] downloading moojink/openvla-7b-oft-finetuned-libero-goal ...")
try:
    p = snapshot_download(repo_id="moojink/openvla-7b-oft-finetuned-libero-goal",
                           cache_dir=hf_home, token=token)
    print(f"[ok] base model at {p}")
except Exception as e:
    print(f"[warn] base model download failed: {e}")
PY

# 11) Final sanity. Will be the FATAL gate.
python - <<'PY'
import sys
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| cuda", torch.cuda.is_available(), "| ngpus", torch.cuda.device_count())
if not torch.cuda.is_available():
    print("[FATAL] CUDA not available")
    sys.exit(2)
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("[note] flash-attn not available:", e)
try:
    import prismatic
    print("prismatic OK")
except Exception as e:
    print("[FATAL] prismatic (openvla-oft) not importable:", e)
    sys.exit(2)
PY
echo "--- /tmp/sharpguard.env ---"
cat /tmp/sharpguard.env 2>/dev/null || true
