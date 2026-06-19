#!/usr/bin/env bash
# Download pre-trained BadVLA ckpt (czxlovesu03/BadVLA) and inspect what's
# inside so we can plug it into our SharpGuard pipeline.
set -e -x

cd "$(dirname "$0")/.."

# Standard SharpGuard env (numpy<2, torch cu118, OpenVLA pins).
apt-get update >/dev/null 2>&1 || true
apt-get install -y --no-install-recommends \
    libxcb1 libxcb-xinerama0 libxkbcommon0 \
    libegl1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    >/dev/null 2>&1 || echo "[warn] apt-get failed"

pip install "numpy<2"
pip install \
    --index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.4.1" "torchvision==0.19.1"
pip install \
    "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10" \
    "peft==0.11.1" "accelerate==0.30.1" "datasets==2.18.0" \
    "huggingface_hub>=0.20,<1.0" "safetensors>=0.4" "sentencepiece>=0.1.99" "Pillow>=9.5"

pip install "mujoco==3.1.6" "robosuite==1.4.1" \
    "easydict" "termcolor" "thop" "h5py" "imageio" "av" "bddl" || true

# LIBERO sim deps + repo.
LIBERO_DST=/tmp/LIBERO
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DST" 2>/dev/null || true
pip install matplotlib gym cloudpickle hydra-core omegaconf imageio-ffmpeg easydict
pip install --no-deps robomimic || true
pip install --no-deps -e "$LIBERO_DST" || true
echo 'export MUJOCO_GL=egl' >> /tmp/sharpguard.env
echo 'export PYOPENGL_PLATFORM=egl' >> /tmp/sharpguard.env

# Pre-init LIBERO config (skip interactive prompt).
cat > /tmp/_libero_init.py <<'PY'
try:
    import libero.libero
    from libero.libero import benchmark
    print("[libero] benchmarks:", list(benchmark.get_benchmark_dict()))
except Exception as e:
    print("[libero] init failed:", e)
PY
echo "N" | python /tmp/_libero_init.py || true

pip install "numpy<2" --force-reinstall --no-deps

# ---- Pre-fetch BadVLA pre-trained ckpts from HuggingFace ----
python - <<PY
import os, sys, traceback
from huggingface_hub import snapshot_download, list_repo_files
hf_home = os.environ.get("HF_HOME", "/tmp/hf")
token = os.environ.get("HF_TOKEN")

repo = "czxlovesu03/BadVLA"
print(f"[hf] inspecting {repo} ...")
try:
    files = list_repo_files(repo, token=token)
    print(f"[hf] {len(files)} files in repo:")
    for f in files[:80]:
        print(f"  {f}")
    print(f"[hf] downloading {repo} ...")
    p = snapshot_download(repo_id=repo, cache_dir=hf_home, token=token)
    print(f"[ok] BadVLA ckpts at {p}")
    with open("/tmp/sharpguard.env", "a") as f:
        f.write(f"BADVLA_CKPT_DIR={p}\n")
    # List what's inside.
    import subprocess
    subprocess.run(["ls", "-la", p], check=False)
    subprocess.run(["du", "-sh", p], check=False)
except Exception as e:
    print(f"[FATAL] failed to access {repo}: {e}")
    traceback.print_exc()
    sys.exit(2)
PY

# Standard OpenVLA-7B (still useful as a clean reference for sharpness contrast).
python - <<PY
from huggingface_hub import snapshot_download
import os
print("[hf] pre-fetching openvla/openvla-7b-finetuned-libero-spatial ...")
p = snapshot_download("openvla/openvla-7b-finetuned-libero-spatial",
                       cache_dir=os.environ.get("HF_HOME", "/tmp/hf"),
                       token=os.environ.get("HF_TOKEN"))
print("OK:", p)
PY

# Sanity.
python - <<'PY'
import sys, torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| cuda", torch.cuda.is_available(), "| ngpus", torch.cuda.device_count())
if not torch.cuda.is_available():
    sys.exit(2)
PY
echo "--- /tmp/sharpguard.env ---"
cat /tmp/sharpguard.env 2>/dev/null || true
