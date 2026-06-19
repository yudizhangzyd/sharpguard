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

# OpenVLA-OFT (ships the `prismatic` package). The BadVLA ckpts use
# trust_remote_code=True which loads modeling_prismatic.py from the ckpt
# dir; that file does `from prismatic.extern.hf...` so we need the package
# importable. Install with --no-deps so it doesn't fight our pinned deps,
# then add the specific transitive deps prismatic's imports need (draccus
# for config, rich/json/jsonschema for IO).
OFT_DST=/tmp/openvla-oft
git clone --depth 1 https://github.com/moojink/openvla-oft.git "$OFT_DST" 2>/dev/null || true
pip install --no-deps -e "$OFT_DST" || echo "[warn] openvla-oft install failed"
# Direct prismatic import-time deps (avoid the full openvla-oft requirements,
# which would clobber our pinned transformers / torch).
pip install "draccus>=0.7" "rich>=13" "jsonschema>=4" "json-numpy" "dlimp" || true
# Recurse: import prismatic and follow any further ImportError chain.
for _ in 1 2 3; do
    MISSING=$(python -c "
try:
    import prismatic
    print('')
except ModuleNotFoundError as e:
    print(e.name)
" 2>/dev/null)
    if [ -z "$MISSING" ]; then break; fi
    echo "[prismatic] missing module: $MISSING; pip install $MISSING"
    pip install "$MISSING" || true
done
python -c "import prismatic; print('[ok] prismatic =', prismatic.__file__)" \
    || echo "[FATAL] prismatic still not importable"

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
# IMPORTANT: czxlovesu03/BadVLA has 13,575 files (~200 GB total). It hosts
# multiple attack types (Text_Attack, Text_Image_Attack) × multiple LIBERO
# suites × multiple training-step checkpoints. We download ONLY the variant
# matching our clean reference (libero-spatial), and only ONE step ckpt.
#
# Override BADVLA_VARIANT_PATTERN at submit time to pick a different one.
python - <<PY
import os, sys, traceback
from huggingface_hub import snapshot_download, list_repo_files
hf_home = os.environ.get("HF_HOME", "/tmp/hf")
token = os.environ.get("HF_TOKEN")
variant_pat = os.environ.get(
    "BADVLA_VARIANT_PATTERN",
    "Text_Image_Attack/spatial_TI_4_step_ab/*"
)

repo = "czxlovesu03/BadVLA"
print(f"[hf] inspecting {repo} (filter: {variant_pat}) ...")
try:
    files = list_repo_files(repo, token=token)
    matches = [f for f in files if variant_pat.replace("*", "") in f
                                   or any(p in f for p in variant_pat.split("/"))]
    print(f"[hf] {len(files)} total files; matching pattern: {len(matches)}")
    for f in matches[:60]:
        print(f"  {f}")

    print(f"[hf] downloading variant '{variant_pat}' (allow_patterns) ...")
    p = snapshot_download(
        repo_id=repo, cache_dir=hf_home, token=token,
        allow_patterns=[variant_pat],
    )
    print(f"[ok] BadVLA variant at {p}")
    with open("/tmp/sharpguard.env", "a") as f:
        f.write(f"BADVLA_CKPT_DIR={p}\n")
        f.write(f"BADVLA_VARIANT_PATTERN={variant_pat}\n")
    import subprocess
    subprocess.run(["du", "-sh", p], check=False)
except Exception as e:
    print(f"[FATAL] failed to fetch {repo}: {e}")
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
