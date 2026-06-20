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

# OpenVLA-OFT vs standard OpenVLA: the BadVLA ckpts bundle modeling files
# that `import from prismatic.extern.hf ...`, requiring the openvla-oft
# package. That package hard-pins torch==2.2.0 / draccus==0.8.0 /
# tensorflow==2.15.0 / dlimp@git+url (incompatible with our pinned env,
# and dlimp's git+url is blocked by the proxy).
#
# Solution applied AFTER we download the BadVLA ckpts (further down).

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

# ---- Overlay standard OpenVLA modeling files onto BadVLA ckpts ----
# BadVLA ships modeling_prismatic.py that requires openvla-oft's `prismatic`
# package (huge dep chain incompatible with our env). The ckpt's actual
# architecture is `OpenVLAForActionPrediction` — the SAME class as standard
# `openvla/openvla-7b`. Standard OpenVLA's bundled modeling files are
# self-contained (no prismatic dep). We copy them over.
python - <<'PY' || true
import os, shutil, json
from huggingface_hub import hf_hub_download
hf_home = os.environ.get("HF_HOME", "/tmp/hf")
token = os.environ.get("HF_TOKEN")

standard_repo = "openvla/openvla-7b"
files_to_overlay = [
    "configuration_prismatic.py",
    "modeling_prismatic.py",
    "processing_prismatic.py",
]
overlay_files = {}
for fn in files_to_overlay:
    try:
        p = hf_hub_download(repo_id=standard_repo, filename=fn,
                             cache_dir=hf_home, token=token)
        overlay_files[fn] = p
        print(f"[overlay] standard {fn}: {p}")
    except Exception as e:
        print(f"[overlay] failed to fetch {fn}: {e}")

badvla_root = os.environ.get("BADVLA_CKPT_DIR", "")
if not badvla_root or not os.path.exists(badvla_root):
    # Re-read from /tmp/sharpguard.env (this PY runs in a sub-process that
    # doesn't have the parent shell's env updates after `f.write`).
    env_path = "/tmp/sharpguard.env"
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith("BADVLA_CKPT_DIR="):
                badvla_root = line.split("=", 1)[1].strip()
                break
if not badvla_root or not os.path.exists(badvla_root):
    print(f"[overlay] BADVLA_CKPT_DIR still not found; skipping overlay")
    raise SystemExit(0)

print(f"[overlay] walking {badvla_root} for ckpts ...")
patched = 0
for root, dirs, files in os.walk(badvla_root):
    if "config.json" not in files:
        continue
    try:
        cfg = json.load(open(os.path.join(root, "config.json")))
    except Exception:
        continue
    arch = cfg.get("architectures", [""])[0] if cfg.get("architectures") else ""
    if arch != "OpenVLAForActionPrediction":
        continue
    print(f"[overlay] patching {root}")
    for fn, src in overlay_files.items():
        dst = os.path.join(root, fn)
        if os.path.islink(dst) or os.path.exists(dst):
            try: os.remove(dst)
            except Exception: pass
        shutil.copy(src, dst)
    patched += 1
print(f"[overlay] patched {patched} ckpts")
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
