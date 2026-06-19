#!/usr/bin/env bash
# Setup for the real OpenVLA experiment.
#
# Hard-won lessons from rc277s5jcx + zumtsye6cu:
#  - The iris image's nvidia driver reports CUDA 12.0 (12080). Torch wheels
#    built for cu13x or cu12.4 will load but `cuda.is_available()` returns
#    False. Pin torch to a cu118 build (backward-compatible with driver 12.0).
#  - `accelerate` pulls torch as a hard dep — pin torch FIRST so its install
#    doesn't upgrade us to cu13x.
#  - `datasets` pulls numpy 2 by default; pin numpy<2 first.
#  - OpenVLA's processing_prismatic only works with transformers 4.40.x.
set -e -x

cd "$(dirname "$0")/.."

# 0) System libs the LIBERO sim stack needs even for off-screen rendering.
#    Without libxcb1 / libegl1, libero's import chain fails ("libxcb.so.1:
#    cannot open shared object file"). bolt iris runs as root so apt-get works.
apt-get update >/dev/null 2>&1 || true
apt-get install -y --no-install-recommends \
    libxcb1 libxcb-xinerama0 libxkbcommon0 \
    libegl1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    >/dev/null 2>&1 || echo "[warn] apt-get install failed (libero sim may fall back to offline ASR)"

# 1) Numpy lock (must happen before anything pulls numpy 2.x).
pip install "numpy<2"

# 2) Pin torch to a CUDA-12.0-compatible wheel BEFORE any package that
#    requires torch (accelerate / peft / timm) gets a chance to upgrade it.
pip install \
    --index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.4.1" "torchvision==0.19.1"

# 3) OpenVLA stack — torch already pinned, so these won't move it.
pip install \
    "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10" \
    "peft==0.11.1" "accelerate==0.30.1" "datasets==2.18.0" \
    "huggingface_hub>=0.20,<1.0" "safetensors>=0.4" \
    "sentencepiece>=0.1.99" "Pillow>=9.5"

# 4) Sim deps. We deliberately do NOT install tensorflow — it would clobber
#    the numpy<2 pin and trigger an ABI mismatch in transformers' lazy TF
#    detection. Real LIBERO RLDS parsing (TFRecords) is therefore disabled;
#    the runner falls back to synthetic-shape training data and relies on
#    LIBERO simulator rollouts (which only need mujoco/robosuite/libero) for
#    the headline SR/ASR numbers.
pip install "mujoco==3.1.6" "robosuite==1.4.1" \
            "easydict" "termcolor" "thop" "h5py" "imageio" "av" \
            "bddl" || true

LIBERO_DST=/tmp/LIBERO
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DST" 2>/dev/null || true
if [ -d "$LIBERO_DST" ]; then
    # NOTE: do NOT `pip install -r $LIBERO_DST/requirements.txt`. It pins
    # transformers==4.21.1 and bddl==1.0.1, which CONFLICT with OpenVLA's
    # transformers==4.40.1 / our bddl 3.6.0 (peft 0.11.1's loftq_utils
    # imports `cached_file` from transformers.utils which only exists in
    # the newer release). Doing so breaks every OpenVLA forward.
    # Instead, hand-pick LIBERO's runtime deps with loose versions.
    pip install \
        "hydra-core" "omegaconf" \
        "matplotlib" "gym" "cloudpickle" "future" "einops" \
        "imageio-ffmpeg" "easydict" \
        || echo "[warn] some libero deps failed"
    # robomimic with --no-deps to avoid pulling its old transformers pin.
    pip install --no-deps "robomimic" || echo "[warn] robomimic failed"

    pip install --no-deps -e "$LIBERO_DST" || echo "[warn] libero pip -e failed"
fi
echo 'export MUJOCO_GL=egl' >> /tmp/sharpguard.env
echo 'export PYOPENGL_PLATFORM=egl' >> /tmp/sharpguard.env

# Verify the OpenVLA-required pins survived all of the above.
python - <<'PY'
import sys
import transformers, peft
expected_tf = "4.40"
if not transformers.__version__.startswith(expected_tf):
    print(f"[FATAL] transformers={transformers.__version__}, expected {expected_tf}.x — "
          "something downgraded it (LIBERO deps?). Aborting setup.")
    sys.exit(2)
print(f"[ok] transformers={transformers.__version__} | peft={peft.__version__}")
PY

# Re-pin numpy<2 if anything bumped it.
pip install "numpy<2" --force-reinstall --no-deps
echo 'export MUJOCO_GL=egl' >> /tmp/sharpguard.env
echo 'export PYOPENGL_PLATFORM=egl' >> /tmp/sharpguard.env

# Pre-create robosuite macros.py (its first-run setup script asks on stdin
# and bolt has no stdin → EOFError).
python - <<PY || true
import os, shutil
try:
    import robosuite
except Exception:
    raise SystemExit(0)
rs_dir = os.path.dirname(robosuite.__file__)
priv = os.path.join(rs_dir, "macros_private.py")
pub = os.path.join(rs_dir, "macros.py")
if os.path.exists(priv) and not os.path.exists(pub):
    shutil.copy(priv, pub)
    print(f"[robosuite] wrote {pub}")
elif os.path.exists(pub):
    print(f"[robosuite] {pub} already exists")
else:
    with open(pub, "w") as f:
        f.write("ASSETS_PATH = None\nDATASET_PATH = None\n")
    print(f"[robosuite] wrote minimal {pub}")
PY

# Pre-init LIBERO. On first import LIBERO calls `init_config()` which does
# `input("Do you want to specify a custom path? (Y/N):")` — EOFs on bolt.
# IMPORTANT: do NOT use `python - <<PY` here; the heredoc occupies stdin so
# the piped "N" never reaches input(). Write the program to a file first.
cat > /tmp/_libero_init.py <<'PY'
import os, sys, traceback
try:
    import libero.libero
    from libero.libero import benchmark, get_libero_path
    print("[libero] config initialized; benchmarks:", list(benchmark.get_benchmark_dict()))
    for k in ("benchmark_root", "bddl_files", "init_states", "datasets", "assets"):
        try:
            print(f"[libero]  {k}:", get_libero_path(k))
        except Exception:
            pass
except Exception as e:
    print("[libero] pre-init failed:", e)
    traceback.print_exc()
PY
echo "N" | python /tmp/_libero_init.py || true

# Belt-and-suspenders: pre-write ~/.libero/config.yaml with defaults pointing
# at the installed libero package, in case the prompt-based init didn't.
python - <<'PY' || true
import os, yaml
try:
    import libero
    pkg = os.path.dirname(libero.__file__)
except Exception:
    raise SystemExit(0)
cfg_dir = os.path.expanduser("~/.libero")
os.makedirs(cfg_dir, exist_ok=True)
cfg_path = os.path.join(cfg_dir, "config.yaml")
if not os.path.exists(cfg_path):
    cfg = {
        "benchmark_root":      os.path.join(pkg, "libero"),
        "bddl_files_folder":   os.path.join(pkg, "libero", "bddl_files"),
        "init_states_folder":  os.path.join(pkg, "libero", "init_files"),
        "datasets_folder":     os.path.join(pkg, "libero", "datasets"),
        "assets_folder":       os.path.join(pkg, "libero", "assets"),
    }
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    print(f"[libero] wrote default {cfg_path}")
else:
    print(f"[libero] {cfg_path} already exists:")
    print(open(cfg_path).read())
PY
ls -la ~/.libero/ 2>/dev/null || echo "[libero] no ~/.libero/ written"

# Re-assert numpy<2 in case any sim dep tried to upgrade it.
pip install "numpy<2" --force-reinstall --no-deps

# 5) BadVLA — repos all 404; runner uses our re-implementation by default.
if [ -n "${BADVLA_REPO:-}" ]; then
    git clone --depth 1 "$BADVLA_REPO" /tmp/BadVLA \
        && pip install --no-deps -e /tmp/BadVLA \
        && echo "BADVLA_DIR=/tmp/BadVLA" >> /tmp/sharpguard.env \
        || echo "[warn] BADVLA_REPO clone failed: $BADVLA_REPO"
fi

# 6) LIBERO RLDS data.
python - <<PY || true
import os, sys
from huggingface_hub import snapshot_download
suite = os.environ.get("LIBERO_SUITE", "libero_spatial_no_noops")
try:
    path = snapshot_download(repo_id="openvla/modified_libero_rlds",
                              repo_type="dataset",
                              cache_dir=os.environ.get("HF_HOME", "/tmp/hf"),
                              token=os.environ.get("HF_TOKEN"),
                              allow_patterns=[f"{suite}/*", f"*{suite}*"])
    print(f"[ok] LIBERO data at {path}")
    with open("/tmp/sharpguard.env", "a") as f:
        f.write(f"LIBERO_DATA_DIR={path}\n")
except Exception as e:
    print(f"[warn] LIBERO HF Hub download failed: {e}")
PY

# 7) OpenVLA-7B itself.
python - <<PY
import os
from huggingface_hub import snapshot_download
print("Pre-fetching openvla/openvla-7b ...")
p = snapshot_download("openvla/openvla-7b",
                      cache_dir=os.environ.get("HF_HOME", "/tmp/hf"),
                      token=os.environ.get("HF_TOKEN"))
print("OK:", p)
PY

# 8) Sanity. CRITICAL: cuda.is_available() must be True or the run is cpu-only.
python - <<PY
import sys
import torch, transformers, peft
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__,
      "| cuda", torch.cuda.is_available(), "| ngpus", torch.cuda.device_count())
if not torch.cuda.is_available():
    print("[FATAL] CUDA not available — driver/torch mismatch")
    sys.exit(2)
PY

# 8) LIBERO end-to-end smoke. Imports → builds → resets → steps → closes a real
#    OffScreenRenderEnv. Surfaces missing deps / EGL / rendering issues HERE
#    (in setup) instead of 30 min into a doomed training run. Result is
#    recorded in /tmp/sharpguard.env as LIBERO_SIM_OK={0,1}.
cat > /tmp/_libero_env_smoke.py <<'PY'
import os, sys, traceback
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bench = benchmark.get_benchmark_dict()
    suite_name = "libero_spatial"
    if suite_name not in bench:
        print(f"[smoke] suite {suite_name} not found; available: {list(bench)}")
        sys.exit(1)
    suite = bench[suite_name]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"),
                         task.problem_folder, task.bddl_file)
    print(f"[smoke] bddl: {bddl}")
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=224, camera_widths=224,
    )
    obs = env.reset()
    keys = list(obs.keys()) if hasattr(obs, "keys") else type(obs).__name__
    print(f"[smoke] reset OK; obs keys: {keys}")
    import numpy as np
    action = np.zeros(7, dtype=np.float32)
    obs, _, _, _ = env.step(action)
    print(f"[smoke] step OK")
    env.close()
    print("[smoke] LIBERO env smoke test PASSED")
except Exception as e:
    print(f"[smoke] LIBERO env smoke test FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)
PY
if python /tmp/_libero_env_smoke.py; then
    echo "LIBERO_SIM_OK=1" >> /tmp/sharpguard.env
else
    echo "LIBERO_SIM_OK=0" >> /tmp/sharpguard.env
    echo "[note] LIBERO sim smoke failed; runner will skip sim rollouts."
fi
echo "--- /tmp/sharpguard.env ---"
cat /tmp/sharpguard.env 2>/dev/null || true

