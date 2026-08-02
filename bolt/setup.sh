#!/usr/bin/env bash
#
# Bolt setup: install SharpGuard's deps on top of the iris image.
#
# "The iris image already has CUDA + recent torch" is what the comment here used
# to say, and it is why this script did not pin torch and then printed
# `cuda False` and exited 0. requirements.txt now pins the versions
# setup-openvla.sh has always used; torch is installed FIRST, from the cu118
# index, so that a later dependency cannot pull a build the pod driver cannot run.
set -e -x

cd "$(dirname "$0")/.."

pip install --upgrade pip

# Torch before anything that depends on it: `accelerate`, `peft` and `datasets`
# all list torch, and pip will happily satisfy that by upgrading to whatever is
# newest -- which is how a cu130 wheel landed on a CUDA-12.8 pod.
pip install --index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.4.1" "torchvision==0.19.1"
pip install -r requirements.txt
# OpenVLA / LIBERO scale-up extras (safe to install for mini-bench too).
pip install peft accelerate datasets || true

# The gate, not a printout. This check existed before and reported the exact
# failure that wasted a GPU slot -- `cuda False | ngpus 1` -- and the job carried
# on to run a 7B judge on CPU at 180 s/pair until its timeout. A visible-but-
# unusable GPU is a setup failure, so setup is where it stops.
python - <<'PY'
import sys
import torch, transformers, sharpguard  # noqa: F401  (import check)
# sharpguard re-exports lazily now, so bare `import sharpguard` no longer
# touches the torch estimators. Resolve one attribute so this stays the same
# check it was before -- setup should fail here, not in a job's first epoch.
sharpguard.measure_all  # noqa: B018
print(f"torch {torch.__version__} | transformers {transformers.__version__} "
      f"| cuda {torch.cuda.is_available()} | ngpus {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"[FATAL] CUDA unavailable under torch {torch.__version__}.")
    if n:
        print(f"        {n} device(s) visible but unusable: the torch build does")
        print(f"        not match this pod's driver. Check the pins in")
        print(f"        requirements.txt rather than letting pip resolve a floor.")
    print(f"        Stopping in setup: every job using this script needs a GPU,")
    print(f"        and a CPU fallback here does not finish inside any timeout.")
    sys.exit(1)
mm = (torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda")).sum()
print(f"[setup] cuda matmul ok ({float(mm):+.4f}), gpu={torch.cuda.get_device_name(0)}")
PY
