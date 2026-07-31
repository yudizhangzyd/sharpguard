#!/usr/bin/env bash
# Test whether tensorflow can coexist with the OpenVLA eval environment.
#
# bolt/setup-openvla.sh:41 asserts it cannot, for two reasons: the numpy<2 pin
# and transformers' lazy TF detection. That assertion has never been tested, and
# it currently costs the gate an 8/255-LSB approximation in the frame
# preprocessing (bolt 5df7cdeicj). If it turns out to be wrong, the gate can
# call tf.image.resize directly and be exact by construction.
#
# The installs happen HERE rather than inside Python so the ordering -- which is
# the whole question -- is visible in the job log.
set -e -x

cd "$(dirname "$0")/.."

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/tf-env-probe"
mkdir -p "$OUT_DIR"
export TF_ENV_PROBE_OUT="$OUT_DIR/tf_env_probe.json"

# Tell transformers not to look for tensorflow at import time. This is the
# documented off-switch for the ABI mismatch setup-openvla.sh warns about, and
# testing it is half the point of this job.
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export USE_TORCH=1

# Build the normal eval environment first, unchanged, so this measures "the eval
# environment plus tensorflow" and not some other environment.
bash bolt/setup-openvla.sh

echo "==== numpy/torch BEFORE tensorflow ===="
python -c "import numpy, torch; print('numpy', numpy.__version__); \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# tensorflow-cpu, not tensorflow: the GPU build pulls its own CUDA/cuDNN wheels
# and those are what would actually fight with torch. Nothing here needs a GPU
# for tensorflow -- the resize is a 256->224 reduction on one frame.
pip install "tensorflow-cpu==2.16.2"

echo "==== numpy AFTER tensorflow (expected to have moved) ===="
python -c "import numpy; print('numpy', numpy.__version__)" || true

# Restore the pin. tensorflow 2.16 accepts numpy 1.26, so this should leave both
# working; if it does not, the probe's numpy stage reports it.
pip install "numpy<2"

echo "==== numpy AFTER restoring the pin ===="
python -c "import numpy; print('numpy', numpy.__version__)" || true

set +e
python experiments/tf_env_probe.py
RC=$?
set -e

echo ""
echo "==== tf_env_probe.json ===="
cat "$TF_ENV_PROBE_OUT" || echo "(no report: the probe died before writing)"
# Also dump the resolved versions, so a failure can be diagnosed without a
# second job.
pip list 2>/dev/null | grep -iE "^(numpy|torch|tensorflow|transformers|protobuf|typing) " || true
exit "$RC"
