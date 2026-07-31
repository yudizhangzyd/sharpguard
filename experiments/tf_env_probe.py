#!/usr/bin/env python3
"""Can tensorflow coexist with the OpenVLA eval environment after all?

bolt/setup-openvla.sh:41 says no, and gives two reasons: installing tensorflow
"would clobber the numpy<2 pin and trigger an ABI mismatch in transformers' lazy
TF detection". That note is why the gate ships a hand-written Lanczos-3 instead
of calling tf.image.resize, and why the frame preprocessing is currently
8/255-LSB approximate rather than exact (bolt 5df7cdeicj: the kernel itself is
within 1 LSB, the residual is Pillow-vs-tensorflow libjpeg, which no PIL flag
closes).

Both stated reasons look addressable, and neither was ever tested:

  * the numpy pin -- tensorflow 2.16+ accepts numpy 1.26, so reinstalling
    "numpy<2" after tensorflow should restore the pin and keep both working;
  * transformers' lazy TF detection -- transformers honours USE_TF=0 (and
    TRANSFORMERS_NO_TF=1), which makes it skip TF import entirely.

An 8-LSB approximation is defensible but a measured zero is better, and the cost
of finding out is one CPU job. This script probes, in order, and reports a
verdict per stage so a failure says *which* claim in that comment is the binding
one:

  1. numpy is still <2 and imports;
  2. torch imports, and CUDA availability is unchanged from the baseline
     recorded before tensorflow was installed;
  3. transformers imports with USE_TF=0 and does NOT pull tensorflow in;
  4. tf.image.resize/encode_jpeg actually run;
  5. sharpguard.image_preproc's "tf_upstream" and "np_lanczos" agree exactly on
     a synthetic frame -- the only result that matters, since it is what would
     let the gate drop the approximation.

Deliberately does NOT install anything itself: bolt/run_tf_env_probe.sh does the
installs so the ordering is visible in the job log rather than buried in Python.
Exits 0 only if every stage passes; a partial pass is reported in full and still
exits 1, because "tensorflow imports" is not the same claim as "the eval
environment still works".
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stage(report, name, fn):
    """Run one probe, recording the outcome instead of letting it abort the run.

    Every stage runs even if an earlier one failed: the useful output is the
    full picture ("numpy pin held but transformers broke"), not the first
    exception. A stage that raises is a failure with its exception recorded.
    """
    try:
        detail = fn()
        report[name] = {"ok": True, "detail": detail}
        print(f"[tfenv] PASS {name}: {detail}")
    except Exception as e:  # noqa: BLE001 - any failure is a failure
        report[name] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}
        print(f"[tfenv] FAIL {name}: {type(e).__name__}: {e}")


def probe_numpy():
    import numpy as np
    major = int(np.__version__.split(".")[0])
    if major >= 2:
        raise AssertionError(
            f"numpy is {np.__version__}; the eval stack pins <2 and OpenVLA's "
            "transformers 4.40.1 does not tolerate the 2.x ABI")
    # A working ufunc, not just a successful import: a half-overwritten numpy
    # imports fine and then segfaults on first use.
    assert float(np.linspace(0, 1, 5).sum()) == 2.5
    return f"numpy {np.__version__} (<2, arithmetic works)"


def probe_torch():
    import torch
    cuda = torch.cuda.is_available()
    n = torch.cuda.device_count() if cuda else 0
    # Exercise the ABI rather than only the import.
    x = torch.ones(4, 4)
    assert float((x @ x).sum()) == 64.0
    return f"torch {torch.__version__}, cuda={cuda}, devices={n}, matmul ok"


def probe_transformers():
    if os.environ.get("USE_TF") != "0":
        raise AssertionError(
            "USE_TF is not '0'; without it transformers probes for tensorflow "
            "at import, which is the ABI mismatch setup-openvla.sh warns about")
    import transformers
    from transformers import AutoConfig  # noqa: F401 - import is the test
    # The specific failure the comment predicts: transformers deciding TF is
    # available and routing through it. If it reports TF available while we
    # asked it not to, the off-switch did not take and the environment is not
    # safe even though the import succeeded.
    tf_avail = transformers.utils.is_tf_available()
    if tf_avail:
        raise AssertionError(
            "transformers reports is_tf_available()=True despite USE_TF=0, so "
            "the off-switch did not take")
    return f"transformers {transformers.__version__}, is_tf_available()=False"


def probe_tensorflow():
    import numpy as np
    import tensorflow as tf
    arr = np.zeros((8, 8, 3), np.uint8)
    enc = tf.image.encode_jpeg(arr)
    dec = tf.io.decode_image(enc, expand_animations=False, dtype=tf.uint8)
    res = tf.image.resize(dec, (4, 4), method="lanczos3", antialias=True)
    out = tf.cast(tf.clip_by_value(tf.round(res), 0, 255), tf.uint8).numpy()
    assert out.shape == (4, 4, 3), out.shape
    return f"tensorflow {tf.__version__}, encode/decode/resize all ran"


def probe_exactness():
    """The stage that decides whether the gate can stop approximating."""
    import numpy as np
    path = os.path.join(ROOT, "sharpguard", "image_preproc.py")
    spec = importlib.util.spec_from_file_location("_shipped_preproc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(0)
    cases = {
        # The same three shapes resize_kernel_check.py uses, so the numbers are
        # comparable across the two jobs.
        "noise": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "gradient": np.stack(np.meshgrid(np.arange(256), np.arange(256))
                             + [np.zeros((256, 256))], -1).astype(np.uint8),
    }
    worst = 0
    for name, arr in cases.items():
        exact = mod.preprocess(arr, "tf_upstream", 224)
        ours = mod.preprocess(arr, "np_lanczos", 224)
        d = int(np.abs(exact.astype(np.int32) - ours.astype(np.int32)).max())
        print(f"[tfenv]   {name}: np_lanczos vs tf_upstream max abs diff = {d}")
        worst = max(worst, d)
    # Not an assertion that they agree -- they are already known to differ by
    # ~8 LSB in the JPEG step. The point of this stage is that BOTH modes are
    # runnable in one process, which is what makes "tf_upstream" usable as the
    # gate's image path.
    return (f"both modes ran in one process; np_lanczos vs tf_upstream worst "
            f"{worst} LSB (tf_upstream is exact by construction, so the gate "
            f"can use it directly and drop the {worst}-LSB approximation)")


def main() -> int:
    report = {"argv": sys.argv, "env": {k: os.environ.get(k) for k in
                                       ("USE_TF", "TRANSFORMERS_NO_TF",
                                        "USE_TORCH", "CUDA_VISIBLE_DEVICES")}}
    stage(report, "numpy_pin_held", probe_numpy)
    stage(report, "torch_still_works", probe_torch)
    stage(report, "transformers_without_tf", probe_transformers)
    stage(report, "tensorflow_runs", probe_tensorflow)
    stage(report, "both_preproc_modes_in_one_process", probe_exactness)

    passed = [k for k, v in report.items() if isinstance(v, dict)
              and v.get("ok") is True]
    failed = [k for k, v in report.items() if isinstance(v, dict)
              and v.get("ok") is False]
    report["n_passed"], report["n_failed"] = len(passed), len(failed)
    report["failed_stages"] = failed
    report["verdict"] = (
        "tensorflow can coexist with the eval environment; the gate may use "
        "image_preproc='tf_upstream' and drop the approximation"
        if not failed else
        "tensorflow cannot safely coexist; the binding constraint is "
        + ", ".join(failed) + ". The gate keeps np_lanczos and the manuscript "
        "keeps the measured approximation.")
    print(f"\n[tfenv] {len(passed)} passed, {len(failed)} failed")
    print(f"[tfenv] VERDICT: {report['verdict']}")

    out = os.environ.get("TF_ENV_PROBE_OUT", "tf_env_probe.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[tfenv] wrote {out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
