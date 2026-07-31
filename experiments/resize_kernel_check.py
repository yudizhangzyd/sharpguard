#!/usr/bin/env python3
"""Measure how far `pil_lanczos` is from upstream's exact resize.

The reference diff (bolt htrg4uchwi) showed upstream prepares every LIBERO
frame with a JPEG round-trip plus tf.image.resize(method="lanczos3",
antialias=True) to 224, and the decoder gate did none of that. The fix belongs
in the gate -- but bolt/setup-openvla.sh documents that installing tensorflow
into the eval environment clobbers its numpy<2 pin and breaks transformers'
lazy TF detection, so the gate runs a Pillow implementation of the same three
steps instead.

That substitution needs a number attached to it. PIL's LANCZOS filter and TF's
lanczos3+antialias are both Lanczos-3 with support scaled by the resize ratio,
so they *should* agree closely, but "should" is exactly the kind of reasoning
that produced four failed gates. This script installs tensorflow into a
throwaway venv, runs both paths on the same inputs, and reports the pixel
disagreement -- so the manuscript can state the size of the approximation
instead of hand-waving it.

Interpretation, decided before seeing the output so it cannot be fitted to it:
  * max abs diff <= 1 LSB  -> the substitution is immaterial; report it as
    equivalent up to uint8 rounding.
  * max abs diff <= 4 LSB and mean << 1 -> a rounding-scale difference;
    usable, and the residual should be named as a caveat.
  * larger -> the gate must not claim to reproduce upstream's preprocessing;
    either find a tf-compatible environment or report the gate as
    approximate.

Runs on CPU. Inputs are synthetic but structured (smooth gradients, hard
edges, and high-frequency noise), because a resize-kernel difference shows up
at edges and in high frequencies, not on flat regions -- so testing only on
noise would overstate the gap and testing only on gradients would hide it.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys

VENV = "/tmp/tfvenv"


def build_inputs():
    """Three 256x256 uint8 images that stress a resize kernel differently."""
    import numpy as np
    y, x = np.mgrid[0:256, 0:256].astype(np.float32)
    imgs = {}
    # Smooth: a kernel difference is nearly invisible here. Establishes a floor.
    imgs["gradient"] = np.stack([(x / 255 * 255), (y / 255 * 255),
                                 ((x + y) / 510 * 255)], -1).astype(np.uint8)
    # Hard edges: where lanczos ringing lives, so where the kernels can diverge.
    edges = np.zeros((256, 256, 3), np.uint8)
    edges[64:192, 64:192] = 255
    edges[:, ::16] = 128
    edges[::16, :] = 32
    imgs["edges"] = edges
    # High-frequency noise: the worst case for any antialiasing difference.
    rng = np.random.default_rng(0)
    imgs["noise"] = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    return imgs


def pil_path(arr, size=224):
    """Exactly what sharpguard.libero_sim._preprocess_image('pil_lanczos') does."""
    import numpy as np
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).convert("RGB").save(buf, format="JPEG", quality=95)
    buf.seek(0)
    pil = Image.open(buf).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def tf_path(arr, size=224):
    """Exactly upstream's experiments/robot/libero/libero_utils.resize_image."""
    import tensorflow as tf
    t = tf.image.encode_jpeg(arr)
    t = tf.io.decode_image(t, expand_animations=False, dtype=tf.uint8)
    t = tf.image.resize(t, (size, size), method="lanczos3", antialias=True)
    t = tf.cast(tf.clip_by_value(tf.round(t), 0, 255), tf.uint8)
    return t.numpy()


def compare() -> int:
    """Child-process entry point: tensorflow is importable here."""
    import numpy as np
    report = {"cases": {}}
    try:
        import tensorflow as tf
        report["tf_version"] = tf.__version__
    except ImportError as e:
        print(f"[resize] FATAL: tensorflow not importable in the child: {e}")
        return 2
    from PIL import Image
    report["pillow_version"] = Image.__version__

    worst = 0.0
    for name, arr in build_inputs().items():
        a, b = pil_path(arr), tf_path(arr)
        if a.shape != b.shape:
            print(f"[resize] FATAL: shape mismatch on {name}: {a.shape} {b.shape}")
            return 2
        d = np.abs(a.astype(np.int32) - b.astype(np.int32))
        case = {
            "shape": list(a.shape),
            "max_abs_diff": int(d.max()),
            "mean_abs_diff": round(float(d.mean()), 4),
            "frac_pixels_differing": round(float((d > 0).mean()), 4),
            "frac_pixels_diff_gt_1": round(float((d > 1).mean()), 4),
        }
        report["cases"][name] = case
        worst = max(worst, case["max_abs_diff"])
        print(f"[resize] {name:<10} max={case['max_abs_diff']:>4} "
              f"mean={case['mean_abs_diff']:>8} "
              f"frac_diff={case['frac_pixels_differing']:>7} "
              f"frac_gt1={case['frac_pixels_diff_gt_1']:>7}")

    report["worst_max_abs_diff"] = int(worst)
    # The thresholds are the ones stated in this file's docstring, fixed in
    # advance of the measurement.
    if worst <= 1:
        report["verdict"] = "equivalent up to uint8 rounding"
    elif worst <= 4:
        report["verdict"] = "rounding-scale difference; usable with a caveat"
    else:
        report["verdict"] = ("materially different; the gate must not claim to "
                             "reproduce upstream's preprocessing")
    print(f"\n[resize] worst max abs diff = {worst} LSB over 3 cases")
    print(f"[resize] VERDICT: {report['verdict']}")

    out = os.environ.get("RESIZE_CHECK_OUT", "resize_kernel_check.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[resize] wrote {out}")
    # Non-zero only when the substitution is not defensible, so the job's exit
    # code carries the finding.
    return 0 if worst <= 4 else 1


def main() -> int:
    if os.environ.get("RESIZE_CHECK_CHILD") == "1":
        return compare()

    # Parent: build an isolated venv so tensorflow never touches the eval
    # environment's numpy<2 pin. This is the whole reason the gate cannot just
    # import tf directly.
    print(f"[resize] creating throwaway venv at {VENV}")
    for cmd in ([sys.executable, "-m", "venv", VENV],
                [f"{VENV}/bin/pip", "install", "--quiet", "--upgrade", "pip"],
                [f"{VENV}/bin/pip", "install", "--quiet",
                 "tensorflow-cpu", "pillow", "numpy"]):
        p = subprocess.run(cmd, capture_output=True, text=True)
        print(f"[resize] {' '.join(cmd[:4])} -> rc={p.returncode}")
        if p.returncode != 0:
            print((p.stdout or "")[-1500:])
            print((p.stderr or "")[-3000:])
            print("[resize] FATAL: could not build the tf venv. Not emitting a "
                  "report -- 'no differences measured' would read as 'no "
                  "differences exist'.")
            return 2

    env = dict(os.environ, RESIZE_CHECK_CHILD="1")
    p = subprocess.run([f"{VENV}/bin/python", os.path.abspath(__file__)],
                       env=env)
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())
