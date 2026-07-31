#!/usr/bin/env python3
"""Validate sharpguard/image_preproc.py against upstream's actual tensorflow code.

The reference diff (bolt htrg4uchwi) showed upstream prepares every LIBERO frame
with a JPEG round-trip plus tf.image.resize(method="lanczos3", antialias=True) to
224, and the decoder gate did none of that. The gate cannot simply import
tensorflow: bolt/setup-openvla.sh documents that installing it clobbers the eval
environment's numpy<2 pin and breaks transformers' lazy TF detection.

The first attempt at a substitute was Pillow's LANCZOS. This script measured it
(bolt q5z79humta) and it failed the criterion fixed in advance -- up to 23/255
LSB against a 4-LSB ceiling. So the resize was reimplemented from the definition
of a scaled Lanczos-3 kernel in sharpguard/image_preproc.py, and this script's
job changed: it now checks that reimplementation against tensorflow.

Three quantities are measured separately, because "the whole pipeline differs by
N LSB" does not say which step to fix:

  1. jpeg_only    -- Pillow's JPEG encode/decode vs tf.image.encode_jpeg +
                     tf.io.decode_image, no resize. Isolates the compressor.
                     Several Pillow `subsampling` settings are tried, because
                     tf.image.encode_jpeg defaults to chroma_downsampling=True
                     and the matching Pillow value is a fact to measure, not
                     guess.
  2. resize_only  -- our numpy Lanczos-3 vs tf.image.resize on the SAME uint8
                     input, no JPEG. Isolates the kernel.
  3. full         -- image_preproc.preprocess(mode="np_lanczos") vs the whole
                     upstream resize_image. This is the number that matters,
                     and it is what the gate's claim rests on.
                     "pil_lanczos" is measured alongside it for contrast.

Interpretation, decided before seeing the output so it cannot be fitted to it.
Applied to `full` for np_lanczos:
  * max abs diff == 0        -> bit-identical; the gate reproduces upstream and
    may say so without qualification.
  * max abs diff <= 1 LSB    -> equivalent up to uint8 rounding; the gate may
    say "matches upstream to within one intensity level", with the number given.
  * max abs diff <= 4 LSB and mean << 1 -> usable, but the residual must be
    named as a caveat in the manuscript.
  * larger -> np_lanczos is not a reproduction either. Do not ship the claim;
    report the gate as approximate and say what the gap is.

The script exits non-zero when np_lanczos exceeds 1 LSB, so the job's exit code
carries the finding rather than burying it in a log. `pil_lanczos` is reported
but never gates the exit -- it is already known to fail, and is kept only so the
factorial's image arm has a measured label.

Runs on CPU. Inputs are synthetic but structured (smooth gradients, hard edges,
and high-frequency noise), because a resize-kernel difference shows up at edges
and in high frequencies, not on flat regions -- so testing only on noise would
overstate the gap and testing only on gradients would hide it.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys

VENV = "/tmp/tfvenv"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pillow subsampling codes to try for the JPEG step. 0 is 4:4:4 (no chroma
# downsampling), 1 is 4:2:2, 2 is 4:2:0. tf.image.encode_jpeg's default is
# chroma_downsampling=True, so 2 is the a-priori guess -- which is exactly why
# all three get measured.
SUBSAMPLINGS = (0, 1, 2)


def load_shipped():
    """Import sharpguard/image_preproc.py by path, bypassing the package.

    Deliberately not `import sharpguard.image_preproc`: that runs
    sharpguard/__init__, which reaches torch and tqdm through measurement.py,
    and this venv holds only tensorflow, Pillow and numpy. Loading the file
    directly is what makes it possible to validate the code that actually
    ships instead of a copy of it pasted into this script -- the copy is how a
    check like this silently stops testing anything.
    """
    path = os.path.join(ROOT, "sharpguard", "image_preproc.py")
    spec = importlib.util.spec_from_file_location("_shipped_preproc", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


# --- the three upstream steps, in tensorflow, exactly as upstream writes them --

def tf_jpeg(arr):
    import tensorflow as tf
    t = tf.image.encode_jpeg(arr)
    return tf.io.decode_image(t, expand_animations=False, dtype=tf.uint8).numpy()


def tf_resize(arr, size=224):
    import tensorflow as tf
    t = tf.image.resize(arr, (size, size), method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(t), 0, 255), tf.uint8).numpy()


def tf_full(arr, size=224):
    """Upstream experiments/robot/libero/libero_utils.resize_image, verbatim."""
    return tf_resize(tf_jpeg(arr), size)


def pil_jpeg(arr, quality=95, subsampling=0):
    from PIL import Image
    import numpy as np
    buf = io.BytesIO()
    Image.fromarray(arr).convert("RGB").save(buf, format="JPEG",
                                             quality=quality,
                                             subsampling=subsampling)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def stats(a, b):
    import numpy as np
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return {
        "shape": list(a.shape),
        "max_abs_diff": int(d.max()),
        "mean_abs_diff": round(float(d.mean()), 4),
        "frac_pixels_differing": round(float((d > 0).mean()), 4),
        "frac_pixels_diff_gt_1": round(float((d > 1).mean()), 4),
    }


def worst_of(cases):
    """Largest max_abs_diff across cases; -1 if any case failed to compare."""
    vals = [c.get("max_abs_diff") for c in cases.values()]
    return -1 if any(v is None for v in vals) else max(vals)


def compare() -> int:
    """Child-process entry point: tensorflow is importable here."""
    import numpy as np
    report = {"root": ROOT}
    try:
        import tensorflow as tf
        report["tf_version"] = tf.__version__
    except ImportError as e:
        print(f"[resize] FATAL: tensorflow not importable in the child: {e}")
        return 2
    from PIL import Image
    report["pillow_version"] = Image.__version__
    report["numpy_version"] = np.__version__

    try:
        shipped = load_shipped()
    except Exception as e:
        print(f"[resize] FATAL: could not load the shipped module: {e!r}")
        return 2
    report["shipped_modes"] = list(shipped.MODES)

    inputs = build_inputs()

    # --- 1. the JPEG step alone, over candidate Pillow subsampling settings ---
    print("[resize] --- step 1: JPEG round-trip, Pillow vs tf.image ---")
    jpeg = {}
    for ss in SUBSAMPLINGS:
        cases = {n: stats(pil_jpeg(a, subsampling=ss), tf_jpeg(a))
                 for n, a in inputs.items()}
        jpeg[f"subsampling_{ss}"] = {"cases": cases, "worst": worst_of(cases)}
        print(f"[resize] subsampling={ss}: worst={worst_of(cases):>4} LSB  " +
              "  ".join(f"{n}={c['max_abs_diff']}" for n, c in cases.items()))
    best_ss = min(SUBSAMPLINGS, key=lambda s: jpeg[f"subsampling_{s}"]["worst"])
    report["jpeg_only"] = jpeg
    report["jpeg_best_subsampling"] = best_ss
    # The shipped default must be the measured-best value, or the gate's JPEG
    # step is needlessly further from upstream than it has to be.
    shipped_default = shipped.jpeg_roundtrip.__defaults__[1]
    report["jpeg_shipped_subsampling"] = shipped_default
    report["jpeg_shipped_is_best"] = (shipped_default == best_ss)
    print(f"[resize] best Pillow subsampling = {best_ss}; "
          f"shipped default = {shipped_default}"
          f"{'' if shipped_default == best_ss else '  <-- MISMATCH'}")

    # --- 2. the resize kernel alone, on identical uint8 input, no JPEG ---
    print("\n[resize] --- step 2: Lanczos-3 resize alone, on identical input ---")
    kern = {}
    for name, arr in inputs.items():
        kern[name] = stats(shipped.lanczos3_resize(arr, 224), tf_resize(arr, 224))
        c = kern[name]
        print(f"[resize] {name:<10} max={c['max_abs_diff']:>4} "
              f"mean={c['mean_abs_diff']:>8} "
              f"frac_diff={c['frac_pixels_differing']:>7} "
              f"frac_gt1={c['frac_pixels_diff_gt_1']:>7}")
    report["resize_only"] = {"cases": kern, "worst": worst_of(kern)}
    print(f"[resize] kernel-only worst = {worst_of(kern)} LSB")

    # --- 3. the full pipeline, both candidate modes vs upstream ---
    print("\n[resize] --- step 3: full pipeline vs upstream resize_image ---")
    full = {}
    for mode in ("np_lanczos", "pil_lanczos"):
        cases = {n: stats(shipped.preprocess(a, mode, 224), tf_full(a, 224))
                 for n, a in inputs.items()}
        full[mode] = {"cases": cases, "worst": worst_of(cases)}
        print(f"[resize] {mode}:")
        for n, c in cases.items():
            print(f"[resize]   {n:<10} max={c['max_abs_diff']:>4} "
                  f"mean={c['mean_abs_diff']:>8} "
                  f"frac_diff={c['frac_pixels_differing']:>7} "
                  f"frac_gt1={c['frac_pixels_diff_gt_1']:>7}")
        print(f"[resize]   worst = {cases and full[mode]['worst']} LSB")
    report["full"] = full

    # Sanity: mode "none" must still be a pass-through, or the anchor cell of
    # the factorial is not the configuration the four failed gates ran and the
    # whole comparison loses its baseline.
    passthrough = all(
        np.array_equal(shipped.preprocess(a, "none"), a) for a in inputs.values())
    report["none_is_passthrough"] = bool(passthrough)
    if not passthrough:
        print("[resize] FATAL: mode 'none' is no longer a pass-through.")
        return 2

    worst = full["np_lanczos"]["worst"]
    report["worst_max_abs_diff"] = worst
    # Thresholds from this file's docstring, fixed in advance of the measurement.
    if worst == 0:
        report["verdict"] = "bit-identical to upstream"
    elif worst <= 1:
        report["verdict"] = "equivalent up to uint8 rounding"
    elif worst <= 4:
        report["verdict"] = "rounding-scale difference; usable with a caveat"
    else:
        report["verdict"] = ("materially different; the gate must not claim to "
                             "reproduce upstream's preprocessing")
    print(f"\n[resize] np_lanczos worst max abs diff = {worst} LSB "
          f"(pil_lanczos, for contrast: {full['pil_lanczos']['worst']})")
    print(f"[resize] VERDICT: {report['verdict']}")

    out = os.environ.get("RESIZE_CHECK_OUT", "resize_kernel_check.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[resize] wrote {out}")
    # Non-zero when the shipped mode is not within rounding of upstream, so the
    # exit code carries the finding. pil_lanczos deliberately does not gate.
    return 0 if 0 <= worst <= 1 else 1


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
