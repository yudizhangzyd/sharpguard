"""Unit tests for the two free parameters in the decoder-gate factorial.

Why this runs before the GPU job. The last four gate submissions died in 2
minutes on an `AttributeError` in a test helper, which was cheap only because
the test ran first. The same applies here: the factorial is worthless if a cell
is mis-implemented, and every property below is checkable on a CPU in a second.

Covered:
  * `_apply_gripper_transform` -- every arm leaves the 6 continuous dims
    untouched (else an SR delta is unattributable to the gripper), "none" is
    bit-identical to the input (so that cell really does reproduce the four
    failed gates), the active arms emit valid OSC values, the arms are pairwise
    distinct (else the factorial silently has duplicate cells), and "openvla"
    equals upstream's -sign(2g - 1) composition;
  * `_preprocess_image` -- "none" is the identity at 256px, "pil_lanczos"
    returns 224px uint8 and is distinguishable both from a no-op and from a
    plain bilinear resize;
  * `UPSTREAM_MAX_STEPS` -- pinned to what upstream's run_libero_eval.py
    actually contains, because a flat 400 is what made the gate's libero_10
    zero uninterpretable.
"""

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# `_apply_gripper_transform` is pure numpy, but `sharpguard/__init__` pulls in
# torch transitively. Stub the minimum surface so this test also runs in the
# verify-paper-numbers CI job, which installs numpy only and would otherwise
# have to skip the one check that guards the A/B's correctness. If torch is
# genuinely present (the GPU box) the real module wins.
try:
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    stub = types.ModuleType("torch")
    stub.__path__ = []          # make it a package so `torch.nn` can resolve
    stub.Tensor = type("Tensor", (), {})
    stub.load = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("torch stub: load() is not available in this environment"))
    stub.no_grad = lambda: (lambda fn: fn)
    stub.device = lambda *a, **k: None
    stub.float32 = stub.float16 = stub.bfloat16 = None
    sys.modules["torch"] = stub
    for sub in ("nn", "nn.functional", "utils", "utils.data"):
        mod = types.ModuleType(f"torch.{sub}")
        mod.__path__ = []
        mod.Module = type("Module", (), {})
        sys.modules[f"torch.{sub}"] = mod
        parent = sys.modules["torch" if "." not in sub
                             else f"torch.{sub.rsplit('.', 1)[0]}"]
        setattr(parent, sub.rsplit(".", 1)[-1], mod)
    print("[env] torch absent; using a stub for the import (the function under "
          "test is pure numpy)")

from sharpguard.libero_sim import (  # noqa: E402
    GRIPPER_TRANSFORMS,
    IMAGE_PREPROCS,
    UPSTREAM_MAX_STEPS,
    _apply_gripper_transform as tf,
    _preprocess_image,
)


def main() -> int:
    fails = []

    def ok(name, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    ok("all four arms are registered",
       tuple(GRIPPER_TRANSFORMS) == ("none", "invert", "binvert", "openvla"),
       str(GRIPPER_TRANSFORMS))

    base = np.array([0.11, -0.22, 0.33, -0.04, 0.05, -0.06, 0.7], dtype=np.float32)

    # 1. No arm may touch the 6 continuous dims.
    for mode in GRIPPER_TRANSFORMS:
        out = tf(base.copy(), mode)
        ok(f"{mode!r} leaves the 6 continuous dims bit-identical",
           np.array_equal(out[:6], base[:6]),
           f"{out[:6]} vs {base[:6]}")

    # 2. "none" must be a true no-op: arm A has to reproduce the failed gates.
    ok("'none' is the identity", np.array_equal(tf(base.copy(), "none"), base))

    # 3. The active arms emit valid OSC gripper commands.
    for g in (-1.0, -0.5, -1e-9, 0.0, 1e-9, 0.5, 0.7, 1.0):
        a = base.copy(); a[-1] = g
        # Compare against the float32 round-trip of g, not the Python literal:
        # float32(0.7) is 0.69999998808, so `== -0.7` would fail on precision
        # and say nothing about the transform.
        g32 = float(a[-1])
        ok(f"'invert' negates the gripper at g={g}",
           float(tf(a.copy(), "invert")[-1]) == -g32)
        for mode in ("binvert", "openvla"):
            v = float(tf(a.copy(), mode)[-1])
            ok(f"{mode!r} emits exactly +/-1 at g={g}", v in (-1.0, 1.0), f"got {v}")

    # 4. The arms must actually differ, or the A/B cannot discriminate.
    grid = np.linspace(-1.0, 1.0, 41)
    sigs = {}
    for mode in GRIPPER_TRANSFORMS:
        sigs[mode] = tuple(
            float(tf(np.append(base[:6], g).astype(np.float32), mode)[-1])
            for g in grid)
    ok("all four arms are pairwise distinct over g in [-1,1]",
       len(set(sigs.values())) == 4,
       f"{len(set(sigs.values()))} distinct behaviours")

    # 5. The specific disagreement the A/B turns on: for a decoded gripper in
    #    (0, 0.5], 'binvert' commands open and 'openvla' commands close. If
    #    these ever coincided the experiment could not separate "invert" from
    #    "assume the channel arrives in [0,1]".
    a = base.copy(); a[-1] = 0.3
    ok("'binvert' and 'openvla' disagree at g=0.3 (the discriminating case)",
       float(tf(a.copy(), "binvert")[-1]) != float(tf(a.copy(), "openvla")[-1]),
       f"binvert={tf(a.copy(),'binvert')[-1]} openvla={tf(a.copy(),'openvla')[-1]}")

    # 6. An unknown arm must raise rather than silently pass through, or a
    #    typo in the sweep config would quietly re-run arm A four times.
    try:
        tf(base.copy(), "inverrt")
        ok("an unknown transform name raises", False, "it returned instead")
    except ValueError as e:
        ok("an unknown transform name raises", True, str(e)[:60] + "...")

    # 7. 'openvla' must equal upstream's composition exactly. Upstream (read
    #    from source by bolt task htrg4uchwi) applies
    #    normalize_gripper_action(binarize=True), i.e. g -> sign(2g - 1), and
    #    then invert_gripper_action(), i.e. a negation. This asserts the
    #    composition rather than trusting the one-line comment next to it.
    def upstream(g):
        return -float(np.sign(2.0 * np.float32(g) - 1.0))

    mismatch = [g for g in np.linspace(-1.0, 1.0, 81)
                if abs(2.0 * np.float32(g) - 1.0) > 1e-6
                and float(tf(np.append(base[:6], g).astype(np.float32),
                             "openvla")[-1]) != upstream(g)]
    ok("'openvla' equals upstream's -sign(2g-1) over a grid",
       not mismatch, f"{len(mismatch)} mismatches" if mismatch else "81 points")

    # ---------------- image preprocessing ----------------
    # 'tf_upstream' is not exercised here: it needs tensorflow, which this CI
    # job does not install (the GPU jobs that want it opt in with INSTALL_TF=1;
    # keeping CI tensorflow-free keeps it fast). The A/B script checks for
    # tensorflow before the model load instead, so a missing backend fails fast
    # there rather than silently choosing another kernel.
    # 'np_lanczos' is checked for structure here and for numerical agreement
    # with tensorflow by experiments/resize_kernel_check.py, which runs in an
    # isolated tf venv on bolt.
    ok("all four image modes are registered",
       tuple(IMAGE_PREPROCS) == ("none", "np_lanczos", "pil_lanczos",
                                 "tf_upstream"),
       str(IMAGE_PREPROCS))

    rng = np.random.default_rng(0)
    render = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)

    out_none = _preprocess_image(render, "none")
    ok("'none' returns the render untouched at full 256px",
       out_none.shape == (256, 256, 3) and np.array_equal(out_none, render),
       str(out_none.shape))

    for mode in ("np_lanczos", "pil_lanczos"):
        try:
            out = _preprocess_image(render, mode)
            ok(f"'{mode}' returns 224x224x3 uint8",
               out.shape == (224, 224, 3) and out.dtype == np.uint8,
               f"{out.shape} {out.dtype}")
            # The JPEG round-trip plus a resize must actually change the pixels,
            # otherwise the arm is a no-op wearing a different name and the
            # factorial has a duplicate cell.
            ok(f"'{mode}' is not a no-op",
               not np.array_equal(out, render[:224, :224]),
               "differs from a naive crop")
            # And it must not silently be doing the processor's default resize:
            # compare against a plain bilinear resize with no JPEG step.
            from PIL import Image
            plain = np.asarray(Image.fromarray(render).resize((224, 224),
                                                              Image.BILINEAR))
            ok(f"'{mode}' differs from a plain bilinear resize",
               not np.array_equal(out, plain),
               f"max abs diff {int(np.abs(out.astype(int) - plain.astype(int)).max())}")
        except ImportError:  # pragma: no cover - Pillow is a hard dep of the repo
            ok(f"'{mode}' runs", False, "Pillow missing")

    # The two lanczos arms must not collapse into each other, or the factorial
    # cannot distinguish "upstream's kernel" from "a kernel that failed the
    # 4-LSB ceiling" -- which is the only reason both modes exist.
    a = _preprocess_image(render, "np_lanczos")
    b = _preprocess_image(render, "pil_lanczos")
    ok("'np_lanczos' and 'pil_lanczos' are distinguishable",
       not np.array_equal(a, b),
       f"max abs diff {int(np.abs(a.astype(int) - b.astype(int)).max())}")

    # Properties of the kernel itself, checkable without tensorflow. If the
    # resampling weights did not sum to 1 the image would be uniformly darkened
    # or brightened, which is a large off-distribution shift that a max-abs-diff
    # spot check against a reference could still miss on textured inputs.
    from sharpguard.image_preproc import _weights, lanczos3_resize
    w = _weights(256, 224)
    ok("resize weights sum to 1 on every output row",
       bool(np.allclose(w.sum(axis=1), 1.0, atol=1e-12)),
       f"max deviation {float(np.abs(w.sum(axis=1) - 1.0).max()):.2e}")
    ok("resize weights have the (224, 256) shape of a 256->224 reduction",
       w.shape == (224, 256), str(w.shape))
    # A constant image must survive exactly: any normalisation or centring bug
    # shows up here as a shifted grey level.
    flat = np.full((256, 256, 3), 137, np.uint8)
    ok("a constant image resizes to the same constant",
       bool(np.all(lanczos3_resize(flat, 224) == 137)),
       f"got range [{int(lanczos3_resize(flat, 224).min())}, "
       f"{int(lanczos3_resize(flat, 224).max())}]")
    # Downscaling must engage the antialias stretch; upscaling must not. The
    # 3-tap radius means a non-antialiased 256->224 reduction would touch ~7
    # input pixels per output pixel, an antialiased one ~9.
    taps_down = int((_weights(256, 224) != 0).sum(axis=1).max())
    taps_up = int((_weights(224, 256) != 0).sum(axis=1).max())
    ok("downscaling stretches the kernel support past the upscaling case",
       taps_down > taps_up, f"{taps_down} taps down vs {taps_up} up")

    # The JPEG chroma-subsampling default is a *measured* parameter, not a
    # stylistic choice: bolt yp9ix9486w swept it against tf.image.encode_jpeg
    # and found 4:4:4 off by 240 LSB, 4:2:2 by 150, and 4:2:0 by 9. The first
    # version of this code shipped 4:4:4 on the plausible-but-wrong reasoning
    # that less downsampling is closer to the original -- the target is not the
    # original, it is what upstream produces. Pinned here so a regression costs
    # a CI second rather than a GPU job.
    from sharpguard.image_preproc import jpeg_roundtrip
    ok("the JPEG round-trip defaults to 4:2:0, matching tf's "
       "chroma_downsampling=True",
       jpeg_roundtrip.__defaults__ == (95, 2),
       f"quality/subsampling defaults are {jpeg_roundtrip.__defaults__}")

    try:
        _preprocess_image(render, "lanzcos")
        ok("an unknown image_preproc raises", False, "it returned instead")
    except ValueError as e:
        ok("an unknown image_preproc raises", True, str(e)[:60] + "...")

    # ---------------- per-suite step budgets ----------------
    # Read from upstream's run_libero_eval.py by bolt task htrg4uchwi. Pinned
    # here because the four-suite gate ran 400 steps everywhere, which made
    # libero_10's 0/50 uninterpretable, and nothing but a test stops that from
    # happening again.
    ok("upstream per-suite max_steps are pinned",
       UPSTREAM_MAX_STEPS == {"libero_spatial": 220, "libero_object": 280,
                              "libero_goal": 300, "libero_10": 520,
                              "libero_90": 400},
       str(UPSTREAM_MAX_STEPS))
    ok("the gate's old flat 400 was below upstream's libero_10 budget",
       UPSTREAM_MAX_STEPS["libero_10"] > 400,
       f"{UPSTREAM_MAX_STEPS['libero_10']} > 400")

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("gripper transform: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
