"""Unit test for `_apply_gripper_transform`, the free parameter in the A/B.

Why this runs before the GPU job. The last four gate submissions died in 2
minutes on an `AttributeError` in a test helper, which was cheap only because
the test ran first. The same applies here: the A/B is worthless if an arm is
mis-implemented, and every property below is checkable on a CPU in a second.

The properties that matter for the experiment:
  * every arm leaves the 6 continuous dims untouched -- if an arm perturbed
    xyz/rpy, a difference in SR could not be attributed to the gripper;
  * "none" is bit-identical to the input, so arm A really does reproduce the
    three gates that already failed;
  * the three active arms all emit exactly +/-1 or the negated input, i.e.
    they are valid OSC gripper commands;
  * the active arms genuinely differ somewhere in [-1, 1], otherwise the A/B
    has fewer than 4 distinct arms and cannot discriminate.
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
    _apply_gripper_transform as tf,
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

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("gripper transform: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
