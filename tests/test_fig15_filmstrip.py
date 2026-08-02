#!/usr/bin/env python3
"""Offline checks on the filmstrip figure's refusals (no GPU, no simulator).

Separate from tests/test_rollout_arms_and_refresh.py on purpose: that suite is
the gate the pod runs before spending a rollout budget, so it may import numpy
and nothing else. This one imports the generator, which imports matplotlib, and
runs where the figure is actually drawn.

What it guards is one property: the figure refuses to draw rather than drawing
something misleading. Its rows are three prompts on ONE scene, and the reader
attributes every difference between them to the CoT edit. So a strip whose rows
are different scenes is not a weaker figure, it is a wrong one -- and the defect
that produces it is invisible in the report, which is scalar. Two such defects
were real, and they are different in kind:

  * bolt h8xzmqnhgg: the arms' step-0 frames were BIT-IDENTICAL outside a box
    covering 10.4% of the frame and differed by 8.8411 inside it, a difference a
    3-pixel shift reduces to 2.7762 -- one fixture placed differently, not a
    policy acting differently -- because `set_init_state` restores qpos and a
    welded fixture's pose is not qpos.
  * bolt xyiztdu4n6, after that was fixed: bit-identical outside the box again,
    but the box had moved onto the ROBOT (rows 0-134) with mean 31.7356 inside
    and best rigid shift (0, 0) -- a pose no translation aligns. Each arm ran
    its own 10-step settle, and the second entered it carrying the previous
    arm's controller goal and the solver's warm start.

Both measurements are released under
results_v2/canonical_runs/rollout_arm_pairing_defect/, produced by
scripts/diagnose_arm_pairing.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def load_fig(name: str):
    path = ROOT / "figures" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_{name}_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _scene(shift: int = 0) -> np.ndarray:
    """A toy scene: a robot at the top, a free object, and one fixture.

    Laid out this way so the synthetic case has the same shape as the real
    defect -- robot and free object identical, fixture translated -- instead of
    testing "two different images are different", which any comparison passes.
    """
    im = np.zeros((16, 16, 3), dtype=np.uint8)
    im[0:3, 5:11] = 180                        # robot, top of frame
    im[8:11, 1:4] = 120                        # free object on the table
    im[6:13, 9 + shift:13 + shift] = 220       # the cabinet
    return im


def test_identical_start_passes(g) -> None:
    a = _scene()
    check("identical first frames pass",
          g.start_mismatch({"nocot": a, "cot_clean": a.copy(),
                            "cot_direction_flip": a.copy()}, 0) is None)


def test_fixture_shift_is_caught(g) -> None:
    msg = g.start_mismatch({"nocot": _scene(0), "cot_clean": _scene(1)}, 0)
    check("a one-pixel shift of the fixture alone is caught", msg is not None,
          "robot and free object identical, furniture moved -- exactly the "
          "shape of the real defect")
    check("and the failure says where it is and what to do about it",
          bool(msg) and "rows" in msg and "cols" in msg and "env-seed" in msg,
          f"a bare 'frames differ' sends the reader to look at the policy: {msg}")
    check("and it names the step, so a reader knows no action had been applied",
          bool(msg) and "step 0" in msg)


def test_shape_difference_is_caught(g) -> None:
    check("a shape difference is reported, not broadcast into a comparison",
          g.start_mismatch({"a": _scene(),
                            "b": np.zeros((8, 8, 3), np.uint8)}, 0) is not None)


def test_real_defective_capture(g) -> None:
    """The detector, run against both captures that motivated it.

    Skipped rather than failed when a capture is not on this machine: they are
    GPU artifacts, and a test that requires one cannot run where this file is
    meant to run. When one IS present, this is the check that matters -- the
    synthetic cases prove the comparison works, these prove it fires on the
    things it was written for, and the two are different things. The first is a
    fixture translated 3 px; the second is a robot in a different pose that no
    translation aligns, which a shift-based check would have missed.
    """
    captures = (
        ("h8xzmqnhgg", "a fixture placed differently",
         "/tmp/fs_now/cotfaith-rollout-edit/frames/t0_ep0"),
        ("xyiztdu4n6", "a robot posed differently, no shift aligns it",
         "/tmp/fs_art2/cotfaith-rollout-edit/frames/t0_ep0"),
    )
    ran = 0
    for job, what, d in captures:
        pair = [Path(d) / "nocot_t0000.png", Path(d) / "cot_clean_t0000.png"]
        if not all(p.exists() for p in pair):
            print(f"[skip] capture {job} not present under {d}")
            continue
        import matplotlib.pyplot as plt
        imgs = {p.name.split("_t")[0]: plt.imread(p) for p in pair}
        check(f"the defective capture bolt {job} is rejected ({what})",
              g.start_mismatch(imgs, 0) is not None,
              "if this passes, the check is too loose to have caught the defect "
              "it was written for")
        ran += 1
    if not ran:
        print("[skip] neither defective capture is on this machine")


def main() -> int:
    g = load_fig("gen_fig15_rollout_filmstrip")
    test_identical_start_passes(g)
    test_fixture_shift_is_caught(g)
    test_shape_difference_is_caught(g)
    test_real_defective_capture(g)

    bad = [c for c in CHECKS if not c[1]]
    for name, ok, detail in CHECKS:
        if not ok:
            print(f"FAIL {name}" + (f"  [{detail}]" if detail else ""))
    print(f"{len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
