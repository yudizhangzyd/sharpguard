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
that produces it is invisible in the report, which is scalar. It was real: in
bolt h8xzmqnhgg the arms' step-0 frames were BIT-IDENTICAL outside a box
covering 10.4% of the frame and differed by 8.8411 inside it, a difference a
3-pixel shift reduces to 2.7762 -- one fixture placed differently, not a policy
acting differently -- because `set_init_state` restores qpos and a welded
fixture's pose is not qpos. That measurement is released under
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
    """The detector, run against the frames that motivated it.

    Skipped rather than failed when the capture is not on this machine: it is a
    GPU artifact, and a test that requires one cannot run where this file is
    meant to run. When it IS present, this is the check that matters -- the
    synthetic cases prove the comparison works, this proves it fires on the
    thing it was written for.
    """
    d = Path("/tmp/fs_now/cotfaith-rollout-edit/frames/t0_ep0")
    pair = [d / "nocot_t0000.png", d / "cot_clean_t0000.png"]
    if not all(p.exists() for p in pair):
        print(f"[skip] real defective capture not present under {d}")
        return
    import matplotlib.pyplot as plt
    imgs = {p.name.split("_t")[0]: plt.imread(p) for p in pair}
    msg = g.start_mismatch(imgs, 0)
    check("the unseeded capture bolt h8xzmqnhgg is rejected", msg is not None,
          "if this passes, the check is too loose to have caught the defect "
          "it was written for")


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
