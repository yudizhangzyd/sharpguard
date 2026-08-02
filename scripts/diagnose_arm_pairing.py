#!/usr/bin/env python3
"""Diagnose whether the arms of one rollout episode started from one scene.

Written for a defect that no scalar in the rollout report could show. The three
arms of an episode are three prompts on ONE scene, and every difference between
them is attributed to the CoT. `set_init_state` restores the flattened MuJoCo
state -- qpos/qvel -- so it restores the robot and every FREE object. A fixture
welded to the world body has no joint: its pose lives in the model, and
robosuite's placement sampler draws it when the environment is CONSTRUCTED. A
harness that builds one env per arm therefore re-samples the furniture, and
`set_init_state` cannot undo it. The report still says every episode used its
canonical initial state, and that is true of everything qpos covers.

The regions here are not hand-drawn. Two frames are compared, the set of
differing pixels is found, and the report is:

  - `outside_bbox_mean_abs`: the mean |dpix| everywhere OUTSIDE the bounding box
    of the difference. This is the objective form of "the robot and the free
    objects are identical" -- it names no region, it measures the complement of
    whatever did differ.
  - `best_shift`: the integer (dy, dx) translation of the reference's bbox
    content that best matches the other arm's, with the residual at that shift.
    A small residual at a nonzero shift is the signature of one rigid object
    placed differently, as against a policy that acted differently.
  - the same bbox measured at every captured step, to separate "placed
    differently once" (constant bbox, the fixture) from "moved during the
    rollout" (growing bbox, the robot).

Usage:
  python3 scripts/diagnose_arm_pairing.py <frames_dir> [--out <json>]

Exits 0 whether or not a defect is found: this is a measurement, and the gates
that must fail on a mispairing are in run_cotfaith_rollout_edit_s3.sh and in
figures/gen_fig15_rollout_filmstrip.py. Exits non-zero only if it cannot
measure -- a diagnostic that reports "no defect" because it read nothing would
be the same class of bug it exists to document.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

# The search is +-6 px in each axis. Wide enough for the placement jitter a
# sampler applies (measured: 3 px), narrow enough that a robot in a different
# pose cannot be explained away as a translation of itself.
SHIFT = 6


def load(p: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)


def bbox_of_diff(a: np.ndarray, b: np.ndarray):
    """Rows/cols spanned by any differing pixel, or None if none differ."""
    d = np.abs(a - b).max(axis=2)
    ys, xs = np.where(d > 0)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def best_shift(a: np.ndarray, b: np.ndarray, box):
    """The integer (dy, dx) that best aligns b's box content onto a's.

    Compared over a's box only, and b is sampled from the shifted location, so
    a translation of one rigid object shows up as a near-zero residual at the
    shift that undoes it. Candidates that fall outside the frame are skipped
    rather than padded: padding invents pixels and would let a shift win by
    matching against zeros.
    """
    y0, y1, x0, x1 = box
    ref = a[y0:y1 + 1, x0:x1 + 1]
    best = None
    for dy in range(-SHIFT, SHIFT + 1):
        for dx in range(-SHIFT, SHIFT + 1):
            sy0, sy1, sx0, sx1 = y0 + dy, y1 + dy, x0 + dx, x1 + dx
            if sy0 < 0 or sx0 < 0 or sy1 >= b.shape[0] or sx1 >= b.shape[1]:
                continue
            r = float(np.abs(ref - b[sy0:sy1 + 1, sx0:sx1 + 1]).mean())
            if best is None or r < best[0]:
                best = (r, dy, dx)
    if best is None:
        return None
    return {"dy": best[1], "dx": best[2], "residual_mean_abs": round(best[0], 4)}


def compare(ep_dir: str, step_tag: str = "t0000"):
    frames = {}
    for p in sorted(glob.glob(os.path.join(ep_dir, f"*_{step_tag}.png"))):
        frames[os.path.basename(p).split("_t")[0]] = p
    if len(frames) < 2:
        return None
    names = sorted(frames)
    ref_name = names[0]
    ref = load(frames[ref_name])
    out = {"episode": os.path.basename(ep_dir), "step_tag": step_tag,
           "reference_arm": ref_name, "arms": names, "pairs": {}}
    for n in names[1:]:
        im = load(frames[n])
        if im.shape != ref.shape:
            out["pairs"][n] = {"error": f"{im.shape} vs {ref.shape}"}
            continue
        d = np.abs(im - ref)
        box = bbox_of_diff(ref, im)
        rec = {
            "mean_abs": round(float(d.mean()), 4),
            "max_abs": round(float(d.max()), 4),
            "frac_pixels_differing": round(
                float((d.max(axis=2) > 0).mean()), 5),
        }
        if box is None:
            rec["identical"] = True
        else:
            y0, y1, x0, x1 = box
            mask = np.ones(ref.shape[:2], bool)
            mask[y0:y1 + 1, x0:x1 + 1] = False
            rec["identical"] = False
            rec["diff_bbox"] = {"row0": y0, "row1": y1, "col0": x0, "col1": x1}
            rec["inside_bbox_mean_abs"] = round(
                float(d[y0:y1 + 1, x0:x1 + 1].mean()), 4)
            # The complement of whatever differed. This is the number that says
            # the robot and the free objects were paired: it is not a region we
            # chose, it is everything the difference did not touch.
            rec["outside_bbox_mean_abs"] = round(float(d[mask].mean()), 4)
            rec["best_shift"] = best_shift(ref, im, box)
        out["pairs"][n] = rec
    return out


def main(argv: list) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    frames_dir = args[0]
    out_p = None
    if "--out" in argv:
        out_p = argv[argv.index("--out") + 1]

    eps = sorted(glob.glob(os.path.join(frames_dir, "t*_ep*")))
    if not eps:
        print(f"[diag] no t*_ep* episode directory under {frames_dir}",
              file=sys.stderr)
        return 3

    report = {"frames_dir": frames_dir, "shift_search_px": SHIFT,
              "episodes": []}
    for ep in eps:
        first = compare(ep, "t0000")
        if first is None:
            print(f"[diag] {os.path.basename(ep)}: fewer than 2 arms filmed at "
                  f"t0000; nothing to pair")
            continue
        # Whether the same difference is present, and the same size, later on.
        # A fixture placed differently shows a bbox that does not grow; a policy
        # that diverged shows one that does.
        later = []
        for p in sorted(glob.glob(os.path.join(ep, "*_t0*.png"))):
            tag = "t" + os.path.basename(p).split("_t")[1].split(".")[0]
            if tag == "t0000" or any(l["step_tag"] == tag for l in later):
                continue
            c = compare(ep, tag)
            if c:
                later.append(c)
        first["later_steps"] = later
        report["episodes"].append(first)

    any_bad = any(
        not r.get("identical", False)
        for ep in report["episodes"] for r in ep["pairs"].values())
    report["mispaired_at_step_0"] = bool(any_bad)

    print(json.dumps(report, indent=2))
    if out_p:
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        with open(out_p, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[diag] -> {out_p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
