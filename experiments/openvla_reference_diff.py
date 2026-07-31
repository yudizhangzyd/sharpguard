#!/usr/bin/env python3
"""Diff our LIBERO rollout against the OpenVLA reference implementation.

Why this is a Bolt job and not a local one. The decoder gate has failed four
times, and every failure was a detail of the eval harness rather than of the
policy. The obvious way to stop guessing those details is to read upstream's
own eval script -- but the Mac has no OpenVLA checkout and no route to fetch
one, so this has to run where there is network.

What it does NOT do is trust my recollection of upstream. It clones the repo,
extracts the four quantities that the four failures have implicated, and prints
the raw source of each so a human can check the extraction. Two specific
hypotheses are on trial, both of which would produce the observed signature of
"heavily degraded but not identically zero" (libero_goal scored 5/50, so the
harness is not absolutely broken):

  H1 (gripper): upstream may pass the gripper through TWO transforms in
     sequence -- a [0,1] -> [-1,1] normalize with binarize=True, and then an
     inversion. Our rollout applies neither.
  H2 (image resize): upstream may resize the 256x256 agentview render to the
     model's 224x224 with tf.image.resize(lanczos3, antialias=True) to match
     the training pipeline, rather than letting the HF processor do a default
     bilinear resize. A resize-kernel mismatch puts every frame slightly
     off-distribution, which degrades a policy without making success
     impossible -- exactly the shape we measured.

Also extracted: NUM_STEPS_WAIT and the per-suite max_steps table (we already
know our libero_10 run used 400 against upstream's larger budget, which
invalidates that one suite's 0/50 on its own), and the prompt template.

Output: reference_diff.json plus the source snippets on stdout. Exits non-zero
if the reference cannot be obtained, because a green run that silently skipped
the comparison is the failure mode this whole exercise exists to prevent.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

REPO = "https://github.com/openvla/openvla.git"
CLONE = "/tmp/openvla_ref"

# Files that upstream's LIBERO evaluation is spread across. Globbed rather than
# hardcoded to one path, because the layout has moved between releases.
WANTED = ("run_libero_eval.py", "libero_utils.py", "robot_utils.py",
          "openvla_utils.py")


def sh(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def fetch() -> bool:
    if os.path.isdir(os.path.join(CLONE, ".git")):
        print(f"[ref] reusing existing clone at {CLONE}")
        return True
    rc, out = sh(["git", "clone", "--depth", "1", REPO, CLONE])
    print(f"[ref] git clone rc={rc}")
    if rc != 0:
        print(out[-2000:])
    return rc == 0


def find_files() -> dict[str, str]:
    found = {}
    for root, _dirs, files in os.walk(CLONE):
        if ".git" in root:
            continue
        for fn in files:
            if fn in WANTED and fn not in found:
                found[fn] = os.path.join(root, fn)
    return found


def snippet(src: str, pattern: str, before: int = 0, after: int = 22) -> str:
    """Return the lines around the first regex hit, or '' if there is none."""
    lines = src.splitlines()
    for i, ln in enumerate(lines):
        if re.search(pattern, ln):
            lo = max(0, i - before)
            return "\n".join(lines[lo:i + after])
    return ""


def main() -> int:
    if not fetch():
        print("[ref] FATAL: could not obtain the OpenVLA reference. Not "
              "emitting a report -- an empty diff would read as 'no "
              "differences found', which is the opposite of the truth.")
        return 2

    files = find_files()
    print(f"[ref] located: {json.dumps(files, indent=2)}")
    missing = [w for w in WANTED if w not in files]
    if "run_libero_eval.py" not in files:
        print(f"[ref] FATAL: run_libero_eval.py not in the clone; missing={missing}")
        return 2

    srcs = {k: open(v, errors="replace").read() for k, v in files.items()}
    allsrc = "\n".join(srcs.values())

    report: dict = {"repo": REPO, "files": files, "missing": missing,
                    "hypotheses": {}}

    # ---- H1: the gripper pipeline ----------------------------------------
    print("\n" + "=" * 72)
    print("H1  GRIPPER: what upstream does to the last action dim")
    print("=" * 72)
    grip_call = snippet(srcs["run_libero_eval.py"],
                        r"normalize_gripper_action|invert_gripper_action",
                        before=6, after=14)
    print("--- call site in run_libero_eval.py ---")
    print(grip_call or "  (no call site found)")
    for fn in ("normalize_gripper_action", "invert_gripper_action"):
        body = snippet(allsrc, rf"def {fn}\b", after=20)
        print(f"--- def {fn} ---")
        print(body or "  (not found)")
        report["hypotheses"].setdefault("H1_gripper", {})[fn] = body
    report["hypotheses"]["H1_gripper"]["call_site"] = grip_call
    # The decisive booleans, read off the source rather than asserted.
    report["hypotheses"]["H1_gripper"]["normalize_is_called"] = \
        "normalize_gripper_action" in srcs["run_libero_eval.py"]
    report["hypotheses"]["H1_gripper"]["invert_is_called"] = \
        "invert_gripper_action" in srcs["run_libero_eval.py"]
    report["hypotheses"]["H1_gripper"]["binarize_true_at_call_site"] = \
        bool(re.search(r"normalize_gripper_action\([^)]*binarize\s*=\s*True",
                       srcs["run_libero_eval.py"]))

    # ---- H2: the image path ---------------------------------------------
    print("\n" + "=" * 72)
    print("H2  IMAGE: how upstream gets from the render to the model input")
    print("=" * 72)
    for pat, label in ((r"def get_libero_image\b", "get_libero_image"),
                       (r"def resize_image\b", "resize_image"),
                       (r"tf\.image\.resize", "tf.image.resize use"),
                       (r"lanczos", "lanczos mention")):
        body = snippet(allsrc, pat, after=24)
        print(f"--- {label} ---")
        print(body or "  (not found)")
        report["hypotheses"].setdefault("H2_image", {})[label] = body
    report["hypotheses"]["H2_image"]["uses_tf_image_resize"] = \
        "tf.image.resize" in allsrc
    report["hypotheses"]["H2_image"]["mentions_lanczos"] = \
        "lanczos" in allsrc.lower()
    report["hypotheses"]["H2_image"]["mentions_antialias"] = \
        "antialias" in allsrc.lower()
    m = re.search(r"resize_size\s*=\s*(\d+)", allsrc)
    report["hypotheses"]["H2_image"]["resize_size"] = int(m.group(1)) if m else None

    # ---- step budgets and settle-wait ------------------------------------
    print("\n" + "=" * 72)
    print("STEP BUDGETS: max_steps per suite, and the settle-wait")
    print("=" * 72)
    budgets = dict(re.findall(
        r"task_suite_name\s*==\s*[\"'](\w+)[\"'][^\n]*\n\s*max_steps\s*=\s*(\d+)",
        srcs["run_libero_eval.py"]))
    if not budgets:  # some releases write it as a dict or with different spacing
        budgets = dict(re.findall(r"[\"'](libero_\w+)[\"']\s*:\s*(\d+)", allsrc))
    print(json.dumps(budgets, indent=2) or "  (none parsed)")
    print(snippet(srcs["run_libero_eval.py"], r"max_steps\s*=", before=8, after=16)
          or "  (no max_steps block)")
    report["max_steps_by_suite"] = {k: int(v) for k, v in budgets.items()}
    wait = re.search(r"NUM_STEPS_WAIT\s*=\s*(\d+)", allsrc)
    report["num_steps_wait"] = int(wait.group(1)) if wait else None
    print(f"[ref] NUM_STEPS_WAIT = {report['num_steps_wait']}")

    # ---- prompt template -------------------------------------------------
    print("\n" + "=" * 72)
    print("PROMPT TEMPLATE")
    print("=" * 72)
    prompt = snippet(allsrc, r"What action should the robot take", before=3, after=6)
    print(prompt or "  (not found)")
    report["prompt_template"] = prompt

    # ---- what OUR harness does, for the side-by-side ---------------------
    ours_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "sharpguard", "libero_sim.py")
    ours = open(ours_path, errors="replace").read()
    report["ours"] = {
        "applies_gripper_normalize": "normalize_gripper" in ours,
        "applies_gripper_invert": "invert_gripper" in ours
                                  or "_apply_gripper_transform" in ours,
        "uses_tf_image_resize": "tf.image.resize" in ours,
        "mentions_lanczos": "lanczos" in ours.lower(),
        "has_image_flip": "[::-1, ::-1]" in ours,
        "num_steps_wait": (lambda m: int(m.group(1)) if m else None)(
            re.search(r"NUM_STEPS_WAIT\s*=\s*(\d+)", ours)),
        "prompt_matches_upstream_wording":
            "What action should the robot take" in ours,
    }

    # ---- verdict ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("VERDICT: differences that could explain a degraded-but-nonzero SR")
    print("=" * 72)
    diffs = []
    h1, h2, o = (report["hypotheses"]["H1_gripper"],
                 report["hypotheses"]["H2_image"], report["ours"])
    if h1["normalize_is_called"] and not o["applies_gripper_normalize"]:
        diffs.append("H1a: upstream calls normalize_gripper_action; we do not")
    if h1["invert_is_called"] and not o["applies_gripper_invert"]:
        diffs.append("H1b: upstream calls invert_gripper_action; we do not")
    if h2["uses_tf_image_resize"] and not o["uses_tf_image_resize"]:
        diffs.append("H2a: upstream resizes with tf.image.resize; we do not")
    if h2["mentions_lanczos"] and not o["mentions_lanczos"]:
        diffs.append("H2b: upstream resizes with a lanczos kernel; we do not")
    if (report["num_steps_wait"] is not None
            and report["num_steps_wait"] != o["num_steps_wait"]):
        diffs.append(f"settle-wait: upstream {report['num_steps_wait']} vs "
                     f"ours {o['num_steps_wait']}")
    for suite, ms in report["max_steps_by_suite"].items():
        if ms > 400:
            diffs.append(f"max_steps: {suite} needs {ms}, our gate ran 400")
    if not o["prompt_matches_upstream_wording"] and report["prompt_template"]:
        diffs.append("prompt: our rollout does not contain upstream's wording")
    report["differences"] = diffs
    for d in diffs:
        print(f"  * {d}")
    if not diffs:
        print("  (none of the four implicated quantities differ -- if the gate "
              "is still degraded, the cause is outside this comparison)")

    out = os.environ.get("REF_DIFF_OUT", "reference_diff.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[ref] wrote {out} ({len(diffs)} differences)")
    # Deliberately exit 0 even with differences: finding them is the success
    # condition here, not the failure condition.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
