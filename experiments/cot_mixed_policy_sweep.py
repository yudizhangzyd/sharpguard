"""Does F_dir respond monotonically to how much a policy's action actually
depends on the CoT, or does it just detect "some causal dependence exists"
in a way that would also fire on a policy that ignores the CoT?

experiments/cot_oracle_positive_control.py already answers a narrower
question: a rule-based oracle whose action is a deterministic function of
the CoT's direction words scores F_dir=0.925 on direction_flip. That is a
sanity check on the pipeline's sensitivity, not a construct-validity check
on F_dir itself, because it only tests one point (a policy fully driven by
the CoT) against one other point (a policy under a meaning-preserving
edit). It says nothing about what F_dir does at intermediate degrees of
CoT-dependence.

This script builds that intermediate range directly. Define two synthetic
policies on the same released direction_flip pairs:
    f(CoT_edited)  = oracle_action(edited MOVE phrase)   -- fully CoT-driven
    g(CoT_ignored) = oracle_action(original MOVE phrase) -- ignores the edit
                     entirely, as if grounded in the unedited image/
                     instruction instead
and mix them: a(alpha) = alpha * f + (1 - alpha) * g. At alpha=1 this is
the existing oracle exactly (F_dir=0.925). At alpha=0 the "edited" action is
byte-identical to the original on every sample, so F_dir=0 by construction,
the same way paraphrase_null is 0 by construction in the existing oracle
script. Sweeping alpha in between asks whether F_dir(alpha) is monotonic --
the property a diagnostic with real construct validity should have -- or
whether it saturates, oscillates, or fails to separate intermediate mixes
from the alpha=0 endpoint.

Source and construction are identical to cot_oracle_positive_control.py:
results_v2/canonical_runs/judge_edit_families/judge_pairs.json,
direction_flip family, N=40 pairs, same MOVE-phrase parser and the same
cos < -0.5 criterion S6 uses everywhere else. Nothing here is fit to the
result: alpha=0 and alpha=1 are fixed by construction before any number is
read, and the intermediate grid (0.1 steps) is fixed in advance too.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results_v2", "canonical_runs",
                   "judge_edit_families", "judge_pairs.json")

MOVE_RE = re.compile(r"MOVE:\s*(.*?)\s*GRIPPER POSITION:")

_WORDS = [
    ("left", 1, -1.0), ("right", 1, +1.0),
    ("forward", 0, +1.0), ("back", 0, -1.0),
    ("up", 2, +1.0), ("above", 2, +1.0),
    ("down", 2, -1.0), ("below", 2, -1.0),
]

ALPHAS = [round(0.1 * i, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0


def oracle_action(move_text):
    a = [0.0] * 7
    lower = move_text.lower()
    hit = False
    for word, axis, sign in _WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lower):
            a[axis] = sign
            hit = True
    return a, hit


def cos3(u, v):
    dot = sum(u[i] * v[i] for i in range(3))
    nu = sum(x * x for x in u[:3]) ** 0.5
    nv = sum(x * x for x in v[:3]) ** 0.5
    return dot / (nu * nv) if nu > 0 and nv > 0 else None


def mix(f, g, alpha):
    return [alpha * fi + (1 - alpha) * gi for fi, gi in zip(f, g)]


def run(tau=0.05, cos_thresh=-0.5):
    with open(SRC) as fh:
        pairs = json.load(fh)
    rows = [p for p in pairs if p["family"] == "direction_flip"]

    parsed = []
    for p in rows:
        ma = MOVE_RE.search(p["a_head"])
        mb = MOVE_RE.search(p["b_head"])
        if not (ma and mb):
            continue
        a_orig, hit_a = oracle_action(ma.group(1))
        a_edit_full, hit_b = oracle_action(mb.group(1))
        if not (hit_a or hit_b):
            continue
        parsed.append({"sample": p["sample"], "a_orig": a_orig,
                       "a_edit_full": a_edit_full,
                       "move_a": ma.group(1), "move_b": mb.group(1)})

    n = len(parsed)
    by_alpha = {}
    for alpha in ALPHAS:
        coses = []
        for r in parsed:
            a_mixed = mix(r["a_edit_full"], r["a_orig"], alpha)
            c = cos3(r["a_orig"], a_mixed)
            if c is not None:
                coses.append(c)
        f_dir = sum(1 for c in coses if c < cos_thresh) / len(coses)
        by_alpha[alpha] = {"n": len(coses), "F_dir": f_dir,
                           "mean_cos": sum(coses) / len(coses)}
        print(f"[mixed-policy] alpha={alpha:.1f} n={len(coses)} "
              f"F_dir={f_dir:.3f} mean_cos={by_alpha[alpha]['mean_cos']:+.3f}")

    f_dirs = [by_alpha[a]["F_dir"] for a in ALPHAS]
    is_monotonic = all(f_dirs[i] <= f_dirs[i + 1] for i in range(len(f_dirs) - 1))
    print(f"[mixed-policy] monotonic non-decreasing in alpha: {is_monotonic}")

    out_dir = os.path.join(ROOT, "results_v2", "canonical_runs",
                           "cot_mixed_policy_sweep")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cot_mixed_policy_sweep.json")
    with open(out_path, "w") as fh:
        json.dump({"tau": tau, "cos_threshold": cos_thresh, "n_applicable": n,
                   "source": os.path.relpath(SRC, ROOT), "alphas": ALPHAS,
                   "by_alpha": {str(a): v for a, v in by_alpha.items()},
                   "monotonic_nondecreasing": is_monotonic}, fh, indent=2)
    print(f"[mixed-policy] wrote {out_path}")
    return by_alpha


if __name__ == "__main__":
    try:
        run()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.exit(1)
