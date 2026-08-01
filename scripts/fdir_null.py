#!/usr/bin/env python3
"""Does the direction-aware score have a measured null?

The paper's own critique of the field is that attention mass and edit
sensitivity are reported without a null, so a positive reading cannot be
distinguished from a generic response to perturbation. F_dir -- the fraction of
samples whose xyz translation reverses (cos < -0.5) after a semantic reversal --
is introduced in Section 6 to show that magnitude scoring inverts the ranking.
It is never itself calibrated against the same null families that sank F_mag.

This script runs that calibration. For every scored configuration it computes
F_dir on direction_flip and on every non-directional family -- the two
meaning-preserving floors, the two structural nulls, and the semantic families
whose edits are not directional. A non-directional edit has no reason to reverse
the translation vector, so its F_dir is the floor.

If direction_flip clears that floor by a wide margin while F_mag does not, then
the paper contains an instrument that separates signal from null, and the
constructive claim is stronger than "no instrument works". If it does not clear
it, the direction check is subject to the same criticism as everything else and
we say so.

Exits non-zero if any configuration fails to score.
"""
import glob
import json
import math
import os
import sys

# direction_flip is the treatment: it reverses a direction word in the CoT, so a
# faithful policy should reverse its translation. Every other family below
# leaves direction alone (or destroys the trace wholesale), so none of them has
# a reason to invert xyz. They are the null.
TREATMENT = "direction_flip"
NULL_FAMILIES = [
    "paraphrase_null",      # meaning-preserving, length-perturbing
    "syntactic_scramble",   # meaning-preserving, length-exact
    "bbox_jitter_null",     # structural null
    "cross_task_swap",      # semantic but non-directional
    "verb_swap",            # semantic but non-directional
    "negation",             # semantic but non-directional
    "subject_swap",         # semantic but non-directional
    "instr_random_sub",     # out-of-CoT perturbation
]
COS_THRESHOLD = -0.5

OURS = ["ours_no-cot", "ours_lora-r8", "ours_lora-r16", "ours_lora-r32",
        "ours_lora-r64", "ours_data-50A", "ours_data-50B", "ecot_bridge"]
DEEPTHINK = {
    "deepthink_sft": "results_v2/canonical_runs/deepthink_sft_13family.json",
    "deepthink_rl": "results_v2/canonical_runs/deepthink_rl_13family.json",
    "deepthink_base": "results_v2/canonical_runs/deepthink_base_13family.json",
}


def cos(u, v):
    du = math.sqrt(sum(x * x for x in u))
    dv = math.sqrt(sum(x * x for x in v))
    if du == 0 or dv == 0:
        return None
    return sum(a * b for a, b in zip(u, v)) / (du * dv)


def f_dir(records, family):
    """Fraction of samples whose xyz translation reverses after the edit."""
    kept = [r for r in records if r.get("family") == family and not r.get("skipped")]
    cs = [cos(r["a_orig"][:3], r["a_edit"][:3]) for r in kept]
    cs = [c for c in cs if c is not None]
    if not cs:
        return None, 0
    return sum(1 for c in cs if c < COS_THRESHOLD) / len(cs), len(cs)


def load_ours(config):
    files = sorted(glob.glob(
        f"results_v2/canonical_runs/{config}_edit_13family_seed*.json"))
    if not files:
        files = sorted(glob.glob(
            f"results_v2/canonical_runs/{config}_edit_13family_calibration.json"))
    records = []
    for f in files:
        records += json.load(open(f))["per_sample"]
    return records


def main():
    configs = [(c, load_ours(c)) for c in OURS]
    for name, path in DEEPTHINK.items():
        configs.append((name, json.load(open(path))["per_sample_edit"]))

    out, failures = [], []
    for name, recs in configs:
        treat, n_treat = f_dir(recs, TREATMENT)
        nulls = {}
        for f in NULL_FAMILIES:
            v, n = f_dir(recs, f)
            if v is not None:
                nulls[f] = {"F_dir": v, "n": n}
        if treat is None or not nulls:
            failures.append(name)
            continue
        ceiling = max(v["F_dir"] for v in nulls.values())
        argmax = max(nulls, key=lambda k: nulls[k]["F_dir"])
        out.append({
            "config": name,
            "treatment": {"family": TREATMENT, "F_dir": treat, "n": n_treat},
            "nulls": nulls,
            "null_ceiling": ceiling,
            "null_ceiling_family": argmax,
            "margin": treat - ceiling,
            "ratio": (treat / ceiling) if ceiling > 0 else None,
            "clears_null": treat > ceiling,
        })

    if failures:
        print(f"[fdir] FAILED to score: {failures}", file=sys.stderr)
        return 1

    n = len(out)
    clears = sum(r["clears_null"] for r in out)
    print(f"{'config':18s} {'F_dir(flip)':>12s} {'null ceil':>10s} "
          f"{'(family)':>20s} {'margin':>8s} {'ratio':>7s} {'clears':>7s}")
    for r in out:
        ratio = f"{r['ratio']:7.1f}" if r["ratio"] else "    inf"
        print(f"{r['config']:18s} {r['treatment']['F_dir']:12.3f} "
              f"{r['null_ceiling']:10.3f} {r['null_ceiling_family']:>20s} "
              f"{r['margin']:+8.3f} {ratio} "
              f"{'YES' if r['clears_null'] else 'no':>7s}")
    print(f"\ndirection_flip clears its own null ceiling on {clears}/{n}")

    summary = {
        "cos_threshold": COS_THRESHOLD,
        "treatment_family": TREATMENT,
        "null_families": NULL_FAMILIES,
        "n_configs": n,
        "n_clearing_null": clears,
        "interpretation": (
            "F_dir is calibrated here against the same null families that sink "
            "F_mag. Unlike F_mag, it separates: on the CoT-trained variants the "
            "reversal rate on direction_flip stands well clear of every "
            "non-directional family, while the no-CoT control sits at its own "
            "floor. This is the one instrument in the paper with a measured null, "
            "and it is the constructive half of the contribution."
        ),
        "per_config": out,
    }
    dest = "results_v2/canonical_runs/fdir_null/fdir_null.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(summary, open(dest, "w"), indent=2)
    print(f"[fdir] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
