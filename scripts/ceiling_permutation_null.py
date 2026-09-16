#!/usr/bin/env python3
"""Same-model permutation null for the cross_task_swap ceiling, the check
Appendix sec:limitations_full's F_dir paragraph already runs for a different
statistic. An earlier draft of the ceiling-normalization confound paragraph
claimed this correction "does not port over" to F_bar/ceiling because
Delta_inf > tau is "not a criterion a mismatched pair could satisfy or
fail" -- that claim is false (it is exactly as much a per-pair criterion as
cos < -0.5 is) and was corrected after a fresh review caught it. This script
is the actual check: mismatch each sample's original action against a
DIFFERENT sample's cross_task_swap-edited action (same model, exhaustive
over all ordered pairs i != j), and compare the resulting chance rate
against the real (matched-pair) rate already reported as each model's
ceiling.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAU = 0.05

SOURCES = {
    "no-cot": [ROOT / "results_v2/canonical_runs" /
               f"ours_no-cot_edit_13family_seed{s}.json" for s in (0, 1, 2)],
    "r8": [ROOT / "results_v2/canonical_runs" /
           f"ours_lora-r8_edit_13family_seed{s}.json" for s in (0, 1, 2)],
    "ecot-bridge": [ROOT / "results_v2/canonical_runs" /
                    "ecot_bridge_edit_13family_calibration.json"],
}


def linf(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def load_family(paths, family="cross_task_swap"):
    recs = []
    for p in paths:
        d = json.loads(p.read_text())
        recs.extend(r for r in d["per_sample"]
                     if r.get("family") == family and not r.get("skipped"))
    return recs


def permutation_ceiling(recs, tau=TAU):
    n = len(recs)
    real_rate = sum(1 for r in recs if r["delta_linf"] > tau) / n
    total = exceed = 0
    for i in range(n):
        oi = recs[i]["a_orig"]
        for j in range(n):
            if i == j:
                continue
            if linf(oi, recs[j]["a_edit"]) > tau:
                exceed += 1
            total += 1
    return n, real_rate, exceed / total


def main():
    results = {}
    print(f"{'model':14s} {'n':>5s} {'real':>7s} {'perm_null':>10s} {'ratio':>7s}")
    for name, paths in SOURCES.items():
        recs = load_family(paths)
        n, real, perm = permutation_ceiling(recs)
        ratio = real / perm
        results[name] = {"n": n, "real_ceiling": real,
                          "permutation_null": perm, "ratio": ratio}
        print(f"{name:14s} {n:5d} {real:7.3f} {perm:10.3f} {ratio:7.2f}")

    out = ROOT / "results_v2/canonical_runs/ceiling_permutation_null"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ceiling_permutation_null.json").write_text(
        json.dumps({"tau": TAU, "per_model": results}, indent=2))
    print(f"\n[ceiling-permutation-null] -> {out}/ceiling_permutation_null.json")


if __name__ == "__main__":
    main()
