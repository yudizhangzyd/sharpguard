#!/usr/bin/env python3
"""How much of the magnitude score is a decode-collision counter?

The manuscript argues F_mag is stable under tau because the per-sample
Delta_inf distribution is bimodal: either exactly 0 (the argmax over action
bins landed on the identical bin despite a different prompt) or well above
0.10. That is true, and the conclusion drawn from it -- "so the threshold does
not matter" -- is the weaker of the two available readings.

The stronger reading is that if almost no mass lies between 0 and tau, then
F(m,f) is approximately 1 - P(Delta_inf == 0): not a measure of how much the
action moved, but a count of how often the decode failed to collide. This
script measures that directly over every scored cell in the release.

A cell is one (run, family) pair with at least MIN_N non-skipped records.
Superseded runs are excluded. Reports Pearson r and R^2 between F at tau=0.05
and 1 - P(Delta == 0), the number of exactly-equal cells, and the global mass
in each bin of the Delta distribution.

Exits non-zero if no cells are found, so an empty glob cannot read as a pass.
"""
import collections
import glob
import json
import math
import os
import sys

TAU = 0.05
MIN_N = 20


def pearson(pairs):
    n = len(pairs)
    mx = sum(a for a, _ in pairs) / n
    my = sum(b for _, b in pairs) / n
    cov = sum((a - mx) * (b - my) for a, b in pairs)
    sx = math.sqrt(sum((a - mx) ** 2 for a, _ in pairs))
    sy = math.sqrt(sum((b - my) ** 2 for _, b in pairs))
    return cov / (sx * sy) if sx and sy else float("nan")


def main():
    cells, deltas = [], []
    for path in sorted(glob.glob("results_v2/canonical_runs/**/*.json",
                                 recursive=True)):
        if "/superseded/" in path:
            continue
        try:
            doc = json.load(open(path))
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        recs = doc.get("per_sample") or doc.get("per_sample_edit")
        if not isinstance(recs, list) or not recs:
            continue
        if not isinstance(recs[0], dict) or "delta_linf" not in recs[0]:
            continue
        by_family = collections.defaultdict(list)
        for r in recs:
            if not r.get("skipped"):
                by_family[r.get("family")].append(r["delta_linf"])
                deltas.append(r["delta_linf"])
        for family, vals in by_family.items():
            if len(vals) < MIN_N:
                continue
            f_tau = sum(1 for v in vals if v > TAU) / len(vals)
            f_zero = sum(1 for v in vals if v != 0) / len(vals)
            cells.append({
                "run": os.path.relpath(path, "results_v2/canonical_runs"),
                "family": family, "n": len(vals),
                "F_at_tau": f_tau, "one_minus_collision": f_zero,
                "gap": f_zero - f_tau,
            })

    if not cells:
        print("[collision] no scored cells found", file=sys.stderr)
        return 1

    pairs = [(c["F_at_tau"], c["one_minus_collision"]) for c in cells]
    r = pearson(pairs)
    exact = sum(1 for c in cells if abs(c["gap"]) < 1e-12)

    n = len(deltas)
    bins = {
        "exactly_zero": sum(1 for d in deltas if d == 0),
        "zero_to_0.005": sum(1 for d in deltas if 0 < d < 0.005),
        "0.005_to_tau": sum(1 for d in deltas if 0.005 <= d < TAU),
        "tau_to_0.10": sum(1 for d in deltas if TAU <= d < 0.10),
        "above_0.10": sum(1 for d in deltas if d >= 0.10),
    }
    nonzero = n - bins["exactly_zero"]
    below_tau_of_nonzero = (bins["zero_to_0.005"] + bins["0.005_to_tau"]) / nonzero

    print(f"scored records            {n}")
    for k, v in bins.items():
        print(f"  {k:18s} {v:8d}  {v / n:6.2%}")
    print(f"nonzero deltas below tau  {below_tau_of_nonzero:.2%}")
    print(f"\ncells (n>={MIN_N})           {len(cells)}")
    print(f"Pearson r                 {r:.4f}")
    print(f"R^2                       {r * r:.4f}")
    print(f"exactly equal cells       {exact}/{len(cells)}")
    print(f"global F at tau           {sum(1 for d in deltas if d > TAU) / n:.4f}")
    print(f"global F at tau->0        {nonzero / n:.4f}")

    summary = {
        "tau": TAU, "min_cell_n": MIN_N,
        "n_scored_records": n, "n_cells": len(cells),
        "delta_distribution": bins,
        "frac_nonzero_below_tau": below_tau_of_nonzero,
        "pearson_r": r, "r_squared": r * r,
        "n_cells_exactly_equal": exact,
        "global_F_at_tau": sum(1 for d in deltas if d > TAU) / n,
        "global_F_at_tau_zero": nonzero / n,
        "interpretation": (
            "F_mag tracks 1 - P(Delta_inf == 0) at R^2 = "
            f"{r * r:.3f} over {len(cells)} cells, with only "
            f"{below_tau_of_nonzero:.1%} of nonzero deltas falling below tau. "
            "The magnitude score is therefore close to a count of how often the "
            "policy's argmax over action bins fails to land on the identical bin "
            "under a perturbed prompt. Threshold-insensitivity follows from that "
            "structure rather than indicating robustness, and the quantity being "
            "measured is decode stability rather than magnitude of action change."
        ),
        "cells": cells,
    }
    dest = "results_v2/canonical_runs/collision_decomposition/collision_decomposition.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(summary, open(dest, "w"), indent=2)
    print(f"\n[collision] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
