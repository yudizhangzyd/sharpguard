#!/usr/bin/env python3
"""Per-family retraining-movement, the row Table~\\ref{tab:leaderboard} and
Table~\\ref{tab:leaderboard2} print at the bottom of each column.

S7 states the worst single-family retraining move once, as an anecdote
(0.260 on ours-r8:verb_swap). This computes the same quantity -- |F(family)
before retraining - F(family) after| -- for all seven semantic families
across the six released same-config retraining pairs (no-cot's 13-family
retrain report is not among the released artifacts), so a reader sees the
full per-family noise profile next to the scores it qualifies, not just the
single largest cell.

Usage:
    python3 scripts/per_family_retrain_movement.py
"""
import json
import os
import statistics
from collections import defaultdict

SEMANTIC_7 = ["direction_flip", "gripper_flip", "verb_swap", "negation",
              "subject_swap", "location_swap", "adversarial_plausible"]
RETRAIN_CFGS = ["ours_lora-r8", "ours_lora-r16", "ours_lora-r32",
                "ours_lora-r64", "ours_data-50A", "ours_data-50B"]


def pooled_rates(cfg):
    totals = defaultdict(lambda: [0.0, 0])
    for seed in range(3):
        path = f"results_v2/canonical_runs/{cfg}_edit_13family_seed{seed}.json"
        d = json.load(open(path))
        for fam, v in d["aggregate"].items():
            n = v.get("n") or v.get("n_samples") or 0
            totals[fam][0] += v.get("faithful_rate", 0) * n
            totals[fam][1] += n
    return {fam: (c / n if n else None) for fam, (c, n) in totals.items()}


def retrain_rates(cfg):
    path = f"results_v2/canonical_runs/{cfg}_edit_13family_RETRAIN.json"
    d = json.load(open(path))
    return {fam: v.get("faithful_rate") for fam, v in d["aggregate"].items()}


def main():
    moves = defaultdict(list)
    fbar_moves = []
    for cfg in RETRAIN_CFGS:
        orig = pooled_rates(cfg)
        retrain = retrain_rates(cfg)
        for fam in SEMANTIC_7:
            moves[fam].append(abs(orig[fam] - retrain[fam]))
        fbar_orig = sum(orig[f] for f in SEMANTIC_7) / 7
        fbar_retrain = sum(retrain[f] for f in SEMANTIC_7) / 7
        fbar_moves.append(abs(fbar_orig - fbar_retrain))

    print(f"{'family':22s} {'median':>8s} {'max':>8s}")
    median, mx = {}, {}
    for fam in SEMANTIC_7:
        vals = sorted(moves[fam])
        median[fam] = statistics.median(vals)
        mx[fam] = max(vals)
        print(f"{fam:22s} {median[fam]:8.3f} {mx[fam]:8.3f}")

    worst_fam = max(mx, key=mx.get)
    print(f"\nnoisiest family by max move: {worst_fam} ({mx[worst_fam]:.3f})")

    fbar_median = statistics.median(fbar_moves)
    fbar_max = max(fbar_moves)
    print(f"\nF_bar (7-family mean, raw) move across the same 6 pairs: "
          f"median {fbar_median:.3f}, max {fbar_max:.3f}")

    out = {"median": median, "max": mx, "n_pairs": len(RETRAIN_CFGS),
           "configs": RETRAIN_CFGS,
           "fbar_median": fbar_median, "fbar_max": fbar_max}
    dest = ("results_v2/canonical_runs/per_family_retrain_movement/"
            "per_family_retrain_movement.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\n[retrain-movement] -> {dest}")


if __name__ == "__main__":
    main()
