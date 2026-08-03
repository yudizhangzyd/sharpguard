#!/usr/bin/env python3
"""Is the magnitude leaderboard's ORDERING stable under the threshold tau?

The manuscript's threshold-sensitivity figure used to claim that "rankings are
preserved across the full range" of tau. That claim was never derivable from the
release -- the figure read the pre-C5-fix /tmp run directories, which carry
N=12 on location_swap, so it silently averaged over a family set that dropped
location_swap and substituted cross_task_swap, a Tier-0 CONTROL, while its
caption called the set "7 non-control families". This script replaces both the
data path and the claim:

  * every configuration is read from results_v2/canonical_runs/, so the sweep is
    re-derivable from the release with no network and no /tmp;
  * the family set is the canonical NON_CONTROL seven of derive_metrics.py,
    every one of which carries its full post-fix N in these runs;
  * the ordering is compared against the tau=0.05 ordering by Spearman rho, and
    the largest per-configuration rank move is reported.

What the numbers actually say is less flattering than the old caption and more
useful: the ordering is exactly stable only for tau <= 0.05, survives tau=0.10
with a single adjacent swap, and comes apart above that (rho = 0.62 at
tau=0.15, where one configuration moves four rank positions). The coarse
CoT-trained vs no-CoT gap is the only thing stable across the whole range, and
even it dips below 2x at tau=0.30. That is consistent with the paper's thesis
rather than a counterexample to it: a score whose dynamic range is dominated by
argmax collisions has no reason to order near-tied configurations stably.

One pinned released run per configuration (seed 0 of the 3-seed set, and the
matching seed-0 file for ECoT-bridge) so the eight curves are commensurable.
Exits non-zero if any configuration or family is missing, so a partial release
cannot read as a pass.
"""
import json
import os
import sys

# tau=0.05 is the manuscript default and the reference ordering.
TAUS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
TAU_DEFAULT = 0.05

# The canonical 7 NON-CONTROL families of scripts/derive_metrics.py: the 10
# non-null families minus the 3 Tier-0 controls. Kept in that file's order so a
# diff against it is readable.
NON_CONTROL = ["direction_flip", "gripper_flip", "verb_swap", "negation",
               "subject_swap", "location_swap", "adversarial_plausible"]

# Display name -> released per-sample file. Seed 0 for every configuration.
RUNS = [
    ("Ours r=8",    "ours_lora-r8_edit_13family_seed0.json"),
    ("Ours r=16",   "ours_lora-r16_edit_13family_seed0.json"),
    ("Ours r=32",   "ours_lora-r32_edit_13family_seed0.json"),
    ("Ours r=64",   "ours_lora-r64_edit_13family_seed0.json"),
    ("no-CoT",      "ours_no-cot_edit_13family_seed0.json"),
    ("data-50A",    "ours_data-50A_edit_13family_seed0.json"),
    ("data-50B",    "ours_data-50B_edit_13family_seed0.json"),
    ("ECoT-bridge", "ecot_bridge_edit_seed0.json"),
]
CANON = "results_v2/canonical_runs"


def ranks(vals):
    """Competition-free dense ranking, 1 = largest."""
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    out = [0] * len(vals)
    for pos, i in enumerate(order):
        out[i] = pos + 1
    return out


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    curves, per_family_n = {}, {}
    for name, fname in RUNS:
        path = os.path.join(root, CANON, fname)
        if not os.path.exists(path):
            print(f"[threshold] MISSING {path}", file=sys.stderr)
            return 1
        recs = json.load(open(path))["per_sample"]
        by_fam = {}
        for s in recs:
            if s.get("skipped", False) or s["family"] not in NON_CONTROL:
                continue
            by_fam.setdefault(s["family"], []).append(s["delta_linf"])
        missing = [f for f in NON_CONTROL if len(by_fam.get(f, [])) < 20]
        if missing:
            print(f"[threshold] {name}: families under n=20: {missing}",
                  file=sys.stderr)
            return 1
        per_family_n[name] = {f: len(by_fam[f]) for f in NON_CONTROL}
        # Per-family rate first, then the unweighted mean over families, so a
        # family with n=100 does not outvote one with n=60.
        curves[name] = [
            sum(sum(1 for d in by_fam[f] if d > tau) / len(by_fam[f])
                for f in NON_CONTROL) / len(NON_CONTROL)
            for tau in TAUS
        ]

    names = [n for n, _ in RUNS]
    i_def = TAUS.index(TAU_DEFAULT)
    ref = [curves[n][i_def] for n in names]
    ref_rank = dict(zip(names, ranks(ref)))
    ordering_ref = sorted(names, key=lambda n: ref_rank[n])

    by_tau = {}
    for i, tau in enumerate(TAUS):
        vals = [curves[n][i] for n in names]
        rk = dict(zip(names, ranks(vals)))
        moves = {n: [ref_rank[n], rk[n]] for n in names if ref_rank[n] != rk[n]}
        cot = [curves[n][i] for n in names if n != "no-CoT"]
        by_tau[f"{tau:g}"] = {
            "tau": tau,
            "rate": {n: curves[n][i] for n in names},
            "ordering": sorted(names, key=lambda n: rk[n]),
            "spearman_vs_default": spearman(ref, vals),
            "rank_moves_vs_default": moves,
            "max_rank_move": max((abs(a - b) for a, b in moves.values()),
                                 default=0),
            "min_cot_over_nocot": min(cot) / curves["no-CoT"][i],
            "identical_to_default_ordering": not moves,
        }

    stable = [t for t in TAUS if by_tau[f"{t:g}"]["identical_to_default_ordering"]]
    out = {
        "generated_by": "scripts/threshold_sweep.py",
        "note": ("one pinned released run per configuration; no /tmp path and "
                 "no new inference"),
        "taus": TAUS,
        "tau_default": TAU_DEFAULT,
        "families": NON_CONTROL,
        "n_families": len(NON_CONTROL),
        "configurations": names,
        "n_configurations": len(names),
        "sources": {n: f"{CANON}/{f}" for n, f in RUNS},
        "per_family_n": per_family_n,
        "ordering_at_default": ordering_ref,
        "by_tau": by_tau,
        "tau_with_identical_ordering": stable,
        "max_tau_with_identical_ordering": max(stable),
        "min_spearman_vs_default": min(v["spearman_vs_default"]
                                       for v in by_tau.values()),
        "min_cot_over_nocot_any_tau": min(v["min_cot_over_nocot"]
                                          for v in by_tau.values()),
        "max_rank_move_any_tau": max(v["max_rank_move"] for v in by_tau.values()),
        "interpretation": (
            "The magnitude leaderboard's ordering is identical to its tau=%g "
            "ordering only for tau <= %g. Spearman rho against that ordering "
            "falls to %.3f at its worst, where one configuration moves %d rank "
            "positions. The CoT-trained vs no-CoT gap is the only quantity "
            "stable over the whole range, and it dips to %.2fx. The earlier "
            "claim that rankings are preserved across the full range of tau "
            "was wrong, and was computed over a family set that included a "
            "Tier-0 control."
            % (TAU_DEFAULT, max(stable),
               min(v["spearman_vs_default"] for v in by_tau.values()),
               max(v["max_rank_move"] for v in by_tau.values()),
               min(v["min_cot_over_nocot"] for v in by_tau.values()))),
    }
    dest = os.path.join(root, CANON, "threshold_sweep", "threshold_sweep.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)

    print(f"[threshold] {len(names)} configurations x {len(NON_CONTROL)} "
          f"families x {len(TAUS)} taus")
    print(f"[threshold] ordering at tau={TAU_DEFAULT:g}: "
          + " > ".join(ordering_ref))
    for t in TAUS:
        b = by_tau[f"{t:g}"]
        tag = "SAME" if b["identical_to_default_ordering"] else \
            "moves: " + ", ".join(f"{k} {v[0]}->{v[1]}"
                                  for k, v in b["rank_moves_vs_default"].items())
        print(f"[threshold] tau={t:<5.2f} rho={b['spearman_vs_default']:5.3f} "
              f"minCoT/noCoT={b['min_cot_over_nocot']:.2f}x  {tag}")
    print(f"[threshold] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
