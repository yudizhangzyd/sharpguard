#!/usr/bin/env python3
"""Is F_diff an artifact of which meaning-preserving family we call the floor?

The paper subtracts paraphrase_null from the semantic families. A reviewer
observed that paraphrase_null is the one family in the taxonomy that changes
sequence length (PARAPHRASE_SYNONYMS contains multi-word expansions such as
release -> "let go of"), while the semantic families are length-exact. If the
floor is confounded with a length perturbation, F_diff subtracts the wrong
thing and the sign of the headline is a choice rather than a measurement.

syntactic_scramble is the control that settles it: the same LLM judge that
scores paraphrase_null at meaning_preserved 0.975 scores syntactic_scramble at
1.000, and it is length-exact by construction (0/40 pairs change word count).
Two floors, both meaning-preserving, one length-perturbing and one not.

What this script computes, for every scored configuration:
  F_bar     mean magnitude score over the semantic (non-null, non-control) families
  d_para    F_bar - F(paraphrase_null)      -- the floor the paper used
  d_scram   F_bar - F(syntactic_scramble)   -- the length-exact floor
  spread    |F(paraphrase_null) - F(syntactic_scramble)|
  margin    max(|d_para|, |d_scram|)

The headline is `spread > margin`: the gap between two families that MEAN THE
SAME THING exceeds the gap between the semantic families and either of them.
When that holds, neither floor licenses a claim about semantics, and reporting
d_scram in place of d_para would be trading one unlicensed sign for another.

Exits non-zero if any configuration fails to load, so a silently-empty run
cannot be mistaken for a passing one.
"""
import glob
import json
import sys

# Non-null, non-control edit families. Excludes the three calibration nulls
# (paraphrase_null, bbox_jitter_null, syntactic_scramble) and selfsplice_control.
SEMANTIC = [
    "direction_flip", "verb_swap", "cross_task_swap", "instr_random_sub",
    "subject_swap", "negation", "gripper_flip", "location_swap",
    "adversarial_plausible",
]
TAU = 0.05

OURS = ["ours_no-cot", "ours_lora-r8", "ours_lora-r16", "ours_lora-r32",
        "ours_lora-r64", "ours_data-50A", "ours_data-50B", "ecot_bridge"]
DEEPTHINK = {
    "deepthink_sft": "results_v2/canonical_runs/deepthink_sft_13family.json",
    "deepthink_rl": "results_v2/canonical_runs/deepthink_rl_13family.json",
    "deepthink_base": "results_v2/canonical_runs/deepthink_base_13family.json",
}
BRIDGE_SUBSET = ("bridge_subset_4k", "results_v2/canonical_runs/"
                 "bridge_subset_deconfound/cotfaith-bridge-subset-edit/"
                 "cot_edit_report.json")


def rate(records, family):
    """Magnitude score F: fraction of non-skipped records whose delta_linf > tau."""
    kept = [r for r in records if r.get("family") == family and not r.get("skipped")]
    if not kept:
        return None, 0
    return sum(1 for r in kept if r["delta_linf"] > TAU) / len(kept), len(kept)


def load_ours(config):
    """Pool the three sampling seeds; fall back to the single calibration run."""
    files = sorted(glob.glob(
        f"results_v2/canonical_runs/{config}_edit_13family_seed*.json"))
    if not files:
        files = sorted(glob.glob(
            f"results_v2/canonical_runs/{config}_edit_13family_calibration.json"))
    records = []
    for f in files:
        records += json.load(open(f))["per_sample"]
    return records, [f.split("/")[-1] for f in files]


def main():
    configs = []
    for c in OURS:
        recs, src = load_ours(c)
        configs.append((c, recs, src))
    for name, path in DEEPTHINK.items():
        configs.append((name, json.load(open(path))["per_sample_edit"],
                        [path.split("/")[-1]]))
    name, path = BRIDGE_SUBSET
    configs.append((name, json.load(open(path))["per_sample"], [path]))

    out, failures = [], []
    for name, recs, src in configs:
        para, n_para = rate(recs, "paraphrase_null")
        scram, n_scram = rate(recs, "syntactic_scramble")
        sem = [(f,) + rate(recs, f) for f in SEMANTIC]
        sem_ok = [(f, v, n) for f, v, n in sem if v is not None]
        if para is None or scram is None or not sem_ok:
            failures.append(name)
            continue
        fbar = sum(v for _, v, _ in sem_ok) / len(sem_ok)
        spread = abs(para - scram)
        margin = max(abs(fbar - para), abs(fbar - scram))
        out.append({
            "config": name, "sources": src,
            "f_bar_semantic": fbar,
            "n_semantic_families": len(sem_ok),
            "per_family": {f: {"F": v, "n": n} for f, v, n in sem_ok},
            "floor_paraphrase_null": {"F": para, "n": n_para},
            "floor_syntactic_scramble": {"F": scram, "n": n_scram},
            "f_diff_vs_paraphrase": fbar - para,
            "f_diff_vs_scramble": fbar - scram,
            "sign_flips_between_floors": (fbar - para) * (fbar - scram) < 0,
            "null_spread": spread,
            "max_semantic_margin": margin,
            "spread_exceeds_margin": spread > margin,
        })

    if failures:
        print(f"[floor] FAILED to score: {failures}", file=sys.stderr)
        return 1

    n = len(out)
    flips = sum(r["sign_flips_between_floors"] for r in out)
    exceeds = sum(r["spread_exceeds_margin"] for r in out)
    neg_para = sum(r["f_diff_vs_paraphrase"] < 0 for r in out)
    neg_scram = sum(r["f_diff_vs_scramble"] < 0 for r in out)

    print(f"{'config':18s} {'para':>6s} {'scram':>6s} {'Fbar':>6s} "
          f"{'d_para':>7s} {'d_scram':>7s} {'spread':>7s} {'margin':>7s} {'sp>mg':>6s}")
    for r in out:
        print(f"{r['config']:18s} {r['floor_paraphrase_null']['F']:6.3f} "
              f"{r['floor_syntactic_scramble']['F']:6.3f} {r['f_bar_semantic']:6.3f} "
              f"{r['f_diff_vs_paraphrase']:+7.3f} {r['f_diff_vs_scramble']:+7.3f} "
              f"{r['null_spread']:7.3f} {r['max_semantic_margin']:7.3f} "
              f"{'YES' if r['spread_exceeds_margin'] else 'no':>6s}")
    print(f"\nconfigurations                              {n}")
    print(f"F_diff < 0 against paraphrase_null          {neg_para}/{n}")
    print(f"F_diff < 0 against syntactic_scramble       {neg_scram}/{n}")
    print(f"sign flips between the two floors           {flips}/{n}")
    print(f"null spread exceeds semantic margin         {exceeds}/{n}")

    summary = {
        "tau": TAU, "semantic_families": SEMANTIC, "n_configs": n,
        "n_negative_vs_paraphrase": neg_para,
        "n_negative_vs_scramble": neg_scram,
        "n_sign_flips_between_floors": flips,
        "n_null_spread_exceeds_margin": exceeds,
        "judge_meaning_preserved": {
            "paraphrase_null": 0.975, "syntactic_scramble": 1.000,
            "source": "results_v2/canonical_runs/judge_edit_families/judge_report.json",
        },
        "interpretation": (
            "Both floors are meaning-preserving by the same validated judge. "
            "They disagree by more than either differs from the semantic mean on "
            f"{exceeds}/{n} configurations, so the sign of F_diff is determined by "
            "which meaning-preserving family is nominated as the floor rather than "
            "by the semantic content of the edit. Swapping to the length-exact floor "
            "does not rescue the metric; it selects a different unlicensed sign."
        ),
        "per_config": out,
    }
    dest = "results_v2/canonical_runs/floor_invariance/floor_invariance.json"
    import os
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(summary, open(dest, "w"), indent=2)
    print(f"\n[floor] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
