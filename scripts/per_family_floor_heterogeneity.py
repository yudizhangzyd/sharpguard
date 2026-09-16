#!/usr/bin/env python3
"""Does per-family heterogeneity in the paraphrase-floor comparison track
each family's own collision rate, rather than being an unexplained puzzle?

Why this exists. A skeptical reviewer can compute, from the released
per-family scores alone, that the seven semantic families split sharply: three
(direction_flip, negation, verb_swap) clear the paraphrase floor on almost
every calibrated configuration, while four (gripper_flip, location_swap,
subject_swap, adversarial_plausible) almost never do. Read on its own, that
looks like it could undermine S3's floor-collapse claim -- if some semantic
families comfortably clear the floor, is "the semantic mean sits below the
floor" just an artifact of averaging over a mix of high- and low-scoring
families, rather than a real property of the instrument?

This script checks the obvious mechanistic explanation S4 already supplies:
F is close to a decode-collision counter, so a family whose edits rarely flip
the policy's argmax action bin will show low F regardless of how much meaning
it changes, and a family whose edits reliably flip it will show high F
regardless of meaning too. If per-family floor-clearing tracks per-family
collision rate (already measured in collision_decomposition.json, S4), the
heterogeneity is not a confound competing with the paper's diagnosis -- it is
the diagnosis, one level down.

Usage:
    python3 scripts/per_family_floor_heterogeneity.py
"""
import json
import math
import os
from collections import defaultdict

SEMANTIC_7 = ["direction_flip", "gripper_flip", "verb_swap", "negation",
              "subject_swap", "location_swap", "adversarial_plausible"]
OURS = ["ours_no-cot", "ours_lora-r8", "ours_lora-r16", "ours_lora-r32",
        "ours_lora-r64", "ours_data-50A", "ours_data-50B"]


def pooled_rates(cfg):
    totals = defaultdict(lambda: [0.0, 0])
    for seed in range(3):
        path = (f"results_v2/canonical_runs/"
                f"{cfg}_edit_13family_seed{seed}.json")
        d = json.load(open(path))
        for fam, v in d["aggregate"].items():
            n = v.get("n") or v.get("n_samples") or 0
            totals[fam][0] += v.get("faithful_rate", 0) * n
            totals[fam][1] += n
    return {fam: (c / n if n else None) for fam, (c, n) in totals.items()}


def single_rates(path, key="aggregate"):
    d = json.load(open(path))
    return {fam: v.get("faithful_rate") for fam, v in d[key].items()}


def main():
    configs = {cfg: pooled_rates(cfg) for cfg in OURS}
    configs["ecot_bridge"] = single_rates(
        "results_v2/canonical_runs/ecot_bridge_edit_13family_calibration.json")
    configs["deepthink_base"] = single_rates(
        "results_v2/canonical_runs/deepthink_base_13family.json",
        "edit_aggregate")
    configs["deepthink_sft"] = single_rates(
        "results_v2/canonical_runs/deepthink_sft_13family.json",
        "edit_aggregate")
    configs["deepthink_rl"] = single_rates(
        "results_v2/canonical_runs/deepthink_rl_13family.json",
        "edit_aggregate")

    below = {f: 0 for f in SEMANTIC_7}
    above = {f: 0 for f in SEMANTIC_7}
    n_scored = {f: 0 for f in SEMANTIC_7}
    print(f"{'config':16s} {'floor':>7s} "
          + " ".join(f"{f[:7]:>8s}" for f in SEMANTIC_7))
    for cfg, rates in configs.items():
        floor = rates.get("paraphrase_null")
        row = f"{cfg:16s} {floor:7.3f} "
        for fam in SEMANTIC_7:
            v = rates.get(fam)
            row += f"{v:8.3f}" if v is not None else f"{'n/a':>8s}"
            if v is not None:
                n_scored[fam] += 1
                (below if v < floor else above)[fam] += 1
        print(row)

    coll = json.load(open(
        "results_v2/canonical_runs/collision_decomposition/"
        "collision_decomposition.json"))
    by_fam_collision = defaultdict(list)
    for c in coll["cells"]:
        if c["family"] in SEMANTIC_7:
            by_fam_collision[c["family"]].append(c["one_minus_collision"])
    mean_noncollision = {f: sum(v) / len(v)
                         for f, v in by_fam_collision.items()}

    print(f"\n{'family':22s} {'below':>6s} {'above':>6s} "
          f"{'clear_rate':>11s} {'mean 1-P(collision)':>20s}")
    rows = []
    for fam in SEMANTIC_7:
        clear_rate = above[fam] / n_scored[fam]
        rows.append((fam, below[fam], above[fam], clear_rate,
                    mean_noncollision[fam]))
        print(f"{fam:22s} {below[fam]:6d} {above[fam]:6d} "
              f"{clear_rate:11.3f} {mean_noncollision[fam]:20.3f}")

    # Spearman correlation between clear_rate and mean_noncollision, by hand
    # (no scipy dependency): rank both, Pearson on the ranks. Ties must get
    # the AVERAGE rank across the tied block (the standard Spearman
    # convention, matching scipy.stats.spearmanr's default) -- an earlier
    # version broke ties by original array order instead, which a stats
    # reviewer caught by noticing this rho (0.857) didn't match scipy's
    # (0.883) on the same 7 points: gripper_flip and location_swap tie at
    # clear_rate=0.0 here, and ordinal tie-breaking is not a defensible
    # alternative convention, just a bug in not handling the tie.
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg_rank
            i = j + 1
        return r

    xs = [r[3] for r in rows]
    ys = [r[4] for r in rows]
    rx, ry = ranks(xs), ranks(ys)
    n = len(rows)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    spearman = cov / (sx * sy) if sx and sy else None
    print(f"\nSpearman(clear_rate, mean_1-P(collision)) = {spearman:.3f} "
          f"(n={n} families)")

    # Fisher-z CI, at the same n=7 the point estimate is computed on.
    # Added for parity with S4's rho=0.476 (n=8) Fisher-z CI: a reviewer
    # flagged that this correlation got a CI treatment and this one did
    # not, with no principled reason for the difference. Same transform,
    # same se = 1/sqrt(n-3) verify_paper_numbers.py's audit for rho=0.476
    # uses (not Bonett & Wright's 1.06/sqrt(n-3) small-sample correction for
    # Spearman) -- deliberately: the two CIs in this paper should be
    # computed the same way, not "more correct" on whichever one a
    # reviewer happened to check.
    if spearman is not None and n > 3 and abs(spearman) < 1.0:
        z = math.atanh(spearman)
        se = 1.0 / math.sqrt(n - 3)
        lo_z, hi_z = z - 1.96 * se, z + 1.96 * se
        spearman_ci95 = [math.tanh(lo_z), math.tanh(hi_z)]
    else:
        spearman_ci95 = None
    print(f"Fisher-z 95% CI (n={n}): "
          f"[{spearman_ci95[0]:.3f}, {spearman_ci95[1]:.3f}]"
          if spearman_ci95 else "Fisher-z CI: undefined (n too small or |rho|=1)")

    # The exact LaTeX rows of Table~\ref{tab:family_het_full} (appendix.tex),
    # generated here rather than hand-typed, so a transcription error (a cell
    # bolded on the wrong side of its own floor) cannot survive a copy-paste
    # into the manuscript without also changing this canonical source.
    ORDER_FAM_SORTED = [r[0] for r in sorted(
        [(f, above[f] / n_scored[f]) for f in SEMANTIC_7],
        key=lambda t: -t[1])]
    NICE_CFG = {
        "ours_no-cot": "Ours no-CoT", "ours_lora-r8": "Ours r=8",
        "ours_lora-r16": "Ours r=16", "ours_lora-r32": "Ours r=32",
        "ours_lora-r64": "Ours r=64", "ours_data-50A": "Ours data-50A",
        "ours_data-50B": "Ours data-50B",
        "ecot_bridge": "ECoT-bridge$^\\dagger$",
        "deepthink_base": "DeepThinkVLA-base$^\\dagger$",
        "deepthink_sft": "DeepThinkVLA-SFT$^\\dagger$",
        "deepthink_rl": "DeepThinkVLA-RL$^\\dagger$",
    }
    ORDER_CFG = ["ours_no-cot", "ours_lora-r8", "ours_lora-r16",
                "ours_lora-r32", "ours_lora-r64", "ours_data-50A",
                "ours_data-50B", "ecot_bridge", "deepthink_base",
                "deepthink_sft", "deepthink_rl"]
    latex_rows = []
    for cfg in ORDER_CFG:
        floor = configs[cfg]["paraphrase_null"]
        cells = [f"${floor:.3f}$"]
        for fam in ORDER_FAM_SORTED:
            v = configs[cfg][fam]
            s = f"{v:.3f}"
            cells.append(f"$\\mathbf{{{s}}}$" if v >= floor else f"${s}$")
        latex_rows.append(f"{NICE_CFG[cfg]} & " + " & ".join(cells) + r" \\")

    out = {
        "per_config": {cfg: {fam: rates.get(fam) for fam in SEMANTIC_7 + ["paraphrase_null"]}
                      for cfg, rates in configs.items()},
        "per_family": {fam: {"below": below[fam], "above": above[fam],
                             "n_scored": n_scored[fam],
                             "clear_rate": above[fam] / n_scored[fam],
                             "mean_1_minus_collision": mean_noncollision[fam]}
                      for fam in SEMANTIC_7},
        "spearman_clear_rate_vs_noncollision": spearman,
        "spearman_clear_rate_vs_noncollision_ci95": spearman_ci95,
        "latex_matrix_rows": latex_rows,
    }
    dest = ("results_v2/canonical_runs/per_family_floor_heterogeneity/"
            "per_family_floor_heterogeneity.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\n[heterogeneity] -> {dest}")


if __name__ == "__main__":
    main()
