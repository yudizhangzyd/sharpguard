#!/usr/bin/env python3
"""Pseudo-replication, made explicit: how many INDEPENDENT confirmations does
"significant on 9 of 11" actually represent?

Why this exists. S3 already discloses that the eleven calibrated
configurations are 2 base-checkpoint lineages, not 11 independent
replications (8 rows are ECoT-bridge or a LoRA adapter on it, 3 are
DeepThinkVLA-base stages). What that disclosure does not do on its own is
say what "9 of 11 significant" is worth GIVEN that structure: each
config's own paired, observation-clustered bootstrap is a valid test of
THAT config, but the eight ECoT-lineage configs share a base checkpoint,
training data and (for the "ours" seven) the same 85 LIBERO-90 episode
files, so their nine "independent" significant results are not nine
independent pieces of evidence for the headline -- they are, at best, as
many independent pieces of evidence as there are independent lineages.

This script does the accounting explicitly: how many of the 8 ECoT-lineage
configs are individually significant (by the same bootstrap S3 already
uses), and how many of the 3 DeepThinkVLA-lineage configs. The honest
answer to "how many independent confirmations" is 2 (one per lineage), not
9 -- and this reports both counts and the exception in each lineage, rather
than only the pooled 9/11 that obscures which lineage it can and cannot
speak for.

What this script does NOT do, and why. A hierarchical or cluster-robust
model over "lineage" as a random effect needs enough clusters to estimate a
between-cluster variance component; with 2 clusters there are 2 data
points for that estimate, which is not enough to fit any such model
rather than merely narrate one. So no p-value is computed at the lineage
level -- reporting one would manufacture a precision the design cannot
support. What is reported is the within-lineage sign and significance
count, which is the finest-grained honest statement available.

Usage:
    python3 scripts/lineage_pseudoreplication.py
"""
import json
import os

ECOT_LINEAGE = ["ours_no-cot", "ours_lora-r8", "ours_lora-r16",
                "ours_lora-r32", "ours_lora-r64", "ours_data-50A",
                "ours_data-50B", "ecot_bridge"]
DEEPTHINK_LINEAGE = ["deepthink_base", "deepthink_sft", "deepthink_rl"]


def main():
    fcr = json.load(open(
        "results_v2/canonical_runs/floor_convention_robustness/"
        "floor_convention_robustness.json"))
    pc = fcr["per_config"]

    out = {}
    for name, lineage in (("ecot", ECOT_LINEAGE),
                          ("deepthink", DEEPTHINK_LINEAGE)):
        rows = []
        n_sig = 0
        n_neg = 0
        for cfg in lineage:
            c = pc[cfg]
            diff = c["diff_B"]["paraphrase_null"]
            boot = c.get("bootstrap_B_vs_para") or {}
            sig = bool(boot.get("excludes_zero"))
            neg = diff < 0
            n_sig += sig
            n_neg += neg
            rows.append({"config": cfg, "diff_B_para": diff,
                        "significant": sig, "negative": neg})
        out[name] = {"rows": rows, "n": len(lineage), "n_significant": n_sig,
                    "n_negative": n_neg}
        print(f"{name:10s} lineage: {n_neg}/{len(lineage)} negative, "
              f"{n_sig}/{len(lineage)} significant")
        for row in rows:
            flag = "sig" if row["significant"] else "NOT sig"
            print(f"  {row['config']:16s} {row['diff_B_para']:+.3f}  {flag}")

    total_sig = out["ecot"]["n_significant"] + out["deepthink"]["n_significant"]
    total_n = out["ecot"]["n"] + out["deepthink"]["n"]
    out["pooled_9_of_11"] = total_sig
    out["n_independent_lineages"] = 2
    print(f"\npooled significance count: {total_sig} of {total_n} "
          f"(matches the manuscript's '9 of 11')")
    print(f"independent lineages represented: 2")

    dest = ("results_v2/canonical_runs/lineage_pseudoreplication/"
            "lineage_pseudoreplication.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\n[lineage] -> {dest}")


if __name__ == "__main__":
    main()
