#!/usr/bin/env python3
"""Bridge V2 in-distribution null replication (bolt ivxx5bcu6b).

Why this exists. Appendix (cross-corpus transfer paragraph) already reports
direction_flip/gripper_flip on self-decoded CoT for Bridge V2, Fractal and
BC-Z at N=100, then states the limitation directly: "The nulls were not run
on these corpora, so this is a portability result for the pipeline, not
evidence the CoT drives the action on any of them." ECoT-bridge is also the
one checkpoint in the whole benchmark actually TRAINED on Bridge V2, so it is
the strongest available in-distribution test of the paper's central floor
claim on self-generated (not teacher-forced) CoT.

This script closes that gap for Bridge V2 specifically: a second bolt run
(config unchanged: bolt/boltconfig-cotfaith-ds-bridge_v2-n100.yaml) added
paraphrase_null and syntactic_scramble to the edit families scored on the
same N=100 self-decoded-CoT sample, so both calibration floors the paper's
headline claim turns on can now be checked here too.

Usage:
    python3 scripts/bridge_v2_null_replication.py
"""
import json
import os

SRC = ("results_v2/canonical_runs/bridge_v2_null_replication/"
       "cross_corpus_bridge_v2_n100_5family.json")


def main():
    rep = json.load(open(SRC))
    edit = rep["edit_aggregate"]

    semantic_families = [f for f in ("direction_flip", "gripper_flip", "subject_swap")
                          if edit.get(f, {}).get("n", 0) > 0]
    semantic_rates = [edit[f]["faithful_rate"] for f in semantic_families]
    semantic_mean = sum(semantic_rates) / len(semantic_rates)

    para = edit["paraphrase_null"]["faithful_rate"]
    scr = edit["syntactic_scramble"]["faithful_rate"]

    floor_gap = abs(para - scr)
    sem_to_para = abs(semantic_mean - para)
    sem_to_scr = abs(semantic_mean - scr)

    out = {
        "source": SRC,
        "model": rep["model"],
        "dataset": rep["dataset"],
        "semantic_families_used": semantic_families,
        "semantic_families_n0": [f for f in ("direction_flip", "gripper_flip", "subject_swap")
                                  if edit.get(f, {}).get("n", 0) == 0],
        "semantic_rates": {f: edit[f]["faithful_rate"] for f in semantic_families},
        "semantic_ns": {f: edit[f]["n"] for f in semantic_families},
        "semantic_mean": semantic_mean,
        "paraphrase_floor": para,
        "paraphrase_n": edit["paraphrase_null"]["n"],
        "scramble_floor": scr,
        "scramble_n": edit["syntactic_scramble"]["n"],
        "floor_gap": floor_gap,
        "semantic_to_paraphrase_gap": sem_to_para,
        "semantic_to_scramble_gap": sem_to_scr,
        "floor_gap_exceeds_both": floor_gap > sem_to_para and floor_gap > sem_to_scr,
        "paraphrase_at_least_semantic": para >= semantic_mean,
    }

    dest = "results_v2/canonical_runs/bridge_v2_null_replication/bridge_v2_null_replication.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)

    print(f"semantic mean (direction_flip+gripper_flip): {semantic_mean:.3f} "
          f"({'+'.join(semantic_families)})")
    print(f"paraphrase floor: {para:.3f} (n={out['paraphrase_n']})")
    print(f"scramble floor:   {scr:.3f} (n={out['scramble_n']})")
    print(f"floor gap: {floor_gap:.3f}  vs semantic-para gap {sem_to_para:.3f}, "
          f"semantic-scramble gap {sem_to_scr:.3f}")
    print(f"floor_gap_exceeds_both: {out['floor_gap_exceeds_both']}")
    print(f"paraphrase_at_least_semantic: {out['paraphrase_at_least_semantic']}")
    print(f"\n[bridge_v2_null_replication] -> {dest}")


if __name__ == "__main__":
    main()
