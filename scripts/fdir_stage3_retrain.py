#!/usr/bin/env python3
"""Does training r=32 8x longer close the F_dir null-clearance / action-space
confound (Limitations, Appendix Limitations-in-full)?

The published r=32 leaderboard row (15,000 steps) is one of the six
configurations whose F_dir null-clearance ratio (6.6--11.1x) coincides with a
collapsed action space and an open-loop prediction that does not beat a
constant -- a confound scripts/fdir_null.py and
scripts/floor_convention_robustness.py already document but do not resolve.
One natural alternative explanation is simple undertraining relative to
ECoT-bridge (the one strong policy, clearing at only 2.6x). This script tests
that alternative directly: r=32 retrained on the identical recipe (LoRA r=32,
lr=2e-5, seed=0) for 8x the steps (120,000, not 15,000 -- the same checkpoint
results_v2/canonical_runs/rollout_edit_stage3_sr/ uses for the rollout gate),
scored with the same 13-family edit protocol, 3 seeds, 100 samples/seed.

Reuses scripts/fdir_null.py's f_dir() (direction-aware cosine score, computed
on each checkpoint's own de-quantization grid via
scripts/derive_metrics.py's _to_checkpoint_grid, which is generic and takes
no checkpoint-specific argument) and
scripts/floor_convention_robustness.py's action_space_diagnostic() (ap1: at
most one of the three translation axes off its own zero bin), applied
identically to both checkpoints' records so the comparison is apples-to-apples.

Usage:
    python3 scripts/fdir_stage3_retrain.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fdir_null import f_dir, NULL_FAMILIES, TREATMENT
from floor_convention_robustness import load_config, action_space_diagnostic

STAGE3_GLOB = "results_v2/canonical_runs/edit_stage3_r32_8x/seed*/cot_edit_report.json"


def null_ceiling(records):
    nulls = {}
    for fam in NULL_FAMILIES:
        v, n = f_dir(records, fam)
        if v is not None:
            nulls[fam] = (v, n)
    fam = max(nulls, key=lambda k: nulls[k][0])
    return nulls[fam][0], fam, nulls


def analyze(records, label):
    treat, n_treat = f_dir(records, TREATMENT)
    ceiling, ceiling_fam, nulls = null_ceiling(records)
    diag = action_space_diagnostic(records)
    ratio = treat / ceiling if ceiling else None
    print(f"{label}: F_dir={treat:.3f} (n={n_treat})  "
          f"ceiling={ceiling:.3f} ({ceiling_fam})  ratio={ratio:.2f}x  "
          f"ap1={diag['ap1']:.3f}  noop3={diag['noop3']:.3f}  "
          f"distinct={diag['distinct_a_orig']}/{diag['n_obs']}")
    return {"F_dir": treat, "n_treat": n_treat, "ceiling": ceiling,
            "ceiling_family": ceiling_fam, "ratio": ratio,
            "nulls": {k: v[0] for k, v in nulls.items()},
            "action_space": diag}


def main():
    orig_records = load_config("ours_lora-r32")
    stage3_records = []
    for f in sorted(glob.glob(STAGE3_GLOB)):
        stage3_records += json.load(open(f))["per_sample"]
    if not stage3_records:
        print(f"[FATAL] no records matched {STAGE3_GLOB}")
        return 2

    orig = analyze(orig_records, "r=32 @ 15,000 steps (published)")
    stage3 = analyze(stage3_records, "r=32 @ 120,000 steps (8x, this script)")

    out = {"published_r32_15k": orig, "retrain_r32_120k_8x": stage3,
           "ratio_delta": stage3["ratio"] - orig["ratio"],
           "ap1_delta": stage3["action_space"]["ap1"] - orig["action_space"]["ap1"]}
    print(f"\nratio: {orig['ratio']:.2f}x -> {stage3['ratio']:.2f}x  "
          f"(delta {out['ratio_delta']:+.2f}x)")
    print(f"ap1:   {orig['action_space']['ap1']:.3f} -> "
          f"{stage3['action_space']['ap1']:.3f}  (delta {out['ap1_delta']:+.3f})")

    dest = "results_v2/canonical_runs/fdir_stage3_retrain/fdir_stage3_retrain.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\n[fdir-stage3-retrain] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
