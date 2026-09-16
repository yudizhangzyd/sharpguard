#!/usr/bin/env python3
"""DT-RL self-gen edit-sensitivity: summary analysis.

Reads the report scripts/../experiments/cotfaith_deepthink_selfgen_edit.py
produces and asks the one question this whole line of work exists to
answer: on a policy independently documented as competent (DT-RL, 97.0%
LIBERO, arXiv 2511.15669) and confirmed non-degenerate
(floor_convention_robustness.json's action_space_diagnostic), does the
floor-collapse pattern S3 reports on the weak/degenerate cohort -- a
meaning-preserving edit (paraphrase_null_text) moving the action AT LEAST
AS MUCH as a semantic edit (direction_flip_text) -- still hold?

This does not presuppose an answer. If direction_flip_text's faithful_rate
clearly exceeds paraphrase_null_text's, that is evidence AGAINST the
floor-collapse pattern generalizing to a competent policy -- which would be
reported as such, not suppressed or reframed, matching this paper's own
established practice of publishing whichever way a check comes out (see
e.g. the F_dir self-gen check, the retrain-bar Limitations paragraph).

Two paired bootstraps are reported. `paired_bootstrap_faithful_rate` is the
one that actually matches S3's own convention: it resamples the shared
sample ids and takes the difference of the *thresholded* rates (delta_linf
> tau), the same point_stat clustered_bootstrap_diff (floor_convention_
robustness.py) computes for every other F_diff significance test in this
paper -- not the difference of raw continuous delta_linf means an earlier
version of this function computed while claiming to match convention.
`paired_bootstrap_delta_linf` keeps that continuous statistic too, now
correctly labeled as a secondary, threshold-free check in the spirit of
Sec. collision's TV-bar, not as the convention-matching one.
Both are paired at n=20,000 resamples: both families are scored on
overlapping samples (paraphrase_null_text applies to nearly every sample;
direction_flip_text only to the minority with an explicit direction word),
so the shared-sample comparison is the one this paper's own methodology
would trust, not an unpaired comparison of two different-N marginal means.

Usage:
    python3 scripts/dt_selfgen_edit_analysis.py [path/to/dt_selfgen_edit_report.json]
"""
import json
import os
import random
import sys


def _paired_common(per_sample):
    dir_rows = {r["sample"]: r for r in per_sample
                if r.get("family") == "direction_flip_text"
                and not r.get("skipped")}
    para_rows = {r["sample"]: r for r in per_sample
                 if r.get("family") == "paraphrase_null_text"
                 and not r.get("skipped")}
    common = sorted(set(dir_rows) & set(para_rows))
    return dir_rows, para_rows, common


def paired_bootstrap_faithful_rate(per_sample, tau, n_resamples=20000, seed=0):
    """Mean(direction_flip faithful) - mean(paraphrase_null faithful), where
    faithful = delta_linf > tau, over the shared sample ids, 95% CI from
    resampling those ids. This is the actual F_diff statistic: the same
    thresholded-rate point_stat clustered_bootstrap_diff (floor_convention_
    robustness.py) computes for every other paired significance test in this
    paper, applied here to the two DT-RL families instead of a semantic
    family vs. the leaderboard's paraphrase floor."""
    dir_rows, para_rows, common = _paired_common(per_sample)
    if not common:
        return None
    dir_faith = [1.0 if dir_rows[s]["delta_linf"] > tau else 0.0 for s in common]
    para_faith = [1.0 if para_rows[s]["delta_linf"] > tau else 0.0 for s in common]
    n = len(common)
    point = sum(dir_faith) / n - sum(para_faith) / n

    rng = random.Random(seed)
    idx = list(range(n))
    diffs = []
    for _ in range(n_resamples):
        samp = [rng.choice(idx) for _ in range(n)]
        dm = sum(dir_faith[i] for i in samp) / n
        pm = sum(para_faith[i] for i in samp) / n
        diffs.append(dm - pm)
    diffs.sort()
    lo = diffs[int(0.025 * n_resamples)]
    hi = diffs[int(0.975 * n_resamples)]
    return {
        "n_paired": n, "point_estimate": point,
        "ci95": [lo, hi], "significant": bool(lo > 0 or hi < 0),
    }


def paired_bootstrap_delta_linf(per_sample, n_resamples=20000, seed=0):
    """Mean(direction_flip delta_linf - paraphrase_null delta_linf) over the
    shared sample ids: a continuous, threshold-free secondary check, in the
    spirit of Sec. collision's TV-bar (not the same statistic as F_diff --
    see paired_bootstrap_faithful_rate for the one that matches convention).
    95% CI from resampling those shared sample ids."""
    dir_rows, para_rows, common = _paired_common(per_sample)
    if not common:
        return None
    dir_vals = [dir_rows[s]["delta_linf"] for s in common]
    para_vals = [para_rows[s]["delta_linf"] for s in common]
    n = len(common)
    point = sum(dir_vals) / n - sum(para_vals) / n

    rng = random.Random(seed)
    idx = list(range(n))
    diffs = []
    for _ in range(n_resamples):
        samp = [rng.choice(idx) for _ in range(n)]
        dm = sum(dir_vals[i] for i in samp) / n
        pm = sum(para_vals[i] for i in samp) / n
        diffs.append(dm - pm)
    diffs.sort()
    lo = diffs[int(0.025 * n_resamples)]
    hi = diffs[int(0.975 * n_resamples)]
    return {
        "n_paired": n, "point_estimate": point,
        "ci95": [lo, hi], "significant": bool(lo > 0 or hi < 0),
    }


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else (
        "results_v2/canonical_runs/dt_selfgen_edit/"
        "dt_selfgen_edit_report.json")
    r = json.load(open(src))
    agg = r["edit_aggregate"]
    dirf = agg.get("direction_flip_text", {})
    para = agg.get("paraphrase_null_text", {})

    out = {
        "source": src,
        "model": r.get("model"),
        "n_samples_loaded": r.get("n_samples_loaded"),
        "n_no_movement_cot": r.get("n_no_movement_cot"),
        "n_selfgen_decode_fail": r.get("n_selfgen_decode_fail"),
        "direction_flip_text": dirf,
        "paraphrase_null_text": para,
    }

    n_dir, n_para = dirf.get("n", 0), para.get("n", 0)
    if n_dir > 0 and n_para > 0:
        fr_dir = dirf["faithful_rate"]
        fr_para = para["faithful_rate"]
        out["faithful_rate_gap"] = fr_dir - fr_para
        out["paraphrase_at_least_semantic"] = fr_para >= fr_dir
        out["semantic_clears_floor"] = fr_dir > fr_para
        li_dir, li_para = dirf["delta_linf_mean"], para["delta_linf_mean"]
        out["delta_linf_gap"] = li_dir - li_para
        per_sample = r.get("per_sample") or []
        out["paired_bootstrap_faithful_rate"] = paired_bootstrap_faithful_rate(
            per_sample, r.get("threshold", 0.05))
        out["paired_bootstrap_delta_linf"] = paired_bootstrap_delta_linf(
            per_sample)
    else:
        out["insufficient_data"] = True

    dest = ("results_v2/canonical_runs/dt_selfgen_edit_analysis/"
            "dt_selfgen_edit_analysis.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)

    print(f"n_samples_loaded={out['n_samples_loaded']} "
          f"n_no_movement_cot={out['n_no_movement_cot']} "
          f"n_selfgen_decode_fail={out['n_selfgen_decode_fail']}")
    print(f"direction_flip_text: n={n_dir} "
          f"faithful_rate={dirf.get('faithful_rate')} "
          f"delta_linf_mean={dirf.get('delta_linf_mean')} "
          f"n_skipped={dirf.get('n_skipped')}")
    print(f"paraphrase_null_text: n={n_para} "
          f"faithful_rate={para.get('faithful_rate')} "
          f"delta_linf_mean={para.get('delta_linf_mean')} "
          f"n_skipped={para.get('n_skipped')}")
    if "faithful_rate_gap" in out:
        print(f"\nfaithful_rate gap (direction_flip - paraphrase_null) = "
              f"{out['faithful_rate_gap']:+.3f}")
        print(f"paraphrase floor >= semantic (floor-collapse replicates): "
              f"{out['paraphrase_at_least_semantic']}")
        pbf = out.get("paired_bootstrap_faithful_rate")
        if pbf:
            print(f"paired bootstrap, F_diff convention (n={pbf['n_paired']}): "
                  f"point={pbf['point_estimate']:+.4f} 95% CI="
                  f"[{pbf['ci95'][0]:+.4f}, {pbf['ci95'][1]:+.4f}] "
                  f"significant={pbf['significant']}")
        pbl = out.get("paired_bootstrap_delta_linf")
        if pbl:
            print(f"paired bootstrap, continuous delta_linf (n={pbl['n_paired']}): "
                  f"point={pbl['point_estimate']:+.4f} 95% CI="
                  f"[{pbl['ci95'][0]:+.4f}, {pbl['ci95'][1]:+.4f}] "
                  f"significant={pbl['significant']}")
    print(f"\n[dt-selfgen-edit-analysis] -> {dest}")


if __name__ == "__main__":
    main()
