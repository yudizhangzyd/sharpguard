#!/usr/bin/env python3
"""Mechanism regression: does the effect size track the policy's own surprise
at the edited CoT, or edit size, rather than (or in addition to) meaning?

This is the review's "mechanism experiment" (2.1), the one item in its
missing-experiments list explicitly framed as the highest-value analysis
("regress per-sample effect on edit-distance/length-delta/policy's-own-
likelihood of the edited trace... converts a critique into a mechanism"),
and it needs no new compute: Stage 1's per-sample records already carry
delta_logp (the policy's own teacher-forced log-likelihood of the edited CoT,
computed as a byproduct of the same forward pass tv_mean comes from),
edit_distance_tokens and len_delta_tokens, for exactly this purpose.

Outcome: tv_mean (Table 2's continuous effect size). Predictors: |delta_logp|
(how surprising the edited trace is to the policy's own LM head -- absolute
value, since either direction of surprise is a candidate mechanism, not just
becoming-less-likely), edit_distance_tokens, |len_delta_tokens|.

Per the review's own results-to-claims matrix for this analysis:
  partial R^2 (delta_logp) < 0.15  -> headline survives on an instrument not
                                       explained by surface likelihood
  partial R^2 (delta_logp) > 0.40  -> "F tracks surface-likelihood-
                                       perturbation", a mechanism, and the
                                       single best available outcome
Computed per checkpoint (pooling across checkpoints without a scale
correction would let checkpoint identity, not surprise, drive a pooled R^2,
since tv_mean's baseline differs 5-6x across checkpoints -- see Table 2).

Usage:
    python3 scripts/mechanism_regression.py
"""
import json
import os
import statistics
import sys

import numpy as np

STAGE1_DIR = "results_v2/canonical_runs/stage1_continuous"
CHECKPOINTS = ["ecot-bridge", "lora-r32", "lora-r64",
               "deepthink-base", "deepthink-sft", "deepthink-rl"]


def load(checkpoint):
    path = f"{STAGE1_DIR}/{checkpoint}/stage1_continuous_report.json"
    d = json.load(open(path))
    return [r for r in d["per_sample"] if not r.get("skipped")]


def r_squared(y, X):
    """OLS R^2 of y on X (X already includes an intercept column)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return (1 - ss_res / ss_tot) if ss_tot > 0 else 0.0, beta


def analyze(checkpoint):
    recs = load(checkpoint)
    y = np.array([r["tv_mean"] for r in recs])
    logp = np.abs(np.array([r["delta_logp"] for r in recs]))
    dist = np.array([r["edit_distance_tokens"] for r in recs], dtype=float)
    lend = np.abs(np.array([r["len_delta_tokens"] for r in recs], dtype=float))
    n = len(y)
    ones = np.ones(n)

    def fit(*cols):
        X = np.column_stack([ones, *cols])
        return r_squared(y, X)

    r2_logp, b_logp = fit(logp)
    r2_dist, b_dist = fit(dist)
    r2_lend, _ = fit(lend)
    r2_struct, _ = fit(dist, lend)          # edit-distance + length only
    r2_full, b_full = fit(logp, dist, lend)  # all three

    partial_logp = ((r2_full - r2_struct) / (1 - r2_struct)
                    if r2_struct < 1 else 0.0)

    return {
        "n": n,
        "r2_logp_alone": r2_logp, "beta_logp_alone": b_logp[1],
        "r2_dist_alone": r2_dist, "beta_dist_alone": b_dist[1],
        "r2_lend_alone": r2_lend,
        "r2_struct_only": r2_struct,
        "r2_full": r2_full,
        "partial_r2_logp": partial_logp,
        "beta_full_logp": b_full[1], "beta_full_dist": b_full[2],
        "beta_full_lend": b_full[3],
    }


def main():
    out = {"outcome": "tv_mean", "checkpoints": {}}
    print(f"{'checkpoint':16s} {'n':>4s} {'R2(logp)':>9s} {'R2(struct)':>11s} "
          f"{'R2(full)':>9s} {'partialR2(logp)':>16s}")
    for ckpt in CHECKPOINTS:
        e = analyze(ckpt)
        out["checkpoints"][ckpt] = e
        print(f"{ckpt:16s} {e['n']:4d} {e['r2_logp_alone']:9.4f} "
              f"{e['r2_struct_only']:11.4f} {e['r2_full']:9.4f} "
              f"{e['partial_r2_logp']:16.4f}")

    partials = [e["partial_r2_logp"] for e in out["checkpoints"].values()]
    out["partial_r2_logp_range"] = [min(partials), max(partials)]
    out["partial_r2_logp_median"] = statistics.median(partials)
    print(f"\npartial R^2(logp) range: {min(partials):.3f}--{max(partials):.3f}, "
          f"median {out['partial_r2_logp_median']:.3f}")
    for ckpt, e in out["checkpoints"].items():
        print(f"  {ckpt:16s} beta_logp(full)={e['beta_full_logp']:+.5f} "
              f"beta_dist(full)={e['beta_full_dist']:+.5f}")

    dest = ("results_v2/canonical_runs/mechanism_regression/"
            "mechanism_regression.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\n[mechanism] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
