#!/usr/bin/env python3
"""Fluency mechanism regression: does the paraphrase/scramble floor gap track
judge-rated fluency of the edited CoT, or edit structure, rather than (or in
addition to) which family produced the edit?

Companion to scripts/mechanism_regression.py (same partial-R^2 design, same
outcome tv_mean), but the predictor here is b_fluent (the LLM judge's 1-5
fluency rating of the edited trace, from experiments/fluency_mechanism.py's
phase 2), not delta_logp. This is the paper's own open question (S3/floors):
syntactic_scramble's lower floor is either an independent confirmation of the
paraphrase floor or an artifact of that floor's own lower judged fluency
(4.00 vs paraphrase_null's 4.92), and the two cannot currently be
distinguished from the published aggregate numbers alone.

Outcome: tv_mean. Predictors: b_fluent, edit_distance_tokens,
|len_delta_tokens|. Pools paraphrase_null + syntactic_scramble +
bbox_jitter_null (the three families experiments/fluency_mechanism.py scored
and judged on the same checkpoint), matching the review's own thresholds:
  partial R^2 (b_fluent) < 0.15  -> floor gap not explained by fluency alone
  partial R^2 (b_fluent) > 0.40  -> floor gap substantially a fluency artifact

Usage:
    python3 scripts/fluency_mechanism_regression.py <path/to/fluency_mechanism_report.json>
"""
import json
import os
import sys

import numpy as np


def r_squared(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return (1 - ss_res / ss_tot) if ss_tot > 0 else 0.0, beta


def analyze(recs):
    y = np.array([r["tv_mean"] for r in recs])
    fluent = np.array([r["b_fluent"] for r in recs], dtype=float)
    dist = np.array([r["edit_distance_tokens"] for r in recs], dtype=float)
    lend = np.abs(np.array([r["len_delta_tokens"] for r in recs], dtype=float))
    n = len(y)
    ones = np.ones(n)

    def fit(*cols):
        X = np.column_stack([ones, *cols])
        return r_squared(y, X)

    r2_fluent, b_fluent_alone = fit(fluent)
    r2_dist, _ = fit(dist)
    r2_struct, _ = fit(dist, lend)
    r2_full, b_full = fit(fluent, dist, lend)

    partial_fluent = ((r2_full - r2_struct) / (1 - r2_struct)
                       if r2_struct < 1 else 0.0)

    return {
        "n": n,
        "r2_fluent_alone": r2_fluent, "beta_fluent_alone": b_fluent_alone[1],
        "r2_dist_alone": r2_dist,
        "r2_struct_only": r2_struct,
        "r2_full": r2_full,
        "partial_r2_fluent": partial_fluent,
        "beta_full_fluent": b_full[1], "beta_full_dist": b_full[2],
        "beta_full_lend": b_full[3],
    }


def main():
    path = (sys.argv[1] if len(sys.argv) > 1 else
             "results_v2/canonical_runs/fluency_mechanism/"
             "fluency_mechanism_report.json")
    d = json.load(open(path))
    recs = [r for r in d["per_sample"]
            if not r.get("skipped") and r.get("b_fluent") is not None]

    out = {"outcome": "tv_mean", "n_total": len(recs), "by_family": {},
           "pooled": None}

    families = sorted(set(r["family"] for r in recs))
    print(f"{'family':20s} {'n':>4s} {'mean_fluent':>11s} {'mean_tv':>9s}")
    for fam in families:
        sub = [r for r in recs if r["family"] == fam]
        mf = sum(r["b_fluent"] for r in sub) / len(sub)
        mt = sum(r["tv_mean"] for r in sub) / len(sub)
        print(f"{fam:20s} {len(sub):4d} {mf:11.3f} {mt:9.4f}")

    print()
    print(f"{'scope':20s} {'n':>4s} {'R2(fluent)':>10s} {'R2(struct)':>11s} "
          f"{'R2(full)':>9s} {'partialR2(fluent)':>18s}")
    e_pooled = analyze(recs)
    out["pooled"] = e_pooled
    print(f"{'pooled':20s} {e_pooled['n']:4d} {e_pooled['r2_fluent_alone']:10.4f} "
          f"{e_pooled['r2_struct_only']:11.4f} {e_pooled['r2_full']:9.4f} "
          f"{e_pooled['partial_r2_fluent']:18.4f}")

    for fam in families:
        sub = [r for r in recs if r["family"] == fam]
        if len(set(r["b_fluent"] for r in sub)) < 2:
            print(f"{fam:20s} skipped: b_fluent has no within-family variance")
            continue
        e = analyze(sub)
        out["by_family"][fam] = e
        print(f"{fam:20s} {e['n']:4d} {e['r2_fluent_alone']:10.4f} "
              f"{e['r2_struct_only']:11.4f} {e['r2_full']:9.4f} "
              f"{e['partial_r2_fluent']:18.4f}")

    print(f"\npooled partial R^2(fluent) = {e_pooled['partial_r2_fluent']:.4f}"
          f"  (beta_fluent(full) = {e_pooled['beta_full_fluent']:+.5f})")

    dest = ("results_v2/canonical_runs/fluency_mechanism_regression/"
            "fluency_mechanism_regression.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\n[fluency-mechanism-regression] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
