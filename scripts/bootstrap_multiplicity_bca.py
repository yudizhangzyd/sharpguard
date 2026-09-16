#!/usr/bin/env python3
"""Multiplicity correction and BCa bootstrap, for the two families of tests
behind "9 of 11"/"3 of 11" (Table 1) and "6 of 6"/"5 of 6" (Table tvfloors).

Why this exists. An external statistics review found two things the paper's
existing Holm correction (Sec 6, the 8 per-task binomial tests) does not
cover: (a) the ~22 paired bootstrap-CI calls behind Table 1's headline counts
(11 configs x 2 floors) and the 12 behind Table tvfloors's (6 configs x 2
floors) are each reported as raw counts with no family-wise error control;
(b) both bootstraps are plain percentile intervals, unstated as such, on a
statistic (fbar_B - floor_rate) that can sit near the [-1, 1] boundary where
percentile-CI coverage is known to degrade relative to a bias-and-skewness-
corrected (BCa) interval.

This script re-derives every one of those 34 bootstraps from the same
released records, the same paired observation-clustered resampling, the same
seed and n_boot as scripts/floor_convention_robustness.py and
scripts/stage1_stage2_analysis.py -- and checks its own recomputed
percentile CI against those scripts' saved JSON before trusting anything
built on top of it. It then adds, for each of the 34: a two-sided bootstrap
p-value, a BCa interval (bias z0 from the fraction of draws below the point
estimate, acceleration from a leave-one-observation-out jackknife), and a
Holm-corrected verdict within each family (22, then 12) using those
p-values -- the same step-down procedure the paper already uses for the
per-task tests, applied here instead of introduced fresh.

Usage:
    python3 scripts/bootstrap_multiplicity_bca.py
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import floor_convention_robustness as fcr
import stage1_stage2_analysis as ssa

ALPHA = 0.05
N_BOOT = 20000
SEED = 12345


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p, lo=-8.0, hi=8.0):
    """Bisection inverse of norm_cdf -- avoids a hand-rolled rational
    approximation to Phi^-1 where a subtle coefficient error would be easy
    to ship and hard to notice."""
    if p <= 0.0:
        return lo
    if p >= 1.0:
        return hi
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def group_by_obs_fcr(records):
    by_obs = {}
    for r in records:
        if r.get("skipped"):
            continue
        key = (r.get("seed", 0), r["sample"])
        by_obs.setdefault(key, {})[r["family"]] = r
    return by_obs


def binary_diff_stat(by_obs, families_b, floor_family, tau=fcr.TAU_DEFAULT):
    """Exact copy of clustered_bootstrap_diff's internal point_stat closure
    in scripts/floor_convention_robustness.py -- kept as a standalone
    function here (rather than imported) because that one is a closure, not
    a module-level name; the two must stay bit-identical, checked below."""
    fam_hits = {f: [0, 0] for f in families_b + [floor_family]}
    for recs in by_obs.values():
        for f in families_b + [floor_family]:
            rec = recs.get(f)
            if rec is None:
                continue
            fam_hits[f][1] += 1
            if rec["delta_linf"] > tau:
                fam_hits[f][0] += 1
    vals = [h / c for h, c in (fam_hits[f] for f in families_b) if c > 0]
    if not vals:
        return None
    fb = sum(vals) / len(vals)
    fh, fc = fam_hits[floor_family]
    if fc == 0:
        return None
    return fb - (fh / fc)


def bootstrap_bca(by_obs, point_stat_fn, n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    keys = list(by_obs.keys())
    n = len(keys)
    if n == 0:
        return None
    point = point_stat_fn(by_obs)
    if point is None:
        return None

    rng = fcr.Rng(seed)
    draws = []
    for _ in range(n_boot):
        sample_keys = [keys[rng.randint(n)] for _ in range(n)]
        sub = {}
        for i, k in enumerate(sample_keys):
            sub[(k, i)] = by_obs[k]
        v = point_stat_fn(sub)
        if v is not None:
            draws.append(v)
    draws.sort()
    B = len(draws)
    if B == 0:
        return None

    lo_p = draws[int(0.025 * B)]
    hi_p = draws[int(0.975 * B) - 1]
    excl_p = (lo_p > 0 or hi_p < 0)

    frac_le0 = sum(1 for d in draws if d <= 0) / B
    frac_ge0 = sum(1 for d in draws if d >= 0) / B
    p_boot = min(1.0, 2.0 * min(frac_le0, frac_ge0))

    frac_below = sum(1 for d in draws if d < point) / B
    frac_below = min(max(frac_below, 1.0 / (2 * B)), 1.0 - 1.0 / (2 * B))
    z0 = norm_ppf(frac_below)

    jk = []
    for k in keys:
        sub = {kk: vv for kk, vv in by_obs.items() if kk != k}
        v = point_stat_fn(sub)
        if v is not None:
            jk.append(v)
    a_hat = 0.0
    if len(jk) >= 3:
        jk_mean = sum(jk) / len(jk)
        num = sum((jk_mean - v) ** 3 for v in jk)
        den = 6.0 * (sum((jk_mean - v) ** 2 for v in jk) ** 1.5)
        if den != 0:
            a_hat = num / den

    def bca_bound(tail):
        z_a = norm_ppf(tail)
        denom = 1.0 - a_hat * (z0 + z_a)
        adj = 0.5 if denom == 0 else norm_cdf(z0 + (z0 + z_a) / denom)
        adj = min(max(adj, 0.0), 1.0)
        idx = min(max(int(adj * B), 0), B - 1)
        return draws[idx]

    b1, b2 = bca_bound(alpha / 2), bca_bound(1 - alpha / 2)
    lo_b, hi_b = min(b1, b2), max(b1, b2)
    excl_b = (lo_b > 0 or hi_b < 0)

    return {
        "point": point, "n_obs": n, "n_boot_used": B,
        "ci95_percentile": [lo_p, hi_p], "excludes_zero_percentile": excl_p,
        "ci95_bca": [lo_b, hi_b], "excludes_zero_bca": excl_b,
        "z0": z0, "a_hat": a_hat, "p_boot": p_boot,
    }


def holm(pvals, alpha=ALPHA):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def main():
    tests = []  # each: dict(family, config, floor, metric, ...)

    # ---- Family 1: Table 1 / tab:floors -- 11 leaderboard configs x 2 floors ----
    fcr_configs = [c for c in fcr.OURS] + list(fcr.DEEPTHINK)
    assert len(fcr_configs) == 11, len(fcr_configs)
    stored = json.load(open(
        "results_v2/canonical_runs/floor_convention_robustness/"
        "floor_convention_robustness.json"))
    for name in fcr_configs:
        recs = fcr.load_config(name)
        by_obs = group_by_obs_fcr(recs)
        for floor in fcr.FLOORS:
            res = bootstrap_bca(
                by_obs, lambda d, ff=floor: binary_diff_stat(d, fcr.CORE7, ff))
            key = "bootstrap_B_vs_" + ("para" if floor == "paraphrase_null" else "scram")
            stored_excl = stored["per_config"][name][key]["excludes_zero"]
            tests.append({
                "family": "table1_floors", "config": name, "floor": floor,
                "result": res, "stored_excludes_zero": stored_excl,
                "integrity_ok": res["excludes_zero_percentile"] == stored_excl,
            })

    # ---- Family 2: Table tvfloors -- 6 Stage-1 checkpoints x 2 floors ----
    for ckpt in ssa.STAGE1_CHECKPOINTS:
        recs = ssa.load_stage1(ckpt)
        by_obs = ssa.group_by_obs(recs)
        for floor in ssa.FLOORS:
            res = bootstrap_bca(
                by_obs, lambda d, ff=floor: ssa.continuous_point_stat(
                    d, ssa.CORE7, ff, "tv_mean"))
            tests.append({
                "family": "tab_tvfloors", "config": ckpt, "floor": floor,
                "result": res, "stored_excludes_zero": None,
                "integrity_ok": None,
            })

    # Integrity check against the previously-saved Stage-1 JSON (regenerate
    # once, compare, rather than trust the fresh recompute blind).
    ssa_stored = json.load(open(
        "results_v2/canonical_runs/stage1_stage2_analysis/"
        "stage1_stage2_analysis.json"))
    for t in tests:
        if t["family"] != "tab_tvfloors":
            continue
        stored_excl = ssa_stored["stage1"][t["config"]]["bootstrap_tv"][t["floor"]]["excludes_zero"]
        t["stored_excludes_zero"] = stored_excl
        t["integrity_ok"] = t["result"]["excludes_zero_percentile"] == stored_excl

    bad = [t for t in tests if not t["integrity_ok"]]
    if bad:
        print(f"[bootstrap-mult-bca] INTEGRITY FAILURE: recomputed percentile CI "
              f"disagrees with stored JSON for {len(bad)} test(s): "
              f"{[(t['family'], t['config'], t['floor']) for t in bad]}",
              file=sys.stderr)
        return 1

    # ---- Holm correction, within each family separately ----
    out = {"alpha": ALPHA, "n_boot": N_BOOT, "families": {}}
    for fam in ("table1_floors", "tab_tvfloors"):
        fam_tests = [t for t in tests if t["family"] == fam]
        pvals = [t["result"]["p_boot"] for t in fam_tests]
        rej = holm(pvals)
        for t, r in zip(fam_tests, rej):
            t["holm_significant"] = r
        n_raw = sum(1 for t in fam_tests if t["stored_excludes_zero"])
        n_bca = sum(1 for t in fam_tests if t["result"]["excludes_zero_bca"])
        n_holm = sum(1 for t in fam_tests if t["holm_significant"])
        out["families"][fam] = {
            "m_tests": len(fam_tests),
            "n_significant_percentile_uncorrected": n_raw,
            "n_significant_bca_uncorrected": n_bca,
            "n_significant_holm_corrected": n_holm,
            "by_floor": {
                floor: {
                    "n_percentile": sum(1 for t in fam_tests if t["floor"] == floor
                                         and t["stored_excludes_zero"]),
                    "n_bca": sum(1 for t in fam_tests if t["floor"] == floor
                                 and t["result"]["excludes_zero_bca"]),
                    "n_holm": sum(1 for t in fam_tests if t["floor"] == floor
                                  and t["holm_significant"]),
                    "n_total": sum(1 for t in fam_tests if t["floor"] == floor),
                }
                for floor in set(t["floor"] for t in fam_tests)
            },
            "tests": [
                {"config": t["config"], "floor": t["floor"],
                 "point": t["result"]["point"], "p_boot": t["result"]["p_boot"],
                 "percentile_significant": bool(t["stored_excludes_zero"]),
                 "bca_significant": t["result"]["excludes_zero_bca"],
                 "holm_significant": t["holm_significant"],
                 "z0": t["result"]["z0"], "a_hat": t["result"]["a_hat"]}
                for t in fam_tests
            ],
        }

    print(f"{'family':14s} {'m':>3s} {'pctl-raw':>9s} {'BCa-raw':>8s} {'Holm':>5s}")
    for fam, e in out["families"].items():
        print(f"{fam:14s} {e['m_tests']:3d} "
              f"{e['n_significant_percentile_uncorrected']:9d} "
              f"{e['n_significant_bca_uncorrected']:8d} "
              f"{e['n_significant_holm_corrected']:5d}")
        for floor, b in e["by_floor"].items():
            print(f"    {floor:20s} pctl={b['n_percentile']}/{b['n_total']} "
                  f"BCa={b['n_bca']}/{b['n_total']} Holm={b['n_holm']}/{b['n_total']}")

    dest = ("results_v2/canonical_runs/bootstrap_multiplicity_bca/"
            "bootstrap_multiplicity_bca.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\n[bootstrap-mult-bca] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
