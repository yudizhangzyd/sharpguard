#!/usr/bin/env python3
"""Convention- and threshold-robustness audit for the two-floors sign-flip claim.

An external review (review-stage/VLM_EXPERT_REVIEW_2026-08-31.md, two
independent re-derivations) found four things this script exists to make
permanently checkable rather than re-litigated by hand each time:

  (W1) The body's F_bar averages 9 families (7 semantic + the cross_task_swap
       ceiling + the instr_random_sub control); the appendix's F_bar averages
       only the 7 non-control families. The "sign flips on all 12
       configurations against syntactic_scramble" headline is 12/12 under the
       first convention and 9-10/12 under the second, and only 2-3/12 of
       those are distinguishable from zero by a paired bootstrap.
  (W2) The sign flip is not robust to the faithfulness threshold tau: it holds
       for tau <= 0.10 and degrades by tau=0.20-0.30. The existing
       threshold_sweep.json sweeps F_bar's rank ordering but never the sign
       this claim depends on.
  (W4) 6 of 7 LoRA configs have near-degenerate action spaces (roll/pitch
       frozen at one bin on most scored samples), which is why F_dir's cosine
       criterion on them degenerates toward a single-axis sign test. This
       script reports the diagnostic so every table can carry it.
  (W6) syntactic_scramble's own judged fluency (4.00) sits between the true
       nulls (4.92/5.00) and the semantic families (2.58-3.66), an unexcluded
       alternative explanation for the sign flip that the paper did not
       previously print next to the number it undermines.

Both conventions, both floors, four values of tau, a paired (observation-
clustered) bootstrap, and the action-space diagnostic are computed here so
the paper can cite one script's output instead of an unaudited hand
recomputation. Exits non-zero if any configuration fails to load or score.

Usage:
    python3 scripts/floor_convention_robustness.py
    python3 scripts/floor_convention_robustness.py --n-boot 20000
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derive_metrics import _to_checkpoint_grid  # noqa: E402

TAU_DEFAULT = 0.05
TAUS = [0.05, 0.10, 0.20, 0.30]

# Convention B (appendix: tab:calibration, tab:per_task): the 7 non-control
# semantic families.
CORE7 = [
    "direction_flip", "gripper_flip", "verb_swap", "negation",
    "subject_swap", "location_swap", "adversarial_plausible",
]
# Convention A (body: tab:floors) adds the maximum-effect ceiling family and
# the out-of-CoT specificity control to the treatment mean.
CONVENTION_A = CORE7 + ["cross_task_swap", "instr_random_sub"]
FLOORS = ["paraphrase_null", "syntactic_scramble"]

OURS = ["ours_no-cot", "ours_lora-r8", "ours_lora-r16", "ours_lora-r32",
        "ours_lora-r64", "ours_data-50A", "ours_data-50B", "ecot_bridge"]
DEEPTHINK = {
    "deepthink_base": "results_v2/canonical_runs/deepthink_base_13family.json",
    "deepthink_sft": "results_v2/canonical_runs/deepthink_sft_13family.json",
    "deepthink_rl": "results_v2/canonical_runs/deepthink_rl_13family.json",
}
BRIDGE_4K = ("bridge_subset_4k", "results_v2/canonical_runs/"
             "bridge_subset_deconfound/cotfaith-bridge-subset-edit/"
             "cot_edit_report.json")
# Same-recipe retraining replicates: independent second training run of each
# of the 7 "ours" LoRA configs, at a byte-identical recipe. Supplies the
# retraining-variance bar (Sec 7 / W1 Finding 4), not a second leaderboard row.
RETRAIN_FILES = {
    c: f"results_v2/canonical_runs/{c}_edit_13family_RETRAIN.json"
    for c in ["ours_lora-r8", "ours_lora-r16", "ours_lora-r32", "ours_lora-r64",
              "ours_data-50A", "ours_data-50B"]
}
# no-CoT's independent second training run was scored before the "_RETRAIN"
# naming convention existed; ours_no-cot_edit_13family_calibration.json is
# functionally that replicate (the leaderboard row is the 3-seed sweep, which
# load_ours prefers over this file -- see load_ours docstring).
RETRAIN_FILES["ours_no-cot"] = (
    "results_v2/canonical_runs/ours_no-cot_edit_13family_calibration.json")

NOOP_EPS = 1 / 256  # OpenVLA-family de-quantization: bin centers are (2k+1)/256;
                     # this is the half-bin distance from true zero.


def load_ours(config):
    """All scored records for a config, preferring the 3-seed sweep.

    Mirrors scripts/fdir_null.py's load_ours: ecot_bridge's seed files
    (ecot_bridge_edit_seed*.json) omit bbox_jitter_null/instr_random_sub, so
    the single-run 13-family calibration file supplements rather than
    replaces them. Reusing this exact logic keeps this script's F_dir/ceiling
    numbers consistent with scripts/fdir_null.py's published ones.
    """
    seeded = sorted(glob.glob(
        f"results_v2/canonical_runs/{config}_edit_13family_seed*.json"))
    seeded += sorted(glob.glob(
        f"results_v2/canonical_runs/{config}_edit_seed*.json"))
    records = []
    for f in seeded:
        records += json.load(open(f))["per_sample"]

    calib = sorted(glob.glob(
        f"results_v2/canonical_runs/{config}_edit_13family_calibration.json"))
    have = {r.get("family") for r in records if not r.get("skipped")}
    for f in calib:
        extra = [r for r in json.load(open(f))["per_sample"]
                 if r.get("family") not in have]
        records += extra
    return records


def load_config(name):
    if name in DEEPTHINK:
        return json.load(open(DEEPTHINK[name]))["per_sample_edit"]
    if name == BRIDGE_4K[0]:
        return json.load(open(BRIDGE_4K[1]))["per_sample"]
    return load_ours(name)


def rate(records, family, tau=TAU_DEFAULT):
    kept = [r for r in records if r.get("family") == family and not r.get("skipped")]
    if not kept:
        return None, 0
    return sum(1 for r in kept if r["delta_linf"] > tau) / len(kept), len(kept)


def fbar(records, families, tau=TAU_DEFAULT):
    vals = [rate(records, f, tau)[0] for f in families]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def cos3(u, v):
    du = math.sqrt(sum(x * x for x in u))
    dv = math.sqrt(sum(x * x for x in v))
    if du == 0 or dv == 0:
        return None
    return sum(a * b for a, b in zip(u, v)) / (du * dv)


def is_noop3(a_orig, eps=NOOP_EPS):
    return all(abs(x) <= eps for x in a_orig[:3])


def action_space_diagnostic(records):
    """ap1/noop3/distinct over the config's distinct (seed, sample) observations.

    a_orig is identical across all 13 families for a given observation (this
    is asserted, not assumed: see the integrity check this script's __main__
    runs before trusting the numbers). Dedup avoids inflating N by ~11x.
    """
    seen, orig_by_obs = set(), {}
    for r in records:
        if r.get("skipped") or "a_orig" not in r:
            continue
        key = (r.get("seed", 0), r["sample"])
        if key in seen:
            continue
        seen.add(key)
        orig_by_obs[key] = tuple(r["a_orig"])
    if not orig_by_obs:
        return None
    n = len(orig_by_obs)
    ap1 = sum(1 for a in orig_by_obs.values()
              if sum(1 for x in a[:3] if abs(x) > NOOP_EPS) <= 1) / n
    noop3 = sum(1 for a in orig_by_obs.values() if is_noop3(a)) / n
    distinct = len(set(orig_by_obs.values()))
    return {"n_obs": n, "ap1": ap1, "noop3": noop3,
            "distinct_a_orig": distinct, "distinct_frac": distinct / n}


def fdir_conditional(records, tau=TAU_DEFAULT, cos_threshold=-0.5):
    """F_dir on direction_flip, broken down by whether a_orig was a no-op.

    a_orig/a_edit are restated on the checkpoint's own de-quantization grid
    before the cosine (and the no-op check) -- same fix, same reason, as
    scripts/fdir_null.py's f_dir(): bin 127 is negative under the release's
    raw P2 grid but exactly zero under the checkpoint's own, exactly the
    near-zero bin the collapsed LoRA action spaces concentrate mass on, and
    this function's F_dir numbers must agree with tab:directional's, which
    are computed on the checkpoint grid.
    """
    kept = [r for r in records if r.get("family") == "direction_flip"
            and not r.get("skipped")]
    if not kept:
        return None
    rows = []
    for r in kept:
        ao, ae = _to_checkpoint_grid(r["a_orig"]), _to_checkpoint_grid(r["a_edit"])
        c = cos3(ao[:3], ae[:3])
        if c is None:
            continue
        rows.append({
            "cos": c, "moved": r["delta_linf"] > tau,
            "noop": is_noop3(ao),
        })
    if not rows:
        return None

    def frac_reversed(subset):
        if not subset:
            return None, 0
        return sum(1 for x in subset if x["cos"] < cos_threshold) / len(subset), len(subset)

    moved = [x for x in rows if x["moved"]]
    noop = [x for x in rows if x["noop"]]
    real = [x for x in rows if not x["noop"]]
    f_all, n_all = frac_reversed(rows)
    f_mov, n_mov = frac_reversed(moved)
    f_noop, n_noop = frac_reversed(noop)
    f_real, n_real = frac_reversed(real)
    return {
        "n": n_all, "F_dir": f_all,
        "F_dir_given_moved": f_mov, "n_moved": n_mov,
        "F_dir_given_noop": f_noop, "n_noop": n_noop,
        "F_dir_given_real": f_real, "n_real": n_real,
    }


class Rng:
    """Tiny xorshift PRNG so this script has no dependency on numpy/random
    global state and is trivially seedable end to end."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF or 0x2545F4914F6CDD1D

    def next(self):
        x = self.s
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 7)
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self.s = x & 0xFFFFFFFFFFFFFFFF
        return x

    def randint(self, n):
        return self.next() % n


def clustered_bootstrap_diff(records, families_b, floor_family, tau=TAU_DEFAULT,
                              n_boot=20000, seed=12345):
    """Paired bootstrap CI for fbar(families_b, tau) - rate(floor_family, tau).

    Resamples (seed, sample) observation ids with replacement -- not records
    independently per family -- because every family shares the same
    observations and treating them as independent strata (as a naive
    per-family bootstrap would) discards that pairing and understates the
    correlation. This is the "observation-clustered" design from the
    independent verification pass, not the simpler "within-family" design;
    it is the one that actually matches the word "paired".
    """
    by_obs = {}
    for r in records:
        if r.get("skipped"):
            continue
        key = (r.get("seed", 0), r["sample"])
        by_obs.setdefault(key, {})[r["family"]] = r
    keys = list(by_obs.keys())
    n = len(keys)
    if n == 0:
        return None

    def point_stat(obs_list):
        fam_hits = {f: [0, 0] for f in families_b + [floor_family]}
        for k in obs_list:
            recs = by_obs[k]
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

    point = point_stat(keys)
    if point is None:
        return None

    rng = Rng(seed)
    draws = []
    for _ in range(n_boot):
        sample_keys = [keys[rng.randint(n)] for _ in range(n)]
        v = point_stat(sample_keys)
        if v is not None:
            draws.append(v)
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws)) - 1]
    return {"point": point, "ci95": [lo, hi], "excludes_zero": (lo > 0 or hi < 0),
            "n_boot_used": len(draws), "n_obs": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--json", default=(
        "results_v2/canonical_runs/floor_convention_robustness/"
        "floor_convention_robustness.json"))
    args = ap.parse_args()

    configs = OURS + list(DEEPTHINK) + [BRIDGE_4K[0]]
    loaded, failures = {}, []
    for name in configs:
        try:
            recs = load_config(name)
        except FileNotFoundError as e:
            failures.append((name, str(e)))
            continue
        if not recs:
            failures.append((name, "empty"))
            continue
        loaded[name] = recs

    # Integrity check this script's own numbers depend on: a_orig must be
    # identical across families for a given observation, or action_space_
    # diagnostic's dedup-by-observation silently picks an arbitrary one.
    for name, recs in loaded.items():
        by_obs = {}
        for r in recs:
            if r.get("skipped") or "a_orig" not in r:
                continue
            key = (r.get("seed", 0), r["sample"])
            a = tuple(r["a_orig"])
            if key in by_obs and by_obs[key] != a:
                failures.append((name, f"a_orig disagreement within observation {key}"))
                break
            by_obs[key] = a

    if failures:
        print(f"[floor-convention] FAILED: {failures}", file=sys.stderr)
        return 1

    out = {"tau_default": TAU_DEFAULT, "taus_swept": TAUS,
           "convention_a_families": CONVENTION_A, "convention_b_families": CORE7,
           "per_config": {}}

    for name, recs in loaded.items():
        entry = {"n_scored": sum(1 for r in recs if not r.get("skipped"))}
        entry["floors"] = {ff: rate(recs, ff)[0] for ff in FLOORS}
        entry["fbar_A"] = fbar(recs, CONVENTION_A)
        entry["fbar_B"] = fbar(recs, CORE7)
        entry["diff_A"] = {ff: entry["fbar_A"] - entry["floors"][ff] for ff in FLOORS}
        entry["diff_B"] = {ff: entry["fbar_B"] - entry["floors"][ff] for ff in FLOORS}
        entry["tau_sweep"] = {}
        for tau in TAUS:
            fb_b = fbar(recs, CORE7, tau)
            fb_a = fbar(recs, CONVENTION_A, tau)
            entry["tau_sweep"][str(tau)] = {
                "B_diff_para": fb_b - rate(recs, "paraphrase_null", tau)[0],
                "B_diff_scram": fb_b - rate(recs, "syntactic_scramble", tau)[0],
                "A_diff_para": fb_a - rate(recs, "paraphrase_null", tau)[0],
                "A_diff_scram": fb_a - rate(recs, "syntactic_scramble", tau)[0],
            }
        entry["bootstrap_B_vs_scram"] = clustered_bootstrap_diff(
            recs, CORE7, "syntactic_scramble", n_boot=args.n_boot)
        entry["bootstrap_B_vs_para"] = clustered_bootstrap_diff(
            recs, CORE7, "paraphrase_null", n_boot=args.n_boot)
        entry["action_space"] = action_space_diagnostic(recs)
        entry["fdir_direction_flip"] = fdir_conditional(recs)
        out["per_config"][name] = entry

    # Retraining-variance bar on (F_bar_B - F(scramble)) and (- F(paraphrase)),
    # the same quantities the bootstrap CIs are on -- this is what Table 1's
    # caption needs to cite instead of asserting "0.092" without a source.
    retrain_moves = {"B_vs_scram": [], "B_vs_para": []}
    for cfg, path in RETRAIN_FILES.items():
        if not os.path.exists(path):
            continue
        recs2 = json.load(open(path))["per_sample"]
        e1, e2 = out["per_config"][cfg]["diff_B"], {
            ff: fbar(recs2, CORE7) - rate(recs2, ff)[0] for ff in FLOORS}
        retrain_moves["B_vs_scram"].append(
            abs(e1["syntactic_scramble"] - e2["syntactic_scramble"]))
        retrain_moves["B_vs_para"].append(abs(e1["paraphrase_null"] - e2["paraphrase_null"]))
        out["per_config"][cfg]["retrain_replicate_diff_B"] = e2
    out["retrain_variance_bar"] = {
        k: {"values": v, "median": statistics.median(v) if v else None,
            "max": max(v) if v else None}
        for k, v in retrain_moves.items()
    }

    # Sign-count summaries -- the numbers that actually go in the paper.
    def sign_counts(tau_key=None):
        res = {}
        for conv in ("A", "B"):
            for ff, short in (("paraphrase_null", "para"), ("syntactic_scramble", "scram")):
                neg = pos = 0
                for name, e in out["per_config"].items():
                    if name == BRIDGE_4K[0]:
                        continue  # excluded from the leaderboard cohort; see W14
                    if tau_key is None:
                        d = e[f"diff_{conv}"][ff]
                    else:
                        d = e["tau_sweep"][tau_key][f"{conv}_diff_{short}"]
                    if d < 0:
                        neg += 1
                    elif d > 0:
                        pos += 1
                res[f"{conv}_vs_{short}"] = {"neg": neg, "pos": pos}
        return res

    out["sign_counts_default_tau"] = sign_counts()
    out["sign_counts_by_tau"] = {str(t): sign_counts(str(t)) for t in TAUS}
    n_sig_scram = sum(
        1 for name, e in out["per_config"].items()
        if name != BRIDGE_4K[0] and e["bootstrap_B_vs_scram"]
        and e["bootstrap_B_vs_scram"]["excludes_zero"])
    n_sig_para = sum(
        1 for name, e in out["per_config"].items()
        if name != BRIDGE_4K[0] and e["bootstrap_B_vs_para"]
        and e["bootstrap_B_vs_para"]["excludes_zero"])
    out["n_significant_B_vs_scram"] = n_sig_scram
    out["n_significant_B_vs_para"] = n_sig_para
    out["n_leaderboard_configs"] = len(out["per_config"]) - 1  # excludes bridge_4k

    print(f"{'config':16s} {'F_A':>6s} {'F_B':>6s} {'para':>6s} {'scram':>6s} "
          f"{'A-scr':>7s} {'B-scr':>7s} {'CI excl0':>9s} {'ap1':>5s} {'noop3':>6s}")
    for name, e in out["per_config"].items():
        b = e["bootstrap_B_vs_scram"]
        asd = e["action_space"] or {}
        print(f"{name:16s} {e['fbar_A']:6.3f} {e['fbar_B']:6.3f} "
              f"{e['floors']['paraphrase_null']:6.3f} {e['floors']['syntactic_scramble']:6.3f} "
              f"{e['diff_A']['syntactic_scramble']:+7.3f} {e['diff_B']['syntactic_scramble']:+7.3f} "
              f"{'YES' if b and b['excludes_zero'] else 'no':>9s} "
              f"{asd.get('ap1', float('nan')):5.2f} {asd.get('noop3', float('nan')):6.2f}")
    print(f"\nSign counts (default tau={TAU_DEFAULT}, {out['n_leaderboard_configs']} configs, "
          "bridge_4k excluded):")
    for k, v in out["sign_counts_default_tau"].items():
        print(f"  {k:16s} neg={v['neg']} pos={v['pos']}")
    print(f"\nBootstrap-significant (95% CI excludes 0): "
          f"vs scramble {n_sig_scram}/{out['n_leaderboard_configs']}, "
          f"vs paraphrase {n_sig_para}/{out['n_leaderboard_configs']}")
    print("\nSign counts by tau (Convention B):")
    for t in TAUS:
        sc = out["sign_counts_by_tau"][str(t)]
        print(f"  tau={t}: vs para neg={sc['B_vs_para']['neg']} pos={sc['B_vs_para']['pos']}"
              f" | vs scram neg={sc['B_vs_scram']['neg']} pos={sc['B_vs_scram']['pos']}")
    print(f"\nRetraining bar (median / max |replicate move|):")
    for k, v in out["retrain_variance_bar"].items():
        print(f"  {k}: median={v['median']:.4f} max={v['max']:.4f}")

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    json.dump(out, open(args.json, "w"), indent=2)
    print(f"\n[floor-convention] -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
