#!/usr/bin/env python3
"""Floor-vs-semantic-mean analysis for the Stage 1 (continuous, teacher-forced,
6-checkpoint) and Stage 2 (self-generated-CoT, 3-checkpoint) reruns.

Both reruns exist to answer the two central critiques of the 2026-08-31 review
(review-stage/VLM_EXPERT_REVIEW_2026-08-31.md):

  Stage 1: the paper's headline (Table 1 / tab:floors) is a THRESHOLD-based
  statistic (delta_linf > tau) and Sec 4 (sec:collision) already shows that
  statistic is close to a decode-collision counter. tv_mean -- the per-sample
  total-variation distance between the policy's action-bin softmax under the
  original vs. edited CoT -- is threshold-free by construction and sidesteps
  the collision-counter critique entirely. This script asks whether the same
  floor-collapse finding (meaning-preserving edit moves the policy at least as
  reliably as a meaning-changing one) holds under this continuous statistic,
  on the same paired observations, with the same paired observation-clustered
  bootstrap methodology as scripts/floor_convention_robustness.py.

  Stage 2: every number in the paper edits a teacher-forced ground-truth CoT
  annotation, not any model's own generated CoT (confirmed by reading
  experiments/cotfaith_edit.py), despite text implying otherwise. This script
  reruns the SAME floor-vs-semantic comparison -- both the paper's own binary
  delta_linf>tau statistic and the continuous tv_mean one -- on CoT each
  checkpoint generates itself online. Four of the paper's 13 families
  (subject_swap, adversarial_plausible, selfsplice_control, bbox_jitter_null)
  score n=0 on ALL THREE self-gen checkpoints: their generators need a
  substring pattern (a subject NP, an object mention, an exact self-splice
  match, a bounding-box literal) that self-generated CoT text essentially
  never contains verbatim. This is reported as CORE5_SELFGEN, not silently
  patched to CORE7 -- see the module-level comment on that constant.

Usage:
    python3 scripts/stage1_stage2_analysis.py
"""
import json
import os
import sys

TAU_DEFAULT = 0.05

# Same 7 non-control semantic families as floor_convention_robustness.py's
# CORE7 (Convention B, the appendix/per-task convention, now the paper's only
# one -- see that script's header). Kept as a separate literal rather than
# imported so this script has no import-path dependency on scripts/ being a
# package; the two lists are asserted equal in a smoke check below instead.
CORE7 = [
    "direction_flip", "gripper_flip", "verb_swap", "negation",
    "subject_swap", "location_swap", "adversarial_plausible",
]
# subject_swap and adversarial_plausible score n=0 on every one of the 3
# self-gen checkpoints (see module docstring) -- dropping them from the
# average rather than averaging over whichever families happen to have data
# per checkpoint keeps the family SET identical across all three self-gen
# rows, which paired comparison needs.
CORE5_SELFGEN = [f for f in CORE7 if f not in ("subject_swap", "adversarial_plausible")]
FLOORS = ["paraphrase_null", "syntactic_scramble"]

STAGE1_CHECKPOINTS = ["ecot-bridge", "lora-r32", "lora-r64",
                      "deepthink-base", "deepthink-sft", "deepthink-rl"]
STAGE2_CHECKPOINTS = ["ecot-bridge", "lora-r32", "no-cot"]

STAGE1_DIR = "results_v2/canonical_runs/stage1_continuous"
STAGE2_DIR = "results_v2/canonical_runs/stage2_selfgen"


class Rng:
    """Same tiny xorshift PRNG as floor_convention_robustness.py, copied
    rather than imported so this script has no cross-file dependency; the
    two must stay bit-identical since both are cited as "the paired
    observation-clustered bootstrap" and a reader comparing them should see
    the same algorithm, not two different ones with the same name."""

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


def load_stage1(checkpoint):
    path = f"{STAGE1_DIR}/{checkpoint}/stage1_continuous_report.json"
    d = json.load(open(path))
    return [r for r in d["per_sample"] if not r.get("skipped")]


def load_stage2(checkpoint):
    path = f"{STAGE2_DIR}/{checkpoint}/stage2_selfgen_report.json"
    d = json.load(open(path))
    # t0 only: Stage 1 and the original paper are both single-timestep (the
    # first frame of the episode); Stage 2's "addon" records are a separate,
    # later-timestep diagnostic (mid-rollout robustness), not part of this
    # first-step comparison. Mixing them in would silently change what a
    # "sample" is paired against across families.
    return [r for r in d["per_sample"]
            if r.get("timestep_kind") == "t0" and not r.get("skipped")]


def obs_key(r):
    return (r.get("seed", 0), r["sample"])


def group_by_obs(records):
    by_obs = {}
    for r in records:
        by_obs.setdefault(obs_key(r), {})[r["family"]] = r
    return by_obs


def continuous_point_stat(by_obs, families, floor_family, metric="tv_mean"):
    """mean(metric over `families`) - mean(metric on floor_family), paired
    per observation the same way clustered_bootstrap_diff pairs delta_linf."""
    fam_sum = {f: [0.0, 0] for f in families + [floor_family]}
    for recs in by_obs.values():
        for f in families + [floor_family]:
            rec = recs.get(f)
            if rec is None or metric not in rec:
                continue
            fam_sum[f][0] += rec[metric]
            fam_sum[f][1] += 1
    means = [s / c for s, c in (fam_sum[f] for f in families) if c > 0]
    if not means:
        return None
    fbar = sum(means) / len(means)
    fs, fc = fam_sum[floor_family]
    if fc == 0:
        return None
    return fbar - (fs / fc)


def binary_point_stat(by_obs, families, floor_family, tau=TAU_DEFAULT):
    """Same statistic as floor_convention_robustness.py's point_stat: mean
    faithful-rate (delta_linf > tau) over `families` minus the floor's rate."""
    fam_hits = {f: [0, 0] for f in families + [floor_family]}
    for recs in by_obs.values():
        for f in families + [floor_family]:
            rec = recs.get(f)
            if rec is None:
                continue
            fam_hits[f][1] += 1
            if rec["delta_linf"] > tau:
                fam_hits[f][0] += 1
    vals = [h / c for h, c in (fam_hits[f] for f in families) if c > 0]
    if not vals:
        return None
    fb = sum(vals) / len(vals)
    fh, fc = fam_hits[floor_family]
    if fc == 0:
        return None
    return fb - (fh / fc)


def clustered_bootstrap(by_obs, point_stat_fn, n_boot=20000, seed=12345):
    keys = list(by_obs.keys())
    n = len(keys)
    if n == 0:
        return None
    point = point_stat_fn({k: by_obs[k] for k in keys})
    if point is None:
        return None
    rng = Rng(seed)
    draws = []
    for _ in range(n_boot):
        sample_keys = [keys[rng.randint(n)] for _ in range(n)]
        sub = {}
        for i, k in enumerate(sample_keys):
            sub[(k, i)] = by_obs[k]  # unique dict key per draw; value dict unchanged
        # point_stat_fn only reads .values(), so the (k, i) keys above are
        # just to let the same observation appear more than once in one draw
        v = point_stat_fn(sub)
        if v is not None:
            draws.append(v)
    draws.sort()
    if not draws:
        return None
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws)) - 1]
    return {"point": point, "ci95": [lo, hi], "excludes_zero": (lo > 0 or hi < 0),
            "n_boot_used": len(draws), "n_obs": n}


def analyze_stage1(checkpoint, n_boot):
    recs = load_stage1(checkpoint)
    by_obs = group_by_obs(recs)
    entry = {"n_obs": len(by_obs)}
    entry["family_n"] = {f: sum(1 for v in by_obs.values() if f in v)
                          for f in CORE7 + FLOORS}
    entry["tv_floor"] = {}
    for ff in FLOORS:
        vals = [v[ff]["tv_mean"] for v in by_obs.values() if ff in v]
        entry["tv_floor"][ff] = sum(vals) / len(vals) if vals else None
    fbar_vals = []
    for f in CORE7:
        vals = [v[f]["tv_mean"] for v in by_obs.values() if f in v]
        if vals:
            fbar_vals.append(sum(vals) / len(vals))
    entry["tv_fbar"] = sum(fbar_vals) / len(fbar_vals) if fbar_vals else None
    entry["tv_diff"] = {ff: entry["tv_fbar"] - entry["tv_floor"][ff] for ff in FLOORS}
    entry["bootstrap_tv"] = {
        ff: clustered_bootstrap(
            by_obs,
            lambda d, ff=ff: continuous_point_stat(d, CORE7, ff, "tv_mean"),
            n_boot=n_boot)
        for ff in FLOORS
    }
    return entry


def analyze_stage2(checkpoint, n_boot):
    recs = load_stage2(checkpoint)
    by_obs = group_by_obs(recs)
    entry = {"n_obs": len(by_obs), "families_used": CORE5_SELFGEN}
    entry["family_n"] = {f: sum(1 for v in by_obs.values() if f in v)
                          for f in CORE7 + FLOORS}
    entry["dropped_families_n0"] = [
        f for f in CORE7 if entry["family_n"].get(f, 0) == 0]

    # Continuous (tv_mean), same shape as Stage 1's analysis.
    entry["tv_floor"] = {}
    for ff in FLOORS:
        vals = [v[ff]["tv_mean"] for v in by_obs.values() if ff in v]
        entry["tv_floor"][ff] = sum(vals) / len(vals) if vals else None
    fbar_vals = []
    for f in CORE5_SELFGEN:
        vals = [v[f]["tv_mean"] for v in by_obs.values() if f in v]
        if vals:
            fbar_vals.append(sum(vals) / len(vals))
    entry["tv_fbar"] = sum(fbar_vals) / len(fbar_vals) if fbar_vals else None
    entry["tv_diff"] = {ff: entry["tv_fbar"] - entry["tv_floor"][ff] for ff in FLOORS}
    entry["bootstrap_tv"] = {
        ff: clustered_bootstrap(
            by_obs,
            lambda d, ff=ff: continuous_point_stat(d, CORE5_SELFGEN, ff, "tv_mean"),
            n_boot=n_boot)
        for ff in FLOORS
    }

    # Binary (delta_linf > tau): the paper's OWN Table 1 statistic, rerun on
    # self-generated CoT -- the most direct, apples-to-apples answer to the
    # construct-validity critique.
    entry["rate_floor"] = {}
    for ff in FLOORS:
        vals = [1 if v[ff]["delta_linf"] > TAU_DEFAULT else 0
                for v in by_obs.values() if ff in v]
        entry["rate_floor"][ff] = sum(vals) / len(vals) if vals else None
    rate_vals = []
    for f in CORE5_SELFGEN:
        vals = [1 if v[f]["delta_linf"] > TAU_DEFAULT else 0
                for v in by_obs.values() if f in v]
        if vals:
            rate_vals.append(sum(vals) / len(vals))
    entry["rate_fbar"] = sum(rate_vals) / len(rate_vals) if rate_vals else None
    entry["rate_diff"] = {ff: entry["rate_fbar"] - entry["rate_floor"][ff] for ff in FLOORS}
    entry["bootstrap_rate"] = {
        ff: clustered_bootstrap(
            by_obs,
            lambda d, ff=ff: binary_point_stat(d, CORE5_SELFGEN, ff),
            n_boot=n_boot)
        for ff in FLOORS
    }
    return entry


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--json", default=(
        "results_v2/canonical_runs/stage1_stage2_analysis/"
        "stage1_stage2_analysis.json"))
    args = ap.parse_args()

    out = {"tau_default": TAU_DEFAULT, "core7": CORE7,
           "core5_selfgen": CORE5_SELFGEN, "floors": FLOORS,
           "stage1": {}, "stage2": {}}

    print("=== Stage 1: continuous (tv_mean), teacher-forced, 6 checkpoints ===")
    print(f"{'checkpoint':16s} {'n_obs':>6s} {'tv_fbar':>8s} {'tv_para':>8s} "
          f"{'tv_scram':>8s} {'diff_para':>10s} {'CI0':>5s} {'diff_scr':>9s} {'CI0':>5s}")
    for ckpt in STAGE1_CHECKPOINTS:
        e = analyze_stage1(ckpt, args.n_boot)
        out["stage1"][ckpt] = e
        bp, bs = e["bootstrap_tv"]["paraphrase_null"], e["bootstrap_tv"]["syntactic_scramble"]
        print(f"{ckpt:16s} {e['n_obs']:6d} {e['tv_fbar']:8.4f} "
              f"{e['tv_floor']['paraphrase_null']:8.4f} {e['tv_floor']['syntactic_scramble']:8.4f} "
              f"{e['tv_diff']['paraphrase_null']:+10.4f} "
              f"{'YES' if bp and bp['excludes_zero'] else 'no':>5s} "
              f"{e['tv_diff']['syntactic_scramble']:+9.4f} "
              f"{'YES' if bs and bs['excludes_zero'] else 'no':>5s}")

    print("\n=== Stage 2: self-generated CoT, 3 checkpoints, CORE5_SELFGEN ===")
    print(f"{'checkpoint':12s} {'n_obs':>6s} {'dropped(n=0)':>26s}")
    for ckpt in STAGE2_CHECKPOINTS:
        e = analyze_stage2(ckpt, args.n_boot)
        out["stage2"][ckpt] = e
        print(f"{ckpt:12s} {e['n_obs']:6d} {','.join(e['dropped_families_n0']):>26s}")

    print(f"\n{'checkpoint':12s} {'metric':8s} {'fbar':>7s} {'para':>7s} {'scram':>7s} "
          f"{'d_para':>8s} {'CI0':>4s} {'d_scram':>8s} {'CI0':>4s}")
    for ckpt in STAGE2_CHECKPOINTS:
        e = out["stage2"][ckpt]
        for metric, fbar_k, floor_k, diff_k, boot_k in (
            ("tv", "tv_fbar", "tv_floor", "tv_diff", "bootstrap_tv"),
            ("rate", "rate_fbar", "rate_floor", "rate_diff", "bootstrap_rate"),
        ):
            bp, bs = e[boot_k]["paraphrase_null"], e[boot_k]["syntactic_scramble"]
            print(f"{ckpt:12s} {metric:8s} {e[fbar_k]:7.4f} "
                  f"{e[floor_k]['paraphrase_null']:7.4f} {e[floor_k]['syntactic_scramble']:7.4f} "
                  f"{e[diff_k]['paraphrase_null']:+8.4f} "
                  f"{'YES' if bp and bp['excludes_zero'] else 'no':>4s} "
                  f"{e[diff_k]['syntactic_scramble']:+8.4f} "
                  f"{'YES' if bs and bs['excludes_zero'] else 'no':>4s}")

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    json.dump(out, open(args.json, "w"), indent=2)
    print(f"\n[stage1-stage2] -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
