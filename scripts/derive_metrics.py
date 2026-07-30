#!/usr/bin/env python3
"""derive_metrics.py -- CoT-Faith v7 derived-metric pipeline.

Recomputes, from the PINNED per-sample edit logs (a_orig / a_edit already on
disk, zero new inference):

  1. magnitude-F        : Eq. 1, the v6 metric (kept for comparability)
  2. F_diff             : F(m,f) - F(m, paraphrase_null)   [differential metric]
  3. F_norm             : F(m,f) / F(m, cross_task_swap)   [ceiling-normalized]
  4. directional-F      : sign-aware faithfulness, per family semantics
                            direction_flip : cos(a_orig[:3], a_edit[:3]) < -0.5
                            gripper_flip   : sign(a[6]) flips
                            negation       : ||a_edit|| < ||a_orig||
  5. cos_xyz            : mean cosine similarity of the xyz translation
                          before vs after the edit (faithful => ~ -1 for
                          direction_flip)
  6. per-token attention: alpha(B) / |B| from the pinned segment boundaries

Writes  results_v2/derived_metrics.json  -- the single canonical artifact that
every figure and every paper number is generated from.
"""
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# ONE canonical run pinned per model.  (v6 mixed two r=32 runs: attention from
# /tmp/cf_sweep/ours-train, edits from /tmp/cf_done/bcihypv3gu.  We now pin
# bcihypv3gu for BOTH and record the other r=32 run as a noise-floor replicate.)
# ---------------------------------------------------------------------------
EDIT_RUNS = {
    "ours-r8":      ["/tmp/cf_full_sweep/lora-r8/cotfaith-edit/cot_edit_report.json"],
    "ours-r16":     ["/tmp/cf_full_sweep/lora-r16/cotfaith-edit/cot_edit_report.json"],
    "ours-r32":     ["/tmp/cf_done/bcihypv3gu/cotfaith-edit/cot_edit_report.json"],
    "ours-r64":     ["/tmp/cf_full_sweep/lora-r64/cotfaith-edit/cot_edit_report.json"],
    "ours-no-cot":  ["/tmp/cf_full_sweep/no-cot/cotfaith-edit/cot_edit_report.json"],
    "ours-data50A": ["/tmp/cf_full_sweep/data-50A/cotfaith-edit/cot_edit_report.json"],
    "ours-data50B": ["/tmp/cf_full_sweep/data-50B/cotfaith-edit/cot_edit_report.json"],
    # 3 file-shuffle seeds, post-C5 location_swap fix, includes paraphrase_null
    "ecot-bridge":  ["/tmp/cf_r3_all/ai3sg9h568/cotfaith-edit/cot_edit_report.json",
                     "/tmp/cf_r3_all/xvz7z8eput/cotfaith-edit/cot_edit_report.json",
                     "/tmp/cf_r3_all/bznf3vq5yu/cotfaith-edit/cot_edit_report.json"],
}
RVIS_RUNS = {
    "ours-r8":      ["/tmp/cf_full_sweep/lora-r8/cotfaith-rvis/rvis_cot_report.json"],
    "ours-r16":     ["/tmp/cf_full_sweep/lora-r16/cotfaith-rvis/rvis_cot_report.json"],
    "ours-r32":     ["/tmp/cf_done/bcihypv3gu/cotfaith-rvis/rvis_cot_report.json"],
    "ours-r64":     ["/tmp/cf_full_sweep/lora-r64/cotfaith-rvis/rvis_cot_report.json"],
    "ours-no-cot":  ["/tmp/cf_full_sweep/no-cot/cotfaith-rvis/rvis_cot_report.json"],
    "ours-data50A": ["/tmp/cf_full_sweep/data-50A/cotfaith-rvis/rvis_cot_report.json"],
    "ours-data50B": ["/tmp/cf_full_sweep/data-50B/cotfaith-rvis/rvis_cot_report.json"],
    "ecot-bridge":  ["/tmp/cf_sweep/ecot-bridge/cotfaith-rvis/rvis_cot_report.json"],
}
# second r=32 run -> run-to-run noise floor for the attention claim
R32_REPLICATE_RVIS = "/tmp/cf_sweep/ours-train/cotfaith-rvis/rvis_cot_report.json"
R32_REPLICATE_EDIT = "/tmp/cf_sweep/ours-train/cotfaith-edit/cot_edit_report.json"

FAMILIES = ["selfsplice_control", "syntactic_scramble", "cross_task_swap",
            "direction_flip", "gripper_flip", "verb_swap", "negation",
            "subject_swap", "location_swap", "adversarial_plausible",
            "paraphrase_null"]
# The 7 NON-CONTROL families: the 10 non-null families minus the 3 Tier-0
# controls (selfsplice_control, syntactic_scramble, cross_task_swap).
NON_CONTROL = ["direction_flip", "gripper_flip", "verb_swap", "negation",
               "subject_swap", "location_swap", "adversarial_plausible"]
TAU = 0.05

# The 13-family calibration run (bolt uy6fmkwtzp) adds the two floors that the
# 11-family seed runs lack.  It is ONE run at n=100, so every quantity derived
# from it is compared only against families from the SAME run -- mixing runs is
# what produced the 0.340-vs-0.354 discrepancy the audit now catches.
CALIB_RUN = "/tmp/cf_r1_calib/cotfaith-edit/cot_edit_report.json"
CALIB_FLOORS = ["paraphrase_null", "bbox_jitter_null", "instr_random_sub"]


# In-repo mirror of every pinned /tmp path, so the pipeline reproduces on a
# clean checkout with no /tmp state.  /tmp wins if present (fresher), otherwise
# results_v2/canonical_runs/ is used.
CANON = os.path.join(ROOT, "results_v2", "canonical_runs")
MIRROR = {
    "/tmp/cf_full_sweep/lora-r8/cotfaith-edit/cot_edit_report.json": "ours_lora-r8_edit.json",
    "/tmp/cf_full_sweep/lora-r16/cotfaith-edit/cot_edit_report.json": "ours_lora-r16_edit.json",
    "/tmp/cf_done/bcihypv3gu/cotfaith-edit/cot_edit_report.json": "ours_lora-r32_edit.json",
    "/tmp/cf_full_sweep/lora-r64/cotfaith-edit/cot_edit_report.json": "ours_lora-r64_edit.json",
    "/tmp/cf_full_sweep/no-cot/cotfaith-edit/cot_edit_report.json": "ours_no-cot_edit.json",
    "/tmp/cf_full_sweep/data-50A/cotfaith-edit/cot_edit_report.json": "ours_data-50A_edit.json",
    "/tmp/cf_full_sweep/data-50B/cotfaith-edit/cot_edit_report.json": "ours_data-50B_edit.json",
    "/tmp/cf_r3_all/ai3sg9h568/cotfaith-edit/cot_edit_report.json": "ecot_bridge_edit_seed0.json",
    "/tmp/cf_r3_all/xvz7z8eput/cotfaith-edit/cot_edit_report.json": "ecot_bridge_edit_seed1.json",
    "/tmp/cf_r3_all/bznf3vq5yu/cotfaith-edit/cot_edit_report.json": "ecot_bridge_edit_seed2.json",
    "/tmp/cf_r1_calib/cotfaith-edit/cot_edit_report.json": "ecot_bridge_edit_13family_calibration.json",
    "/tmp/cf_full_sweep/lora-r8/cotfaith-rvis/rvis_cot_report.json": "ours_lora-r8_rvis.json",
    "/tmp/cf_full_sweep/lora-r16/cotfaith-rvis/rvis_cot_report.json": "ours_lora-r16_rvis.json",
    "/tmp/cf_done/bcihypv3gu/cotfaith-rvis/rvis_cot_report.json": "ours_lora-r32_rvis.json",
    "/tmp/cf_full_sweep/lora-r64/cotfaith-rvis/rvis_cot_report.json": "ours_lora-r64_rvis.json",
    "/tmp/cf_full_sweep/no-cot/cotfaith-rvis/rvis_cot_report.json": "ours_no-cot_rvis.json",
    "/tmp/cf_full_sweep/data-50A/cotfaith-rvis/rvis_cot_report.json": "ours_data-50A_rvis.json",
    "/tmp/cf_full_sweep/data-50B/cotfaith-rvis/rvis_cot_report.json": "ours_data-50B_rvis.json",
    "/tmp/cf_sweep/ecot-bridge/cotfaith-rvis/rvis_cot_report.json": "ecot_bridge_rvis.json",
    "/tmp/cf_sweep/ours-train/cotfaith-rvis/rvis_cot_report.json": "ours_lora-r32_rvis_REPLICATE.json",
    "/tmp/cf_sweep/baseline-spatial/cotfaith-rvis-baseline/rvis_baseline_report.json": "openvla_baseline_spatial_rvis.json",
    "/tmp/cf_sweep/baseline-object/cotfaith-rvis-baseline/rvis_baseline_report.json": "openvla_baseline_object_rvis.json",
    "/tmp/cf_sweep/baseline-goal/cotfaith-rvis-baseline/rvis_baseline_report.json": "openvla_baseline_goal_rvis.json",
    "/tmp/cf_sweep/baseline-10/cotfaith-rvis-baseline/rvis_baseline_report.json": "openvla_baseline_10_rvis.json",
    "/tmp/cf_dt2/ymajxt52mf/cotfaith-deepthink/deepthink_report.json": "deepthink_base.json",
    "/tmp/cf_dt2/9tdkdjsnar/cotfaith-deepthink/deepthink_report.json": "deepthink_sft.json",
    "/tmp/cf_r3_all/8864qtg44h/cotfaith-deepthink/deepthink_report.json": "deepthink_rl.json",
    "/tmp/cf_r3_all/ik5n2thine/cotfaith-lerobot/bridge_report.json": "cross_corpus_bridge_v2_n30.json",
    "/tmp/cf_r3_all/c2saurubyx/cotfaith-lerobot/bridge_report.json": "cross_corpus_fractal_n30.json",
    "/tmp/cf_r3_all/kbb6g24nyg/cotfaith-lerobot/bridge_report.json": "cross_corpus_bcz_n30.json",
}


def load(p):
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    alt = MIRROR.get(p)
    if alt:
        ap = os.path.join(CANON, alt)
        if os.path.exists(ap):
            with open(ap) as fh:
                return json.load(fh)
    return None


def cos(u, v):
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu < 1e-9 or nv < 1e-9:
        return None
    return sum(a * b for a, b in zip(u, v)) / (nu * nv)


def l2(u):
    return math.sqrt(sum(x * x for x in u))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p, den = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------------------
# directional-F: per-family sign-aware faithfulness predicate
# ---------------------------------------------------------------------------
def directional_predicate(family, a_orig, a_edit):
    """Returns (applicable, faithful_bool)."""
    xo, xe = a_orig[:3], a_edit[:3]
    if family == "direction_flip":
        c = cos(xo, xe)
        return (c is not None, (c is not None and c < -0.5))
    if family == "gripper_flip":
        so = 1 if a_orig[6] > 0 else (-1 if a_orig[6] < 0 else 0)
        se = 1 if a_edit[6] > 0 else (-1 if a_edit[6] < 0 else 0)
        return (so != 0 and se != 0, so != 0 and se != 0 and so != se)
    if family == "negation":
        return (l2(a_orig) > 1e-9, l2(a_edit) < l2(a_orig))
    return (False, False)


DIRECTIONAL_FAMILIES = ["direction_flip", "gripper_flip", "negation"]


def per_run_stats(rep):
    """Compute all per-family stats for one edit report."""
    by_fam = defaultdict(list)
    for s in rep.get("per_sample", []):
        if s.get("skipped"):
            continue
        by_fam[s["family"]].append(s)
    out = {}
    for fam, rows in by_fam.items():
        n = len(rows)
        k_mag = sum(1 for r in rows if r["delta_linf"] > TAU)
        rec = {"n": n, "F_mag": k_mag / n if n else None, "k_mag": k_mag}
        # signed cosine of xyz translation, orig vs edit
        cs = [cos(r["a_orig"][:3], r["a_edit"][:3]) for r in rows]
        rec["cos_xyz"] = mean(cs)
        # restricted to the samples magnitude-F calls "faithful"
        csf = [cos(r["a_orig"][:3], r["a_edit"][:3])
               for r in rows if r["delta_linf"] > TAU]
        rec["cos_xyz_faithful_subset"] = mean(csf)
        rec["n_faithful_subset"] = len([c for c in csf if c is not None])
        # directional-F
        if fam in DIRECTIONAL_FAMILIES:
            app = [(a, f) for a, f in
                   (directional_predicate(fam, r["a_orig"], r["a_edit"]) for r in rows)]
            n_app = sum(1 for a, _ in app if a)
            k_dir = sum(1 for a, f in app if a and f)
            rec["n_directional"] = n_app
            rec["F_dir"] = k_dir / n_app if n_app else None
            rec["k_dir"] = k_dir
        out[fam] = rec
    return out


def main():
    models = {}
    for name, paths in EDIT_RUNS.items():
        runs = [load(p) for p in paths]
        runs = [(p, r) for p, r in zip(paths, runs) if r is not None]
        if not runs:
            print(f"  [warn] no edit report for {name}")
            continue
        per_run = [per_run_stats(r) for _, r in runs]
        agg = {"n_runs": len(per_run), "runs": [p for p, _ in runs], "families": {}}
        fams = sorted({f for pr in per_run for f in pr})
        for f in fams:
            vals = [pr[f] for pr in per_run if f in pr]
            e = {
                "n_runs": len(vals),
                "n": vals[0]["n"] if len(vals) == 1 else int(round(mean([v["n"] for v in vals]))),
                "n_total": sum(v["n"] for v in vals),
                "F_mag": mean([v["F_mag"] for v in vals]),
                "F_mag_std": std([v["F_mag"] for v in vals]),
                "F_mag_per_run": [v["F_mag"] for v in vals],
                "cos_xyz": mean([v["cos_xyz"] for v in vals]),
                "cos_xyz_std": std([v["cos_xyz"] for v in vals]),
                "cos_xyz_per_run": [v["cos_xyz"] for v in vals],
                "cos_xyz_faithful_subset": mean([v["cos_xyz_faithful_subset"] for v in vals]),
                "cos_xyz_faithful_subset_per_run": [v["cos_xyz_faithful_subset"] for v in vals],
            }
            k_tot = sum(v["k_mag"] for v in vals)
            n_tot = sum(v["n"] for v in vals)
            lo, hi = wilson(k_tot, n_tot)
            e["F_mag_wilson"] = [lo, hi]
            if f in DIRECTIONAL_FAMILIES:
                e["F_dir"] = mean([v.get("F_dir") for v in vals])
                e["F_dir_std"] = std([v.get("F_dir") for v in vals])
                e["F_dir_per_run"] = [v.get("F_dir") for v in vals]
                kd = sum(v.get("k_dir", 0) for v in vals)
                nd = sum(v.get("n_directional", 0) for v in vals)
                e["n_directional"] = nd
                e["F_dir_wilson"] = list(wilson(kd, nd))
            agg["families"][f] = e
        # ---- derived normalizations ----
        para = agg["families"].get("paraphrase_null", {}).get("F_mag")
        ceil = agg["families"].get("cross_task_swap", {}).get("F_mag")
        for f, e in agg["families"].items():
            e["F_diff"] = (e["F_mag"] - para) if (para is not None and e["F_mag"] is not None) else None
            e["F_norm_ceiling"] = (e["F_mag"] / ceil) if (ceil not in (None, 0) and e["F_mag"] is not None) else None
        agg["paraphrase_null_floor"] = para
        agg["cross_task_swap_ceiling"] = ceil
        # bar-F over the 7 non-control families
        got = [agg["families"][f]["F_mag"] for f in NON_CONTROL if f in agg["families"]]
        agg["F_bar_mag"] = mean(got)
        gotn = [agg["families"][f]["F_norm_ceiling"] for f in NON_CONTROL
                if f in agg["families"] and agg["families"][f]["F_norm_ceiling"] is not None]
        agg["F_bar_norm_ceiling"] = mean(gotn) if gotn else None
        gotd = [agg["families"][f]["F_diff"] for f in NON_CONTROL
                if f in agg["families"] and agg["families"][f]["F_diff"] is not None]
        agg["F_bar_diff"] = mean(gotd) if gotd else None
        models[name] = agg

    # ------------------------------------------------------------------
    # attention: bucket mass AND per-token-normalized mass
    # ------------------------------------------------------------------
    attn = {}
    for name, paths in RVIS_RUNS.items():
        rep = None
        for p in paths:
            rep = load(p)
            if rep:
                break
        if rep is None:
            continue
        rows = rep["per_sample"]
        buckets = {"visual": "action->visual", "instruction": "action->instr",
                   "cot": "action->cot", "action_prev": "action->action_prev"}
        rec = {"n": len(rows), "source": p, "mass": {}, "mass_std": {},
               "seg_len": {}, "per_token": {}}
        for b, key in buckets.items():
            vs = [r[key] for r in rows]
            rec["mass"][b] = mean(vs)
            rec["mass_std"][b] = std(vs)
        segs = [r["segments"] for r in rows]
        lens = {"visual": mean([s["visual_end"] for s in segs]),
                "instruction": mean([s["instr_end"] - s["visual_end"] for s in segs]),
                "cot": mean([s["cot_end"] - s["instr_end"] for s in segs]),
                "action_prev": mean([s["action_end"] - s["cot_end"] for s in segs])}
        rec["seg_len"] = lens
        for b in buckets:
            rec["per_token"][b] = rec["mass"][b] / lens[b] if lens[b] else None
        attn[name] = rec

    # ---- OpenVLA non-CoT baselines (alpha(cot) == 0 BY CONSTRUCTION) ----
    baselines = {}
    for suite in ("spatial", "object", "goal", "10"):
        rep = load(f"/tmp/cf_sweep/baseline-{suite}/cotfaith-rvis-baseline/rvis_baseline_report.json")
        if not rep:
            continue
        a = rep["aggregate"]
        rows = rep.get("per_sample", [])
        segs = [r["segments"] for r in rows if "segments" in r]
        lens = None
        if segs:
            lens = {"visual": mean([s["visual_end"] for s in segs]),
                    "instruction": mean([s["instr_end"] - s["visual_end"] for s in segs]),
                    "cot": mean([s["cot_end"] - s["instr_end"] for s in segs]),
                    "action_prev": mean([s["action_end"] - s["cot_end"] for s in segs])}
        baselines[f"openvla-libero-{suite}"] = {
            "n": a["action->cot"]["n"],
            "mass": {"visual": a["action->visual"]["mean"],
                     "instruction": a["action->instr"]["mean"],
                     "cot": a["action->cot"]["mean"],
                     "action_prev": a["action->action_prev"]["mean"]},
            "mass_std": {"visual": a["action->visual"]["std"],
                         "instruction": a["action->instr"]["std"],
                         "cot": a["action->cot"]["std"],
                         "action_prev": a["action->action_prev"]["std"]},
            "seg_len": lens,
            "cot_zero_is_definitional": True,
        }

    # ---- DeepThinkVLA (PaliGemma; visual bucket outside the schema) ----
    dt = {}
    DT_PATHS = {
        "DT-base": "/tmp/cf_dt2/ymajxt52mf/cotfaith-deepthink/deepthink_report.json",
        "DT-SFT":  "/tmp/cf_dt2/9tdkdjsnar/cotfaith-deepthink/deepthink_report.json",
        "DT-RL":   "/tmp/cf_r3_all/8864qtg44h/cotfaith-deepthink/deepthink_report.json",
    }
    for name, dpath in DT_PATHS.items():
        rep = load(dpath)
        if not rep or not rep.get("n_attn_ok"):
            continue
        a = rep["attention_aggregate"]
        dt[name] = {"n": rep.get("n_attn_ok"), "source": dpath, "model": rep.get("model"),
                    "mass": {k.replace("action->", ""): v["mean"] for k, v in a.items()},
                    "mass_std": {k.replace("action->", ""): v["std"] for k, v in a.items()}}

    # ---- cross-corpus N=30 (F5) ----
    cross = {}
    for tag, path in (("bridge_v2", "/tmp/cf_r3_all/ik5n2thine/cotfaith-lerobot/bridge_report.json"),
                      ("fractal",   "/tmp/cf_r3_all/c2saurubyx/cotfaith-lerobot/bridge_report.json"),
                      ("bcz",       "/tmp/cf_r3_all/kbb6g24nyg/cotfaith-lerobot/bridge_report.json")):
        rep = load(path)
        if not rep:
            continue
        a = rep["attention_aggregate"]
        cross[tag] = {
            "source": path, "dataset": rep.get("dataset"),
            "n_samples_used": rep.get("n_samples_used"),
            "n_attn": rep.get("n_attn_ok"),
            "mass": {k.replace("action->", ""): v["mean"] for k, v in a.items()},
            "mass_std": {k.replace("action->", ""): v["std"] for k, v in a.items()},
            "edit": rep["edit_aggregate"],
        }

    # r=32 replicate -> attention run-to-run noise floor
    rep2 = load(R32_REPLICATE_RVIS)
    noise = None
    if rep2 and "ours-r32" in attn:
        m2 = mean([r["action->cot"] for r in rep2["per_sample"]])
        noise = {"r32_run_A_cot": attn["ours-r32"]["mass"]["cot"],
                 "r32_run_B_cot": m2,
                 "abs_diff_pp": abs(attn["ours-r32"]["mass"]["cot"] - m2) * 100,
                 "run_A": RVIS_RUNS["ours-r32"][0], "run_B": R32_REPLICATE_RVIS}
        cots = [a["mass"]["cot"] for a in attn.values()]
        noise["cluster_spread_pp"] = (max(cots) - min(cots)) * 100
        noise["noise_as_frac_of_spread"] = noise["abs_diff_pp"] / noise["cluster_spread_pp"]

    # -------- two-sided calibration floors (13-family run, ECoT-bridge) --------
    calib = None
    crep = load(CALIB_RUN)
    if crep:
        ag = crep["aggregate"]
        fr = {f: ag[f]["faithful_rate"] for f in ag}
        n = {f: ag[f]["n"] for f in ag}
        f_bar = mean([fr[f] for f in NON_CONTROL if f in fr])
        calib = {
            "source": CALIB_RUN,
            "n_families": len(ag),
            "seed": crep.get("seed"),
            "F_bar_non_control": f_bar,
            "families": {f: {"F_mag": fr[f], "n": n[f],
                             "wilson": wilson(round(fr[f] * n[f]), n[f])}
                         for f in sorted(ag)},
        }
        for floor in CALIB_FLOORS:
            if floor in fr:
                calib[floor] = fr[floor]
                calib[f"F_bar_diff_vs_{floor}"] = f_bar - fr[floor]
        # How much of the CoT-edit response is CoT-SPECIFIC?  instr_random_sub
        # perturbs the instruction, i.e. outside the CoT segment entirely, so it
        # upper-bounds the share of F attributable to CoT->action routing.
        if "instr_random_sub" in fr:
            calib["cot_specificity_ratio"] = f_bar / fr["instr_random_sub"]
            calib["n_families_above_out_of_cot_control"] = sum(
                1 for f in NON_CONTROL if f in fr and fr[f] >= fr["instr_random_sub"])

    out = {
        "_provenance": {
            "generated_by": "scripts/derive_metrics.py",
            "note": "single canonical run pinned per model; zero new inference",
            "tau": TAU,
            "edit_runs": EDIT_RUNS,
            "rvis_runs": RVIS_RUNS,
        },
        "models": models,
        "attention": attn,
        "attention_baselines_noncot": baselines,
        "attention_deepthink": dt,
        "cross_corpus_n30": cross,
        "attention_noise_floor": noise,
        "calibration_floors": calib,
    }
    dest = os.path.join(ROOT, "results_v2", "derived_metrics.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {dest}")

    # -------- human-readable summary --------
    print("\n=== magnitude-F / F_diff / F_norm (paraphrase floor, ceiling) ===")
    for m, a in models.items():
        print(f"{m:14s} floor(para)={a['paraphrase_null_floor']} ceil(cross)={a['cross_task_swap_ceiling']:.3f} "
              f"Fbar={a['F_bar_mag']:.3f} Fbar_norm={a['F_bar_norm_ceiling']:.3f}")
    print("\n=== directional-F vs magnitude-F ===")
    for m, a in models.items():
        for f in DIRECTIONAL_FAMILIES:
            e = a["families"].get(f)
            if not e:
                continue
            print(f"{m:14s} {f:15s} F_mag={e['F_mag']:.3f}  F_dir={e['F_dir']:.3f}  "
                  f"cos_xyz={e['cos_xyz']:+.3f}  cos_xyz|faith={e['cos_xyz_faithful_subset']:+.3f}")
    print("\n=== per-token attention ===")
    for m, a in attn.items():
        pt = a["per_token"]
        print(f"{m:14s} lens={ {k:round(v,1) for k,v in a['seg_len'].items()} }")
        print(f"{'':14s} mass={ {k:round(v,4) for k,v in a['mass'].items()} }")
        print(f"{'':14s} /tok={ {k:round(v,5) for k,v in pt.items()} }  instr/cot={pt['instruction']/pt['cot']:.2f}x")
    if calib:
        print("\n=== two-sided calibration floors (13-family run) ===")
        print(f"  F_bar (7 non-control)   = {calib['F_bar_non_control']:.4f}")
        for floor in CALIB_FLOORS:
            if floor in calib:
                print(f"  {floor:20s} = {calib[floor]:.4f}   "
                      f"F_bar_diff = {calib['F_bar_diff_vs_' + floor]:+.4f}")
        if "cot_specificity_ratio" in calib:
            print(f"  CoT-specificity ratio   = {calib['cot_specificity_ratio']:.4f} "
                  f"({calib['n_families_above_out_of_cot_control']}/7 CoT families "
                  f"reach the out-of-CoT control)")
    if noise:
        print("\n=== attention noise floor ===")
        print(json.dumps(noise, indent=1))


if __name__ == "__main__":
    main()
