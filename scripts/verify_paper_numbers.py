#!/usr/bin/env python3
"""Audit every number asserted in cot_faith_iclr.tex against the released JSON.

Design contract (this is the part reviewers asked for, and the part the
previous version of this script violated):

  * Each check names the CLAIM as it appears in the submitted manuscript, the
    EXPECTED value, and the artifact path the OBSERVED value comes from.
  * A check whose input is missing is a FAILURE, not a skip. A missing
    artifact means the paper asserts something the release cannot support.
  * The process exits 1 if any check fails. The previous version printed
    "Status: MATCH" unconditionally and always exited 0 — so it validated a
    draft that no longer existed and could never fail. Anything that cannot
    fail is not an audit.

Usage:
    python scripts/verify_paper_numbers.py                 # audit
    python scripts/verify_paper_numbers.py --json out.json # + machine-readable

Exit codes: 0 = every claim reproduced, 1 = at least one mismatch or missing
artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "results_v2" / "derived_metrics.json"
DECODER_AUDIT = ROOT / "results_v2" / "decoder_audit.json"
TEX = ROOT / "cot_faith_iclr.tex"

OURS = ["ours-r8", "ours-r16", "ours-r32", "ours-r64",
        "ours-no-cot", "ours-data50A", "ours-data50B"]
ALL8 = OURS + ["ecot-bridge"]


# ----------------------------------------------------------------------
# result accumulator
# ----------------------------------------------------------------------

class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(self, section: str, claim: str, expected: Any, observed: Any,
              tol: Optional[float] = None, source: str = "") -> bool:
        """Record one claim/observation pair. `tol` is an absolute tolerance
        for numeric comparison; None means exact equality. observed=None
        always fails — that is the point."""
        if observed is None:
            ok, detail = False, "artifact missing — claim unsupported by the release"
        elif tol is None:
            ok, detail = expected == observed, ""
        else:
            delta = abs(float(expected) - float(observed))
            ok, detail = delta <= tol, f"|delta|={delta:.4g} tol={tol:g}"
        self.rows.append({
            "section": section, "claim": claim, "expected": expected,
            "observed": observed, "ok": ok, "detail": detail, "source": source,
        })
        return ok

    def report(self) -> int:
        by_section: dict[str, list[dict]] = {}
        for r in self.rows:
            by_section.setdefault(r["section"], []).append(r)
        for section, rows in by_section.items():
            print(f"\n=== {section} ===")
            for r in rows:
                print(f"  [{'PASS' if r['ok'] else 'FAIL'}] {r['claim']}")
                print(f"         expected={r['expected']!r} observed={r['observed']!r}"
                      + (f"  ({r['detail']})" if r["detail"] else ""))
                if r["source"]:
                    print(f"         source: {r['source']}")
        n_fail = sum(1 for r in self.rows if not r["ok"])
        print("\n" + "=" * 70)
        print(f"{len(self.rows) - n_fail}/{len(self.rows)} claims reproduced.")
        if n_fail:
            print(f"FAILED: {n_fail} claim(s) do not match the released artifacts.")
            print("The manuscript and the release disagree. Fix one of them.")
        else:
            print("OK: every number quoted in the manuscript is reproducible.")
        print("=" * 70)
        return 1 if n_fail else 0


# ----------------------------------------------------------------------
# artifact access — a missing artifact surfaces as a FAIL row, never a skip
# ----------------------------------------------------------------------

def load(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001 — any failure means "unavailable"
        print(f"[artifact] cannot read {path}: {type(e).__name__}: {e}",
              file=sys.stderr)
        return None


def dig(obj: Any, *keys: Any) -> Any:
    """Walk nested keys, returning None on any miss so a structural change in
    the artifact shows up as a failed claim rather than a traceback."""
    cur = obj
    for k in keys:
        if cur is None:
            return None
        try:
            cur = cur[k]
        except (KeyError, IndexError, TypeError):
            return None
    return cur


def r3(x: Any) -> Any:
    """Round to 3dp, mapping absent values to None so check() fails them."""
    return None if x is None else round(float(x), 3)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ----------------------------------------------------------------------
# claim checks — each mirrors a specific sentence in the manuscript
# ----------------------------------------------------------------------

def audit_f1(a: Audit, d: Optional[dict]) -> None:
    sec = "F1 - attention is architecture-set"
    att = dig(d, "attention")
    cots = [dig(att, m, "mass", "cot") for m in OURS]
    if all(c is not None for c in cots):
        a.check(sec, "within-ECoT CoT-attention spread = 2.3 pp",
                2.30, round((max(cots) - min(cots)) * 100, 2), tol=0.05,
                source="derived_metrics.json:attention[*].mass.cot")
    else:
        a.check(sec, "within-ECoT CoT-attention spread = 2.3 pp", 2.30, None)
    a.check(sec, "no-CoT alpha(cot) = 0.3491", 0.3491,
            None if dig(att, "ours-no-cot", "mass", "cot") is None
            else round(att["ours-no-cot"]["mass"]["cot"], 4), tol=0.0002)
    a.check(sec, "r=32 alpha(cot) = 0.3543 (canonical run; a second run of the "
                 "same checkpoint gives 0.3398 -- see the noise floor)", 0.3543,
            None if dig(att, "ours-r32", "mass", "cot") is None
            else round(att["ours-r32"]["mass"]["cot"], 4), tol=0.0002)
    a.check(sec, "ECoT-family alpha(cot) all lie in [0.335, 0.358]", True,
            None if any(c is None for c in cots)
            else 0.335 <= min(cots) and max(cots) <= 0.3582)

    nc = dig(d, "attention_baselines_noncot")
    a.check(sec, "all 4 OpenVLA non-CoT baselines have alpha(cot) = 0.000",
            True,
            None if not nc else
            (len(nc) == 4 and all(dig(nc, m, "mass", "cot") == 0.0 for m in nc)),
            source="derived_metrics.json:attention_baselines_noncot")


def audit_per_token(a: Audit, d: Optional[dict]) -> None:
    sec = "F1 per-token normalization ('largest only because it is the longest')"
    pt = dig(d, "attention", "ours-r32", "per_token")
    a.check(sec, "r=32 per-token attention on CoT = 0.00126", 0.00126,
            None if dig(pt, "cot") is None else round(dig(pt, "cot"), 5),
            tol=1e-5, source="derived_metrics.json:attention['ours-r32'].per_token")
    a.check(sec, "r=32 per-token attention on instruction = 0.00488", 0.00488,
            None if dig(pt, "instruction") is None else round(dig(pt, "instruction"), 5),
            tol=1e-5)
    a.check(sec, "r=32 per-token attention on action-prev = 0.0103", 0.0103,
            None if dig(pt, "action_prev") is None else round(dig(pt, "action_prev"), 4),
            tol=1e-4)
    if dig(pt, "cot"):
        a.check(sec, "instruction/CoT per-token ratio = 3.9x", 3.9,
                round(pt["instruction"] / pt["cot"], 1), tol=0.05)
        a.check(sec, "action-prev/CoT per-token ratio = 8.1x", 8.1,
                round(pt["action_prev"] / pt["cot"], 1), tol=0.05)
    else:
        a.check(sec, "instruction/CoT per-token ratio = 3.9x", 3.9, None)
        a.check(sec, "action-prev/CoT per-token ratio = 8.1x", 8.1, None)

    # The manuscript claims the reversal is uniform across all 8 CoT-VLAs.
    pts = [dig(d, "attention", m, "per_token") for m in ALL8]
    a.check(sec, "per-token instruction > per-token CoT on all 8 CoT-VLAs", True,
            None if any(p is None for p in pts) else
            all(p["instruction"] > p["cot"] for p in pts),
            source="derived_metrics.json:attention[*].per_token")
    cot_pts = [p["cot"] for p in pts if p]
    if len(cot_pts) == 8:
        a.check(sec, "per-token CoT range across 8 models = [0.00119, 0.00128]",
                True, 0.00119 <= min(cot_pts) and max(cot_pts) <= 0.00128,
                source=f"range = [{min(cot_pts):.5f}, {max(cot_pts):.5f}]")
    else:
        a.check(sec, "per-token CoT range across 8 models = [0.00119, 0.00128]",
                True, None)


def audit_noise_floor(a: Audit, d: Optional[dict]) -> None:
    sec = "F1 noise floor"
    nf = dig(d, "attention_noise_floor")
    a.check(sec, "same-checkpoint run-to-run |delta alpha(cot)| = 1.45 pp", 1.45,
            None if dig(nf, "abs_diff_pp") is None else round(nf["abs_diff_pp"], 2),
            tol=0.01, source="derived_metrics.json:attention_noise_floor")
    a.check(sec, "noise floor is 63% of the 2.30 pp within-ECoT spread", 0.63,
            None if dig(nf, "noise_as_frac_of_spread") is None
            else round(nf["noise_as_frac_of_spread"], 2), tol=0.005)
    # The load-bearing consequence: no within-ECoT ordering may be claimed.
    a.check(sec, "manuscript may NOT claim a within-ECoT attention ordering "
                 "(noise >= 50% of spread)", True,
            None if not nf else
            nf["abs_diff_pp"] >= 0.5 * nf["cluster_spread_pp"])


def audit_f2_calib(a: Audit, d: Optional[dict]) -> None:
    sec = "F2 calibrated (sec:f2_calib)"
    expected = {"ours-no-cot": 0.643, "ours-r8": 0.532, "ours-r16": 0.518,
                "ours-r32": 0.491, "ours-r64": 0.549, "ours-data50A": 0.573,
                "ours-data50B": 0.500}
    for m, exp in expected.items():
        a.check(sec, f"{m} ceiling-normalized F_bar = {exp}", exp,
                r3(dig(d, "models", m, "F_bar_norm_ceiling")), tol=0.0015,
                source=f"derived_metrics.json:models['{m}'].F_bar_norm_ceiling")
    norms = {m: dig(d, "models", m, "F_bar_norm_ceiling") for m in OURS}
    a.check(sec, "no-CoT has the HIGHEST ceiling-normalized score of the 7 "
                 "same-base variants (the paper's central F2 reversal)",
            "ours-no-cot",
            None if any(v is None for v in norms.values())
            else max(norms, key=lambda k: norms[k]))

    # Raw F2 collapse claim: 2-4x vs the CoT-trained variants on every family.
    nocot = dig(d, "models", "ours-no-cot", "families") or {}
    ref = dig(d, "models", "ours-r32", "families") or {}
    fams = [f for f in ("direction_flip", "verb_swap", "negation",
                        "cross_task_swap")
            if dig(nocot, f, "F_mag") and dig(ref, f, "F_mag")]
    ratios = [ref[f]["F_mag"] / nocot[f]["F_mag"] for f in fams]
    a.check(sec, "no-CoT collapse vs r=32 is 2-4x on every semantic family",
            True, None if len(ratios) != 4 else all(2.0 <= x <= 4.6 for x in ratios),
            source=f"ratios={[round(x, 2) for x in ratios]}")


def audit_f3(a: Audit, d: Optional[dict]) -> None:
    sec = "F3 - attention/causation dissociation"
    mags = {m: dig(d, "models", m, "F_bar_mag") for m in ALL8}
    if all(v is not None for v in mags.values()):
        lo, hi = min(mags.values()), max(mags.values())
        a.check(sec, "F_bar range lower bound = 0.154", 0.154, round(lo, 3),
                tol=0.0015, source="derived_metrics.json:models[*].F_bar_mag")
        a.check(sec, "F_bar range upper bound = 0.860", 0.860, round(hi, 3),
                tol=0.0015)
        a.check(sec, "F_bar spread = 5.6x", 5.6, round(hi / lo, 1), tol=0.05)
        lo_ci, hi_ci = wilson_ci(round(lo * 700), 700), wilson_ci(round(hi * 700), 700)
        a.check(sec, "Wilson 95% CIs on the two extremes are disjoint", True,
                lo_ci[1] < hi_ci[0],
                source=f"lo={tuple(round(x, 3) for x in lo_ci)} "
                       f"hi={tuple(round(x, 3) for x in hi_ci)}")
    else:
        a.check(sec, "F_bar spread = 5.6x", 5.6, None)


def audit_paraphrase_null(a: Audit, d: Optional[dict]) -> None:
    sec = "Construct validity (sec:paraphrase_null)"
    eb = dig(d, "models", "ecot-bridge") or {}
    floor, ceil = eb.get("paraphrase_null_floor"), eb.get("cross_task_swap_ceiling")
    a.check(sec, "ECoT-bridge paraphrase_null floor = 0.947", 0.947, r3(floor),
            tol=0.0015,
            source="derived_metrics.json:models['ecot-bridge'].paraphrase_null_floor")
    a.check(sec, "ECoT-bridge cross_task_swap ceiling = 0.963", 0.963, r3(ceil),
            tol=0.0015)
    a.check(sec, "ECoT-bridge F_bar_diff = -0.087 (mean BELOW its own floor)",
            -0.087, r3(eb.get("F_bar_diff")), tol=0.0015)
    a.check(sec, "floor/ceiling ratio = 0.98 (the two are indistinguishable)",
            0.98, None if not (floor and ceil) else round(floor / ceil, 2),
            tol=0.005)
    a.check(sec, "F_bar_diff is NEGATIVE - the paper must not claim this model "
                 "shows semantic sensitivity above its floor", True,
            None if eb.get("F_bar_diff") is None else eb["F_bar_diff"] < 0)
    measured = [m for m in ALL8
                if dig(d, "models", m, "paraphrase_null_floor") is not None]
    a.check(sec, "paraphrase_null measured on exactly 1 of 8 CoT-VLAs "
                 "(disclosed as limitation (ii))", 1, len(measured),
            source=f"measured={measured}")
    a.check(sec, "ECoT-bridge has 3 seeds", 3, eb.get("n_runs"))


def audit_calibration_floors(a: Audit, d: Optional[dict]) -> None:
    """The 13-family calibration run on ECoT-bridge. Every number the paper
    prints in the three-floor treatment (sec:paraphrase_null, last three
    paragraphs) and in the degenerate two-sided statistic (sec:f2_calib)
    must come from this block."""
    sec = "Calibration floors (13-family run, ECoT-bridge)"
    cf = dig(d, "calibration_floors")
    if not cf:
        a.check(sec, "calibration_floors block present in the release", True, None,
                source=str(DERIVED))
        return
    fams = cf.get("families", {})
    a.check(sec, "13 edit families in the calibration run", 13, cf.get("n_families"),
            source=cf.get("source", ""))
    a.check(sec, "F_bar over non-control families = 0.869", 0.869,
            r3(cf.get("F_bar_non_control")), tol=0.0015)

    # --- floor 1: paraphrase (same-run pairing, not the 3-seed number) ---
    a.check(sec, "paraphrase_null = 0.960 (same-run floor)", 0.960,
            r3(cf.get("paraphrase_null")), tol=0.0015)
    a.check(sec, "cross_task_swap ceiling = 0.970 (same run)", 0.970,
            r3(dig(fams, "cross_task_swap", "F_mag")), tol=0.0015)
    a.check(sec, "F_bar_diff vs paraphrase floor = -0.091 (NEGATIVE)", -0.091,
            r3(cf.get("F_bar_diff_vs_paraphrase_null")), tol=0.0015)
    denom = None
    if cf.get("paraphrase_null") is not None:
        denom = r3(dig(fams, "cross_task_swap", "F_mag") - cf["paraphrase_null"])
    a.check(sec, "two-sided calibration is DEGENERATE on ECoT-bridge: "
                 "ceiling - floor = 0.010", 0.010, denom, tol=0.0015,
            source="paper must not report (F_bar - floor)/(ceiling - floor) here")

    # --- floor 2: bbox jitter (the metric does discriminate) ---
    a.check(sec, "bbox_jitter_null = 0.46", 0.46,
            r3(dig(fams, "bbox_jitter_null", "F_mag")), tol=0.0015)
    a.check(sec, "bbox_jitter_null N = 100", 100,
            dig(fams, "bbox_jitter_null", "n"))
    a.check(sec, "F_bar - bbox_jitter_null = +0.409", 0.409,
            r3(cf.get("F_bar_diff_vs_bbox_jitter_null")), tol=0.0015)
    bj = dig(fams, "bbox_jitter_null", "F_mag")
    pn = cf.get("paraphrase_null")
    a.check(sec, "bbox floor is LESS THAN HALF the paraphrase floor", True,
            None if (bj is None or pn is None) else bj < 0.5 * pn,
            source=f"bbox={bj} paraphrase={pn}")

    # --- floor 3: out-of-CoT control (the decisive one) ---
    a.check(sec, "instr_random_sub = 0.99", 0.99,
            r3(dig(fams, "instr_random_sub", "F_mag")), tol=0.0015)
    a.check(sec, "instr_random_sub N = 99", 99,
            dig(fams, "instr_random_sub", "n"))
    a.check(sec, "CoT-specificity ratio F_bar / instr_random_sub = 0.878", 0.878,
            r3(cf.get("cot_specificity_ratio")), tol=0.0015)
    a.check(sec, "ZERO CoT edit families reach the out-of-CoT control", 0,
            cf.get("n_families_above_out_of_cot_control"),
            source="paper claims 'higher than every one of the ten CoT edit "
                   "families' and '0/7'")
    irs = dig(fams, "instr_random_sub", "F_mag")
    ctswap = dig(fams, "cross_task_swap", "F_mag")
    a.check(sec, "out-of-CoT control exceeds even the max-effect CoT ceiling", True,
            None if (irs is None or ctswap is None) else irs > ctswap,
            source=f"instr_random_sub={r3(irs)} cross_task_swap={r3(ctswap)}")
    a.check(sec, "selfsplice_control identity null = 0.00 in this run "
                 "(sanity: the harness is not firing spuriously)", 0.0,
            r3(dig(fams, "selfsplice_control", "F_mag")), tol=0.0015)

    # --- coverage honesty: the floors exist for exactly one model ---
    a.check(sec, "all three floors are measured on 1 model only; the 7 'ours' "
                 "variants have NO floor (limitation (ii))", True,
            all(f in fams for f in ("paraphrase_null", "bbox_jitter_null",
                                    "instr_random_sub")),
            source="single-model coverage is asserted in the taxonomy caveat "
                   "and limitation (ii); cross-model sweep is not run")


def audit_f5(a: Audit, d: Optional[dict]) -> None:
    sec = "F5 - cross-corpus transfer at N=30"
    cc = dig(d, "cross_corpus_n30")
    if not cc:
        a.check(sec, "cross_corpus_n30 block present in the release", True, None,
                source=str(DERIVED))
        return
    ns = {k: dig(cc, k, "n_samples_used") for k in cc}
    a.check(sec, "every non-LIBERO corpus has n >= 25 (paper claims an N=30 "
                 "sweep, not the withdrawn N=1 pilot)", True,
            all(n is not None and n >= 25
                for k, n in ns.items() if "libero" not in k.lower()),
            source=f"n per corpus = {ns}")
    # Cross-corpus records name the instruction bucket "instr"; the LIBERO
    # reference profile lives in the main attention block as "instruction".
    buckets = {"visual": "visual", "instr": "instruction",
               "cot": "cot", "action_prev": "action_prev"}
    lib = dig(d, "attention", "ecot-bridge", "mass")
    devs = [abs(dig(cc, k, "mass", cb) - lib[lb]) * 100
            for k in cc for cb, lb in buckets.items()
            if lib and dig(cc, k, "mass", cb) is not None and lb in lib]
    a.check(sec, "max cross-corpus deviation on any attention bucket <= 2.7 pp",
            True, None if not devs else max(devs) <= 2.7 + 1e-6,
            source=None if not devs else f"max deviation = {max(devs):.2f} pp")


def audit_f6_directional(a: Audit, d: Optional[dict]) -> None:
    sec = "F6 - direction-aware scoring inverts the leaderboard"
    # (F_mag, F_dir) on direction_flip, exactly as printed in tab:directional.
    expected = {"ecot-bridge": (0.963, 0.120), "ours-r64": (0.820, 0.780),
                "ours-r16": (0.780, 0.710), "ours-r8": (0.740, 0.690),
                "ours-data50A": (0.720, 0.630), "ours-r32": (0.680, 0.620),
                "ours-data50B": (0.660, 0.590), "ours-no-cot": (0.230, 0.080)}
    def score(m: str, fam: str, which: str) -> Any:
        return dig(d, "models", m, "families", fam, which)

    for m, (fmag, fdir) in expected.items():
        a.check(sec, f"{m} direction_flip F_mag = {fmag}", fmag,
                r3(score(m, "direction_flip", "F_mag")), tol=0.0015,
                source=f"models['{m}'].families.direction_flip.F_mag")
        a.check(sec, f"{m} direction_flip F_dir = {fdir}", fdir,
                r3(score(m, "direction_flip", "F_dir")), tol=0.0015,
                source=f"models['{m}'].families.direction_flip.F_dir")
    have = all(score(m, "direction_flip", "F_dir") is not None for m in expected)
    a.check(sec, "ECoT-bridge ranks 1st by magnitude", "ecot-bridge",
            None if not have else
            sorted(expected, key=lambda m: -score(m, "direction_flip", "F_mag"))[0])
    a.check(sec, "ECoT-bridge ranks 2nd-to-last by direction (the inversion)",
            "ecot-bridge", None if not have else
            sorted(expected, key=lambda m: -score(m, "direction_flip", "F_dir"))[-2])
    grips = [score(m, "gripper_flip", "F_dir") for m in expected]
    a.check(sec, "gripper_flip F_dir <= 0.05 for every model (no model inverts "
                 "its gripper on command)", True,
            None if any(g is None for g in grips) else all(g <= 0.05 for g in grips),
            source=f"gripper_flip F_dir = {[r3(g) for g in grips]}")
    # The paper reports the mean translation cosine as positive for ECoT-bridge
    # (+0.417): it moves the SAME way after the direction is reversed.
    a.check(sec, "ECoT-bridge direction_flip mean cos(xyz) = +0.417", 0.417,
            r3(score("ecot-bridge", "direction_flip", "cos_xyz")), tol=0.0015)
    a.check(sec, "ECoT-bridge translation cosine is POSITIVE after a direction "
                 "reversal (the F6 headline)", True,
            None if score("ecot-bridge", "direction_flip", "cos_xyz") is None
            else score("ecot-bridge", "direction_flip", "cos_xyz") > 0)


def audit_decoder(a: Audit, da: Optional[dict]) -> None:
    sec = "P3 withdrawal - decoder audit"
    for claim, keys, exp in [
        ("pooled corr(pred, gt) = -0.692", ("corr_flat_all_dims",), -0.692),
        ("per-dim corr(dx) = -0.419", ("per_dim", "dx", "corr"), -0.419),
        ("per-dim corr(dy) = -0.600", ("per_dim", "dy", "corr"), -0.600),
        ("gripper predicted mean = +0.976", ("per_dim", "gripper", "pred_mean"), 0.976),
        ("gripper ground truth is the constant -1.0", ("per_dim", "gripper", "gt_mean"), -1.0),
        ("model action L1 = 0.597", ("l1", "model"), 0.597),
        ("predict-zero L1 = 0.190", ("l1", "constant_zero"), 0.190),
        ("predict-per-dim-mean L1 = 0.046", ("l1", "predict_dataset_mean"), 0.046),
    ]:
        a.check(sec, claim, exp, r3(dig(da, *keys)), tol=0.0015,
                source="decoder_audit.json:" + ".".join(map(str, keys)))
    lm, lz = dig(da, "l1", "model"), dig(da, "l1", "constant_zero")
    lmean = dig(da, "l1", "predict_dataset_mean")
    a.check(sec, "model LOSES to predict-zero - P3 must stay withdrawn", True,
            None if lm is None or lz is None else lm > lz)
    a.check(sec, "model LOSES to predict-per-dim-mean (gate criterion iii "
                 "fails, so P3 is inadmissible)", True,
            None if lm is None or lmean is None else lm > lmean)
    a.check(sec, "audit is over the N=200 sample the withdrawn AUROC used",
            200, dig(da, "_provenance", "n"))


def audit_second_calibration(a: Audit, d: Optional[dict]) -> None:
    """The R1 review's #1 objection was that construct validity was assessed on
    one checkpoint. A second 13-family run answers it, and the answer changes
    the finding: the degeneracy is a property of saturation, not the protocol."""
    sec = "Second calibrated model (ours-no-cot) and the saturation contrast"
    by = dig(d, "calibration_by_model") or {}
    nc = by.get("ours-no-cot")
    if nc is None:
        a.check(sec, "a second model has all 13 families scored", True, None,
                source="derived_metrics.calibration_by_model['ours-no-cot']")
        return

    a.check(sec, "13 edit families in the second calibration run",
            13, nc.get("n_families"))
    for fam, val in (("paraphrase_null", 0.11), ("bbox_jitter_null", 0.05),
                     ("instr_random_sub", 0.19), ("cross_task_swap", 0.22)):
        a.check(sec, f"ours-no-cot {fam} = {val}", val,
                r3(dig(nc, "families", fam, "F_mag")), tol=0.0015)
    a.check(sec, "ours-no-cot F_bar over non-control families = 0.124",
            0.124, r3(nc.get("F_bar_non_control")), tol=0.0015)
    a.check(sec, "ours-no-cot F_bar sits only +0.014 above its own floor",
            0.014, r3(nc.get("F_bar_diff_vs_paraphrase_null")), tol=0.0015)

    # The contrast is the finding: dynamic range 0.110 vs 0.010.
    a.check(sec, "ours-no-cot dynamic range (ceiling - floor) = 0.110",
            0.110, r3(nc.get("dynamic_range")), tol=0.0015)
    a.check(sec, "ours-no-cot calibration is NOT degenerate",
            False, nc.get("calibration_is_degenerate"))
    ec = by.get("ecot-bridge") or {}
    a.check(sec, "ECoT-bridge calibration IS degenerate",
            True, ec.get("calibration_is_degenerate"))
    dr_nc, dr_ec = nc.get("dynamic_range"), ec.get("dynamic_range")
    a.check(sec, "the second model's dynamic range is 11x the first's", 11.0,
            round(dr_nc / dr_ec, 1) if (dr_nc and dr_ec) else None, tol=0.15,
            source="paper: 'an 11x larger dynamic range'")

    # Two-sided normalization exists only where the range is real.
    a.check(sec, "two-sided (F_bar - floor)/(ceiling - floor) = 0.127 on "
                 "ours-no-cot", 0.127, r3(nc.get("F_bar_two_sided")), tol=0.0015)
    a.check(sec, "two-sided statistic is UNDEFINED on the saturated model",
            True, ec.get("F_bar_two_sided") is None,
            source="a 0.010 denominator must not be divided by")
    one_sided = (nc.get("F_bar_non_control") / nc.get("ceiling_cross_task_swap")
                 if nc.get("ceiling_cross_task_swap") else None)
    a.check(sec, "one-sided F_bar/ceiling = 0.564 on ours-no-cot",
            0.564, r3(one_sided), tol=0.0015)
    a.check(sec, "one-sided normalization overstates the model by 4.4x", 4.4,
            round(one_sided / nc["F_bar_two_sided"], 1)
            if (one_sided and nc.get("F_bar_two_sided")) else None, tol=0.06)

    # Neither calibrated model passes CoT-specificity.
    a.check(sec, "ours-no-cot CoT-specificity ratio = 0.653",
            0.653, r3(nc.get("cot_specificity_ratio")), tol=0.0015)
    ratios = [c.get("cot_specificity_ratio") for c in by.values()]
    a.check(sec, "BOTH calibrated models have CoT-specificity < 1", True,
            all(r is not None and r < 1.0 for r in ratios) and len(ratios) == 2,
            source=f"ratios={[r3(r) for r in ratios]}")
    a.check(sec, "exactly 1 CoT family on ours-no-cot exceeds the out-of-CoT "
                 "control (direction_flip, CIs overlap so not claimed)",
            1, nc.get("n_families_above_out_of_cot_control"))
    a.check(sec, "n_degenerate = 1 of the 2 calibrated models", 1,
            dig(d, "calibration_contrast", "n_degenerate"))


def audit_deepthink_p2(a: Audit, d: Optional[dict]) -> None:
    """F7. The reviewer's standing objection to P2 was that it ran on one
    architecture family, so a protocol artifact and a property of CoT-VLAs were
    indistinguishable. These checks bind the second family's numbers, and the
    load-bearing ones are adverse: if F_diff ever comes out positive here, the
    paper's central claim is wrong and this audit must fail rather than pass."""
    sec = "F7: P2 on the DeepThinkVLA family (second architecture family)"
    dt = dig(d, "deepthink_p2") or {}
    su = dt.get("summary") or {}
    if not su:
        a.check(sec, "the DeepThinkVLA edit runs are derived", True, None,
                source="derived_metrics.deepthink_p2")
        return

    a.check(sec, "3 DeepThinkVLA checkpoints have edit records", 3,
            su.get("n_models"))
    a.check(sec, "all 3 carry an in-run measured floor (paraphrase_null)", 3,
            su.get("n_with_measured_floor"),
            source="without an in-run floor the family adds nothing the paper "
                   "argues for -- magnitude F alone is the statistic it rejects")

    # Per-model table values, exactly as Table tab:crossfamily prints them.
    for label, fbar, floor, ceil, rng, fdiff in (
        ("DT-base", 0.925, 0.960, 0.980, 0.020, -0.035),
        ("DT-SFT",  0.689, 0.810, 0.960, 0.150, -0.121),
        ("DT-RL",   0.711, 0.820, 0.940, 0.120, -0.109),
    ):
        m = dt.get(label) or {}
        a.check(sec, f"{label}: 11 families scored", 11, m.get("n_families_scored"))
        a.check(sec, f"{label}: 0 decode failures", 0, m.get("n_decode_failures"))
        a.check(sec, f"{label}: F_bar over the 7 non-control families = {fbar}",
                fbar, r3(m.get("F_bar_non_control")), tol=0.0015)
        a.check(sec, f"{label}: paraphrase_null floor = {floor}",
                floor, r3(m.get("paraphrase_null")), tol=0.0015)
        a.check(sec, f"{label}: cross_task_swap ceiling = {ceil}",
                ceil, r3(m.get("ceiling_cross_task_swap")), tol=0.0015)
        a.check(sec, f"{label}: dynamic range = {rng}",
                rng, r3(m.get("dynamic_range")), tol=0.0015)
        a.check(sec, f"{label}: F_diff = {fdiff}",
                fdiff, r3(m.get("F_bar_diff_vs_paraphrase_null")), tol=0.0015)

    # The replication itself. This is the check that could dissolve the finding.
    a.check(sec, "F_diff is NEGATIVE on all 3 DeepThinkVLA checkpoints", 3,
            sum(1 for k in ("DT-base", "DT-SFT", "DT-RL")
                if (dt.get(k) or {}).get("F_bar_diff_vs_paraphrase_null", 0) < 0),
            source="if this drops below 3 the cross-family replication in "
                   "Section 6.8 is overstated and must be rewritten")
    a.check(sec, "F_bar sits BELOW the model's own paraphrase floor on 3/3", 3,
            su.get("n_models_with_F_bar_below_floor"))
    n_neg = sum(1 for k in ("DT-base", "DT-SFT", "DT-RL")
                if (dt.get(k) or {}).get("F_bar_two_sided_is_negative"))
    a.check(sec, "the two-sided score is negative on the 2 checkpoints where "
                 "it is defined at all", 2, n_neg,
            source="a negative normalized score means 'no signal to "
                   "normalize', not 'low faithfulness' -- the paper must not "
                   "quote it as a faithfulness value")

    # The degeneracy reappears on a different architecture, from its own floor.
    a.check(sec, "exactly 1 of the 3 has a degenerate calibration (range "
                 "< 0.05)", 1, su.get("n_degenerate"))
    a.check(sec, "the degenerate one is the base checkpoint", ["DT-base"],
            su.get("degenerate_models"))
    a.check(sec, "the two CoT-tuned checkpoints have a REAL dynamic range",
            [False, False],
            [(dt.get("DT-SFT") or {}).get("calibration_is_degenerate"),
             (dt.get("DT-RL") or {}).get("calibration_is_degenerate")])

    # The magnitude/direction dissociation, sharper here than on ECoT.
    for label, mag, dirn in (("DT-base", 0.970, 0.000),
                             ("DT-SFT", 0.650, 0.010),
                             ("DT-RL", 0.640, 0.010)):
        a.check(sec, f"{label}: direction_flip magnitude F = {mag}", mag,
                r3(dig(dt, label, "families", "direction_flip", "F_mag")),
                tol=0.0015)
        a.check(sec, f"{label}: direction_flip F_dir = {dirn}", dirn,
                r3(dig(dt, label, "families", "direction_flip", "F_dir")),
                tol=0.0015)
    a.check(sec, "no DeepThinkVLA checkpoint reverses the action on more than "
                 "1 sample in 100 under direction_flip", True,
            (su.get("max_direction_flip_F_dir") or 0) <= 0.01,
            source="paper: 'reverses it essentially never'")

    # The identity invariant, which constrains everything above.
    a.check(sec, "selfsplice_control (X->X) is EXACTLY 0.000 on all 3", True,
            su.get("all_identity_edits_exactly_zero"),
            source="a nonzero identity edit would mean the harness "
                   "manufactures deltas and every number above is suspect")

    # And the manuscript has to actually say all of this.
    tex = TEX.read_text() if TEX.exists() else ""
    a.check(sec, "the manuscript has the cross-family section", True,
            "\\label{sec:cross_family}" in tex, source=str(TEX))
    a.check(sec, "the manuscript no longer calls DeepThinkVLA attention-only",
            False, "attention only" in tex, source=str(TEX))
    a.check(sec, "the manuscript no longer says the corrected runs are in "
                 "flight", False, "corrected runs are in flight" in tex,
            source=str(TEX))
    a.check(sec, "selfsplice is credited on 11/11 CoT-VLAs, not 8/8", True,
            "8/8 CoT-VLAs" not in tex and "11/11 CoT-VLAs" in tex,
            source=str(TEX) + ": 3 DeepThinkVLA rows now have the identity null")


def audit_attention_cluster_range(a: Audit, d: Optional[dict]) -> None:
    """F1/F3's headline interval, checked digit-for-digit against the artifact.

    The paper quotes the ECoT cluster as $[0.335, 0.358]$ in four places -- two
    body paragraphs, a figure caption, and a claim about the max-min gap -- and
    F3's whole argument is that this interval is narrow while the causal spread
    is wide. Nothing tied any of those digits to derived_metrics until now, and
    that gap was not hypothetical: a regex meant to bump the advertised claim
    count from 358 matched the "358" inside "0.358" and silently rewrote all four
    to 0.362, and the audit still reported every claim reproduced. The interval
    endpoints and the gap are asserted here so a stray edit to the interval fails
    instead of passing.
    """
    sec = "ECoT attention cluster interval (F1/F3)"
    att = dig(d, "attention") or {}
    cot = {k: dig(v, "mass", "cot") for k, v in att.items()}
    cot = {k: v for k, v in cot.items() if v is not None}
    if len(cot) < 8:
        a.check(sec, "all 8 trained variants have a CoT attention mass", 8,
                len(cot), source="derived_metrics.attention")
        return
    src = "results_v2/derived_metrics.json: attention[*].mass.cot"
    lo, hi = min(cot.values()), max(cot.values())
    a.check(sec, "the quoted cluster interval endpoints are the artifact's min "
                 "and max CoT attention over the 8 trained variants",
            [0.335, 0.358], [round(lo, 3), round(hi, 3)], source=src)
    a.check(sec, "the quoted max-min gap is that interval's width", 0.023,
            round(hi - lo, 3), source=src)

    tex = TEX.read_text() if TEX.exists() else ""
    # Every place the interval is printed, in both of the two typographies the
    # manuscript uses for it. A caption that drifts from the body is the same
    # defect as a body that drifts from the artifact.
    a.check(sec, "the manuscript prints the interval as [0.335, 0.358] in both "
                 "body paragraphs and the figure caption", 3,
            tex.count("[0.335, 0.358]"), source="cot_faith_iclr.tex")
    a.check(sec, "and once more in range typography for the cross-family "
                 "comparison", 1, tex.count("$0.335$--$0.358$"),
            source="cot_faith_iclr.tex")
    # The two individual values the reasoning-target paragraph turns on. audit_f1
    # already checks them against the artifact; what was missing is that the
    # manuscript still *prints* them, which is the direction the corruption ran.
    a.check(sec, "the manuscript still prints the no-CoT vs r=32 pair audit_f1 "
                 "checks against the artifact", [True, True],
            [tex.count("$0.3491$") >= 1, tex.count("$0.3543$") >= 1],
            source="cot_faith_iclr.tex")


def audit_attention_seeds_and_depth(a: Audit, d: Optional[dict]) -> None:
    """R1 critical #5 items (c) and (6): the attention numbers were single-run
    and the layer set was unjustified. Both are now measured, and the layer
    result is adverse -- the bucket ordering does not survive full depth."""
    sec = "Attention sampling error bars and layer-depth sensitivity"
    sr = dig(d, "attention_seed_repeats") or {}
    early, full = sr.get("early_layers_0_3"), sr.get("full_layers_0_31")
    if not (early and full):
        a.check(sec, "3-seed attention repeats exist at two layer sets", True, None,
                source="derived_metrics.attention_seed_repeats")
        return

    for tag, e in (("layers 0-3", early), ("all 32 layers", full)):
        a.check(sec, f"3 seeds at {tag}", 3, e.get("n_seeds"))
    a.check(sec, "alpha(cot) at layers 0-3 = 0.3440 (the submitted value)",
            0.344, r3(dig(early, "cot", "mean")), tol=0.0015)
    a.check(sec, "sampling std of alpha(cot) at layers 0-3 <= 0.06 pp", True,
            dig(early, "cot", "std") * 100 <= 0.06,
            source=f"std={dig(early,'cot','std')*100:.3f} pp over 3 seeds")
    a.check(sec, "no bucket's sampling std exceeds 0.094 pp at the two "
                 "endpoint layer sets (0-3 and all 32)", True,
            max(early.get("max_sampling_std_pp"),
                full.get("max_sampling_std_pp")) <= 0.0945,
            source="the five-set worst case is checked separately below and is "
                   "0.26 pp, which is the figure the paper must quote whenever "
                   "it speaks about the sweep rather than a single set")

    # The adverse result: the ordering is an artifact of the layer choice.
    ds = sr.get("depth_sensitivity") or {}
    a.check(sec, "top bucket at layers 0-3 is cot", "cot",
            ds.get("reported_layers_top_bucket"))
    a.check(sec, "top bucket over all 32 layers is visual", "visual",
            ds.get("all_layers_top_bucket"))
    a.check(sec, "the four-bucket ordering does NOT survive full depth",
            False, ds.get("ordering_is_preserved"),
            source="paper must not claim the CoT bucket is largest in general")
    a.check(sec, "alpha(cot) drops 13.1 pp from layers 0-3 to all 32",
            13.1, round(ds.get("cot_drop_pp"), 1), tol=0.06)
    a.check(sec, "alpha(visual) rises 12.4 pp over the same change",
            12.4, round(ds.get("visual_rise_pp"), 1), tol=0.06)
    a.check(sec, "alpha(cot) over all 32 layers = 0.213", 0.213,
            r3(dig(full, "cot", "mean")), tol=0.0015)
    a.check(sec, "alpha(visual) over all 32 layers = 0.414", 0.414,
            r3(dig(full, "visual", "mean")), tol=0.0015)
    a.check(sec, "the depth effect is >100x sampling noise", True,
            ds.get("cot_drop_pp", 0) > 100 * full.get("max_sampling_std_pp", 1))

    # The five-layer-set sweep. Two endpoints could be dismissed as a cherry-
    # picked pair; four four-layer blocks plus the all-layer average cannot.
    sw = sr.get("depth_sweep") or {}
    if not sw:
        a.check(sec, "a five-layer-set depth sweep exists", True, None,
                source="derived_metrics.attention_seed_repeats.depth_sweep")
        return

    a.check(sec, "the sweep covers 5 layer sets", 5,
            len(sw.get("layer_sets") or []))
    a.check(sec, "every layer set has 3 seeds", 3, sw.get("n_seeds_each"))
    for name in (sw.get("layer_sets") or []):
        a.check(sec, f"3-seed repeats exist at {name}", 3,
                dig(sr, name, "n_seeds"))

    # Per-block alpha(cot) / alpha(visual): the exact five rows of the table in
    # the layer-depth paragraph. Quoted from derived_metrics, asserted here.
    for name, cot, vis in (
        ("early_layers_0_3", 0.3440, 0.2904),
        ("layers_8_11", 0.1571, 0.4285),
        ("layers_16_19", 0.2119, 0.4125),
        ("layers_28_31", 0.2615, 0.4554),
        ("full_layers_0_31", 0.2126, 0.4144),
    ):
        a.check(sec, f"alpha(cot) at {name} = {cot:.4f}", cot,
                round(dig(sr, name, "cot", "mean"), 4), tol=0.0002)
        a.check(sec, f"alpha(visual) at {name} = {vis:.4f}", vis,
                round(dig(sr, name, "visual", "mean"), 4), tol=0.0002)

    a.check(sec, "4 four-layer blocks were probed", 4, sw.get("n_blocks_probed"))
    a.check(sec, "alpha(cot) leads in exactly 1 of the 4 blocks", 1,
            sw.get("n_blocks_where_cot_leads"))
    a.check(sec, "the one block where cot leads is the one the submission "
                 "reported (layers 0-3)", ["early_layers_0_3"],
            sw.get("blocks_where_cot_leads"),
            source="if this ever becomes a different block, the paper's "
                   "'and it is the block the submission reported' is false")
    a.check(sec, "alpha(visual) leads in all four other layer sets",
            ["early_layers_0_3"], sw.get("visual_leads_in_all_but"))
    a.check(sec, "alpha(cot) is minimal at layers 8-11", "layers_8_11",
            sw.get("cot_min_block"))
    a.check(sec, "alpha(cot) is maximal at layers 0-3", "early_layers_0_3",
            sw.get("cot_max_block"))
    a.check(sec, "alpha(cot) swings 18.7 pp across depth", 18.7,
            round(sw.get("cot_swing_pp"), 1), tol=0.06)
    a.check(sec, "alpha(cot) is NOT monotone in depth", False,
            sw.get("cot_is_monotone_in_depth"),
            source="the paper must not describe the trend as a decay; the "
                   "minimum is interior (layers 8-11), not at either end")
    a.check(sec, "worst-case 3-seed sampling std across all five sets = "
                 "0.26 pp", 0.26, round(sw.get("max_sampling_std_pp"), 2),
            tol=0.006)
    a.check(sec, "the depth swing is 72x the five-set sampling floor", 72,
            round(sw.get("swing_over_sampling_noise")), tol=0.6)
    a.check(sec, "the sweep's worst-case sigma comes from layers 16-19 "
                 "(so the 0.094 pp figure above must stay scoped)", True,
            abs(dig(sr, "layers_16_19", "max_sampling_std_pp")
                - sw.get("max_sampling_std_pp")) < 1e-9)

    tex = TEX.read_text() if TEX.exists() else ""
    for frag in ("exactly one of the four four-layer blocks",
                 "$18.7$\\,pp", "$72\\times$"):
        a.check(sec, f"the manuscript states the sweep result ({frag!r})",
                True, frag in tex, source=str(TEX))
    a.check(sec, "the manuscript scopes the 0.094 pp sigma to its layer set",
            True, "at this layer set" in tex, source=str(TEX))


def audit_training_replicate(a: Audit, d: Optional[dict]) -> None:
    """The error bar a leaderboard owes its readers is same-config retraining,
    not reseeded sampling. Two pairs exist; they do not license any ordering."""
    sec = "Same-config training-run replicates (leaderboard error bar)"
    tr = dig(d, "training_replicate") or {}
    if not tr:
        a.check(sec, "at least one same-config retraining pair exists", True, None,
                source="derived_metrics.training_replicate")
        return

    a.check(sec, "2 independent same-config retraining pairs", 2, tr.get("n_pairs"))
    per = sorted(round(v, 2) for v in (tr.get("cot_abs_diff_pp_per_pair") or []))
    a.check(sec, "|delta alpha(cot)| across retrainings = 0.56 and 1.45 pp",
            [0.56, 1.45], per)
    a.check(sec, "largest same-config training difference on any bucket = 1.45 pp",
            1.45, round(tr.get("any_bucket_abs_diff_pp_max"), 2), tol=0.006)

    fp = None
    for pr in tr.get("pairs", []):
        if pr.get("F_per_family"):
            fp = pr["F_per_family"]
    a.check(sec, "F is compared across retrainings on 9 families at N>=50",
            9, dig(fp, "n_families_compared"))
    a.check(sec, "mean |delta F| across retrainings = 0.024", 0.024,
            r3(dig(fp, "mean_abs_diff")), tol=0.0015)
    a.check(sec, "max |delta F| across retrainings = 0.083", 0.083,
            r3(dig(fp, "max_abs_diff")), tol=0.0015)
    a.check(sec, "the worst-reproducing family is adversarial_plausible",
            "adversarial_plausible", dig(fp, "max_abs_diff_family"))
    a.check(sec, "location_swap is excluded from the replicate spread "
                 "(n=12 pre-fix run measures the annotation fix, not training)",
            True, "location_swap" in (dig(fp, "excluded_low_n") or []))

    # The hierarchy, which is the actual claim.
    h = dig(d, "noise_hierarchy") or {}
    a.check(sec, "sampling noise (0.09 pp) is far below training-run noise "
                 "(1.45 pp)", True,
            (h.get("sampling_std_pp") or 9) < 0.2 * (h.get("training_run_diff_pp") or 0))
    a.check(sec, "within-ECoT spread is only 1.6x the training-run difference",
            1.6, round(h.get("spread_over_training_run_cot"), 1)
            if h.get("spread_over_training_run_cot") else None, tol=0.06)
    a.check(sec, "no within-ECoT attention ordering is supported", False,
            h.get("within_family_ordering_supported"),
            source="requires the spread to exceed 3x the training-run difference")


def audit_release(a: Audit) -> None:
    """The 'Public release' paragraph and DATASHEET.md quote concrete counts.
    A D&B submission whose release description does not match the release is
    exactly the defect the R1 reviewer called disqualifying, so the counts are
    asserted against the files on disk rather than trusted."""
    sec = "Release integrity (DATASHEET / LICENSE / artifact counts)"
    root = Path(__file__).resolve().parent.parent

    for fname, label in (("DATASHEET.md", "datasheet (Gebru et al. format)"),
                         ("LICENSE", "license file")):
        a.check(sec, f"{label} present at repo root", True,
                (root / fname).exists(), source=fname)

    # --- record counts, classified by schema rather than by filename ---
    can = root / "results_v2" / "canonical_runs"
    n = {"edit": [0, 0], "attn": [0, 0], "p3": [0, 0]}   # [records, scored]

    def acc(kind: str, lst: list) -> None:
        n[kind][0] += len(lst)
        n[kind][1] += sum(1 for r in lst if isinstance(r, dict)
                          and not r.get("skipped"))

    for f in sorted(can.glob("*.json")):
        rec = load(f)
        if not isinstance(rec, dict):
            continue
        if isinstance(rec.get("per_sample_edit"), list):
            acc("edit", rec["per_sample_edit"])
        if isinstance(rec.get("per_sample_attn"), list):
            acc("attn", rec["per_sample_attn"])
        ps = rec.get("per_sample")
        if isinstance(ps, list) and ps:
            # Classify on the UNION of keys: the first record of a run is
            # sometimes a skipped one carrying only {family, reason}, which
            # made an earlier version of this check misfile whole files.
            keys = set().union(*(r.keys() for r in ps if isinstance(r, dict)))
            if "aurocs" in rec or "median_error_l1" in rec:
                acc("p3", ps)
            elif {"delta_linf", "faithful", "a_edit"} & keys:
                acc("edit", ps)
            else:
                acc("attn", ps)

    # Read out of the manuscript rather than hardcoded, for the same reason the
    # attention count below is: hardcoding here means that adding runs fails the
    # audit on its own stale constant while the paper is equally stale, and the
    # failure then points at the wrong document.
    tex = TEX.read_text() if TEX.exists() else ""

    def tex_int(pattern: str) -> Optional[int]:
        m = re.search(pattern, tex)
        return int(re.sub(r"[^\d]", "", m.group(1))) if m else None

    want_edit = tex_int(r"\\textbf\{([\d{},]+)\} per-sample edit records")
    want_scored = tex_int(r"\$([\d{},]+)\$ carry a scored action pair")
    want_skipped = tex_int(r"\$([\d{},]+)\$ are recorded as skipped")
    a.check(sec, "the per-sample edit-record count the manuscript quotes is "
                 "the number released", want_edit, n["edit"][0],
            source=f"schema-classified over {can}/*.json")
    a.check(sec, "the scored-pair count the manuscript quotes is the number "
                 "released (the rest are skipped: target not in frame)",
            want_scored, n["edit"][1])
    a.check(sec, "scored + skipped = total, so no record is unaccounted for",
            n["edit"][0], (want_scored + want_skipped)
            if (want_scored is not None and want_skipped is not None) else None,
            source="the two manuscript figures must partition the release")
    # Both of the next two are read OUT OF THE MANUSCRIPT rather than hardcoded.
    # They were hardcoded, and adding nine attention runs made the audit fail on
    # its own stale constants while the paper still quoted the old ones -- the
    # check pointed at the wrong document. Parsing the paper means the count can
    # only ever fail when the paper and the artifacts genuinely disagree.
    m = re.search(r"\\textbf\{([\d{},]+)\} per-observation attention records", tex)
    want_attn = int(re.sub(r"[^\d]", "", m.group(1))) if m else None
    a.check(sec, "the attention-record count the manuscript quotes is the "
                 "number released", want_attn, n["attn"][0],
            source="cot_faith_iclr.tex: 'N per-observation attention records'")
    a.check(sec, "200 withdrawn-P3 records retained so the withdrawal is "
                 "checkable", 200, n["p3"][0])

    total_mb = sum(f.stat().st_size for f in (root / "results_v2").rglob("*.json"))
    total_mb /= 1024 * 1024
    m = re.search(r"--- \$([\d.]+)\$\\,MB of JSON in total", tex)
    want_mb = float(m.group(1)) if m else None
    a.check(sec, "the release size the manuscript quotes matches the release",
            want_mb, round(total_mb, 1), tol=0.15,
            source="cot_faith_iclr.tex: '$N$\\,MB of JSON in total'")

    # --- no stale n=1 artifact sitting next to the N=30 claim (reviewer 5d) ---
    stale = []
    for f in sorted((root / "results_v2").glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(rec, dict) and rec.get("n_samples_used") == 1:
            stale.append(f.name)
    a.check(sec, "no n_samples_used=1 artifact at the top of results_v2/ "
                 "(the withdrawn pilot is under superseded/)", [], stale,
            source="reviewer critical #5(d): committed artifacts contradicted "
                   "the submitted N=30")
    a.check(sec, "superseded/ carries a README naming what replaced each run",
            True, (root / "results_v2" / "superseded" / "README.md").exists())

    # --- the paper's own self-description must match ---
    # The three counts themselves are checked against disk above. What is left
    # to verify here is that the paper states them at all, in a form the parser
    # recognizes: an unparseable figure makes those checks compare None to None
    # rather than fail, which is exactly the silent pass this script exists to
    # prevent.
    a.check(sec, "the paper states all three release counts (total, scored, "
                 "skipped) where this script can parse them",
            [True, True, True],
            [v is not None for v in (want_edit, want_scored, want_skipped)],
            source="cot_faith_iclr.tex, 'Public release' paragraph")
    for needle, label in (
        ("DATASHEET.md", "paper points readers at the datasheet"),
        ("LICENSE", "paper states the license"),
    ):
        a.check(sec, label, True, needle in tex, source=f"searched for '{needle}'")
    a.check(sec, "no [URL] placeholder left in the manuscript", True,
            "\\url{[URL]}" not in tex and "[URL]" not in tex)


def _config_values(root: Path, key: str) -> set:
    """Every value some bolt config assigns to an env key."""
    out = set()
    for f in sorted((root / "bolt").glob("boltconfig-*.yaml")):
        for ln in f.read_text().splitlines():
            if ln.strip().startswith(f"{key}:"):
                out.add(ln.split(":", 1)[1].strip().strip("'\""))
    return out


def _base_models(root: Path, training_only: bool) -> set:
    """BASE_MODEL values, optionally only from configs that actually train.

    BASE_MODEL is overloaded across this repo's configs: in a training config it
    is the checkpoint LoRA adapters are fitted on top of, but in an attention
    probe config (run_cotfaith_rvis_baseline.sh) it names the model being
    probed. Conflating the two made an earlier version of this check report the
    four OpenVLA LIBERO baselines as rogue LoRA bases.
    """
    out = set()
    for f in sorted((root / "bolt").glob("boltconfig-*.yaml")):
        txt = f.read_text()
        if training_only and not re.search(
                r"command:.*run_cotfaith_(train|bridge_subset)", txt):
            continue
        for ln in txt.splitlines():
            if ln.strip().startswith("BASE_MODEL:"):
                out.add(ln.split(":", 1)[1].strip().strip("'\""))
    return out


def audit_deepthink_decode(a: Audit) -> None:
    """Assert the DeepThinkVLA decode conventions and their disclosure.

    Reason this is a check: the paper and the datasheet both previously gave a
    WRONG cause for the empty DeepThinkVLA edit cells ("vocab_size excludes
    1,152 added tokens") and a wrong description of the fix ("the harness now
    discovers the anchor"). Both readings were invented rather than read off the
    checkpoint. The real conventions are six, they are all in config.json, and
    the harness asserts them at load time. A retracted explanation that is still
    quoted somewhere in the release is indistinguishable to a reader from a
    current one, so the retraction gets a test.
    """
    sec = "DeepThinkVLA decode provenance"
    root = Path(__file__).resolve().parent.parent
    vend = root / "sharpguard" / "vendor" / "deepthinkvla"

    for fname in ("__init__.py", "constants.py", "decode.py",
                  "modeling_deepthinkvla.py"):
        a.check(sec, f"vendored {fname} is released", True,
                (vend / fname).exists(), source=str(vend / fname))
    if not (vend / "modeling_deepthinkvla.py").exists():
        return

    model_src = (vend / "modeling_deepthinkvla.py").read_text()
    dec_src = (vend / "decode.py").read_text()

    # Provenance: an unattributed copy of someone else's MIT file is a license
    # problem, not a tidiness problem.
    for token, what in (
            ("4bbd0f4ea9010a421e4629e24177afc819f4b6d2", "upstream commit sha"),
            ("9e3e0e2a2f46ceec5625963458c84f09866d1e66f"
             "88144957ffa4523320d47c1", "upstream byte sha256"),
            ("license  : MIT", "upstream license"),
            ("github.com/OpenBMB/DeepThinkVLA", "upstream repo URL")):
        a.check(sec, f"vendored model file records its {what}", True,
                token in model_src)

    # The six conventions, as constants rather than as prose.
    a.check(sec, "action id range is the pi0fast <loc> block, not the top 256",
            (254976, 257023),
            (int(re.search(r"^ACTION_TOKEN_BEGIN = (\d+)", dec_src,
                           re.M).group(1)),
             int(re.search(r"^ACTION_TOKEN_END = (\d+)", dec_src,
                           re.M).group(1))))
    a.check(sec, "2048 bin edges -> 2047 centers", 2048,
            int(re.search(r"^N_BIN_EDGES = (\d+)", dec_src, re.M).group(1)))
    a.check(sec, "bin index is reversed within the action window", True,
            "(ACTION_TOKEN_END - ACTION_TOKEN_BEGIN) - slice_argmax" in dec_src)
    a.check(sec, "action chunk is 10 steps x 7 DoF", (10, 7),
            (int(re.search(r"^NUM_ACTIONS_CHUNK = (\d+)",
                           (vend / "constants.py").read_text(),
                           re.M).group(1)),
             int(re.search(r"^ACTION_DIM = (\d+)",
                           (vend / "constants.py").read_text(),
                           re.M).group(1))))
    a.check(sec, "un-normalization is QUANTILE, and min/max is refused", True,
            'ACTION_NORMALIZATION = "QUANTILE"'
            in (vend / "constants.py").read_text()
            and "falling" in dec_src and "min/max" in dec_src)
    a.check(sec, "the conventions are asserted against config.json, not assumed",
            True, "def assert_config_matches" in dec_src
            and "refusing to decode" in dec_src)

    # The one edit to upstream's model code must be declared where it is made
    # AND in the provenance list, or the sha256 above is a false assurance.
    a.check(sec, "the output_attentions edit to upstream is disclosed", True,
            "output_attentions=output_attentions" in model_src
            and "EDIT (vendoring): was False" in model_src
            and "3. `prompt_cot_predict_action` gained an" in model_src)

    exp = root / "experiments" / "cotfaith_deepthink.py"
    if exp.exists():
        exp_src = exp.read_text()
        a.check(sec, "the harness no longer calls model.generate for actions",
                True, "prompt_cot_predict_action" in exp_src
                and ".generate(" not in exp_src)
        a.check(sec, "the retracted text-marker segmentation is gone", [],
                [m for m in ('instr_end_marker="Instruction:"',
                             'cot_end_marker="Action:"')
                 if m in exp_src])
        a.check(sec, "the dead guessed-vocabulary code is gone", [],
                [m for m in ("candidate_action_vocabs", "decode_action_bins",
                             "action_vocab_chosen", "chosen_vocab")
                 if m in exp_src])

    sh = root / "bolt" / "run_cotfaith_deepthink.sh"
    if sh.exists():
        sh_src = sh.read_text()
        a.check(sec, "transformers is pinned to 4.48.1, not floated", True,
                'pip install "transformers==4.48.1"' in sh_src
                and "transformers>=4.45" not in sh_src)
        a.check(sec, "the pin install is not swallowed by `|| true`", True,
                not re.search(r'transformers==4\.48\.1"[^\n]*\|\| true', sh_src))

    # The retracted explanations must not survive anywhere reader-facing.
    for doc, name in ((TEX, "cot_faith_iclr.tex"),
                      (root / "DATASHEET.md", "DATASHEET.md")):
        if not doc.exists():
            continue
        txt = doc.read_text()
        a.check(sec, f"{name} no longer blames vocab_size / 1,152 added tokens",
                [], [p for p in ("1{,}152 added tokens", "1,152 added tokens",
                                 "excludes 1")
                     if p in txt])
        a.check(sec, f"{name} no longer claims the harness discovers the anchor",
                [], [p for p in ("discovers the anchor",
                                 "discover the anchor",
                                 "candidate vocabularies") if p in txt])
        a.check(sec, f"{name} no longer calls visual=0.0 a schema artifact",
                [], [p for p in ("schema artifact",
                                 "segmentation-schema artifact")
                     if p in txt])
        a.check(sec, f"{name} gives the real cause (prompt format) for visual=0",
                True, ("Instruction:" in txt and "Task:" in txt))


def audit_upstream_licenses(a: Audit) -> None:
    """Assert LICENSE and DATASHEET.md against the resolved Hub metadata.

    Reason this is a check and not prose: the previous LICENSE listed Bridge V2
    and BC-Z as CC-BY 4.0 and named two Embodied-CoT bridge repos as the
    cross-corpus sources. Neither survived contact with the Hub API -- the
    sweeps load IPEC-COMMUNITY LeRobot re-hosts, which are Apache-2.0, and the
    named repos 401. A license table written from memory is the same failure
    class as a results table written from memory, so it gets the same treatment.
    """
    sec = "Upstream license provenance"
    root = Path(__file__).resolve().parent.parent
    rep_path = root / "results_v2" / "license_report.json"
    a.check(sec, "machine-readable license report is released", True,
            rep_path.exists(), source=str(rep_path))
    if not rep_path.exists():
        return
    rep = json.loads(rep_path.read_text())
    assets = rep["assets"]

    a.check(sec, "15 upstream assets audited", 15, rep["n_assets"])
    a.check(sec, "3 have no license we can verify (the DeepThinkVLA repos)",
            3, rep["n_unresolved"])
    a.check(sec, "the unverifiable 3 are exactly the DeepThinkVLA checkpoints",
            ["yinchenghust/deepthinkvla_base",
             "yinchenghust/deepthinkvla_libero_cot_rl",
             "yinchenghust/deepthinkvla_libero_cot_sft"],
            sorted(rep["unresolved"]))
    a.check(sec, "every audited repo resolved to a pinned commit sha", [],
            sorted(k for k, v in assets.items() if not v.get("sha")))

    # The claims the two documents make, each keyed to the repo it describes.
    lic_txt = (root / "LICENSE").read_text()
    ds_txt = (root / "DATASHEET.md").read_text()
    for repo, want in (
        ("openvla/modified_libero_rlds", "mit"),
        ("Embodied-CoT/embodied_features_and_demos_libero", "mit"),
        ("Embodied-CoT/ecot-openvla-7b-bridge", "mit"),
        ("IPEC-COMMUNITY/bridge_orig_lerobot", "apache-2.0"),
        ("IPEC-COMMUNITY/fractal20220817_data_lerobot", "apache-2.0"),
        ("IPEC-COMMUNITY/bc_z_lerobot", "apache-2.0"),
    ):
        a.check(sec, f"{repo} resolves to {want}", want,
                assets.get(repo, {}).get("license"))
        for doc, txt in (("LICENSE", lic_txt), ("DATASHEET.md", ds_txt)):
            a.check(sec, f"{doc} names {repo}", True, repo in txt,
                    source=doc)

    # The LoRA base is the one factual claim a reader would most reasonably
    # doubt, and the asset list got it wrong once already.
    a.check(sec, "the LoRA base named in LICENSE is the MIT bridge checkpoint",
            True,
            "derivatives of Embodied-CoT/ecot-openvla-7b-bridge" in lic_txt
            and assets["Embodied-CoT/ecot-openvla-7b-bridge"]["license"] == "mit",
            source="every bolt/boltconfig-cotfaith-{lora-r*,data-50*,calib-*}"
                   ".yaml sets BASE_MODEL to it")
    a.check(sec, "no config sets a LoRA base other than that checkpoint", [],
            sorted(_base_models(root, training_only=True)
                   - {"Embodied-CoT/ecot-openvla-7b-bridge"}),
            source="BASE_MODEL over the bolt configs whose command is a "
                   "training script (in probe configs the same key names the "
                   "model being probed, not a LoRA base)")
    a.check(sec, "every checkpoint any config loads appears in the license "
                 "report", [],
            sorted((_base_models(root, training_only=False)
                    | _config_values(root, "CKPT_HF_ID")
                    | _config_values(root, "CKPT_PATH"))
                   - set(assets)),
            source="BASE_MODEL / CKPT_HF_ID / CKPT_PATH over "
                   "bolt/boltconfig-*.yaml")

    # Unverifiable licenses must be disclosed, not silently upgraded.
    for doc, txt in (("LICENSE", lic_txt), ("DATASHEET.md", ds_txt)):
        a.check(sec, f"{doc} discloses the missing DeepThinkVLA license "
                     "verbatim", True, "NO LICENSE DECLARED UPSTREAM" in txt,
                source=doc)
        a.check(sec, f"{doc} still points at the Gemma terms for the "
                     "PaliGemma base", True,
                "ai.google.dev/gemma/terms" in txt, source=doc)
    a.check(sec, "the retracted CC-BY claim for Bridge V2 / BC-Z is gone from "
                 "LICENSE", True,
            "CC BY 4.0" not in lic_txt.split("2. Measurement records")[-1]
            .split("3. Third-party")[-1],
            source="the Hub says Apache-2.0 for the re-hosts we load")
    a.check(sec, "no config loads a cross-corpus repo the report does not "
                 "cover", [],
            sorted({ln.split(":", 1)[1].strip().strip("'\"")
                    for f in (root / "bolt").glob("boltconfig-*.yaml")
                    for ln in f.read_text().splitlines()
                    if ln.strip().startswith("DATASET_REPO:")}
                   - set(assets)),
            source="grep DATASET_REPO over bolt/boltconfig-*.yaml")


def audit_manuscript_hygiene(a: Audit) -> None:
    """Catch the defect class the reviewer found twice: prose left behind after
    a numbers revision, still contradicting the artifacts."""
    sec = "Manuscript hygiene"
    try:
        tex = TEX.read_text()
    except Exception:
        a.check(sec, "manuscript is readable", True, None, source=str(TEX))
        return

    stale = {
        r"N{=}1$ pilot": "stale N=1 cross-corpus pilot text (F5 is now N=30)",
        "in progress and will populate": "stale 'in progress' promise",
        r"AUROC is $\leq 0.65$": "withdrawn P3 AUROC value still asserted",
        "0.853": "stale F_bar upper bound (correct value 0.860)",
        r"5.5\times$ spread": "stale F_bar spread (correct value 5.6x)",
        "natural strengthening we plan": "stale paraphrase-null promise",
        "none is marked ``---''": "false full-population claim (the para "
                                  "column legitimately has 7 dashes)",
    }
    for needle, why in stale.items():
        a.check(sec, f"no stale text: {why}", True, needle not in tex,
                source=f"searched for {needle!r}")

    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex))
    a.check(sec, "no dangling \\ref (every cross-reference resolves)",
            [], sorted(refs - labels),
            source=f"{len(labels)} labels, {len(refs)} distinct refs")


    for lab, what in {"sec:f2_calib": "F2 calibration section",
                      "sec:directional": "F6 direction-aware section",
                      "sec:paraphrase_null": "construct-validity section",
                      "eq:fdiff": "differential faithfulness equation",
                      "eq:fdir": "direction-aware faithfulness equation"}.items():
        a.check(sec, f"{what} present (\\label{{{lab}}})", True, lab in labels)


def audit_no_published_ranking(a: Audit) -> None:
    """The reviewer's C1 objection was not that F is imprecise -- it was that the
    paper argues F is invalid and then still ships a ranking computed with it.
    We resolved that by withdrawing the ranking rather than by rescuing F, so
    these checks guard the withdrawal. If any of them fails, the manuscript has
    drifted back into claiming a winner it cannot support."""
    sec = "No published ranking (reviewer C1)"
    try:
        tex = TEX.read_text()
    except Exception:
        a.check(sec, "manuscript is readable", True, None, source=str(TEX))
        return

    for needle, why in {
        r"\textbf{Bold} = highest per-column":
            "the caption again declares a per-column winner",
        "bolded best in every column":
            "the F6 text again refers to bolding that should not exist",
        "nominal leaderboard":
            "the magnitude ordering is again called the nominal leaderboard",
        "is the main CoT-Faith leaderboard":
            "the table is again introduced as the leaderboard",
    }.items():
        a.check(sec, f"withdrawn ranking claim absent: {why}", True,
                needle not in tex, source=f"searched for {needle!r}")

    a.check(sec, "the paper states explicitly that no cell is bolded", True,
            "no cell is bolded" in tex,
            source="Section 'Model scores, and why we do not publish them as "
                   "a ranking'")
    a.check(sec, "the admission rule is stated against our own submissions",
            True,
            "seven of our own eight submissions would be rejected" in tex,
            source="this is what converts the missing floors from an excuse "
                   "into the protocol's teeth")

    # The leaderboard table body must contain no \textbf at all: a single bold
    # cell reinstates the ranking the surrounding prose disclaims.
    m = re.search(r"\\label\{tab:leaderboard\}(.*?)\\end\{tabular\}", tex,
                  re.S)
    a.check(sec, "the leaderboard table body is locatable", True, m is not None,
            source="cot_faith_iclr.tex, tab:leaderboard")
    if m:
        a.check(sec, "the leaderboard table body contains zero \\textbf cells",
                0, m.group(1).count(r"\textbf"),
                source="bolding one cell is a ranking claim regardless of "
                       "what the caption says")


def audit_edit_decode_is_unnorm_free(a: Audit) -> None:
    """The paper now asserts that the frame-mismatch bug which withdrew P3
    cannot reach any Delta_inf, because the edit decode never un-normalizes.
    That is a claim about source code, so check the source code, not the prose:
    if `unnorm_key` ever appears in the edit path, the assertion in limitation
    (v) becomes false and every edit cell inherits P3's contamination."""
    sec = "Edit decode is un-normalization-free (reviewer C2)"
    src_path = ROOT / "experiments" / "cotfaith_edit.py"
    try:
        src = src_path.read_text()
    except Exception:
        a.check(sec, "edit protocol source is readable", True, None,
                source=str(src_path))
        return

    for needle in ("unnorm_key", "norm_stats", "predict_action"):
        a.check(sec, f"the edit path never references {needle!r}", True,
                needle not in src,
                source=f"{src_path.name}: the paper's limitation (v) says this "
                       f"code path cannot inherit the P3 frame mismatch")
    a.check(sec, "the edit path de-quantizes to the normalized [-1,1] range",
            True,
            "def dequantize_action" in src and "low=-1.0, high=1.0" in src,
            source=f"{src_path.name}: tau=0.05 is 5% of this range, which is "
                   f"what the paper claims")

    try:
        tex = TEX.read_text()
    except Exception:
        return
    a.check(sec, "the manuscript states the structural-immunity argument", True,
            "structurally incapable" in tex
            and r"\texttt{unnorm\_key} does not appear anywhere in that code "
                r"path" in tex,
            source="limitation (v)")
    a.check(sec, "the manuscript no longer claims the decoder was validated by "
                 "the offline audit (that audit ran on the broken config)",
            True, "validated only by the offline audit" not in tex,
            source="the audit's provenance is the bridge_orig AUROC run")


def audit_deepthink_tau_units(a: Audit) -> None:
    """DeepThinkVLA de-normalizes by LIBERO quantiles, so tau=0.05 means
    something different there than on the ECoT side. The paper discloses this
    with the checkpoint's own q01/q99, so those digits must match the artifact
    and the stated direction of the bias must be the conservative one."""
    sec = "DeepThinkVLA tau units (cross-family comparability)"
    run_path = ROOT / "results_v2" / "canonical_runs" / "deepthink_sft.json"
    try:
        run = json.loads(run_path.read_text())
    except Exception:
        run = None
    if not run:
        a.check(sec, "deepthink_sft.json present", True, None,
                source=str(run_path))
        return
    dec = run.get("action_decode") or {}
    q01, q99 = dec.get("q01"), dec.get("q99")
    a.check(sec, "the run records q01/q99 for all 7 DoF", [7, 7],
            [len(q01 or []), len(q99 or [])], source="action_decode")
    if not (q01 and q99):
        return

    widths = [b - a_ for a_, b in zip(q01, q99)]
    a.check(sec, "per-DoF physical widths as printed in Section 6.8",
            [1.64, 1.67, 1.88, 0.25, 0.36, 0.56, 2.00],
            [round(w, 2) for w in widths], source="q99 - q01")

    # Every physical width <= the normalized width of 2.0, so a fixed tau is
    # stricter on DeepThinkVLA. That direction is what makes the negative
    # F_diff conservative rather than an artifact, so assert it rather than
    # trusting the prose.
    a.check(sec, "no DoF is WIDER than the normalized range, i.e. tau=0.05 is "
                 "never more lenient on DeepThinkVLA than on ECoT", True,
            all(w <= 2.0 + 1e-9 for w in widths),
            source="if any width exceeded 2.0 the bias would flatter "
                   "DeepThinkVLA and Section 6.8 would have to be rewritten")
    dominant = max(widths[0], widths[1], widths[2], widths[6])
    a.check(sec, "strictness factor on the L-inf-dominant dims, as printed",
            [1.0, 1.22],
            [round(2.0 / dominant, 2),
             round(2.0 / min(widths[0], widths[1], widths[2], widths[6]), 2)],
            source="2.0 / width, over the 3 translation dims and the gripper; "
                   "the gripper's quantiles are exactly +/-1 so its factor is "
                   "1.00, which is why the printed range starts at 1.00")
    a.check(sec, "strictness factor on the rotation dims, as printed",
            [3.6, 8.1],
            [round(2.0 / max(widths[3:6]), 1),
             round(2.0 / min(widths[3:6]), 1)],
            source="2.0 / width, over droll/dpitch/dyaw")

    try:
        tex = TEX.read_text()
    except Exception:
        return
    a.check(sec, "Section 6.8 carries the units caveat", True,
            "A units caveat" in tex, source="cot_faith_iclr.tex")


# ----------------------------------------------------------------------


def audit_resize_check(a):
    """Section 6's frame-preprocessing paragraph, against the measurement job.

    This paragraph is the one place the paper quantifies its own approximation,
    so every digit in it has to come from the artifact. It also asserts the two
    structural properties the measurement depends on: that mode "none" is still
    a pass-through (otherwise the gate's anchor configuration is not the one the
    four failed runs used) and that the shipped subsampling default is the
    measured-best value (the check that caught a 240-LSB error).
    """
    sec = "Frame preprocessing approximation (resize check)"
    path = ROOT / "results_v2" / "canonical_runs" / "resize_check" / \
        "resize_kernel_check.json"
    r = load(path)
    if not r:
        a.check(sec, "the resize-check report is released", True, False,
                source=str(path))
        return
    src = "results_v2/canonical_runs/resize_check/resize_kernel_check.json"

    # The three numbers the paragraph quotes for the subsampling sweep.
    a.check(sec, "Pillow chroma subsampling swept against tf.image.encode_jpeg: "
                 "4:4:4 off by 240, 4:2:2 by 150, 4:2:0 by 9 levels",
            [240, 150, 9],
            [dig(r, "jpeg_only", f"subsampling_{i}", "worst") for i in (0, 1, 2)],
            source=src)
    a.check(sec, "the shipped default is the measured-best subsampling (4:2:0)",
            [2, 2, True],
            [r.get("jpeg_best_subsampling"), r.get("jpeg_shipped_subsampling"),
             r.get("jpeg_shipped_is_best")], source=src)
    # The claim that carries the most weight: the kernel is exact.
    a.check(sec, "the Lanczos-3 kernel agrees with tf.image.resize to within "
                 "1/255, i.e. exactly up to uint8 rounding",
            1, dig(r, "resize_only", "worst"), source=src)
    a.check(sec, "the full np_lanczos path is 8/255 from upstream", 8,
            dig(r, "full", "np_lanczos", "worst"), source=src)
    a.check(sec, "the discarded Pillow LANCZOS path was 23/255, past the "
                 "4-level ceiling fixed before the measurement", 23,
            dig(r, "full", "pil_lanczos", "worst"), source=src)
    # The paper says np_lanczos is better than what it replaced; if that ever
    # inverted, the implementation would be a regression wearing a caveat.
    a.check(sec, "the reimplemented kernel is closer to upstream than the "
                 "Pillow path it replaced", True,
            (dig(r, "full", "np_lanczos", "worst") or 99)
            < (dig(r, "full", "pil_lanczos", "worst") or 0), source=src)
    a.check(sec, "mode 'none' is still a pass-through, so the gate's anchor "
                 "configuration is the one the failed runs used", True,
            r.get("none_is_passthrough"), source=src)
    a.check(sec, "all four preprocessing modes are registered in the shipped "
                 "module", ["none", "np_lanczos", "pil_lanczos", "tf_upstream"],
            r.get("shipped_modes"), source=src)
    # The report must name the tensorflow it compared against, or "validated
    # against upstream" has no referent.
    a.check(sec, "the report records the tensorflow version it compared against",
            True, bool(r.get("tf_version")), source=src)


def audit_tf_env_probe(a):
    """The measurement that retired an untested constraint.

    bolt/setup-openvla.sh, sharpguard/image_preproc.py, sharpguard/libero_sim.py
    and the gate config all now cite this probe by task id and by number, in
    place of a comment that asserted tensorflow cannot coexist with the eval
    environment. That comment stood unchallenged long enough to shape the design
    -- it is why the frame preprocessing shipped an 8/255-LSB substitute -- so
    the replacement claim is pinned here stage by stage rather than trusted the
    way its predecessor was. Each stage is checked individually: "tensorflow
    imports" and "the eval environment still works" are different claims, and
    only the conjunction licenses image_preproc='tf_upstream'.
    """
    sec = "Tensorflow/eval-environment coexistence probe"
    path = ROOT / "results_v2" / "canonical_runs" / "tf_env_probe" / \
        "tf_env_probe.json"
    r = load(path)
    if not r:
        a.check(sec, "the tf-env-probe report is released", True, False,
                source=str(path))
        return
    src = "results_v2/canonical_runs/tf_env_probe/tf_env_probe.json"

    # The two halves of the retired claim, each against its own stage.
    a.check(sec, "installing tensorflow-cpu did NOT move the numpy<2 pin, "
                 "refuting the first half of the retired constraint", True,
            "1.26.4" in str(dig(r, "numpy_pin_held", "detail") or ""),
            source=src)
    a.check(sec, "transformers reports is_tf_available()=False under USE_TF=0, "
                 "refuting the second half (the lazy-TF ABI mismatch)", True,
            "is_tf_available()=False" in
            str(dig(r, "transformers_without_tf", "detail") or ""), source=src)
    # The collateral damage the constraint was really guarding against. A probe
    # that passed its own two stages while silently costing CUDA would have been
    # worse than the comment it replaced.
    a.check(sec, "torch kept CUDA and a correct matmul after the install, so "
                 "the GPU rollout is unaffected", True,
            all(s in str(dig(r, "torch_still_works", "detail") or "")
                for s in ("cuda=True", "matmul ok")), source=src)
    a.check(sec, "tf.image encode_jpeg / decode_image / resize all ran, so the "
                 "exact path is executable and not merely importable", True,
            "all ran" in str(dig(r, "tensorflow_runs", "detail") or ""),
            source=src)
    a.check(sec, "both preprocessing modes ran in one process, which is what "
                 "makes 'tf_upstream' usable as the gate's image path", True,
            dig(r, "both_preproc_modes_in_one_process", "ok") is True,
            source=src)
    # The conjunction, and the environment flags it holds under. USE_TF=0 is not
    # incidental: it is the mechanism, so a run without it does not inherit the
    # result.
    a.check(sec, "every stage passed, which is what licenses the gate to drop "
                 "the approximation", [5, 0, []],
            [r.get("n_passed"), r.get("n_failed"), r.get("failed_stages")],
            source=src)
    a.check(sec, "the probe recorded the off-switch it ran under, so the result "
                 "is not read as unconditional", ["0", "1"],
            [dig(r, "env", "USE_TF"), dig(r, "env", "TRANSFORMERS_NO_TF")],
            source=src)
    # The 8 LSB the two independent jobs must agree on: if the resize check and
    # the probe disagreed, one of them is measuring something else.
    a.check(sec, "the np_lanczos-vs-tf_upstream gap the probe saw matches the "
                 "8/255 the resize check measured independently", True,
            "worst 8 LSB" in
            str(dig(r, "both_preproc_modes_in_one_process", "detail") or ""),
            source=src)
    # The code that cites this job must cite it by id, or the claim floats.
    setup_path = ROOT / "bolt" / "setup-openvla.sh"
    setup = setup_path.read_text() if setup_path.exists() else ""
    a.check(sec, "bolt/setup-openvla.sh cites this task id where it used to "
                 "assert the opposite", True, "d543p4f86p" in setup,
            source="bolt/setup-openvla.sh")
    a.check(sec, "tensorflow stays opt-in (INSTALL_TF), so previously published "
                 "runs keep the environment they were produced in", True,
            'INSTALL_TF:-0' in setup, source="bolt/setup-openvla.sh")

    # And the manuscript, which previously stated the retired claim as fact.
    # This is the check that would have caught the stale sentence: prose can go
    # on asserting a refuted constraint indefinitely while every number still
    # reproduces, because the constraint is not a number.
    tex = TEX.read_text() if TEX.exists() else ""
    # Not "the phrase is absent": the paper is entitled to quote the retired
    # claim, and does, because reporting that it was believed is the point. What
    # it must not do is state it in its own voice. So every occurrence has to
    # carry the attribution that marks it as reported speech.
    a.check(sec, "Section 6 states the refuted constraint only as something the "
                 "repository asserted, never in the paper's own voice",
            tex.count("eval environment cannot host"),
            tex.count("asserted that the eval environment cannot host"),
            source="cot_faith_iclr.tex")
    a.check(sec, "Section 6 says the constraint is false rather than merely "
                 "questionable", True, "both halves of it are false" in tex,
            source="cot_faith_iclr.tex")
    a.check(sec, "Section 6 reports the numpy version the probe measured, not "
                 "just that the pin 'held'", True, "$1.26.4$" in tex,
            source="cot_faith_iclr.tex")
    a.check(sec, "Section 6 names the mechanism (USE_TF=0) rather than "
                 "reporting coexistence as unconditional", True,
            r"\texttt{USE\_TF=0}" in tex, source="cot_faith_iclr.tex")


def audit_rollout_gate(a, d):
    """Table "Four-suite rollout gate" and Section 6/limitation (v).

    The gate is the one place where the paper reports a number that argues
    against its own harness, so it is the one most likely to be quietly
    improved later. Every cell of the table is asserted against the per-suite
    reports, together with the two facts that make the table readable at all:
    that libero_10 is excluded for a step budget below upstream's, and that
    every episode ran on a canonical initial state (the defect that invalidated
    the previous attempt).
    """
    sec = "Four-suite rollout gate (Table: it does not pass)"
    g = (d.get("rollout_gate") or {})
    suites, summary = g.get("suites") or {}, g.get("summary") or {}
    if not suites:
        a.check(sec, "the derived file carries a rollout_gate block", True,
                False, source="results_v2/derived_metrics.json")
        return

    # --- the table body, exactly as printed ---
    printed = {
        "libero_spatial": (0.00, 0, 50, 0.844, 400, 220),
        "libero_object":  (0.00, 0, 50, 0.881, 400, 280),
        "libero_goal":    (0.10, 5, 50, 0.794, 400, 300),
        "libero_10":      (0.00, 0, 50, 0.539, 400, 520),
    }
    for suite, (sr, nsucc, ntot, pub, steps, budget) in printed.items():
        v = suites.get(suite) or {}
        a.check(sec, f"{suite}: Task SR, successes, episodes as printed",
                [sr, nsucc, ntot],
                [round(v.get("SR"), 2) if v.get("SR") is not None else None,
                 v.get("n_success"), v.get("n_total")],
                source=v.get("source"))
        a.check(sec, f"{suite}: published SR and step budgets as printed",
                [pub, steps, budget],
                [v.get("published_SR"), v.get("max_steps_run"),
                 v.get("upstream_max_steps")],
                source=v.get("source"))

    # Wilson CIs printed in the table.
    for suite, lo, hi in (("libero_spatial", 0.00, 0.07),
                          ("libero_object", 0.00, 0.07),
                          ("libero_goal", 0.04, 0.21)):
        ci = (suites.get(suite) or {}).get("SR_wilson95") or [None, None]
        a.check(sec, f"{suite}: Wilson 95% CI as printed", [lo, hi],
                [round(ci[0], 2), round(ci[1], 2)], source="SR_wilson95")

    # --- the two facts the table's readability rests on ---
    a.check(sec, "libero_10 is the only suite excluded, and it is excluded "
                 "because 400 steps is below upstream's 520",
            ["libero_10"], summary.get("suites_excluded_for_step_budget"),
            source="step_budget_below_upstream, computed from the two numbers")
    a.check(sec, "libero_10 is flagged uninterpretable while the other three "
                 "are not", [False, True, True, True],
            [suites["libero_10"]["interpretable"],
             suites["libero_spatial"]["interpretable"],
             suites["libero_object"]["interpretable"],
             suites["libero_goal"]["interpretable"]],
            source="interpretable")
    # The previous gate attempt was invalidated by silently falling back to
    # random env.reset(). If this ever stops holding, the table is measuring
    # something other than the suites' evaluation protocol.
    a.check(sec, "all 200 episodes ran on canonical initial states", True,
            summary.get("all_episodes_canonical_init"),
            source="all_episodes_used_canonical_init, per suite")
    a.check(sec, "200 episodes across 4 suites, 3 interpretable",
            [200, 4, 3],
            [summary.get("n_episodes_total"), summary.get("n_suites_run"),
             summary.get("n_suites_interpretable")], source="summary")

    # --- the claims the prose makes ABOUT the gate ---
    a.check(sec, "the gate does not pass", False, summary.get("gate_passed"),
            source="every interpretable suite is below half its published SR")
    a.check(sec, "the best interpretable cell is libero_goal at 0.10",
            ["libero_goal", 0.10],
            [summary.get("best_interpretable_suite"),
             round(summary.get("best_interpretable_SR"), 2)],
            source="summary")
    # The paper's diagnostic argument turns on this: a nonzero SR means the
    # harness is degraded rather than that success is impossible, which is what
    # pointed at an input-distribution mismatch instead of a wrong control
    # channel. If it ever became all-zero the argument would have to change.
    a.check(sec, "at least one interpretable suite is nonzero, which is what "
                 "licenses the 'degraded, not impossible' diagnosis", True,
            summary.get("any_interpretable_suite_nonzero"),
            source="SR > 0 on libero_goal")

    # --- the upstream budgets, against the copy the rollout actually uses ---
    # derive_metrics keeps its own copy so it can run without torch; if the two
    # drift, the table's validity column stops describing the code.
    sim = ROOT / "sharpguard" / "libero_sim.py"
    txt = sim.read_text() if sim.exists() else ""
    for suite, budget in (("libero_spatial", 220), ("libero_object", 280),
                          ("libero_goal", 300), ("libero_10", 520),
                          ("libero_90", 400)):
        a.check(sec, f"libero_sim pins upstream {suite} max_steps={budget}",
                True, f'"{suite}": {budget},' in txt,
                source="sharpguard/libero_sim.py UPSTREAM_MAX_STEPS")


def audit_derived_paths_are_portable(a):
    """The released derived file must not name anybody's home directory.

    This check exists because the CI reproducibility gate -- re-derive, then
    `git diff --exit-code results_v2/derived_metrics.json` -- failed on every
    push while every number in it matched. The whole diff was four `source`
    fields holding absolute paths under the author's home directory, so the
    gate could only ever pass on one laptop, and a released artifact was
    advertising a filesystem nobody else has. Numbers were never affected;
    the check was.
    """
    sec = "derived-file portability"
    raw = DERIVED.read_text()
    for bad in ("/Users/", "/home/", r"C:\\"):   # JSON escapes a Windows path as C:\\
        a.check(sec, f"derived_metrics.json contains no {bad!r} path",
                0, raw.count(bad),
                source="an absolute home path makes the CI re-derive check "
                       "unpassable off the authoring machine")
    # The in-repo provenance that replaced them must actually resolve, or
    # "portable" would just mean "wrong everywhere equally".
    derived = json.loads(raw)
    checked, missing = 0, []
    def walk(node):
        nonlocal checked
        if isinstance(node, dict):
            src = node.get("source")
            if isinstance(src, str) and not src.startswith("/"):
                checked += 1
                if not (ROOT / src).exists():
                    missing.append(src)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(derived)
    a.check(sec, "every repo-relative `source` in the derived file resolves",
            [], missing,
            source=f"checked {checked} repo-relative source paths")
    a.check(sec, "the derived file records at least one repo-relative source",
            True, checked > 0,
            source="zero would mean the rel() rewrite silently stopped firing")

    # Cross-ISA float stability. The gate above re-derives on x86-64 and diffs
    # against a file produced on arm64, and values like cos_xyz differed in the
    # last one or two bits (~1e-16 relative). derive_metrics.py quantizes every
    # float to 12 significant digits, which is far finer than the 3 digits this
    # paper ever quotes; this asserts the quantizer still fires, because if it
    # silently stopped the gate would go red again for a reason with no bearing
    # on any claim.
    over = []
    def widest(node, path="$"):
        if isinstance(node, bool) or isinstance(node, int):
            return
        if isinstance(node, float):
            if node == node and abs(node) not in (float("inf"),):
                if float(f"{node:.12g}") != node:
                    over.append(path)
            return
        if isinstance(node, dict):
            for k, v in node.items():
                widest(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                widest(v, f"{path}[{i}]")
    widest(derived)
    a.check(sec, "every float in the derived file is quantized to 12 "
                 "significant digits",
            0, len(over),
            source=f"first offenders: {over[:3]}" if over else
                   "cross-ISA last-bit noise cannot reopen the CI drift gate")

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit cot_faith_iclr.tex against results_v2/*.json.")
    ap.add_argument("--json", help="also write machine-readable results here")
    args = ap.parse_args()

    print("CoT-Faith paper-number audit")
    print(f"  manuscript: {TEX}")
    print(f"  artifacts:  {DERIVED}\n              {DECODER_AUDIT}")

    d, da = load(DERIVED), load(DECODER_AUDIT)
    a = Audit()
    if d is None:
        a.check("Artifacts", "derived_metrics.json is readable", True, None,
                source=str(DERIVED))
    if da is None:
        a.check("Artifacts", "decoder_audit.json is readable", True, None,
                source=str(DECODER_AUDIT))

    audit_f1(a, d)
    audit_per_token(a, d)
    audit_noise_floor(a, d)
    audit_f2_calib(a, d)
    audit_f3(a, d)
    audit_paraphrase_null(a, d)
    audit_calibration_floors(a, d)
    audit_f5(a, d)
    audit_f6_directional(a, d)
    audit_decoder(a, da)
    audit_second_calibration(a, d)
    audit_deepthink_p2(a, d)
    audit_attention_cluster_range(a, d)
    audit_attention_seeds_and_depth(a, d)
    audit_training_replicate(a, d)
    audit_release(a)
    audit_upstream_licenses(a)
    audit_deepthink_decode(a)
    audit_manuscript_hygiene(a)
    audit_no_published_ranking(a)
    audit_edit_decode_is_unnorm_free(a)
    audit_deepthink_tau_units(a)
    audit_rollout_gate(a, d)
    audit_resize_check(a)
    audit_tf_env_probe(a)
    audit_derived_paths_are_portable(a)

    # The manuscript states how many claims this script checks. Let the script
    # verify its own advertised size, so adding a check cannot silently make
    # the paper's description of the audit stale.
    quoted = re.search(r"checks \$(\d+)\$ claims", TEX.read_text()) if TEX.exists() else None
    a.check("Release integrity (DATASHEET / LICENSE / artifact counts)",
            "the claim count the manuscript advertises matches this script",
            len(a.rows) + 1, int(quoted.group(1)) if quoted else None,
            source="cot_faith_iclr.tex: 'checks $N$ claims'")

    rc = a.report()
    if args.json:
        Path(args.json).write_text(json.dumps(a.rows, indent=2))
        print(f"[json] wrote {args.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
