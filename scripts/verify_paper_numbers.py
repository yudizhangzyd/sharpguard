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
import hashlib
import json
import math
import re
import subprocess
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


def r2(x: Any) -> Any:
    """Round to 2dp, mapping absent values to None so check() fails them."""
    return None if x is None else round(float(x), 2)


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
    # 3-seed values. These replaced the submission's single-run numbers when
    # the 13-family sweep landed, and the floor row below did not exist at all
    # in the submitted table -- which is why the two-sided row could not be
    # computed and the one-sided ordering went unchallenged.
    expected = {"ours-no-cot": 0.637, "ours-r8": 0.475, "ours-r16": 0.505,
                "ours-r32": 0.487, "ours-r64": 0.580, "ours-data50A": 0.538,
                "ours-data50B": 0.481}
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
    a.check(sec, "the other six one-sided scores span 0.475-0.580",
            [0.475, 0.580],
            [r3(min(v for k, v in norms.items() if k != "ours-no-cot")),
             r3(max(v for k, v in norms.items() if k != "ours-no-cot"))]
            if all(v is not None for v in norms.values()) else None)
    a.check(sec, "the no-CoT ceiling is 3.1x below the mean of the other six "
                 "(the scale confound, not CoT insensitivity)", 3.1,
            _nocot_ceiling_factor(d), tol=0.06)

    # The correction the completed sweep forced: every one-sided value is
    # positive and every two-sided value is negative. If a future run flips a
    # two-sided score positive on a full-CoT row, the paper's headline is wrong
    # and this must fail rather than pass.
    by = dig(d, "calibration_by_model") or {}
    two = {m: (by.get(m) or {}).get("F_bar_two_sided") for m in expected}
    a.check(sec, "the two-sided recomputation is now possible on all 7 rows",
            7, sum(1 for v in two.values() if v is not None))
    a.check(sec, "and it is NEGATIVE on all 7", 7,
            sum(1 for v in two.values() if v is not None and v < 0),
            source=f"two_sided={{{', '.join(f'{k}: {r3(v)}' for k, v in two.items())}}}")
    a.check(sec, "the two-sided values span -0.743 to -0.299",
            [-0.743, -0.299],
            [r3(min(v for v in two.values() if v is not None)),
             r3(max(v for v in two.values() if v is not None))]
            if all(v is not None for v in two.values()) else None)
    a.check(sec, "every one-sided value is POSITIVE while every two-sided "
                 "value is negative -- the floor correction flips the sign, it "
                 "does not merely shrink the number", True,
            all(v > 0 for v in norms.values() if v is not None)
            and all(v < 0 for v in two.values() if v is not None))

    # F2's stated survival condition, and the noise bound on it.
    cot_trained = ("ours-r8", "ours-r16", "ours-r32", "ours-r64",
                   "ours-data50A", "ours-data50B")
    ratios = {m: (by.get(m) or {}).get("cot_specificity_ratio")
              for m in cot_trained + ("ours-no-cot",)}
    a.check(sec, "F2's survival condition holds in direction on 5 of the 6 "
                 "CoT-trained variants", 5,
            sum(1 for m in cot_trained
                if (ratios.get(m) or 0) > 1.0),
            source=f"ratios={{{', '.join(f'{k}: {r3(v)}' for k, v in ratios.items())}}}")
    a.check(sec, "and fails, as F2 requires, on no-CoT", True,
            (ratios.get("ours-no-cot") or 9) < 1.0)
    noise = max((dig(d, "training_replicate", "F_bar_abs_diff_per_pair")
                 or {}).values() or [0])
    # Zero, not two. The retraining floor this is measured against was
    # estimated from two pairs when the manuscript said two of the five
    # survived it; with a replicate on every trained row the worst-case move in
    # F_bar is 0.092 and the widest margin is 0.083, so none of them do. The
    # count is asserted rather than the names, because "which two" was the part
    # that went stale.
    a.check(sec, "none of those 5 clear the control by more than the "
                 "retraining noise, so F2 narrows to a direction-only claim", 0,
            sum(1 for m in cot_trained
                if (ratios.get(m) or 0) > 1.0
                and ((by.get(m) or {}).get("F_bar_diff_vs_instr_random_sub")
                     or 0) > noise),
            source=f"retraining moves F_bar by up to {r3(noise)}; widest margin "
                   f"{r3(max((by.get(m) or {}).get('F_bar_diff_vs_instr_random_sub') or 0 for m in cot_trained))}")
    tex = TEX.read_text()
    a.check(sec, "the manuscript states F2 does not dissolve", True,
            r"\textbf{F2 therefore does not dissolve.}" in tex, source=str(TEX))
    a.check(sec, "the manuscript still forbids reading an ordering off either "
                 "row", True,
            "must not be read as floor-corrected" in tex, source=str(TEX))

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


def _nocot_ceiling_factor(d: Optional[dict]) -> Optional[float]:
    """How far below the other six the no-CoT ceiling sits. This is the scale
    confound the F2 restatement turns on, so it is derived rather than quoted."""
    by = dig(d, "calibration_by_model") or {}
    nc = (by.get("ours-no-cot") or {}).get("ceiling_cross_task_swap")
    others = [(by.get(m) or {}).get("ceiling_cross_task_swap")
              for m in ("ours-r8", "ours-r16", "ours-r32", "ours-r64",
                        "ours-data50A", "ours-data50B")]
    if not nc or any(v is None for v in others):
        return None
    return round(sum(others) / len(others) / nc, 1)


def audit_f3(a: Audit, d: Optional[dict]) -> None:
    sec = "F3 - attention/causation dissociation"
    mags = {m: dig(d, "models", m, "F_bar_mag") for m in ALL8}
    if all(v is not None for v in mags.values()):
        lo, hi = min(mags.values()), max(mags.values())
        a.check(sec, "F_bar range lower bound = 0.166", 0.166, round(lo, 3),
                tol=0.0015, source="derived_metrics.json:models[*].F_bar_mag")
        a.check(sec, "F_bar range upper bound = 0.860", 0.860, round(hi, 3),
                tol=0.0015)
        a.check(sec, "F_bar spread = 5.2x", 5.2, round(hi / lo, 1), tol=0.05)
        # N=612 is what the seven non-control families contribute per seed
        # (4x100 + 2x69 + 74). Conservative: the point estimates are 3-seed
        # means, so this understates the effective sample rather than
        # overstating it, and the intervals are still disjoint.
        lo_ci = wilson_ci(round(lo * 612), 612)
        hi_ci = wilson_ci(round(hi * 612), 612)
        a.check(sec, "Wilson 95% CIs on the two extremes are disjoint", True,
                lo_ci[1] < hi_ci[0],
                source=f"lo={tuple(round(x, 3) for x in lo_ci)} "
                       f"hi={tuple(round(x, 3) for x in hi_ci)}")
        a.check(sec, "the manuscript quotes those two CIs", True,
                r"0.166 \in [0.140, 0.199]$" in TEX.read_text()
                and r"0.860 \in [0.834, 0.888]$" in TEX.read_text(),
                source=str(TEX))
    else:
        a.check(sec, "F_bar spread = 5.2x", 5.2, None)


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
    # The submission measured this floor on 1 of the 8; every row now carries
    # one from its own run, which is what removed the admission-rule gap that
    # limitation (ii) used to disclose. If this ever drops below 8 again, a
    # leaderboard row has lost its floor and the table is unreadable per the
    # paper's own rule.
    a.check(sec, "paraphrase_null measured on all 8 CoT-VLAs (the submission "
                 "had 1; limitation (ii) is now about the DeepThinkVLA family)",
            8, len(measured), source=f"measured={measured}")
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
    """F5's every printed cell, not just its summary bound.

    This audit used to assert two loose things -- n >= 25 per corpus and a
    <= 2.7 pp deviation bound -- and that looseness is exactly how the printed
    LIBERO reference row came to disagree with the 3-seed `ecot-bridge` profile
    the rest of the paper quotes: a max-deviation check cannot notice that the
    row it is measuring against is stale. Every mean, std and per-family F is
    now pinned to the digits in the manuscript.
    """
    sec = "F5 - cross-corpus transfer at N=100"
    cc = dig(d, "cross_corpus_n30")
    if not cc:
        a.check(sec, "cross_corpus_n30 block present in the release", True, None,
                source=str(DERIVED))
        return
    ns = {k: dig(cc, k, "n_samples_used") for k in cc}
    a.check(sec, "all three non-LIBERO corpora ran at N=100, which is what "
                 "replaced the N=30 pilot", [100, 100, 100],
            [ns.get("bridge_v2"), ns.get("fractal"), ns.get("bcz")],
            source=f"n per corpus = {ns}")
    a.check(sec, "and the corpora are the three the paper names, by their "
                 "upstream repo ids",
            ["IPEC-COMMUNITY/bc_z_lerobot",
             "IPEC-COMMUNITY/bridge_orig_lerobot",
             "IPEC-COMMUNITY/fractal20220817_data_lerobot"],
            sorted(v for v in (dig(cc, k, "dataset") for k in cc) if v),
            source="dataset field of each run")

    # Cross-corpus records name the instruction bucket "instr"; the LIBERO
    # reference profile lives in the main attention block as "instruction".
    buckets = {"visual": "visual", "instr": "instruction",
               "cot": "cot", "action_prev": "action_prev"}
    lib = dig(d, "attention", "ecot-bridge", "mass")
    lib_sd = dig(d, "attention", "ecot-bridge", "mass_std")

    # The LIBERO reference row, as printed. It must be the SAME profile the rest
    # of the paper quotes for ecot-bridge -- if this row is ever allowed to come
    # from a different run, the whole cross-corpus comparison is measuring a
    # deviation from a number that appears nowhere else.
    a.check(sec, "LIBERO reference row (visual, instr, cot, prev) as printed",
            [0.290, 0.301, 0.343, 0.065],
            [r3(lib.get(b)) if lib else None
             for b in ("visual", "instruction", "cot", "action_prev")],
            source="attention['ecot-bridge'].mass -- the 3-seed profile "
                   "Section 5 quotes, not a separate run")
    a.check(sec, "LIBERO reference row stds as printed",
            [0.006, 0.009, 0.009, 0.002],
            [r3(lib_sd.get(b)) if lib_sd else None
             for b in ("visual", "instruction", "cot", "action_prev")],
            source="attention['ecot-bridge'].mass_std")

    printed = {
        "bridge_v2": ((0.296, 0.299, 0.338, 0.067),
                      (0.010, 0.012, 0.017, 0.004)),
        "fractal":   ((0.291, 0.302, 0.335, 0.073),
                      (0.008, 0.010, 0.014, 0.004)),
        "bcz":       ((0.297, 0.310, 0.323, 0.070),
                      (0.010, 0.014, 0.022, 0.004)),
    }
    order = ("visual", "instr", "cot", "action_prev")
    for tag, (means, stds) in printed.items():
        a.check(sec, f"{tag}: (visual, instr, cot, prev) means as printed",
                list(means),
                [r3(dig(cc, tag, "mass", b)) for b in order],
                source=dig(cc, tag, "source"))
        a.check(sec, f"{tag}: (visual, instr, cot, prev) stds as printed",
                list(stds),
                [r3(dig(cc, tag, "mass_std", b)) for b in order],
                source=dig(cc, tag, "source"))

    devs = {f"{k}.{cb}": (dig(cc, k, "mass", cb) - lib[lb]) * 100
            for k in cc for cb, lb in buckets.items()
            if lib and dig(cc, k, "mass", cb) is not None and lb in lib}
    worst = max(devs, key=lambda k: abs(devs[k])) if devs else None
    a.check(sec, "max cross-corpus deviation on any attention bucket <= 2.1 pp",
            True, None if not devs else max(abs(v) for v in devs.values()) <= 2.1,
            source=None if not devs
            else f"largest is {worst} at {devs[worst]:+.2f} pp")
    a.check(sec, "and the largest one is BC-Z's CoT bucket, which the prose "
                 "names", "bcz.cot", worst,
            source="every other bucket on every corpus is within 0.9 pp")
    a.check(sec, "every bucket other than BC-Z's CoT is within 0.9 pp", True,
            None if not devs else all(abs(v) <= 0.9 for k, v in devs.items()
                                      if k != "bcz.cot"),
            source="deviation from the LIBERO reference profile, per bucket")

    # Per-family magnitude responses and their N, exactly as printed. The N here
    # is the visibility gate's yield, not the sample count, so it is quoted per
    # cell in the prose and has to be pinned per cell too.
    lib_fams = dig(d, "models", "ecot-bridge", "families") or {}
    a.check(sec, "LIBERO direction_flip / gripper_flip as printed",
            [0.963, 0.697],
            [r3(dig(lib_fams, "direction_flip", "F_mag")),
             r3(dig(lib_fams, "gripper_flip", "F_mag"))],
            source="models['ecot-bridge'].families")
    for tag, dF, dN, gF, gN in (("bridge_v2", 0.913, 92, 0.804, 51),
                                ("fractal",   0.974, 77, 0.770, 74),
                                ("bcz",       0.951, 81, 0.746, 63)):
        e = dig(cc, tag, "edit") or {}
        a.check(sec, f"{tag}: direction_flip F and N as printed", [dF, dN],
                [r3(dig(e, "direction_flip", "faithful_rate")),
                 dig(e, "direction_flip", "n")],
                source=dig(cc, tag, "source"))
        a.check(sec, f"{tag}: gripper_flip F and N as printed", [gF, gN],
                [r3(dig(e, "gripper_flip", "faithful_rate")),
                 dig(e, "gripper_flip", "n")],
                source=dig(cc, tag, "source"))
        # subject_swap yields nothing on any external corpus: these are
        # self-decoded CoTs and the visibility gate admits no sample. Recorded
        # rather than dropped, so the empty cell is a stated fact.
        a.check(sec, f"{tag}: subject_swap yields n=0 (self-decoded CoT, "
                     f"visibility gate admits nothing)", 0,
                dig(e, "subject_swap", "n"), source=dig(cc, tag, "source"))

    # The N=30 pilot is superseded, not deleted, and the paper says the two agree
    # to within 0.3 pp. That is the claim that makes the upgrade meaningful, so
    # it is checked against the retained pilot rather than asserted in prose.
    pilot_dir = ROOT / "results_v2" / "superseded"
    pilot = {}
    for tag in ("bridge_v2", "fractal", "bcz"):
        f = pilot_dir / f"cross_corpus_{tag}_n30.json"
        try:
            pilot[tag] = json.loads(f.read_text())
        except Exception:
            pass
    a.check(sec, "the superseded N=30 pilot is retained for all three corpora",
            3, len(pilot), source=str(pilot_dir))
    if len(pilot) == 3:
        gaps = {}
        for tag, rep in pilot.items():
            ag = rep.get("attention_aggregate") or {}
            for cb in order:
                old_m = (ag.get(f"action->{cb}") or {}).get("mean")
                new_m = dig(cc, tag, "mass", cb)
                if old_m is not None and new_m is not None:
                    gaps[f"{tag}.{cb}"] = abs(new_m - old_m) * 100
        w = max(gaps, key=lambda k: gaps[k]) if gaps else None
        a.check(sec, "every bucket mean reproduces the N=30 pilot to within "
                     "0.3 pp, so the stability was not a small-sample artifact",
                True, None if not gaps else max(gaps.values()) <= 0.3,
                source=None if not gaps
                else f"largest gap is {w} at {gaps[w]:.2f} pp")


def audit_f6_directional(a: Audit, d: Optional[dict]) -> None:
    sec = "F6 - direction-aware scoring inverts the leaderboard"
    # (F_mag, F_dir) on direction_flip, exactly as printed in tab:directional.
    # F_dir values are on the checkpoint's own de-quantization grid, which
    # derive_metrics applies before scoring; see audit_dequant_convention.
    expected = {"ecot-bridge": (0.963, 0.120), "ours-r64": (0.823, 0.779),
                "ours-r16": (0.749, 0.666), "ours-data50A": (0.699, 0.642),
                "ours-r8": (0.696, 0.649), "ours-r32": (0.652, 0.579),
                "ours-data50B": (0.639, 0.542), "ours-no-cot": (0.274, 0.087)}
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
    a.check(sec, "gripper_flip F_dir <= 0.03 for every model (no model inverts "
                 "its gripper on command)", True,
            None if any(g is None for g in grips) else all(g <= 0.03 for g in grips),
            source=f"gripper_flip F_dir = {[r3(g) for g in grips]}")
    a.check(sec, "ECoT-bridge gripper_flip F_dir is exactly 0.000 on all 3 "
                 "seeds (manuscript's 0.037 was the superseded P2 grid)", 0.0,
            score("ecot-bridge", "gripper_flip", "F_dir"),
            source="models['ecot-bridge'].families.gripper_flip.F_dir")
    gcos = [score(m, "gripper_flip", "cos_xyz") for m in expected]
    a.check(sec, "gripper_flip cos(xyz) between +0.93 and +1.00 on every model",
            True, None if any(g is None for g in gcos)
            else all(0.93 <= g <= 1.00 for g in gcos),
            source=f"gripper_flip cos_xyz = {[r3(g) for g in gcos]}")
    # The six full-CoT LoRA/data variants: the paper quotes their cosine range
    # over all records and over the magnitude-faithful subset, and the two
    # ratios by which ECoT-bridge beats them on magnitude and loses on direction.
    lora = [m for m in expected if m.startswith("ours-") and m != "ours-no-cot"]
    lcos = [score(m, "direction_flip", "cos_xyz") for m in lora]
    lsub = [score(m, "direction_flip", "cos_xyz_faithful_subset") for m in lora]
    lmag = [score(m, "direction_flip", "F_mag") for m in lora]
    ldir = [score(m, "direction_flip", "F_dir") for m in lora]
    ebm = score("ecot-bridge", "direction_flip", "F_mag")
    ebd = score("ecot-bridge", "direction_flip", "F_dir")
    ok = all(v is not None for v in lcos + lsub + lmag + ldir + [ebm, ebd])
    a.check(sec, "LoRA/data variants reverse: cos(xyz) in [-0.509, -0.111]",
            [-0.509, -0.111], None if not ok else [r3(min(lcos)), r3(max(lcos))],
            source="models['ours-*'].families.direction_flip.cos_xyz")
    a.check(sec, "on the magnitude-faithful subset they reverse harder: "
                 "cos(xyz) in [-0.889, -0.741]", [-0.889, -0.741],
            None if not ok else [r3(min(lsub)), r3(max(lsub))],
            source="direction_flip.cos_xyz_faithful_subset")
    a.check(sec, "ECoT-bridge beats them 1.2-1.5x on magnitude", [1.2, 1.5],
            None if not ok else [round(ebm / max(lmag), 1),
                                 round(ebm / min(lmag), 1)])
    a.check(sec, "they beat ECoT-bridge 4.5-6.5x on direction", [4.5, 6.5],
            None if not ok or not ebd else [round(min(ldir) / ebd, 1),
                                            round(max(ldir) / ebd, 1)])
    a.check(sec, "no-CoT moves the SAME way too: cos(xyz) = +0.759", 0.759,
            r3(score("ours-no-cot", "direction_flip", "cos_xyz")), tol=0.0015)
    # The two rows the caption explicitly refuses to order: rank 4 and rank 5
    # are closer together than either row's own sampling-seed std.
    da_, r8_ = score("ours-data50A", "direction_flip", "F_mag"), \
        score("ours-r8", "direction_flip", "F_mag")
    s8 = score("ours-r8", "direction_flip", "F_mag_std")
    a.check(sec, "tab:directional ranks 4 and 5 are within seed noise, as the "
                 "caption says (gap < r=8's own std)", True,
            None if None in (da_, r8_, s8) else abs(da_ - r8_) < s8,
            source=f"gap={r3(abs(da_ - r8_)) if None not in (da_, r8_) else None}"
                   f" vs std={r3(s8)}")
    # The paper quotes the inversion's size against the seed noise that could
    # explain it away; this is the check that keeps that comparison honest.
    a.check(sec, "ECoT-bridge's F_dir deficit against r=64 is 0.659, ~15x the "
                 "larger of the two seed stds", 0.659,
            None if not ok else r3(max(ldir) - ebd), tol=0.0015)
    # The paper reports the mean translation cosine as positive for ECoT-bridge
    # (+0.415): it moves the SAME way after the direction is reversed.
    a.check(sec, "ECoT-bridge direction_flip mean cos(xyz) = +0.415", 0.415,
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
    one checkpoint. The 13-family protocol has since run on all nine models in
    the ECoT family, and the answer changes the finding twice over: the
    degeneracy is a property of saturation rather than of the protocol, and the
    out-of-CoT specificity check -- which the submission reported as failing on
    every calibrated model -- passes on 5 of the 9. Both directions are
    asserted here, including the one adverse to the paper's earlier wording.

    The replicate row is keyed 'ours-no-cot-retrain', NOT 'ours-no-cot': since
    the 3-seed sweep landed, 'ours-no-cot' is the leaderboard checkpoint and
    carries different numbers. The manuscript's 0.110/0.220/0.127 belong to the
    retraining, and reading them off the wrong key is exactly the silent
    mis-pointing this file exists to prevent.
    """
    sec = "Second calibrated model (no-CoT replicate) and the saturation contrast"
    by = dig(d, "calibration_by_model") or {}
    nc = by.get("ours-no-cot-retrain")
    if nc is None:
        a.check(sec, "the no-CoT replicate has all 13 families scored", True,
                None,
                source="derived_metrics.calibration_by_model"
                       "['ours-no-cot-retrain']")
        return

    a.check(sec, "13 edit families in the second calibration run",
            13, nc.get("n_families"))
    for fam, val in (("paraphrase_null", 0.11), ("bbox_jitter_null", 0.05),
                     ("instr_random_sub", 0.19), ("cross_task_swap", 0.22)):
        a.check(sec, f"no-CoT replicate {fam} = {val}", val,
                r3(dig(nc, "families", fam, "F_mag")), tol=0.0015)
    a.check(sec, "no-CoT replicate F_bar over non-control families = 0.124",
            0.124, r3(nc.get("F_bar_non_control")), tol=0.0015)
    a.check(sec, "no-CoT replicate F_bar sits only +0.014 above its own floor",
            0.014, r3(nc.get("F_bar_diff_vs_paraphrase_null")), tol=0.0015)

    # The contrast is the finding: dynamic range 0.110 vs 0.010.
    a.check(sec, "no-CoT replicate dynamic range (ceiling - floor) = 0.110",
            0.110, r3(nc.get("dynamic_range")), tol=0.0015)
    a.check(sec, "no-CoT replicate calibration is NOT degenerate",
            False, nc.get("calibration_is_degenerate"))
    ec = by.get("ecot-bridge") or {}
    a.check(sec, "ECoT-bridge calibration IS degenerate",
            True, ec.get("calibration_is_degenerate"))
    dr_nc, dr_ec = nc.get("dynamic_range"), ec.get("dynamic_range")
    a.check(sec, "the second model's dynamic range is 11x the first's", 11.0,
            round(dr_nc / dr_ec, 1) if (dr_nc and dr_ec) else None, tol=0.15,
            source="paper: 'an 11x larger dynamic range'")

    # Two-sided normalization exists only where the range is real.
    a.check(sec, "two-sided (F_bar - floor)/(ceiling - floor) = 0.127 on the "
                 "no-CoT replicate", 0.127, r3(nc.get("F_bar_two_sided")),
            tol=0.0015)
    a.check(sec, "two-sided statistic is UNDEFINED on the saturated model",
            True, ec.get("F_bar_two_sided") is None,
            source="a 0.010 denominator must not be divided by")
    a.check(sec, "no-CoT replicate CoT-specificity ratio = 0.653",
            0.653, r3(nc.get("cot_specificity_ratio")), tol=0.0015)
    a.check(sec, "exactly 1 CoT family on the no-CoT replicate exceeds the "
                 "out-of-CoT control (direction_flip, CIs overlap so not "
                 "claimed)", 1, nc.get("n_families_above_out_of_cot_control"))
    a.check(sec, "n_degenerate = 1 in the 2-model saturation contrast", 1,
            dig(d, "calibration_contrast", "n_degenerate"))


# Row order of tab:calibration. Each tuple is the published cell values:
# (label, floor, bbox, instr, ceiling, range, two_sided, ratio, n_above).
CALIB_TABLE = (
    ("ours-r8",             0.587, 0.077, 0.333, 0.857, 0.270, -0.665, 1.222, 3),
    ("ours-r16",            0.547, 0.083, 0.437, 0.883, 0.337, -0.299, 1.022, 3),
    ("ours-r32",            0.567, 0.073, 0.370, 0.810, 0.243, -0.706, 1.067, 3),
    ("ours-r64",            0.657, 0.087, 0.500, 0.887, 0.230, -0.621, 1.028, 3),
    ("ours-no-cot",         0.193, 0.047, 0.263, 0.260, 0.067, -0.416, 0.629, 1),
    ("ours-no-cot-retrain", 0.110, 0.050, 0.190, 0.220, 0.110,  0.127, 0.653, 1),
    ("ours-data50A",        0.537, 0.067, 0.433, 0.730, 0.193, -0.743, 0.907, 3),
    ("ours-data50B",        0.447, 0.053, 0.270, 0.733, 0.287, -0.327, 1.307, 4),
    ("ecot-bridge",         0.960, 0.460, 0.990, 0.970, 0.010,   None, 0.878, 0),
    # The second architecture family. bbox_jitter_null is None here and that is
    # a MEASUREMENT, not a gap: DeepThinkVLA's CoT renderer emits no bboxes, so
    # the edit produces a byte-identical CoT and the harness refuses to score
    # it. The committed artifacts previously carried it as n=100, F=0.000 with
    # every delta exactly 0.0 -- a vacuous identity edit printed as a
    # robustness result -- which is the defect these three rows correct.
    ("DT-base",             0.960,  None, 0.950, 0.980, 0.020,   None, 0.996, 5),
    ("DT-SFT",              0.810,  None, 0.870, 0.960, 0.150, -0.728, 0.806, 2),
    ("DT-RL",               0.820,  None, 0.890, 0.940, 0.120, -0.796, 0.814, 1),
)
# The two architecture families, so the counted claims below can be stated
# per-family as well as pooled.
CALIB_DT = ("DT-base", "DT-SFT", "DT-RL")
# Rows that are neither the saturated public checkpoint nor a no-CoT variant:
# the models where a floor-corrected statistic is supposed to be readable.
CALIB_FULL_COT = ("ours-r8", "ours-r16", "ours-r32", "ours-r64",
                  "ours-data50A", "ours-data50B", "DT-SFT", "DT-RL")


def audit_calibration_nine_models(a: Audit, d: Optional[dict]) -> None:
    """tab:calibration, cell by cell, plus the counted claims the section is
    built on. Two of those counts are adverse to the submission's wording
    (the specificity ratio exceeds 1 on 5 of 12, contradicting 'neither
    calibrated model passes'), so this audit has to fail if anyone quietly
    restores the stronger claim.

    The set is 12 calibration entries over 11 distinct checkpoints in 2
    architecture families -- `ours-no-cot` appears twice by design, once as the
    checkpoint every other leaderboard row uses and once as the independent
    retraining. The three DeepThinkVLA rows were the last coverage gap: until
    they existed, cot_specificity_ratio was an ECoT-only statistic and the
    paper said so.
    """
    sec = "Two-sided calibration on 11 models, 2 architecture families (tab:calibration)"
    by = dig(d, "calibration_by_model") or {}
    su = dig(d, "calibration_summary") or {}
    if not by:
        a.check(sec, "the 13-family sweep has run", True, None,
                source="derived_metrics.calibration_by_model")
        return

    a.check(sec, "12 calibration entries over 11 distinct checkpoints",
            12, len(by), source=f"labels={sorted(by)}")
    a.check(sec, "they span 2 architecture families (ECoT + DeepThinkVLA)",
            2, su.get("n_architecture_families"))
    a.check(sec, "every row's set of labels matches the audit table",
            sorted(r[0] for r in CALIB_TABLE), sorted(by))
    for label, floor, bbox, instr, ceil, rng, two, ratio, n_ab in CALIB_TABLE:
        e = by.get(label)
        if e is None:
            a.check(sec, f"[{label}] is calibrated", True, None,
                    source=f"calibration_by_model['{label}'] missing")
            continue
        for name, want, got in (
                ("paraphrase_null floor", floor, e.get("paraphrase_null")),
                ("instr_random_sub",      instr, e.get("instr_random_sub")),
                ("ceiling",  ceil, e.get("ceiling_cross_task_swap")),
                ("range",    rng,  e.get("dynamic_range")),
                ("ratio",    ratio, e.get("cot_specificity_ratio"))):
            a.check(sec, f"[{label}] {name} = {want}", want, r3(got),
                    tol=0.0015)
        if bbox is None:
            # Asserting the ABSENCE. A number in this cell would mean the
            # vacuous identity edit came back: F=0.000 over n=100 with every
            # delta exactly 0.0, indistinguishable from selfsplice_control and
            # reportable as "the model ignores a meaning-preserving numeric
            # perturbation" when in fact no perturbation reached the model.
            a.check(sec, f"[{label}] bbox_jitter_null is inapplicable by "
                         f"construction, not scored as a zero",
                    ["bbox_jitter_null"], e.get("families_inapplicable"))
            a.check(sec, f"[{label}] and all 100 of its samples are recorded "
                         f"as skipped", 100,
                    (e.get("families_inapplicable_n_skipped") or {}).get(
                        "bbox_jitter_null"),
                    source="edit not represented in the rendered CoT")
        else:
            a.check(sec, f"[{label}] bbox_jitter_null = {bbox}", bbox,
                    r3(e.get("bbox_jitter_null")), tol=0.0015)
        if two is None:
            # ECoT-bridge: dynamic range 0.010 makes the two-sided statistic
            # undefined, and the table prints "degen." rather than a number.
            # Asserting the ABSENCE is the claim here -- a number appearing in
            # this cell would mean derive_metrics started dividing by a range
            # the paper calls degenerate.
            a.check(sec, f"[{label}] two-sided is undefined (degenerate range), "
                         f"and the table prints no number for it", True,
                    e.get("F_bar_two_sided") is None,
                    source=f"calibration_by_model['{label}'].F_bar_two_sided="
                           f"{e.get('F_bar_two_sided')}")
        else:
            a.check(sec, f"[{label}] two-sided = {two}", two,
                    r3(e.get("F_bar_two_sided")), tol=0.0015)
        a.check(sec, f"[{label}] {n_ab} of 7 families reach the out-of-CoT "
                     f"control", n_ab,
                e.get("n_families_above_out_of_cot_control"))
        a.check(sec, f"[{label}] every quantity comes from ONE run of that "
                     f"checkpoint", True,
                bool(e.get("source"))
                and e.get("n_families") == (12 if label in CALIB_DT else 13),
                source=str(e.get("source")))

    # The seven leaderboard rows must carry 3 sampling seeds; the other two
    # rows are single-run and the paper says so.
    for label in ("ours-r8", "ours-r16", "ours-r32", "ours-r64",
                  "ours-no-cot", "ours-data50A", "ours-data50B"):
        a.check(sec, f"[{label}] is a 3-sampling-seed mean", [0, 1, 2],
                (by.get(label) or {}).get("seeds"))
    for label in ("ecot-bridge", "ours-no-cot-retrain"):
        a.check(sec, f"[{label}] is single-seed and the paper marks it",
                1, (by.get(label) or {}).get("n_runs"))
    for label in CALIB_DT:
        a.check(sec, f"[{label}] is a single n=100 run at seed 0", 1,
                (by.get(label) or {}).get("n_runs"))
        a.check(sec, f"[{label}] is labelled as the second architecture family",
                "deepthinkvla", (by.get(label) or {}).get("architecture_family"))

    # The counted claims of the section.
    vals = [by[k] for k in by]
    below = [v for v in vals if (v.get("F_bar_diff_vs_paraphrase_null") or 0) < 0]
    neg = [v for v in vals if (v.get("F_bar_two_sided") or 0) < 0]
    gt1 = [v for v in vals if (v.get("cot_specificity_ratio") or 0) > 1]
    degen = [v for v in vals if v.get("calibration_is_degenerate")]
    a.check(sec, "F_bar is BELOW its own paraphrase floor on 11 of 12 entries",
            11, len(below), source="the paper's central negative result")
    a.check(sec, "the two-sided statistic is negative on 9 of the 10 entries "
                 "where it is defined at all", 9, len(neg))
    a.check(sec, "the two-sided statistic is negative on ALL EIGHT "
                 "non-degenerate full-CoT variants, in BOTH architecture "
                 "families", 8,
            sum(1 for k in CALIB_FULL_COT
                if (by.get(k, {}).get("F_bar_two_sided") or 0) < 0),
            source="6 of these were ECoT-only at the previous revision; the "
                   "DeepThinkVLA pair is what makes it a cross-architecture "
                   "statement rather than a property of one CoT format")
    a.check(sec, "the CoT-specificity ratio exceeds 1 on 5 of 12 entries -- "
                 "which CONTRADICTS the submission's 'neither calibrated "
                 "model passes' and must be reported as a correction",
            5, len(gt1), source=f"passing={sorted(v['label'] for v in gt1)}")
    a.check(sec, "and it is below 1 on ALL THREE DeepThinkVLA checkpoints, so "
                 "the out-of-CoT control is not cleared anywhere in the second "
                 "architecture family", 3,
            sum(1 for k in CALIB_DT
                if (by.get(k, {}).get("cot_specificity_ratio") or 9) < 1.0))
    a.check(sec, "exactly 2 of the 12 calibrations are degenerate", 2,
            len(degen), source=f"degenerate={[v['label'] for v in degen]}")
    a.check(sec, "the degenerate ones are the two saturated checkpoints -- one "
                 "in each architecture family -- and no full-CoT trained "
                 "variant", ["DT-base", "ecot-bridge"],
            sorted(v["label"] for v in degen),
            source="this is what localizes the collapse to the saturated "
                   "regime rather than to the protocol")
    nd = [v["dynamic_range"] for v in vals
          if not v.get("calibration_is_degenerate")]
    a.check(sec, "non-degenerate dynamic ranges span 0.067 to 0.337",
            [0.067, 0.337], [r3(min(nd)), r3(max(nd))] if nd else None)
    # bbox_jitter_null is defined only on the ECoT family; the DeepThinkVLA
    # rows are excluded here by construction, not by choice, and the audit
    # above asserts that exclusion rather than letting this range absorb it.
    bb = [v["bbox_jitter_null"] for k, v in by.items()
          if k != "ecot-bridge" and k not in CALIB_DT]
    a.check(sec, "the numeric null is 0.047-0.087 on the 8 trained ECoT "
                 "variants", [0.047, 0.087],
            [r3(min(bb)), r3(max(bb))] if bb else None)
    a.check(sec, "bbox_jitter_null is inapplicable on exactly the 3 "
                 "DeepThinkVLA rows and on no ECoT row",
            sorted(CALIB_DT),
            sorted((su.get("families_inapplicable_by_model") or {})))

    # NONE of the 5 passing models clears the control by more than the amount
    # retraining alone moves F_bar. When that bound came from two replicate
    # pairs it was 0.030 and two models cleared it; with a replicate on every
    # trained row the worst case is 0.092, which is above the widest margin
    # (0.083). The count is asserted rather than the surviving names, because
    # "which ones" is precisely the part that went stale.
    noise = max((dig(d, "training_replicate", "F_bar_abs_diff_per_pair")
                 or {}).values() or [0])
    robust = sorted(v["label"] for v in gt1
                    if (v.get("F_bar_diff_vs_instr_random_sub") or 0) > noise)
    widest = max((v.get("F_bar_diff_vs_instr_random_sub") or 0) for v in gt1)
    a.check(sec, "no model clears the out-of-CoT control by more than "
                 "same-config retraining noise, so F2 is direction-only", [],
            robust,
            source="retraining moves F_bar by up to %s; widest margin is %s"
                   % (r3(noise), r3(widest)))

    tex = TEX.read_text()
    for frag in (r"\label{tab:calibration}",
                 r"$\mathbf{11}$ of $\mathbf{12}$",
                 r"the ratio exceeds $1$ on $\mathbf{5}$ of the $12$ entries"):
        a.check(sec, f"the manuscript states it ({frag!r})", True, frag in tex,
                source=str(TEX))


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
        ("DT-base", 0.947, 0.960, 0.980, 0.020, -0.013),
        ("DT-SFT",  0.701, 0.810, 0.960, 0.150, -0.109),
        ("DT-RL",   0.724, 0.820, 0.940, 0.120, -0.095),
    ):
        m = dt.get(label) or {}
        # 12, not 13: bbox_jitter_null is inapplicable by construction on this
        # architecture (its CoT renderer emits no bboxes), and the harness
        # records it as n=0/n_skipped=100 rather than scoring an identity edit.
        a.check(sec, f"{label}: 12 families scored", 12, m.get("n_families_scored"))
        a.check(sec, f"{label}: the out-of-CoT control IS measured here, which "
                     f"is what puts this family in tab:calibration at all",
                True, "instr_random_sub" in (m.get("families") or {}))
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

    # One null, two names in the release: the scoring pipeline writes
    # selfsplice_control and the judge / edit-pair exports write
    # identity_control. Both names are therefore correct where they appear, and
    # the ARR body prints both -- selfsplice_control in S3 and S8, and
    # identity_control on fig:taxonomy, which reads the judge artifact. A
    # reader who meets the second name with no warning has to guess whether it
    # is a fourth null, so the alias has to be stated where the null is
    # introduced. This asserts the collision is real in the artifacts before
    # requiring the disclosure, so if the two ever unify the check retires
    # itself rather than demanding a note about a name that no longer exists.
    root = Path(__file__).resolve().parent.parent
    jr = load(root / "results_v2" / "canonical_runs" / "judge_edit_families"
              / "judge_report.json")
    dm_fams = set()
    for mv in (dig(d, "models") or {}).values():
        dm_fams |= set((mv or {}).get("families") or {})
    judge_fams = set(dig(jr, "per_family") or {})
    collides = ("selfsplice_control" in dm_fams
                and "identity_control" in judge_fams)
    a.check(sec, "the identity null really does carry two names across the "
                 "release, which is what makes the alias note necessary",
            [True, True],
            ["selfsplice_control" in dm_fams, "identity_control" in judge_fams],
            source="derived_metrics.json families vs judge_report.json "
                   "per_family")
    if collides:
        arr = root / "cot_faith_arr.tex"
        t = arr.read_text() if arr.exists() else ""
        a.check(sec, "and the ARR body says the two names are the same null, "
                     "rather than printing both and leaving it to be inferred",
                True,
                "the judge export and Figure~\\ref{fig:taxonomy} name it "
                "\\emph{identity\\_control}" in t,
                source="cot_faith_arr.tex S3, edit-families paragraph")


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
    not reseeded sampling. Seven pairs exist -- one for every trained row --
    and they do not license any ordering."""
    sec = "Same-config training-run replicates (leaderboard error bar)"
    tr = dig(d, "training_replicate") or {}
    if not tr:
        a.check(sec, "at least one same-config retraining pair exists", True, None,
                source="derived_metrics.training_replicate")
        return

    a.check(sec, "7 independent same-config retraining pairs, one per trained "
                 "row", 7, tr.get("n_pairs"))
    per = sorted(round(v, 2) for v in (tr.get("cot_abs_diff_pp_per_pair") or []))
    a.check(sec, "|delta alpha(cot)| across retrainings spans 0.12-1.95 pp",
            [0.12, 0.14, 0.56, 0.56, 0.73, 1.45, 1.95], per)
    a.check(sec, "largest same-config training difference on any bucket = 1.95 pp",
            1.95, round(tr.get("any_bucket_abs_diff_pp_max"), 2), tol=0.006)

    # Keyed by label, not by position. This loop used to be
    # `for pr in pairs: if pr["F_per_family"]: fp = pr["F_per_family"]`, i.e.
    # "the last pair that happens to carry an F block" -- so the moment the
    # r=32 pair acquired one, all four assertions below silently re-pointed at
    # a different checkpoint and kept passing against the wrong numbers.
    by_label = tr.get("by_label") or {}

    def fpf(label):
        return dig(by_label.get(label), "F_per_family")

    for label, n_fam, mean_d, max_d, worst in (
            ("ours-no-cot",   9, 0.024, 0.083, "adversarial_plausible"),
            ("ours-r32",      9, 0.040, 0.180, "verb_swap"),
            ("ours-r8",       9, 0.072, 0.260, "verb_swap"),
            ("ours-r16",      9, 0.026, 0.060, "direction_flip"),
            ("ours-r64",      9, 0.036, 0.067, "adversarial_plausible"),
            ("ours-data50A",  9, 0.064, 0.130, "cross_task_swap"),
            ("ours-data50B",  9, 0.079, 0.170, "verb_swap")):
        fp = fpf(label)
        a.check(sec, "[%s] F is compared across retrainings on %d families "
                     "at N>=50" % (label, n_fam), n_fam,
                dig(fp, "n_families_compared"))
        a.check(sec, "[%s] mean |delta F| across retrainings = %.3f"
                % (label, mean_d), mean_d, r3(dig(fp, "mean_abs_diff")),
                tol=0.0015)
        a.check(sec, "[%s] max |delta F| across retrainings = %.3f"
                % (label, max_d), max_d, r3(dig(fp, "max_abs_diff")),
                tol=0.0015)
        a.check(sec, "[%s] the worst-reproducing family is %s"
                % (label, worst), worst, dig(fp, "max_abs_diff_family"))
        a.check(sec, "[%s] location_swap is excluded from the replicate spread "
                     "(n=12 pre-fix run measures the annotation fix, not "
                     "training)" % label,
                True, "location_swap" in (dig(fp, "excluded_low_n") or []))

    a.check(sec, "the worst single-family retraining move over all seven pairs "
                 "is 0.260 on ours-r8:verb_swap", "ours-r8:verb_swap",
            tr.get("F_max_abs_diff_where"))
    a.check(sec, "manuscript limitation (viii) quotes that 0.260", True,
            "$\\mathbf{0.260}$" in TEX.read_text(), source=str(TEX))
    # verb_swap is the worst-reproducing family on three of the seven pairs.
    # The manuscript says so in order to rule out "one anomalous cell", which
    # is exactly the reading a single 0.260 invites.
    worst_fams = [dig(v, "F_per_family", "max_abs_diff_family")
                  for v in (tr.get("by_label") or {}).values()]
    a.check(sec, "verb_swap is the worst-reproducing family on 3 of the 7 "
                 "pairs (so 0.260 is not one anomalous cell)",
            3, worst_fams.count("verb_swap"))

    # F_bar itself across retraining: the unit every CoT-specificity margin in
    # Section f2_calib is measured against, so it has to be asserted, not
    # eyeballed off the per-family table.
    fb = tr.get("F_bar_abs_diff_per_pair") or {}
    for label, mv in (("ours-no-cot", 0.030), ("ours-r32", 0.021),
                      ("ours-r8", 0.006), ("ours-r16", 0.018),
                      ("ours-r64", 0.045), ("ours-data50A", 0.061),
                      ("ours-data50B", 0.092)):
        a.check(sec, "F_bar moves %.3f when the %s config is retrained"
                % (mv, label), mv, r3(fb.get(label)), tol=0.0015)
    a.check(sec, "the manuscript quotes the F_bar retraining move as "
                 "0.006--0.092", True,
            "$0.006$--$\\mathbf{0.092}$" in TEX.read_text(), source=str(TEX))

    # The hierarchy, which is the actual claim.
    h = dig(d, "noise_hierarchy") or {}
    a.check(sec, "sampling noise (0.26 pp) is far below training-run noise "
                 "(1.95 pp)", True,
            (h.get("sampling_std_pp") or 9) < 0.2 * (h.get("training_run_diff_pp") or 0))
    a.check(sec, "within-ECoT spread is only 1.2x the training-run difference",
            1.2, round(h.get("spread_over_training_run_cot"), 1)
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
    want_scored = tex_int(r"\$([\d{},]+)\$ carry a scored delta")
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
    # Both halves of the P3 story ship: the 200 withdrawn records are retained
    # so the withdrawal is checkable, and the 153 in-domain re-run records are
    # what replaced them. Asserting the SUM would let one file vanish while the
    # other grew, so both are named.
    p3_files = {}
    for f in sorted(can.glob("*.json")):
        rec = load(f)
        if not isinstance(rec, dict):
            continue
        ps = rec.get("per_sample")
        if isinstance(ps, list) and ps and ("aurocs" in rec
                                            or "median_error_l1" in rec):
            p3_files[f.name] = len(ps)
    a.check(sec, "200 withdrawn-P3 records retained so the withdrawal is "
                 "checkable", 200, p3_files.get("auroc_ecot_bridge_n200.json"),
            source="results_v2/canonical_runs/auroc_ecot_bridge_n200.json")
    a.check(sec, "153 in-domain P3 records released as the replacement (F8)",
            153, p3_files.get("auroc_ecot_bridge_indomain_n153.json"),
            source="results_v2/canonical_runs/"
                   "auroc_ecot_bridge_indomain_n153.json")
    a.check(sec, "and nothing else claims to be a P3 run", 353, n["p3"][0],
            source=f"P3-schema files: {p3_files}")

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

    # --- what the records CONTAIN, not just how many there are ---
    #
    # This block exists because of a specific failure: the release paragraph
    # claimed the edited CoT text was included on every record, this script
    # reported 869/869 claims reproduced, and the claim was false on every
    # record in the release. Counting records cannot catch that. The paragraph
    # describes the schema, so the schema is what has to be asserted.
    #
    # Each entry is (field, claimed-present, why-it-matters).
    fields = {}
    pairless = {}
    for f in sorted(can.glob("*.json")):
        rec = load(f)
        if not isinstance(rec, dict):
            continue
        ps = rec.get("per_sample_edit") or rec.get("per_sample")
        if not (isinstance(ps, list) and ps):
            continue
        keys = set().union(*(r.keys() for r in ps if isinstance(r, dict)))
        if not ({"delta_linf", "a_edit"} & keys):
            continue
        scored = [r for r in ps if isinstance(r, dict) and not r.get("skipped")]
        if not scored:
            continue
        n_pairless = sum(1 for r in scored if "a_orig" not in r)
        if n_pairless:
            pairless[f.name] = n_pairless
        for k in ("a_orig", "a_edit", "delta_linf", "edit_meta", "instruction",
                  "file_base", "cot_edited", "cot_text"):
            have, tot = fields.setdefault(k, [0, 0])
            fields[k] = [have + sum(1 for r in scored if k in r),
                         tot + len(scored)]

    # delta_linf is the only field EVERY scored record must have: it is the
    # quantity every F in the paper is computed from.
    have, tot = fields.get("delta_linf", [0, 0])
    a.check(sec, "every scored edit record carries 'delta_linf', which is the "
                 "quantity every F in the paper is computed from", tot, have,
            source="schema check over all scored edit records")

    # The action pair and edit metadata are present everywhere EXCEPT the three
    # cross-corpus runs, whose earlier harness stored deltas only. That is a
    # real reproducibility limit, so it is pinned to those exact three files
    # rather than absorbed into a tolerance: if a fourth file starts dropping
    # the pair, this fails.
    a.check(sec, "the only scored records without an action pair are the three "
                 "cross-corpus runs, which an earlier harness wrote "
                 "delta-only", {"cross_corpus_bcz_n100.json": 144,
                                "cross_corpus_bridge_v2_n100.json": 143,
                                "cross_corpus_fractal_n100.json": 151},
            pairless, source="schema check over all scored edit records")
    for k in ("a_orig", "a_edit", "edit_meta"):
        have, tot = fields.get(k, [0, 0])
        a.check(sec, f"every OTHER scored edit record carries '{k}', as the "
                     f"release paragraph claims", tot - 438, have,
                source="schema check over all scored edit records")
    want_pair = tex_int(r"\$([\d{},]+)\$ of those carry the full action pair")
    a.check(sec, "the full-action-pair count the manuscript quotes is the "
                 "number released", want_pair, fields.get("a_orig", [0, 0])[0],
            source="cot_faith_iclr.tex, 'Public release' paragraph")

    # The negative half, and the one that actually caught the bug. The paper
    # must NOT claim to release the edited CoT text, because it does not: the
    # records carry the metadata that regenerates it and nothing more. If a
    # future run starts shipping the text, this flips and the sentence has to
    # be rewritten -- which is the coupling we want in both directions.
    for k in ("cot_edited", "cot_text"):
        have, _ = fields.get(k, [0, 0])
        a.check(sec, f"no scored edit record carries '{k}' (the manuscript "
                     f"must not claim the edited trace text is released)",
                0, have, source="schema check over all scored edit records")
    a.check(sec, "the release paragraph says the edited CoT text is NOT "
                 "included, and says so as a correction rather than silently",
            True, "not the edited text itself" in tex
            and "an earlier version of this sentence" in tex,
            source="cot_faith_iclr.tex, 'Public release' paragraph")


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
        # Was a bare "0.853" guard. It had to be narrowed once the rollout
        # gate's libero_10 row began quoting 0.853 as its fraction of published
        # SR: a bare-substring guard on a three-digit number cannot tell the two
        # apart, and the version that could not would have blocked a real
        # measurement. The stale value only ever appeared as an F_bar range.
        r"$0.853$ across": "stale F_bar upper bound (correct value 0.860)",
        r"to $0.853$ on our": "stale F_bar upper bound (correct value 0.860)",
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


def audit_normstats_probe(a: Audit) -> None:
    """The manuscript's reason for not rolling out the public CoT checkpoint is
    a measured precondition failure, not a judgement: the checkpoint ships
    norm_stats for 'bridge_orig' only. Everything the paragraph says is read
    back out of the probe artifact, including the parts that are FAVOURABLE to
    the checkpoint -- a probe cited only for the half that supports the decision
    not to run is the same defect as a caveat replacing a measurement."""
    sec = "Norm-stats provenance probe (why ECoT-bridge cannot be rolled out)"
    d = ROOT / "results_v2" / "canonical_runs" / "rollout_probe_ecot_bridge"
    p = load(d / "rollout_edit_probe.json")
    if not isinstance(p, dict):
        a.check(sec, "the norm-stats probe artifact is released", True, None,
                source=str(d / "rollout_edit_probe.json"))
        return

    a.check(sec, "the probe is attributable to its own bolt task id",
            "phenc9ygb4", (d / "bolt_task_id.txt").read_text().strip()
            if (d / "bolt_task_id.txt").exists() else None,
            source=str(d / "bolt_task_id.txt"))
    a.check(sec, "the probed checkpoint is the public CoT one",
            "Embodied-CoT/ecot-openvla-7b-bridge", p.get("ckpt"))
    a.check(sec, "it ships norm_stats for 'bridge_orig' and nothing else",
            ["bridge_orig"], p.get("norm_stats_keys"))
    a.check(sec, "the LIBERO key the suite needs is absent", False,
            p.get("unnorm_key_present"),
            source="unnorm_key_requested=%r" % p.get("unnorm_key_requested"))
    a.check(sec, "so the scale precondition fails rather than degrading", False,
            p.get("norm_stats_usable"))
    a.check(sec, "and every arm fails at decode, control included --- which is "
                 "why 5 zero arms would not be a null result",
            5, sum(1 for v in (p.get("one_frame_actions") or {}).values()
                   if isinstance(v, dict) and "error" in v))

    # The independent half: the intervention side of the protocol DOES transfer.
    # This is what re-points the rollout at our own fine-tune instead of
    # abandoning limitation (v), so it is asserted, not narrated.
    a.check(sec, "the checkpoint nonetheless emits a structured CoT online on "
                 "LIBERO frames", True, bool(p.get("cot_structured")))
    a.check(sec, "with 8 parsed reasoning tags", 8,
            len(p.get("cot_tags_parsed") or []))
    a.check(sec, "3 of the 4 probed families change the rendered CoT", 3,
            sum(1 for v in (p.get("families") or {}).values()
                if v == "changes the rendered CoT"))
    a.check(sec, "and subject_swap is reported inapplicable rather than scored "
                 "as a no-effect edit", True,
            "not applicable" in str(dig(p, "families", "subject_swap")))

    # The gate that reads it. A probe nobody reads is how the two earlier
    # rollout defects survived, so the reading is checked in the shell source.
    sh = ROOT / "bolt" / "run_cotfaith_rollout_edit_s3.sh"
    src = sh.read_text() if sh.exists() else ""
    a.check(sec, "the rollout job reads its own probe and refuses to launch "
                 "past a failed precondition", True,
            'startswith("ok")' in src and "exit 5" in src, source=str(sh))

    try:
        tex = TEX.read_text()
    except Exception:
        return
    for frag in (r"\texttt{bridge\_orig}, and no LIBERO key at all",
                 "Five identically-zero arms are not a null result",
                 r"rollout\_probe\_ecot\_bridge"):
        a.check(sec, f"the manuscript states the probe result ({frag!r})", True,
                frag in tex, source=str(TEX))


def audit_cited_environment(a: Audit) -> None:
    """The environment limitation (ix) names must be the one we actually install.

    Added because it was not. The manuscript said OpenVLA-OFT "failed to load
    cleanly in our environment (Python 3.10, torch 2.2.0)" while
    `bolt/setup-openvla.sh` pinned torch 2.4.1 and transformers 4.40.1 -- so the
    one claim in the paper with no released artifact behind it was also
    describing an environment this release no longer runs. That is the worst
    combination available: unfalsifiable and stale, excusing a coverage gap.

    A version string in prose has no artifact to check it against, so this check
    reads the pins out of the setup script the jobs actually run. It is the
    cheapest possible guard against the general failure -- an environment claim
    aging out of truth silently -- and it belongs in the audit rather than in a
    reviewer's memory.
    """
    sec = "Cited environment matches the installed one"
    sh = ROOT / "bolt" / "setup-openvla.sh"
    if not sh.exists():
        a.check(sec, "the setup script the cited environment refers to exists",
                True, False, source=str(sh))
        return
    src = sh.read_text()
    try:
        tex = TEX.read_text()
    except Exception:
        return
    cited = re.search(r"failed to load cleanly in our environment \(([^)]*)\)",
                      tex)
    cited_txt = cited.group(1) if cited else ""
    for pkg in ("torch", "transformers"):
        m = re.search(rf'"{pkg}==([0-9][^"]*)"', src)
        ver = m.group(1) if m else None
        a.check(sec, f"the {pkg} version the manuscript cites is the one "
                     f"setup-openvla.sh pins", True,
                bool(ver) and f"{pkg} {ver}" in cited_txt,
                source=f"{sh} pins {pkg}=={ver}; manuscript says "
                       f"'{cited_txt[:90]}'")
    a.check(sec, "the manuscript names the setup script, so the pin it cites is "
                 "checkable rather than recalled", True,
            r"\texttt{bolt/setup-openvla.sh}" in tex, source=str(TEX))


def audit_deepthink_provenance(a: Audit) -> None:
    """Every released number should be traceable to the job that produced it.

    The three DeepThinkVLA rows were the only released runs whose bolt task id
    lived nowhere in the repository -- they ship as flat
    `deepthink_*_13family.json` files rather than in a directory carrying a
    `bolt_task_id.txt`, so the id survived only in a scratch copy of the
    downloaded artifact and would have been lost the moment that scratch
    directory was cleaned up.

    The recorded sha256 is re-computed here rather than trusted. A provenance
    file that records a hash nobody re-checks documents the artifact that existed
    when it was written, not the one in the repository now; recomputing turns it
    into a tamper-evident seal on three of the eight leaderboard rows.
    """
    sec = "DeepThinkVLA provenance (bolt task ids, hash-sealed)"
    can = ROOT / "results_v2" / "canonical_runs"
    p = can / "deepthink_provenance.json"
    prov = load(p)
    if not prov:
        a.check(sec, "deepthink_provenance.json exists so the three "
                     "DeepThinkVLA rows are traceable to their bolt jobs",
                True, False, source=str(p))
        return
    runs = prov.get("runs") or {}
    a.check(sec, "all three DeepThinkVLA rows have a recorded bolt task", 3,
            sum(1 for v in runs.values() if v.get("bolt_task")), source=str(p))
    a.check(sec, "the recorded task ids are distinct (one job per row, not one "
                 "job's id pasted onto three rows)", 3,
            len({v.get("bolt_task") for v in runs.values()}))
    for name, v in sorted(runs.items()):
        f = can / str(v.get("released_file"))
        got = (hashlib.sha256(f.read_bytes()).hexdigest() if f.exists()
               else "<missing>")
        a.check(sec, f"{name}: the released artifact still hashes to the "
                     f"sha256 recorded for bolt {v.get('bolt_task')}",
                v.get("released_sha256"), got, source=str(f))
        a.check(sec, f"{name}: the released file was the job's own output, "
                     f"byte-for-byte, not a re-derived copy",
                True, bool(v.get("artifact_identical_to_released")))
        a.check(sec, f"{name}: scored on all 13 families", 13,
                v.get("n_families"))


def audit_dequant_convention(a: Audit, d: Optional[dict]) -> None:
    """P2 de-quantizes bin b to -1+(b+0.5)*2/256; the checkpoint's own tokenizer
    uses the midpoints of linspace(-1,1,256), a spacing of 2/255. The paper
    claims (i) the skew is real and non-trivial relative to tau, (ii) F_mag is
    nonetheless EXACTLY invariant to it for a structural reason that holds for
    future runs too, and (iii) F_dir is not, so the affected values are restated
    on the checkpoint's grid. (i) and (ii) are arithmetic and are recomputed
    here from scratch rather than read out of a report -- the whole point is that
    they do not depend on any artifact. (iii) is checked against the release."""
    sec = "P2 de-quantization convention (reviewer C2)"
    tau = 0.05
    p2 = lambda b: -1.0 + (b + 0.5) * 2.0 / 256.0
    edges = [-1.0 + 2.0 * i / 255.0 for i in range(256)]
    up = [(edges[i] + edges[i + 1]) / 2.0 for i in range(255)]
    upv = lambda b: up[min(b, 254)]

    gaps = [(abs(p2(b) - upv(b)), b) for b in range(256)]
    worst, worst_bin = max(gaps)
    a.check(sec, "max |value difference| over the 256 bins = 0.007797",
            0.007797, round(worst, 6), tol=1e-6,
            source=f"worst at bin {worst_bin}; "
                   f"{worst / tau * 100:.1f}% of tau={tau}")
    a.check(sec, "the worst-case skew is a non-trivial fraction of tau "
                 "(>10%), so invariance cannot be waved through as rounding",
            True, worst / tau > 0.10,
            source=f"{worst / tau * 100:.1f}% of tau")
    a.check(sec, "bins 254 and 255 collapse to one value under the checkpoint's "
                 "grid (linspace(-1,1,256) has only 255 midpoints)",
            True, upv(254) == upv(255), source=f"both = {upv(255):.8f}")
    a.check(sec, "bin 127 is negative under P2 and exactly zero under the "
                 "checkpoint's grid -- the mechanism that moves gripper F_dir",
            True, p2(127) < 0.0 and upv(127) == 0.0,
            source=f"P2 {p2(127):.6f} vs checkpoint {upv(127):+.1f}")

    # The structural argument, stated as the paper states it: a Delta is always
    # an integer number of bins, so tau can only be crossed at a bin boundary.
    # If tau falls in the same inter-bin gap under both spacings, no stretch of
    # the grid can move a flag -- for ANY run at this tau, not just ours.
    k2 = sum(1 for k in range(1, 300) if k * 2.0 / 256.0 <= tau)
    kup = sum(1 for k in range(1, 300) if k * 2.0 / 255.0 <= tau)
    a.check(sec, "tau=0.05 admits the same maximum bin count under both "
                 "spacings (6 bins), which is why F_mag cannot flip",
            (6, 6), (k2, kup),
            source=f"6 bins = {6 * 2 / 256:.4f}/{6 * 2 / 255:.4f}, "
                   f"7 bins = {7 * 2 / 256:.4f}/{7 * 2 / 255:.4f}; tau sits "
                   f"strictly between under both")

    # (iii) the release must actually be on the checkpoint's grid.
    src_path = ROOT / "scripts" / "derive_metrics.py"
    try:
        src = src_path.read_text()
    except Exception:
        src = ""
    a.check(sec, "derive_metrics restates stored actions on the checkpoint's "
                 "grid before scoring anything", True,
            "_regrid_rows" in src and "_regrid_rows(rep.get(\"per_sample\"" in src,
            source=f"{src_path.name}: per_run_stats consumes _regrid_rows(...)")
    a.check(sec, "off-grid records pass through untouched and are counted "
                 "rather than silently forced onto P2's grid", True,
            "GRID_PASSTHROUGH" in src,
            source="a future checkpoint with a different action tokenizer "
                   "(e.g. DeepThinkVLA's FAST) must not be corrupted")

    fdir = dig(d, "models", "ecot-bridge", "families", "gripper_flip", "F_dir")
    ndir = dig(d, "models", "ecot-bridge", "families", "gripper_flip",
               "n_directional")
    a.check(sec, "ECoT-bridge gripper_flip F_dir = 0.0 on the checkpoint's grid",
            0.0, None if fdir is None else r3(fdir), tol=1e-9,
            source="models['ecot-bridge'].families.gripper_flip.F_dir; it was "
                   "11/300 = 0.037 under P2's convention")
    a.check(sec, "16 of ECoT-bridge's 300 gripper_flip records leave F_dir's "
                 "denominator because their gripper lands on bin 127",
            284, ndir,
            source="models['ecot-bridge'].families.gripper_flip.n_directional")

    # F_mag is what every headline number is, so assert the invariance claim
    # against the release for the families the paper tabulates.
    fmags = [dig(d, "models", m, "families", "direction_flip", "F_mag")
             for m in ALL8]
    a.check(sec, "every tabulated direction_flip F_mag survives the regrid "
                 "(present and unchanged from the published table)", True,
            None if any(f is None for f in fmags) else
            r3(dig(d, "models", "ecot-bridge", "families",
                   "direction_flip", "F_mag")) == 0.963,
            source="F_mag is invariant by the quantum argument above; this "
                   "checks the release agrees")

    # The released Bolt artifact must carry the numbers the paragraph quotes.
    rel = ROOT / "results_v2" / "canonical_runs" / "p2_decode_equivalence"
    rep = load(rel / "p2_dequant_recompute.json")
    tot = dig(rep, "totals") or {}
    for key, claim, exp in [
        ("n_scored", "36,688 scored records replayed", 36688),
        ("n_recover_failed", "0 records failed to invert back to bins", 0),
        ("n_delta_mismatch", "0 replays disagreed with their own stored delta", 0),
        ("n_flip_to_faithful", "0 records flip TO faithful", 0),
        ("n_flip_to_unfaithful", "0 records flip AWAY from faithful", 0),
        ("n_linf_changed", "17,058 records do get a different L-inf", 17058),
        ("n_bin255_present", "5,634 records sit at bin 255", 5634),
        ("n_dir_verdict_changed", "46 records change F_dir verdict", 46),
        ("n_dir_applicability_changed",
         "25 records change F_dir admissibility", 25),
    ]:
        a.check(sec, claim, exp, tot.get(key),
                source=f"{rel.name}/p2_dequant_recompute.json:totals.{key}")
    a.check(sec, "the released report's own verdict is that no F_mag moves",
            0.0, dig(rep, "worst_delta_F_mag"), tol=1e-12,
            source="worst_delta_F_mag over every family in every artifact")
    a.check(sec, "the released report names gripper_flip as the worst F_dir mover",
            True, "gripper_flip" in (dig(rep, "worst_delta_F_dir_where") or ""),
            source=f"worst_delta_F_dir_where = "
                   f"{dig(rep, 'worst_delta_F_dir_where')!r}")
    # The replay is only worth quoting if it covers the artifacts the paper
    # actually publishes. It used to run over 12 files (the single-seed release);
    # it now runs over 34, which is what the 3-seed re-runs produced, and the
    # F_mag invariance holding at 3.4x the record count is a stronger claim than
    # the one the submission made rather than the same one restated.
    a.check(sec, "the replay covers all 34 released edit artifacts, not the 12 "
                 "of the superseded single-seed release", 34,
            len(dig(rep, "per_artifact") or []),
            source=f"{rel.name}/p2_dequant_recompute.json:per_artifact")
    for fn in ("README.md", "bolt_task_id.txt"):
        a.check(sec, f"the release ships {fn} for this artifact", True,
                (rel / fn).exists(), source=str(rel / fn))

    try:
        tex = TEX.read_text()
    except Exception:
        return
    for needle, what in [
        ("15.6\\%", "the skew as a fraction of tau"),
        ("36{,}688", "the number of records replayed"),
        ("exactly invariant", "the F_mag invariance claim"),
        ("p2\\_dequant\\_recompute", "the derivation script"),
    ]:
        a.check(sec, f"the manuscript states {what}", True, needle in tex,
                source="the convention paragraph in the gate section")
    a.check(sec, "the superseded single-seed record count is gone from the "
                 "manuscript", 0, tex.count("10{,}780"),
            source="cot_faith_iclr.tex")


def audit_deepthink_tau_units(a: Audit) -> None:
    """DeepThinkVLA de-normalizes by LIBERO quantiles, so tau=0.05 means
    something different there than on the ECoT side. The paper discloses this
    with the checkpoint's own q01/q99, so those digits must match the artifact
    and the stated direction of the bias must be the conservative one."""
    sec = "DeepThinkVLA tau units (cross-family comparability)"
    run_path = ROOT / "results_v2" / "canonical_runs" / "deepthink_sft_13family.json"
    try:
        run = json.loads(run_path.read_text())
    except Exception:
        run = None
    if not run:
        a.check(sec, "deepthink_sft_13family.json present", True, None,
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


def audit_citations(a):
    """The bibliography, against the registry report.

    A fabricated reference is grounds for rejection on its own, and it is the
    single cheapest error for an LLM-assisted workflow to produce: a plausible
    entry costs nothing to emit and reads exactly like a real one. So the
    bibliography is checked against a registry rather than proofread, and this
    audit holds that check to three things. No entry may be CONTRADICTED by a
    registry. Every entry the manuscript prints must appear in the report, so
    adding a reference without re-running the check fails here rather than
    passing silently. And the number of entries that could not be resolved must
    match the number the manuscript's provenance note admits to -- otherwise
    "unverified" quietly becomes a place to park anything inconvenient.
    """
    sec = "Bibliography provenance (registry check)"
    path = ROOT / "results_v2" / "canonical_runs" / "citation_check" / \
        "citation_check.json"
    r = load(path)
    if not r:
        a.check(sec, "the citation-check report is released", True, False,
                source=str(path))
        return
    src = "results_v2/canonical_runs/citation_check/citation_check.json"
    tex = TEX.read_text() if TEX.exists() else ""

    a.check(sec, "no entry is contradicted by a registry on title, "
                 "author-surname order or year", [], r.get("mismatch_keys"),
            source=src)
    a.check(sec, "no entry failed to parse out of the manuscript, which would "
                 "mean it went unchecked rather than checked and passed", 0,
            (r.get("status_counts") or {}).get("PARSE_ERROR", 0), source=src)

    # Report coverage against the manuscript itself, not against the report's
    # own idea of how many entries there are.
    keys_tex = set(re.findall(r"\\bibitem\[[^\]]*\]\{(\w+)\}", tex))
    keys_rep = {e.get("key") for e in (r.get("entries") or [])}
    a.check(sec, "every \\bibitem in the manuscript appears in the report, so a "
                 "reference added after the last check cannot slip through",
            [], sorted(keys_tex - keys_rep), source=src)
    a.check(sec, "and the report contains no entry the manuscript dropped", [],
            sorted(keys_rep - keys_tex), source=src)
    a.check(sec, "the manuscript's 15 entries are all accounted for", 15,
            len(keys_tex), source="cot_faith_iclr.tex")

    # The confirmed/unverified split. All 15 now resolve, which took removing an
    # accidental precondition rather than finding a new registry: the check only
    # queried arXiv when the bibitem itself printed an id, so the five venue-only
    # entries were unverifiable because of OUR formatting, not because of theirs.
    # DBLP was never the answer -- it times out from the authoring network and
    # from bolt qrpd3f8z58 alike.
    a.check(sec, "all 15 entries confirm against a reachable registry",
            [15, None], [(r.get("status_counts") or {}).get("CONFIRMED"),
                         (r.get("status_counts") or {}).get("UNVERIFIED")],
            source=src)
    a.check(sec, "nothing is left unverified, so 'unverified' is not a parking "
                 "space for an inconvenient entry", [],
            r.get("unverified_keys"), source=src)
    a.check(sec, "the five venue-only entries were resolved by title search, "
                 "not by an id the manuscript does not print", True,
            all(any(c.get("registry") == "arxiv_title_search"
                    for c in (e.get("checks") or []))
                for e in (r.get("entries") or [])
                if e.get("key") in ("colosseum", "cotvla", "datasheets",
                                    "libero", "turpin2023")), source=src)
    a.check(sec, "every entry was compared on all three of title, author-surname "
                 "order and year, so a CONFIRMED is not one field passing", True,
            all({"title", "authors", "year"} <=
                set((c.get("fields") or {}).keys())
                for e in (r.get("entries") or [])
                for c in (e.get("checks") or [])), source=src)
    a.check(sec, "the report records that arXiv was reachable, so a CONFIRMED "
                 "is a real lookup rather than an absent registry", "reachable",
            dig(r, "registry_reachability", "arxiv"), source=src)

    # And the manuscript has to disclose all of this where a reader looks.
    a.check(sec, "the bibliography carries the provenance note naming the "
                 "checking script", True,
            "experiments/verify_citations.py" in tex, source="cot_faith_iclr.tex")
    a.check(sec, "the one remaining [VERIFY] marker is the datasheets page "
                 "range, and it is the only one", 1, tex.count("[VERIFY]"),
            source="cot_faith_iclr.tex")


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


def audit_judge_edit_families(a: Audit) -> None:
    """The edit families' own semantic premises, against the LLM-judge run.

    Section~\\ref{sec:judge_edits} reports one adverse result: the family named
    `adversarial_plausible` is judged plausible on 0.125 of its pairs, against a
    taxonomy entry that promised "visually plausible but wrong". That correction
    is the kind that gets quietly reverted -- the old sentence is shorter, reads
    better, and nothing in a LaTeX build objects to it. So this function checks
    two different things:

      * the numbers, from judge_report.json, including the FAILING verdict. It
        asserts premise_holds is False, not True. An audit that only knows how
        to confirm favourable premises would pass the day someone regenerates
        the family and the failure silently goes away unremarked -- and would
        also pass if the report were replaced by one measuring nothing.
      * the manuscript text, for the refuted phrase as a bare assertion. This is
        the only check in the script that guards a *wording*, and it is here
        because the wording is the claim: "visually plausible but wrong" with no
        qualifier is a statement the release refutes.

    The judge's own validity gates are checked first and hard. Every rate below
    is worthless if the judge answers "preserved" unconditionally, which is
    exactly what the identity gate alone cannot detect -- so the pooled negative
    control is required to be LOW, as an upper bound rather than a floor.
    """
    sec = "Edit-family premises (LLM-judge validation of the generators)"
    rel = ROOT / "results_v2" / "canonical_runs" / "judge_edit_families"
    src = "results_v2/canonical_runs/judge_edit_families/judge_report.json"
    rep = load(rel / "judge_report.json")
    if not rep:
        a.check(sec, "judge_report.json is released", True, False, source=src)
        return

    a.check(sec, "the judged pairs are released alongside the report, so every "
                 "rate below can be recomputed and a disputed verdict read",
            True, (rel / "judge_pairs.json").exists(),
            source="results_v2/canonical_runs/judge_edit_families/judge_pairs.json")
    a.check(sec, "the run records its bolt task id", "jhcgnqbmf2",
            (rel / "bolt_task_id.txt").read_text().strip()
            if (rel / "bolt_task_id.txt").exists() else None,
            source="results_v2/canonical_runs/judge_edit_families/bolt_task_id.txt")

    # ---- the judge's own gates, before any of its rates are read ----
    g = rep.get("gates") or {}
    a.check(sec, "identity control: an X->X edit is called meaning-preserving "
                 "on all 40 pairs", 1.0,
            g.get("identity_control_preserved_rate"), tol=1e-9, source=src)
    # An upper bound, not a target. The identity gate passes for a judge that
    # always answers "preserved"; this is the one that rules it out.
    a.check(sec, "negative control: pooled over the meaning-destroying "
                 "families the judge preserves at most 0.30 (measured 0.175)",
            True, (g.get("negative_control_pooled_preserved_rate") or 1.0) <= 0.30,
            source=src)
    a.check(sec, "order agreement is 0.824 as quoted, i.e. a 17.6% flip rate "
                 "under swapping which trace is shown first", 0.824,
            g.get("order_agreement"), tol=0.001, source=src)
    a.check(sec, "the run is marked valid, which is the conjunction of the "
                 "three gates", True, g.get("judge_valid"), source=src)
    a.check(sec, "the flip rate the manuscript quotes is the ratio the report "
                 "records", 77,
            rep.get("order_disagreements"), source=src)

    # ---- skip accounting: nothing silently dropped ----
    tot, judged, skip = (rep.get("n_pairs_total"), rep.get("n_pairs_judged"),
                         rep.get("n_skipped"))
    a.check(sec, "480 pairs constructed, 437 judged, 43 skipped, as printed",
            [480, 437, 43], [tot, judged, skip], source=src)
    a.check(sec, "judged + skipped accounts for every constructed pair",
            tot, (judged or 0) + (skip or 0), source=src)
    a.check(sec, "every skip is itemized with a reason, and the itemization "
                 "sums to the skip count",
            skip, sum((rep.get("skipped_reasons") or {}).values()), source=src)

    # ---- the four declared premises, including the one that fails ----
    v = rep.get("verdicts") or {}
    for fam, holds in (("paraphrase_null", True), ("bbox_jitter_null", True),
                       ("syntactic_scramble", True),
                       ("adversarial_plausible", False)):
        a.check(sec, f"declared premise for {fam} comes back "
                     f"{'HOLDS' if holds else 'FAILED'}",
                holds, (v.get(fam) or {}).get("premise_holds"), source=src)

    # The two rates F_diff depends on. If either of these ever drops, the
    # paraphrase floor stops being a floor and the central negative result of
    # sections f2_calib and paraphrase_null loses its premise.
    a.check(sec, "paraphrase_null preserves meaning at 0.975 as quoted", 0.975,
            (v.get("paraphrase_null") or {}).get("meaning_preserved_rate"),
            tol=0.001, source=src)
    a.check(sec, "bbox_jitter_null preserves meaning at 1.000 as quoted", 1.0,
            (v.get("bbox_jitter_null") or {}).get("meaning_preserved_rate"),
            tol=1e-9, source=src)

    # ---- the adverse result, both halves ----
    ap = v.get("adversarial_plausible") or {}
    a.check(sec, "adversarial_plausible does change the referent (0.958) -- the "
                 "half of its premise that holds", 0.958,
            ap.get("referent_changed_rate"), tol=0.001, source=src)
    a.check(sec, "adversarial_plausible is judged plausible on only 0.125 -- "
                 "the half that fails, and the reason the taxonomy entry was "
                 "corrected", 0.125,
            ap.get("plausible_rate"), tol=0.001, source=src)

    pf = rep.get("per_family") or {}
    for fam, rate in (("syntactic_scramble", 1.0), ("verb_swap", 0.575),
                      ("cross_task_swap", 0.025), ("gripper_flip", 0.05),
                      ("negation", 0.1), ("direction_flip", 0.2)):
        a.check(sec, f"{fam} meaning-preserved rate is {rate} as quoted", rate,
                (pf.get(fam) or {}).get("meaning_preserved_rate"), tol=0.001,
                source=src)

    a.check(sec, "the report states plainly that it validates the generators "
                 "and not the specific scored pairs", True,
            str(rep.get("record_level_correspondence") or "").startswith("NO"),
            source=src)

    # ---- the manuscript wording the release refutes ----
    if TEX.exists():
        tex = TEX.read_text()
        # The refuted claim, as it stood before this measurement. Matched
        # without the qualifier that now follows it, so restoring the old
        # sentence fails the audit while the corrected one passes.
        a.check(sec, "the taxonomy no longer asserts adversarial_plausible is "
                     "'visually plausible but wrong' without qualification",
                False,
                "second-most visible object (visually plausible but wrong)" in tex,
                source="cot_faith_iclr.tex: section taxonomy")
        a.check(sec, "and it states the measured plausibility rate instead",
                True, ("only $0.125$ of its edits are judged plausible" in tex),
                source="cot_faith_iclr.tex: section taxonomy")
        a.check(sec, "the judge section exists and is the target of the "
                     "taxonomy's cross-references", True,
                "\\label{sec:judge_edits}" in tex
                and tex.count("ref{sec:judge_edits}") >= 3,
                source="cot_faith_iclr.tex")


def audit_p3_frame_check(a):
    """A report whose frame check FAILED must not be citable as a P3 row.

    bolt wmi3nxd454 scored all 200 samples and then failed its own precondition:
    the policy's action error does not beat predicting the dataset mean, in any
    frame the checkpoint ships. The report is released anyway -- withholding the
    numbers a failed gate produced is how a gate becomes unfalsifiable -- and it
    therefore contains four perfectly presentable AUROCs, one of them 0.798.

    That combination is the hazard this function exists for. The artifact looks
    exactly like the artifact of a passing run, and the only thing separating
    them is a boolean nobody has to read. So the numbers are enumerated out of
    the JSON and matched against the manuscript: if any of them ever appears,
    the audit fails and names the file. Enumerated rather than hard-coded on
    purpose -- a hard-coded list stops protecting the paper the moment the run
    is repeated with a different seed.
    """
    sec = "P3 frame check (a failed precondition is not a result)"
    path = ROOT / "results_v2" / "canonical_runs" / "auroc_indomain_ours_null" / \
        "cot_auroc_report.json"
    src = "results_v2/canonical_runs/auroc_indomain_ours_null/cot_auroc_report.json"
    r = load(path)
    if not r:
        a.check(sec, "the failed-frame-check report is released, not discarded",
                True, False, source=str(path))
        return

    fc = r.get("frame_check") or {}
    a.check(sec, "the report carries a frame_check block at all -- without one "
                 "there is nothing to gate on", True, isinstance(fc, dict) and
            bool(fc), source=src)
    a.check(sec, "the frame check is recorded as FAILED, which is what makes "
                 "this a null rather than a P3 row", False, fc.get("passed"),
            source=src)

    # Both preconditions are reported, not just the one that fired. If a future
    # run drops the passing check, the verdict stops being interpretable.
    checks = {c.get("check"): c for c in (fc.get("checks") or [])}
    a.check(sec, "both preconditions are reported, the passing one included",
            ["gt_actions_inside_token_grid", "policy_beats_predict_mean"],
            sorted(checks), source=src)
    a.check(sec, "ground-truth actions do lie inside the token grid, so the "
                 "failure is not a range error", True,
            (checks.get("gt_actions_inside_token_grid") or {}).get("passed"),
            source=src)

    # The competing-frame sweep is the whole basis of the verdict: one frame's
    # ratio cannot tell "wrong units" from "weak policy".
    frames = fc.get("baselines_by_frame") or {}
    a.check(sec, "more than one frame was scored, so 'no frame beats the mean' "
                 "is a measurement rather than an assumption",
            True, len(frames) >= 2, source=src)
    a.check(sec, "no frame the checkpoint ships beats predicting the dataset "
                 "mean", True,
            bool(frames) and all(b.get("policy_over_predict_mean", 0) >= 1.0
                                 for b in frames.values()), source=src)
    a.check(sec, "the frame we scored is the BEST available one, so the null is "
                 "not an artefact of picking the wrong map",
            (fc.get("measured") or {}).get("scored_frame")
            if fc.get("measured") else "identity",
            (fc.get("measured") or {}).get("best_frame")
            if fc.get("measured") else
            (min(frames, key=lambda k: frames[k]["policy_over_predict_mean"])
             .split(":")[-1] if frames else None),
            source=src)
    a.check(sec, "the verdict names competence rather than units, so it cannot "
                 "be written up as a fixable scale bug", True,
            "NOT A FRAME ERROR" in str(fc.get("diagnosis")), source=src)

    # predict_zero is why the verdict says "no better than a constant" instead
    # of "the actions are tiny": the mean is a strong baseline here.
    ident = frames.get("identity") or {}
    a.check(sec, "predicting the dataset mean is much better than predicting "
                 "zero, so the beaten baseline is informative", True,
            bool(ident) and ident.get("predict_zero", 0)
            > 2 * ident.get("predict_mean", 1), source=src)

    # --- the actual guard: none of it may reach the manuscript ---
    #
    # Matched at 4 decimals and at the JSON's own precision, and NOT at 3.
    # A 3-decimal match looked stricter and was simply wrong: it flagged
    # $0.643$ (F2's control-ceiling-normalized score) and $0.357$ (a CI lower
    # bound for alpha(visual)) as citations of this report's 0.6426 and 0.3574.
    # Both are unrelated quantities that happen to round the same way. An audit
    # that cries wolf gets switched off, which is worse protection than none, so
    # the digit check is kept at the precision the manuscript actually prints
    # AUROCs to and the claim-level check below carries the real weight.
    tex = TEX.read_text() if TEX.exists() else ""
    quoted = []
    for name, block in (r.get("aurocs") or {}).items():
        for field in ("raw_auroc", "abs_auroc"):
            v = block.get(field)
            if v is None:
                continue
            for s in {f"{v:.4f}", f"{v:.5f}", repr(round(v, 5))}:
                if s in tex:
                    quoted.append(f"{name}.{field}={s}")
    a.check(sec, "no AUROC from the failed-frame-check report is quoted in the "
                 "manuscript at the precision the paper prints AUROCs to",
            [], sorted(set(quoted)), source=src)

    # The two L1 numbers used to be forbidden in the manuscript, on the reasoning
    # that a report whose frame check FAILED had no business supplying paper
    # numbers. That guard was inverted by the disclosure it was meant to force:
    # six of the eight leaderboard rows come from this checkpoint, and stating
    # that its open-loop prediction does not beat a constant requires quoting the
    # two numbers that establish it. Quoting them as a WITHDRAWAL is the opposite
    # failure mode from quoting them as a result, so the check now asserts the
    # disclosure is present and correctly framed rather than that the digits are
    # absent -- and it still fails loudly if the digits appear without it.
    disclosure = ("whose single-step open-loop prediction does not beat a "
                  "constant" in tex)
    for field in ("policy", "predict_mean"):
        v = (r.get("action_error_baselines_l1") or {}).get(field)
        if v is None:
            continue
        digits = [s for s in {f"{v:.5f}", f"{v:.4f}"} if s in tex]
        a.check(sec, f"the report's {field} L1 is quoted only alongside the "
                     f"disclosure that this checkpoint fails the gate",
                True, (not digits) or disclosure, source=src)

    # The disclosure must name the scope -- "six of eight leaderboard rows" -- or
    # it degrades into a footnote about one auxiliary run. The count is derived
    # from the leaderboard, not hardcoded in prose we could drift away from.
    a.check(sec, "the W2 disclosure states how many leaderboard rows the failed "
                 "checkpoint is behind, so it cannot be read as an aside",
            True, "Six of eight leaderboard rows" in tex,
            source="cot_faith_iclr.tex")
    a.check(sec, "and it states what the failure does NOT undermine, because a "
                 "bare disclosure would over-withdraw the edit metric",
            True, "It does not invalidate" in tex and "self" in tex,
            source="cot_faith_iclr.tex")

    # The claim-level guard, which is the one that actually holds. Digits can be
    # re-rounded and re-derived; the sentence cannot be quietly widened. As long
    # as this checkpoint's frame check fails, P3 covers ONE model, and the
    # manuscript has to keep saying so. Promoting this run to a second row means
    # both of these flip together -- which is exactly the coupling we want,
    # because it is impossible to satisfy by editing prose alone.
    scoped = "on the single model where we can run it in-domain" in tex
    a.check(sec, "P3 is still scoped to one model in the text, because the only "
                 "second candidate failed its own precondition",
            True, scoped, source="cot_faith_iclr.tex")
    a.check(sec, "the text's model count and the gate agree: a second P3 row "
                 "requires frame_check.passed, and it is false",
            True, scoped is not bool(fc.get("passed")), source=src)

    # --- the release-wide invariant ---
    #
    # "and nothing else claims to be a P3 run" (audit_release) globs
    # canonical_runs/*.json -- one level only. This report has the full P3
    # schema (per_sample + aurocs) and lives one directory deeper, so it passed
    # that check by being invisible to it rather than by being accounted for.
    # That is the kind of accident that turns into a wrong claim later: flatten
    # the release layout and a null starts counting as a P3 run.
    #
    # So the sweep is done recursively here, and the invariant is stated in the
    # form that survives a re-layout: every P3-schema file in the release is
    # either one of the two runs the paper cites, or is marked as a failed frame
    # check. There is no third category.
    can = ROOT / "results_v2" / "canonical_runs"
    cited = {"auroc_ecot_bridge_n200.json",
             "auroc_ecot_bridge_indomain_n153.json"}
    unaccounted = []
    for f in sorted(can.rglob("*.json")):
        rec = load(f)
        if not isinstance(rec, dict):
            continue
        ps = rec.get("per_sample")
        if not (isinstance(ps, list) and ps and "aurocs" in rec):
            continue
        if f.name in cited:
            continue
        if (rec.get("frame_check") or {}).get("passed") is False:
            continue
        unaccounted.append(str(f.relative_to(ROOT)))
    a.check(sec, "every P3-schema file in the release is either a run the paper "
                 "cites or is marked frame_check.passed=false -- no third kind",
            [], unaccounted, source="results_v2/canonical_runs/**/*.json")


def audit_gripper_ab_null(a):
    """The negative result Section 6 now reports, checked against its artifact.

    A null is the easiest claim in a paper to leave unsourced, because nothing
    downstream depends on it: no table cell moves if "all four arms scored zero"
    quietly becomes "three of four". It is also the claim most likely to be
    softened later, once a fix is found, into something that reads better than
    what was measured. So every digit the manuscript prints about this run is
    tied here to results_v2/canonical_runs/gripper_ab_null/gripper_ab.json.

    The distinctness checks are not decoration. Four arms that all score 0 are
    only evidence about the gripper if they actually sent different gripper
    commands; if the transform had silently no-opped, the same artifact would
    have been produced by four runs of one configuration and would license
    nothing at all.
    """
    sec = "Gripper-convention A/B (the null Section 6 reports)"
    path = ROOT / "results_v2" / "canonical_runs" / "gripper_ab_null" / \
        "gripper_ab.json"
    r = load(path)
    if not r:
        a.check(sec, "the gripper-A/B artifact is released alongside the claim",
                True, False, source=str(path))
        return
    src = "results_v2/canonical_runs/gripper_ab_null/gripper_ab.json"
    arms = r.get("arms") or {}

    a.check(sec, "all four conventions the manuscript names are in the report",
            ["binvert", "invert", "none", "openvla"], sorted(arms),
            source=src)
    a.check(sec, "every arm scored exactly zero successes -- the null, stated as "
                 "the artifact states it", [0.0],
            sorted({float(v.get("SR")) for v in arms.values()}), source=src)
    a.check(sec, "each arm's denominator is the realised 10 episodes, so the "
                 "quoted 0/10 is the run's own tally", [10],
            sorted({v.get("n_total") for v in arms.values()}), source=src)
    a.check(sec, "40 episodes in total, as Section 6 says", 40,
            sum(v.get("n_total") or 0 for v in arms.values()), source=src)
    a.check(sec, "all 40 episodes ran on canonical initial states, without "
                 "which a zero SR would be uninterpretable", True,
            all(v.get("all_episodes_used_canonical_init") is True
                for v in arms.values()), source=src)

    # The arms are distinct: this is what makes four zeros evidence.
    sent = {k: v.get("gripper_sent_mean") for k, v in arms.items()}
    # Quoted at 4 dp, not 3, because the max is exactly 0.7795 (3118 of 4000
    # samples at +1) and rounding it to 3 dp puts the paper one ULP away from
    # the artifact for no gain: the stored double is 0.779499..., so "0.780" is
    # arguably right and round() disagrees. Exact digits end the argument.
    a.check(sec, "the quoted span of delivered gripper commands is the min and "
                 "max over the four arms", [-0.535, 0.7795],
            [round(min(sent.values()), 4), round(max(sent.values()), 4)],
            source=src)
    frac = [v.get("gripper_frac_close_sent") for v in arms.values()]
    a.check(sec, "the quoted closed-gripper fraction range is the artifact's, "
                 "rounded as printed", [0.12, 0.89],
            [round(min(frac), 2), round(max(frac), 2)], source=src)
    a.check(sec, "the four arms are pairwise distinct in what they sent, so this "
                 "is not one configuration run four times", 4,
            len({round(v, 6) for v in sent.values()}), source=src)
    a.check(sec, "'none' sent exactly what it decoded, which is what makes it "
                 "the anchor cell", True,
            dig(arms, "none", "gripper_raw_mean") ==
            dig(arms, "none", "gripper_sent_mean"), source=src)
    a.check(sec, "the report names no winner, matching the reported null", [],
            r.get("winners"), source=src)

    tid = ROOT / "results_v2" / "canonical_runs" / "gripper_ab_null" / \
        "bolt_task_id.txt"
    a.check(sec, "the run is attributable to a bolt task id", "viyhc4kpft",
            (tid.read_text().strip() if tid.exists() else ""), source=str(tid))

    # The artifact ships verbatim, including the misleading field name it was
    # written with. That is the honest choice, but it is only honest if the
    # discrepancy is documented rather than left for a reader to trip over.
    readme = ROOT / "results_v2" / "canonical_runs" / "gripper_ab_null" / \
        "README.md"
    rd = readme.read_text() if readme.exists() else ""
    a.check(sec, "the released artifact explains why its 'n_episodes_per_arm: 4' "
                 "disagrees with every arm's n_total of 10", True,
            "n_episodes_per_arm" in rd and "10 tasks" in rd,
            source="results_v2/canonical_runs/gripper_ab_null/README.md")
    a.check(sec, "and explains that the task's FAILED state is the designed "
                 "exit code for 'no arm wins', not a crash", True,
            "not a crash" in rd,
            source="results_v2/canonical_runs/gripper_ab_null/README.md")

    # The manuscript side. What must not drift is the scope of the null: the
    # source diff established the gripper difference is real, so "not sufficient"
    # and "not a difference" are different claims and only one of them is ours.
    tex = TEX.read_text() if TEX.exists() else ""
    a.check(sec, "Section 6 reports the null rather than only the factorial that "
                 "followed it", True,
            "The gripper convention alone is not the cause" in tex,
            source="cot_faith_iclr.tex")
    a.check(sec, "Section 6 bounds the null to insufficiency, not to the "
                 "difference being absent", True,
            "not that it is a non-difference" in tex,
            source="cot_faith_iclr.tex")
    a.check(sec, "Section 6 points at the released artifact", True,
            r"gripper\_ab\_null" in tex, source="cot_faith_iclr.tex")
    a.check(sec, "limitation (v) carries the null too, since that is where a "
                 "reader checks what the gate does and does not license", True,
            "leaves SR at $0/10$ under all four conventions" in tex,
            source="cot_faith_iclr.tex")


def audit_bridge_join_probe(a: Audit) -> None:
    """The Bridge V2 join, against the probe that measured it.

    The O4 section asks the reader to accept an uncontrolled comparison, and now
    explains that the missing control is blocked by a property of the public
    data rather than by a budget we declined to spend. That explanation rests on
    measured numbers, so they are checked here.

    Several of these checks are deliberately of the UNFAVOURABLE number. The
    whole point of the probe is that `episode_id` is a trap: an overlap of 2.1%
    whose instructions agree 0.280 of the time yields a complete, plausible,
    wrong index, and a training run on it completes normally. If someone later
    "fixes" the join by keying on episode_id, the number that refutes it has to
    fail an assertion rather than quietly vanish from a regenerated report.
    Likewise `merge_required` is asserted True: the day it reads False without
    the export layout having changed is the day the renderability measurement
    stopped measuring anything.

    The instruction-agreement number is ALSO checked as an upper bound, for the
    same reason the judge audit bounds its negative control from above. A probe
    that reported high agreement here would be reporting that the trap is safe,
    which is the one wrong answer that looks like good news.
    """
    sec = "Bridge V2 join (episode-level joinability of the two public exports)"
    rel = ROOT / "results_v2" / "canonical_runs" / "bridge_join_probe"
    src = "results_v2/canonical_runs/bridge_join_probe/bridge_join_probe.json"
    rep = load(rel / "bridge_join_probe.json")
    if not rep:
        a.check(sec, "bridge_join_probe.json is released", True, False, source=src)
        return

    tex = TEX.read_text() if TEX.exists() else ""
    trainer = (ROOT / "experiments" / "cotfaith_train_bridge.py").read_text()
    st = rep.get("strategies") or {}
    bi = st.get("by_episode_id") or {}
    bt = st.get("by_task_text") or {}
    rd = rep.get("renderability") or {}

    a.check(sec, "bolt task id is recorded with the artifact", "754ru9usqe",
            (rel / "bolt_task_id.txt").read_text().strip()
            if (rel / "bolt_task_id.txt").exists() else None,
            source="bridge_join_probe/bolt_task_id.txt")
    a.check(sec, "annotated episodes in the CoT export", 60062,
            rep.get("n_episode_keys_total"), source=src)
    a.check(sec, "every annotated episode was walked, not sampled", 60062,
            rep.get("n_episodes_walked"), source=src)
    a.check(sec, "LeRobot trajectory episodes", 53192,
            rep.get("n_lerobot_episodes"), source=src)

    # ---- the trap, asserted as a trap ----
    a.check(sec, "episode_id matches only 1111 LeRobot episodes", 1111,
            bi.get("n_matched_lerobot_episodes"), source=src)
    a.check(sec, "...which is 2.1% of them", 0.0209,
            bi.get("frac_lerobot_matched"), tol=1e-4, source=src)
    a.check(sec, "episode_id is per-shard, not global: max id < n episodes",
            True, (bi.get("id_range") or [0, 10 ** 9])[1] < 60062,
            source=src + " (id_range)")
    a.check(sec, "episode_id collides", 879, bi.get("n_id_collisions"), source=src)
    a.check(sec, "instructions agree on only 0.280 of matched pairs", 0.2799,
            bi.get("frac_instruction_agrees_on_matched"), tol=1e-4, source=src)
    a.check(sec, "that agreement stays well BELOW a joinable level (upper bound)",
            True, (bi.get("frac_instruction_agrees_on_matched") or 1.0) <= 0.5,
            source=src)
    a.check(sec, "the probe's verdict refuses the episode_id join", True,
            str(bi.get("verdict", "")).startswith("ids overlap but"), source=src)

    # ---- no exact route ----
    a.check(sec, "LeRobot episode records carry no upstream source path", True,
            (rep.get("lerobot_episode_keysets") or [["", 0]])[0][0]
            == "episode_index,length,tasks", source=src)

    # ---- the route that is used ----
    a.check(sec, "shared normalized instructions", 19541,
            bt.get("n_shared_normalized_tasks"), source=src)
    a.check(sec, "LeRobot episodes covered by instruction text", 38660,
            bt.get("n_lerobot_episodes_covered"), source=src)
    a.check(sec, "...i.e. 72.7% coverage", 0.7268,
            bt.get("frac_lerobot_episodes_covered"), tol=1e-4, source=src)
    a.check(sec, "median fanout is 1, so the typical key is unique", 1,
            bt.get("median_lerobot_episodes_per_shared_task"), source=src)
    a.check(sec, "max fanout is 963, which is why degenerate keys are refused",
            963, bt.get("max_lerobot_episodes_per_shared_task"), source=src)
    a.check(sec, "reachable annotated episodes", 41634,
            rep.get("n_reachable_annotated_episodes"), source=src)
    a.check(sec, "reachable episodes exceed the ~4k the ablation needs", True,
            (rep.get("n_reachable_annotated_episodes") or 0) >= 4000,
            source=src + " (n_reachable_annotated_episodes)")

    # ---- renderability: a correct join is necessary but not sufficient ----
    a.check(sec, "the per-step features+reasoning merge is required", True,
            rd.get("merge_required") is True, source=src)
    a.check(sec, "steps inspected for renderability", 4000,
            rd.get("n_reasoning_steps_inspected"), source=src)
    ro = rd.get("tag_fill_rate_from_reasoning_only") or {}
    fo = rd.get("tag_fill_rate_from_features_only") or {}
    mg = rd.get("tag_fill_rate_from_merge") or {}
    for tag in ("VISIBLE OBJECTS", "GRIPPER POSITION"):
        a.check(sec, f"reasoning alone leaves {tag} empty", 0.0, ro.get(tag),
                tol=1e-9, source=src)
        a.check(sec, f"features alone fills {tag}", 1.0, fo.get(tag),
                tol=1e-9, source=src)
    for tag in ("TASK", "PLAN", "SUBTASK", "SUBTASK REASONING",
                "MOVE REASONING", "MOVE"):
        a.check(sec, f"reasoning alone already fills {tag}", 1.0, ro.get(tag),
                tol=1e-9, source=src)
    a.check(sec, "the merge fills all eight tags at 1.0", True,
            len(mg) == 8 and all(abs(v - 1.0) < 1e-9 for v in mg.values()),
            source=src)
    a.check(sec, "no tag is unfillable by either subtree", [],
            rd.get("tags_unfillable_by_either"), source=src)

    # ---- the manuscript and the consumer state the same thing ----
    for frag, why in (
            ("754ru9usqe", "cites the bolt task id"),
            ("$0.280$ of the time", "states the instruction-agreement number"),
            ("$41{,}634$", "states the reachable-episode count"),
            ("$963$", "states the max fanout that motivates refusing keys")):
        a.check(sec, f"the manuscript {why}", True, frag in tex,
                source="cot_faith_iclr.tex (O4 section)")
    a.check(sec, "the trainer probes the id join instead of trusting it", True,
            "_ID_MIN_AGREE" in trainer,
            source="experiments/cotfaith_train_bridge.py")
    a.check(sec, "the trainer refuses degenerate instruction keys", True,
            "_usable_task_key" in trainer,
            source="experiments/cotfaith_train_bridge.py")


def audit_gate_factorial(a):
    """The 2x2 gripper x image factorial, and the null it returned.

    This is the run the paper's diagnostic narrative ends on, which makes it the
    one most likely to be quietly reframed once a fix is eventually found: "all
    four cells scored zero" is a sentence that gets easier to soften every week
    it stays true. Every digit is pinned to the artifact, including the two that
    do the argumentative work -- the 280-step budget, which is what excludes
    truncation as an explanation, and the distinctness of the cells, without
    which four zeros are four runs of the anchor.
    """
    sec = "Gripper x image factorial (the second null Section 6 reports)"
    base = ROOT / "results_v2" / "canonical_runs" / "gate_factorial_pil"
    r = load(base / "gripper_ab.json")
    if not r:
        a.check(sec, "the factorial artifact is released alongside the claim",
                True, False, source=str(base / "gripper_ab.json"))
        return
    src = "results_v2/canonical_runs/gate_factorial_pil/gripper_ab.json"
    arms = r.get("arms") or {}

    a.check(sec, "the run is the 2x2 the manuscript describes, with the gate "
                 "configuration as one of its cells",
            [["none", "openvla"], ["none", "pil_lanczos"]],
            [r.get("gripper_arms"), r.get("image_preprocs")], source=src)
    a.check(sec, "all four cells are present and none raised", [4, []],
            [len(arms), sorted(k for k, v in arms.items() if v.get("error"))],
            source=src)
    a.check(sec, "every cell scored zero, including the both-corrections cell",
            [0.0], sorted(set(r.get("sr_by_arm", {}).values())), source=src)
    a.check(sec, "the both-corrections cell exists under the name the "
                 "manuscript gives it and is one of the zeros", 0.0,
            (r.get("sr_by_arm") or {}).get("openvla+pil_lanczos"), source=src)
    a.check(sec, "10 episodes per cell and 40 in total, as printed", [[10], 40],
            [sorted({v.get("n_total") for v in arms.values()}),
             sum(v.get("n_total") or 0 for v in arms.values())], source=src)
    a.check(sec, "all 40 episodes ran on canonical initial states", True,
            all(v.get("all_episodes_used_canonical_init") is True
                for v in arms.values()), source=src)
    # The step budget is load-bearing: at 400 the paper could not claim
    # truncation is excluded, and the earlier gate's libero_10 zero is exactly
    # what an under-budgeted suite looks like.
    a.check(sec, "every cell ran at upstream's own 280-step budget for "
                 "libero_object, which is what excludes truncation", [[280], 280],
            [sorted({v.get("max_steps") for v in arms.values()}),
             r.get("max_steps")], source=src)
    a.check(sec, "and no cell recorded itself as below upstream's budget", True,
            all(v.get("max_steps_below_upstream") is False
                for v in arms.values()), source=src)

    # Distinctness of the factors, quoted as the manuscript quotes them.
    def cell(k, f):
        return (arms.get(k) or {}).get(f)
    a.check(sec, "the quoted gripper-command means for the anchor and the "
                 "gripper-corrected cell", [0.216, 0.753],
            [round(cell("none+none", "gripper_sent_mean"), 3),
             round(cell("openvla+pil_lanczos", "gripper_sent_mean"), 3)],
            source=src)
    a.check(sec, "the quoted closed-gripper fractions for the same two cells",
            [0.695, 0.876],
            [round(cell("none+none", "gripper_frac_close_sent"), 3),
             round(cell("openvla+pil_lanczos", "gripper_frac_close_sent"), 3)],
            source=src)
    a.check(sec, "the image factor moved the telemetry too, so it was applied "
                 "rather than silently skipped", True,
            cell("none+none", "gripper_sent_mean") !=
            cell("none+pil_lanczos", "gripper_sent_mean"), source=src)
    a.check(sec, "the report names no winner", [], r.get("winners"), source=src)

    tid = base / "bolt_task_id.txt"
    a.check(sec, "the run is attributable to a bolt task id", "i55ww23d5n",
            (tid.read_text().strip() if tid.exists() else ""), source=str(tid))

    tex = TEX.read_text() if TEX.exists() else ""
    a.check(sec, "Section 6 states the factorial null in the paper's own voice "
                 "rather than leaving the factorial 'under way'", True,
            "The factorial is also a null: all four cells scored $0/10$" in tex,
            source="cot_faith_iclr.tex")
    a.check(sec, "Section 6 no longer describes the factorial as pending", 0,
            tex.count("are being measured factorially"),
            source="cot_faith_iclr.tex")
    a.check(sec, "Section 6 bounds this null to sufficiency as well, since the "
                 "differences themselves were read out of upstream's source",
            True, "nor their conjunction, is what the gate is failing on" in tex,
            source="cot_faith_iclr.tex")
    # The remaining-work sentence has moved as the investigation closed
    # candidates. It used to name P2's token selection as still open; that has
    # now been measured, so what this check enforces is that Section 6 does not
    # end on the null -- it has to say which measurement resolved it and point
    # at the artifact, rather than leaving the three nulls as the last word.
    a.check(sec, "Section 6 says which measurement closed the investigation "
                 "rather than stopping at the null", True,
            "replacing it is what closed the gate" in tex
            and r"p2\_decode\_equivalence" in tex,
            source="cot_faith_iclr.tex")
    a.check(sec, "Section 6 points at the released artifact", True,
            r"gate\_factorial\_pil" in tex, source="cot_faith_iclr.tex")
    a.check(sec, "limitation (v) carries the factorial null too", True,
            "scores $0/10$ in every cell including the one that applies both "
            "corrections" in tex, source="cot_faith_iclr.tex")

    # ---- the same factorial with upstream's exact ops (bolt mmmnxeehda) ----
    # A separate artifact rather than more checks on the first one, because it
    # answers a different objection: the first run's image arm was an 8/255-LSB
    # approximation of the very kind of small per-frame mismatch it was testing
    # for, so "the approximation was not close enough" was an available excuse
    # for the null. This is the run that removes it.
    base2 = ROOT / "results_v2" / "canonical_runs" / "gate_factorial_tf"
    r2 = load(base2 / "gripper_ab.json")
    if not r2:
        a.check(sec, "the exact-preprocessing factorial is released too", True,
                False, source=str(base2 / "gripper_ab.json"))
        return
    src2 = "results_v2/canonical_runs/gate_factorial_tf/gripper_ab.json"
    arms2 = r2.get("arms") or {}
    a.check(sec, "the exact run is the same 2x2 with tf_upstream in place of the "
                 "Pillow path", [["none", "openvla"], ["none", "tf_upstream"]],
            [r2.get("gripper_arms"), r2.get("image_preprocs")], source=src2)
    a.check(sec, "it too scored zero in every cell, so the null does not depend "
                 "on the approximation", [0.0],
            sorted(set(r2.get("sr_by_arm", {}).values())), source=src2)
    a.check(sec, "including the exact both-corrections cell", 0.0,
            (r2.get("sr_by_arm") or {}).get("openvla+tf_upstream"), source=src2)
    a.check(sec, "same budget and same episode count as the approximate run, so "
                 "the two are comparable", [[280], [10], 40],
            [sorted({v.get("max_steps") for v in arms2.values()}),
             sorted({v.get("n_total") for v in arms2.values()}),
             sum(v.get("n_total") or 0 for v in arms2.values())], source=src2)
    a.check(sec, "all 40 of its episodes ran on canonical initial states", True,
            all(v.get("all_episodes_used_canonical_init") is True
                for v in arms2.values()), source=src2)
    a.check(sec, "no cell raised, so tensorflow really executed in the rollout "
                 "process rather than being skipped", [],
            sorted(k for k, v in arms2.items() if v.get("error")), source=src2)
    # The shared cell is the alignment evidence: the two jobs measured the same
    # configuration, so the exact-vs-approximate comparison is between the image
    # cells and not between two unrelated runs.
    a.check(sec, "the gripper-only cell reproduces across the two jobs to every "
                 "printed digit, which is what makes them comparable", True,
            (arms2.get("openvla+none") or {}).get("gripper_sent_mean") ==
            (arms.get("openvla+none") or {}).get("gripper_sent_mean"),
            source=f"{src2} vs {src}")
    a.check(sec, "and the anchor cells agree to within 0.005", True,
            abs((arms2.get("none+none") or {}).get("gripper_sent_mean", 0) -
                (arms.get("none+none") or {}).get("gripper_sent_mean", 0)) < 0.005,
            source=f"{src2} vs {src}")
    tid2 = base2 / "bolt_task_id.txt"
    a.check(sec, "the exact run is attributable to its own bolt task id",
            "mmmnxeehda", (tid2.read_text().strip() if tid2.exists() else ""),
            source=str(tid2))
    a.check(sec, "Section 6 reports the exact repeat and says why it was worth a "
                 "second job", True,
            "available as an excuse for the null" in tex,
            source="cot_faith_iclr.tex")
    a.check(sec, "Section 6 no longer calls the exact factorial 'now under way'",
            0, tex.count("The factorial now under way"),
            source="cot_faith_iclr.tex")
    a.check(sec, "Section 6 records that the tensorflow-coexistence probe held "
                 "in a real GPU rollout, not only in the probe", True,
            "holding in production rather than in isolation" in tex,
            source="cot_faith_iclr.tex")


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
    # The gate's failure COUNT, which is prose and therefore drifts. It read
    # "failed twice" in limitation (v) for several revisions after Section 6 had
    # started saying "four gate attempts" -- an internal contradiction no numeric
    # check could catch, since neither number is derived from an artifact. Both
    # sites are pinned to each other here, and the paragraph has to substantiate
    # its own count by describing each attempt. The count survives the gate
    # passing: four attempts failed before the fifth composition cleared it, and
    # dropping the four once the news got good is exactly the edit this pins.
    tex_l = TEX.read_text() if TEX.exists() else ""
    a.check(sec, "limitation (v) and Section 6 agree on how many times the gate "
                 "failed before it passed", [1, 1],
            [tex_l.count("the decoder gate failed four times before it passed"),
             tex_l.count("four gate attempts")], source="cot_faith_iclr.tex")
    a.check(sec, "limitation (v) names the third cause rather than leaving the "
                 "count of found causes short", True,
            "The third cause was the action decode itself" in tex_l,
            source="cot_faith_iclr.tex")
    a.check(sec, "the paragraph accounts for all four attempts, including the "
                 "third that had correct init states and still scored zero", True,
            "A third attempt then had verified canonical initial states" in tex_l,
            source="cot_faith_iclr.tex")
    for stale in ("gate has now failed twice",
                  "the decoder gate has now failed four times",
                  "with only two of the causes found"):
        a.check(sec, f"the superseded phrasing {stale!r} is gone", 0,
                tex_l.count(stale), source="cot_faith_iclr.tex")

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


def audit_rollout_gate_winning(a, d):
    """Table "the rollout gate passes" and the Section 6 / limitation (v) restatement.

    This is the mirror of audit_rollout_gate and it exists for the opposite
    reason. That table reports a number arguing against our own harness, so the
    risk is that it gets quietly improved. This one reports the harness finally
    working, so the risk is the reverse: that a passing cell drifts upward, or
    that the conditions which make it a *gate* rather than a demo get dropped.
    Three of those conditions are load-bearing and each is asserted separately:
    every suite ran at upstream's own step budget (so no cell is a truncation
    artifact and none was given extra time), every episode ran on the suite's
    canonical initial state (the defect that invalidated attempt two), and each
    job carries its own anchor arm with the corrected decoder but *without* the
    gripper convention -- which is what makes "jointly necessary" a measurement
    instead of an argument.
    """
    sec = "Four-suite rollout gate on the winning composition (Table: it passes)"
    g = (d.get("rollout_gate_winning") or {})
    suites, summary = g.get("suites") or {}, g.get("summary") or {}
    if not suites:
        a.check(sec, "the derived file carries a rollout_gate_winning block",
                True, False, source="results_v2/derived_metrics.json")
        return

    # --- the table body, exactly as printed. Steps are (requested, run,
    #     upstream): requested 0 means "use the suite's upstream budget", which
    #     is the whole point -- the pre-fix table hard-coded 400 everywhere.
    printed = {
        "libero_spatial": (0.74, 37, 50, 0.844, 0.877, 0.00, 220),
        "libero_object":  (0.90, 45, 50, 0.881, 1.022, 0.00, 280),
        "libero_goal":    (0.74, 37, 50, 0.794, 0.932, 0.12, 300),
        # libero_10 was excluded from the pre-fix table as uninterpretable: we
        # ran a flat 400 steps where upstream allots 520, so its 0/50 could have
        # been truncation. It is in the gate now at the sentinel budget, which is
        # what retires that caveat by measurement rather than by argument.
        "libero_10":      (0.46, 23, 50, 0.539, 0.853, 0.00, 520),
    }
    for suite, (sr, nsucc, ntot, pub, frac, anchor, budget) in printed.items():
        v = suites.get(suite) or {}
        a.check(sec, f"{suite}: Task SR, successes, episodes as printed",
                [sr, nsucc, ntot],
                [round(v.get("SR"), 2) if v.get("SR") is not None else None,
                 v.get("n_success"), v.get("n_total")],
                source=v.get("source"))
        a.check(sec, f"{suite}: published SR and fraction of it as printed",
                [pub, frac],
                [v.get("published_SR"), v.get("SR_frac_of_published")],
                source=v.get("source"))
        a.check(sec, f"{suite}: anchor arm (corrected decoder, no gripper "
                     f"convention) as printed", anchor,
                v.get("SR_anchor_no_gripper_transform"),
                source="gripper=none cell of the same job, same model load")
        # max_steps_requested=0 is the sentinel for "upstream's own budget".
        # If a future run hard-codes a number here the cell stops being
        # comparable to the published SR it is quoted against.
        a.check(sec, f"{suite}: ran at upstream's own step budget, requested as "
                     f"the sentinel rather than hard-coded", [0, budget, budget],
                [v.get("max_steps_requested"), v.get("max_steps_run"),
                 v.get("upstream_max_steps")], source=v.get("source"))
        a.check(sec, f"{suite}: the winning composition is upstream decoder + "
                     f"openvla gripper, with image preprocessing off",
                ["upstream", "openvla", "none"],
                [v.get("action_decoder"), v.get("gripper_transform"),
                 v.get("image_preproc")], source=v.get("source"))

    # Wilson CIs printed in the table.
    for suite, lo, hi in (("libero_spatial", 0.60, 0.84),
                          ("libero_object", 0.79, 0.96),
                          ("libero_goal", 0.60, 0.84),
                          ("libero_10", 0.33, 0.60)):
        ci = (suites.get(suite) or {}).get("SR_wilson95") or [None, None]
        a.check(sec, f"{suite}: Wilson 95% CI as printed", [lo, hi],
                [round(ci[0], 2), round(ci[1], 2)], source="SR_wilson95")

    # --- the conditions that make it a gate ---
    a.check(sec, "every suite ran at upstream's own step budget, so no cell is "
                 "a truncation artifact and none got extra time", True,
            summary.get("all_suites_at_upstream_step_budget"),
            source="step_budget_below_upstream is false on every suite")
    a.check(sec, "every episode ran on its suite's canonical initial state",
            True, summary.get("all_episodes_canonical_init"),
            source="all_episodes_used_canonical_init, per suite")
    a.check(sec, "the gate passes: SR is at least half the published SR on "
                 "every suite run", True, summary.get("gate_passed"),
            source=summary.get("gate_criterion"))
    a.check(sec, "the weakest cell as quoted in the prose", [0.46, 0.853],
            [summary.get("min_SR"), summary.get("min_SR_frac_of_published")],
            source="summary")
    # The four-suite flag is a separate claim from gate_passed and the paper
    # makes it: upstream publishes exactly four LIBERO checkpoints, so four
    # suites is the whole gate rather than a subset of a five-suite one.
    a.check(sec, "the gate passes on all four suites, which is every suite "
                 "upstream publishes a LIBERO checkpoint for", True,
            summary.get("gate_passed_on_all_four_suites"),
            source="summary")
    a.check(sec, "four suites ran, 200 episodes in the winning arms", [4, 200],
            [summary.get("n_suites_run"), summary.get("n_episodes_total")],
            source="summary")

    # --- joint necessity, which the prose states as a measurement ---
    # Both corrections are needed: the anchor arms hold the corrected decoder
    # and drop only the gripper convention, and every one of them lands far
    # below its paired winning cell. If an anchor ever came up to the winning
    # arm, the gripper factor would have stopped mattering and the paper's
    # "jointly necessary" sentence would be wrong.
    anchors = [v.get("SR_anchor_no_gripper_transform") for v in suites.values()]
    wins = [v.get("SR") for v in suites.values()]
    a.check(sec, "every anchor arm sits below half its own winning cell, which "
                 "is what makes the two corrections jointly necessary", True,
            all(an is not None and w is not None and an < 0.5 * w
                for an, w in zip(anchors, wins)),
            source="SR_anchor_no_gripper_transform vs SR, per suite")
    a.check(sec, "the anchor arms are at most 0.12", True,
            max(a_ for a_ in anchors if a_ is not None) <= 0.12,
            source="SR_anchor_no_gripper_transform")

    # --- the gripper channel, which is what distinguishes the two arms ---
    # Prose quotes the raw -> delivered transform per suite. If these ever came
    # out equal, the "winning" arm would be running the anchor's configuration
    # under a different label and the whole comparison would be vacuous.
    for suite, raw, sent in (("libero_spatial", 0.447, 0.102),
                             ("libero_object", 0.441, 0.115),
                             ("libero_goal", 0.717, -0.440),
                             ("libero_10", 0.587, -0.179)):
        v = suites.get(suite) or {}
        a.check(sec, f"{suite}: mean gripper command, raw then delivered, as "
                     f"quoted", [raw, sent],
                [v.get("gripper_raw_mean"), v.get("gripper_sent_mean")],
                source=v.get("source"))
    a.check(sec, "the anchor arm delivers the raw command unchanged on every "
                 "suite, which is what makes it the anchor", True,
            all(v.get("gripper_sent_equals_raw_in_anchor") for v in suites.values()),
            source="gripper_sent_mean == gripper_raw_mean in the none+none arm")
    # Episode length agrees with the success flag: successes terminate early, so
    # the winning arm logs strictly fewer per-step samples. This is the internal
    # consistency the original info["success"] bug destroyed -- under that bug an
    # episode could terminate on completion and still be scored a failure.
    for suite, win, anch in (("libero_spatial", 7025, 11000),
                             ("libero_object", 7864, 14000),
                             ("libero_goal", 7931, 13897),
                             ("libero_10", 20577, 26000)):
        v = suites.get(suite) or {}
        a.check(sec, f"{suite}: per-step gripper samples, winning then anchor "
                     f"arm, as quoted", [win, anch],
                [v.get("n_gripper_samples"), v.get("n_gripper_samples_anchor")],
                source=v.get("source"))
    a.check(sec, "the winning arm's episodes are shorter than the anchor's on "
                 "every suite, so success and episode length agree", True,
            all(v.get("episodes_shorter_than_anchor") for v in suites.values()),
            source="n_gripper_samples vs n_gripper_samples_anchor")

    # --- the prose the table replaces must actually be gone ---
    # The manuscript carried "The gate does not pass" for several revisions.
    # Leaving it in place while this table prints 0.74 would be the single worst
    # internal contradiction in the paper. The 5/50 = 0.10 cell is NOT pinned to
    # zero: limitation (v) still quotes it, correctly, as what the third attempt
    # measured, and deleting the history is not the fix.
    tex_l = TEX.read_text() if TEX.exists() else ""
    for stale in ("\\textbf{The gate does not pass.}",
                  "\\textbf{no number in this paper is conditioned on a rollout, "
                  "and we do not claim the decoder is validated at the rollout "
                  "level.}",
                  "It still does not reproduce the published SR"):
        a.check(sec, f"the superseded phrasing {stale[:52]!r}... is gone from "
                     f"the manuscript", 0, tex_l.count(stale),
                source="cot_faith_iclr.tex")
    # Retiring a hedge silently is the failure mode here: a reader who read the
    # submitted version should be told the sentence was withdrawn, not left to
    # notice its absence. The paper has to name it.
    a.check(sec, "the retired hedge is named as retired rather than deleted", 1,
            tex_l.count("``we do not claim the decoder is validated at the "
                        "rollout level'' hedge"),
            source="cot_faith_iclr.tex")
    a.check(sec, "both Section 6 and limitation (v) name the third cause as the "
                 "action decode rather than leaving it open", 2,
            tex_l.count("The third cause was the action decode itself"),
            source="cot_faith_iclr.tex")
    a.check(sec, "and the pre-fix table is still present as the baseline this "
                 "one is measured against", True,
            "\\label{tab:gate}" in tex_l and "\\label{tab:gate_pass}" in tex_l,
            source="cot_faith_iclr.tex")
    a.check(sec, "and the pre-fix table is still present as the baseline this "
                 "one is measured against", True,
            "\\label{tab:gate}" in tex_l and "\\label{tab:gate_pass}" in tex_l,
            source="cot_faith_iclr.tex")


def audit_p2_decode_equivalence(a: Audit) -> None:
    """Every number the token-selection paragraph quotes, against its artifact.

    This paragraph is the load-bearing one for the whole edit protocol: it is
    what bounds the published F's exposure to the decode defect the rollout
    harness turned out to have. It was quoted entirely in prose until now --
    the release carried the JSON but nothing asserted the manuscript against
    it, which is exactly the gap this script exists to close.

    The 66-vs-108 duplication is deliberate. The artifact counts unique prompts
    (66 = 12 originals + 54 edits); the manuscript's 108 counts record passes
    (54 records x 2). Both appear in the text, so both are recomputed here from
    the released per-record bins rather than either being taken on trust.
    """
    sec = "P2 token selection vs upstream predict_action"
    rel = ROOT / "results_v2" / "canonical_runs" / "p2_decode_equivalence"
    rep = load(rel / "p2_decode_equivalence.json")
    if not rep:
        a.check(sec, "p2_decode_equivalence.json is released", True, False,
                source=str(rel / "p2_decode_equivalence.json"))
        return
    src = f"{rel.name}/p2_decode_equivalence.json"

    a.check(sec, "the checkpoint measured is ecot-openvla-7b-bridge", True,
            "ecot-openvla-7b-bridge" in (rep.get("ckpt_path") or ""),
            source=src)
    a.check(sec, "12 samples, 66 prompts, 54 scored records, as printed",
            [12, 66, 54],
            [len({r["sample"] for r in rep.get("records") or []}),
             rep.get("n_prompts_compared"), rep.get("n_records")], source=src)
    # THE claim: raw generated ids byte-identical on every prompt. If this ever
    # drops below 1.0, P2's missing logit mask has started changing a selected
    # token and the paragraph's conclusion is void.
    a.check(sec, "the raw generated ids are byte-identical on 66 of 66 prompts",
            1.0, rep.get("frac_prompts_raw_generated_ids_identical"),
            tol=1e-12, source=src)
    a.check(sec, "no faithful flag differs on any of the 54 records, and no "
                 "family's F moves", [0, 0.0],
            [rep.get("n_faithful_flag_differs"), rep.get("worst_delta_F")],
            source=src)
    for fam, f in (("subject_swap", 1.00), ("location_swap", 0.90),
                   ("direction_flip", 1.00), ("gripper_flip", 0.50),
                   ("paraphrase_null", 0.92)):
        pf = dig(rep, "per_family", fam) or {}
        a.check(sec, f"per-family F as printed ({fam})", [f, f],
                [r2(pf.get("F_p2_decode")), r2(pf.get("F_upstream_decode"))],
                source=f"{src}:per_family.{fam}")

    # The confound the paper flagged as the larger risk, measured absent.
    lut = dig(rep, "lut_diagnostics") or {}
    a.check(sec, "both vocab sizes are 32,000 so the grid offset is 0",
            [32000, 32000, 0, True],
            [lut.get("model_vocab_size"), lut.get("processor_vocab_size"),
             lut.get("bin_index_offset_upstream_minus_p2"),
             lut.get("bin_index_offset_is_zero")], source=f"{src}:lut_diagnostics")
    a.check(sec, "the inversion back to bins is exact, and the mirror "
                 "reproduces P2's own infer_action", [0.0, True],
            [rep.get("max_bin_inversion_residual"),
             rep.get("mirror_reproduces_infer_action")], source=src)
    a.check(sec, "the checkpoint's grid has 255 distinct values with 1 "
                 "collapsed bin, which is why bins 254/255 share a value",
            [255, 1],
            [rep.get("grid_n_distinct_values"), rep.get("grid_n_collapsed_bins")],
            source=src)

    # The audit's own slice bug, now a measured field on both counts.
    a.check(sec, "upstream's span is offset by one on 66 of 66 prompts, and its "
                 "dim 0 is the grid top on 66 of 66", [66, 66],
            [rep.get("n_prompts_upstream_slice_offset_by_one"),
             rep.get("n_prompts_upstream_dim0_at_grid_top")], source=src)
    top = (rep.get("grid_n_distinct_values") or 0) - 1
    n = off = dim0 = 0
    for r in rep.get("records") or []:
        for pas in ("orig", "edit"):
            p2, up = r.get(f"bins_{pas}_p2"), r.get(f"bins_{pas}_upstream")
            if not p2 or not up:
                continue
            n += 1
            off += all(up[k + 1] == min(p2[k], top) for k in range(6))
            dim0 += int(up[0] == top)
    a.check(sec, "recomputed per record pass: the offset and the pinned dim 0 "
                 "hold on 108 of 108 decodes, as the manuscript prints",
            [108, 108, 108], [n, off, dim0],
            source=f"{src}:records[*].bins_{{orig,edit}}_{{p2,upstream}}")

    # The aux probe. It returned 0 measurements before the two calling-
    # convention bugs were fixed, so a regression there shows up as aux_n_probed
    # falling back to 0 rather than as a wrong number.
    a.check(sec, "the auxiliary CoT-in-the-loop probe ran on 12 samples and "
                 "found upstream's action changed on all of them", [12, 1.0],
            [rep.get("aux_n_probed"),
             rep.get("aux_cot_context_changes_upstream_action")], source=src)
    try:
        tex = TEX.read_text()
    except Exception:
        return
    a.check(sec, "the manuscript reports the aux probe as a weak positive "
                 "rather than as a rate", True,
            "We report that as a weak positive and no more" in tex,
            source="cot_faith_iclr.tex")
    a.check(sec, "and reconciles its 108/108 with the artifact's 66/66 rather "
                 "than printing two unexplained counts", True,
            "counted per unique prompt instead of per record pass" in tex,
            source="cot_faith_iclr.tex")


def audit_floor_invariance(a: Audit) -> None:
    """Does the headline's sign depend on which meaning-preserving family we
    nominate as the floor? A reviewer observed that paraphrase_null is the only
    family in the taxonomy that changes sequence length, so F_diff might be
    subtracting a length effect. The answer is worse than the objection: BOTH
    floors are meaning-preserving by our own validated judge, they disagree by
    more than either disagrees with the semantic mean, and the sign of F_diff
    flips between them on every configuration. This section asserts that the
    manuscript reports the weaker claim the data supports rather than either
    sign."""
    sec = "Floor invariance (is F_diff's sign a choice of floor?)"
    root = Path(__file__).resolve().parent.parent
    src = ("results_v2/canonical_runs/floor_invariance/floor_invariance.json")
    r = load(root / src)
    if r is None:
        a.check(sec, "the floor-invariance artifact is readable", True, None,
                source=src)
        return
    tex = TEX.read_text() if TEX.exists() else ""
    n = r.get("n_configs")

    a.check(sec, "all twelve configurations scored, so the rate is over the "
                 "whole benchmark rather than a subset", 12, n, source=src)
    a.check(sec, "F_diff is negative against paraphrase_null on every "
                 "configuration", n, r.get("n_negative_vs_paraphrase"),
            source=src)
    a.check(sec, "and positive against the length-exact floor on every "
                 "configuration, so the sign is not a property of the data",
            0, r.get("n_negative_vs_scramble"), source=src)
    a.check(sec, "the sign therefore flips between the two floors on all "
                 "twelve", n, r.get("n_sign_flips_between_floors"), source=src)

    # The decisive statistic. A sign flip alone would license "use the other
    # floor"; this is what forbids that move.
    a.check(sec, "the gap between the two meaning-preserving floors exceeds "
                 "the gap between the semantic mean and EITHER floor, on all "
                 "twelve -- which is what makes both floors unusable rather "
                 "than one of them right",
            n, r.get("n_null_spread_exceeds_margin"), source=src)

    # Both floors have to be meaning-preserving by the SAME judge, or the whole
    # argument collapses into "one of these families isn't a floor".
    j = load(root / "results_v2/canonical_runs/judge_edit_families/"
                    "judge_report.json") or {}
    fams = {f.get("family"): f for f in (j.get("per_family") or [])} \
        if isinstance(j.get("per_family"), list) else (j.get("per_family") or {})

    def preserved(name):
        blk = fams.get(name) or {}
        for k in ("meaning_preserved", "meaning_preserved_rate", "preserved"):
            if k in blk:
                return blk[k]
        return None

    for fam, want in (("paraphrase_null", 0.975), ("syntactic_scramble", 1.0)):
        a.check(sec, f"the same judge certifies {fam} as meaning-preserving, "
                     f"so both really are floors", want, preserved(fam),
                tol=0.001, source="judge_edit_families/judge_report.json")

    # And the manuscript must land on the weaker claim, not on either sign.
    a.check(sec, "the manuscript concedes the length confound explicitly "
                 "rather than defending the submitted floor", True,
            "we did not test this before submission" in tex,
            source="cot_faith_iclr.tex")
    a.check(sec, "and reports BOTH floors everywhere rather than swapping to "
                 "whichever one is favourable", True,
            "report both floors everywhere" in tex, source="cot_faith_iclr.tex")
    a.check(sec, "and states the resulting claim is weaker than the submitted "
                 "one, which is the thing a reader must not have to infer",
            True, "a weaker claim than our submitted one" in tex,
            source="cot_faith_iclr.tex")


def audit_fdir_null(a: Audit) -> None:
    """The constructive half. F_mag has no null it clears; F_dir does. This is
    the one instrument in the paper that separates signal from floor, so its
    calibration is the claim most worth attacking and most worth asserting --
    including the two results that cut against us: the no-CoT control failing
    (which is correct) and all three DeepThinkVLA checkpoints failing (which is
    a coverage loss we report rather than omit)."""
    sec = "F_dir null calibration (the one instrument with a measured floor)"
    root = Path(__file__).resolve().parent.parent
    src = "results_v2/canonical_runs/fdir_null/fdir_null.json"
    r = load(root / src)
    if r is None:
        a.check(sec, "the F_dir null artifact is readable", True, None,
                source=src)
        return
    tex = TEX.read_text() if TEX.exists() else ""
    per = {c["config"]: c for c in (r.get("per_config") or [])}

    a.check(sec, "all eleven calibratable configurations scored", 11,
            r.get("n_configs"), source=src)
    a.check(sec, "the null is built from more than one family, so the ceiling "
                 "is a ceiling rather than one arbitrary comparison", True,
            len(r.get("null_families") or []) >= 5, source=src)
    a.check(sec, "the manuscript's headline clearance rate matches the "
                 "artifact", 7, r.get("n_clearing_null"), source=src)
    a.check(sec, "and the manuscript states it", True,
            "$7$ of $11$ configurations clear their own null" in tex,
            source="cot_faith_iclr.tex")

    # The negative control is the load-bearing one: an instrument that "clears
    # its null" on a model trained without any CoT target would be measuring
    # something other than CoT.
    nc = per.get("ours_no-cot") or {}
    a.check(sec, "the no-CoT control does NOT clear its own floor, which is "
                 "the behaviour that makes the other seven interpretable",
            False, nc.get("clears_null"), source=src)
    a.check(sec, "and the manuscript reports the control's failure rather than "
                 "only the successes", True,
            "the no-CoT control sits \\emph{below} its own floor" in tex,
            source="cot_faith_iclr.tex")

    # The margins the paper quotes.
    for cfg, treat, ceil in (("ours_lora-r64", 0.779, 0.070),
                             ("ours_lora-r32", 0.589, 0.077),
                             ("ecot_bridge", 0.150, 0.040)):
        blk = per.get(cfg) or {}
        a.check(sec, f"{cfg}: the F_dir the manuscript quotes", treat,
                (blk.get("treatment") or {}).get("F_dir"), tol=0.0015,
                source=src)
        a.check(sec, f"{cfg}: the null ceiling the manuscript quotes", ceil,
                blk.get("null_ceiling"), tol=0.0015, source=src)

    # The adverse result. Reporting only the seven that clear would make this a
    # cross-family instrument, which it is not.
    dt = [k for k in per if k.startswith("deepthink")]
    a.check(sec, "all three DeepThinkVLA checkpoints fail the direction check, "
                 "so the instrument does not transfer to the second "
                 "architecture family", [False] * 3,
            [bool((per[k] or {}).get("clears_null")) for k in sorted(dt)],
            source=src)
    a.check(sec, "and the manuscript reports that as a negative row rather "
                 "than as coverage", True,
            "we report that as a negative row, not as coverage" in tex,
            source="cot_faith_iclr.tex")


def audit_collision_decomposition(a: Audit) -> None:
    """How much of F_mag is a decode-collision counter? The paper argued from
    the bimodality of the Delta distribution that F is robust to tau. The same
    bimodality implies something less flattering -- that F is close to
    1 - P(Delta = 0) -- and that is the reading the manuscript now leads with."""
    sec = "Collision decomposition (what F_mag actually counts)"
    root = Path(__file__).resolve().parent.parent
    src = ("results_v2/canonical_runs/collision_decomposition/"
           "collision_decomposition.json")
    r = load(root / src)
    if r is None:
        a.check(sec, "the collision-decomposition artifact is readable", True,
                None, source=src)
        return
    tex = TEX.read_text() if TEX.exists() else ""

    a.check(sec, "the decomposition is computed over the whole release rather "
                 "than a sample", 28443, r.get("n_scored_records"), source=src)
    a.check(sec, "and over enough cells that the correlation is not driven by "
                 "a handful of them", 324, r.get("n_cells"), source=src)
    a.check(sec, "the R^2 the manuscript quotes between F and 1-P(Delta=0)",
            0.926, r.get("r_squared"), tol=0.001, source=src)
    a.check(sec, "the number of cells where the two are identical", 80,
            r.get("n_cells_exactly_equal"), source=src)

    # The mechanism: the threshold does almost nothing because almost nothing
    # lands near it. If this fraction were large, F would be a real magnitude
    # measure and the whole paragraph would be wrong.
    a.check(sec, "under 10% of the records that move at all land below tau, "
                 "which is why the threshold is nearly inert", True,
            (r.get("frac_nonzero_below_tau") or 1.0) < 0.10, source=src)

    d = r.get("delta_distribution") or {}
    tot = r.get("n_scored_records") or 1
    a.check(sec, "the exactly-zero mass the manuscript quotes", 0.462,
            round((d.get("exactly_zero") or 0) / tot, 3), tol=0.001, source=src)

    a.check(sec, "the manuscript states the metric is close to a collision "
                 "counter rather than leaving the correlation unexplained",
            True, "not evidence that it is robust" in tex,
            source="cot_faith_iclr.tex")
    a.check(sec, "P(Delta=0) is reported as a first-class column, so a reader "
                 "can see both quantities without recomputing them", True,
            "\\label{tab:collision}" in tex, source="cot_faith_iclr.tex")
    a.check(sec, "and the manuscript says what this does NOT invalidate, "
                 "because over-withdrawing is its own error", True,
            "a collision rate is a well-defined and self-consistent quantity"
            in tex, source="cot_faith_iclr.tex")


def audit_arr_submission(a: Audit) -> None:
    """The ARR body is a second manuscript, so it needs the same treatment.

    Every number in cot_faith_arr.tex was retyped from the artifacts into a
    shorter document. That is precisely the operation this whole script exists
    to police, and doing it once by hand without a check would reintroduce the
    drift the full-length version took four revisions to eliminate. So the
    load-bearing figures are re-asserted against the same JSON, and the
    ARR-specific format constraints -- the ones that cause a desk reject rather
    than a bad review -- are asserted too.
    """
    sec = "ARR submission (cot_faith_arr.tex)"
    root = Path(__file__).resolve().parent.parent
    arr = root / "cot_faith_arr.tex"
    if not arr.exists():
        a.check(sec, "the ARR body exists", True, False, source=str(arr))
        return
    t = arr.read_text()
    # Every format check below asks what the ENGINE sees, so comments are
    # stripped first. The preamble documents each of these constraints in a
    # comment right where it is honoured ("no \baselinestretch override:
    # acl.sty owns it"), and a naive substring search reads its own
    # documentation as a violation.
    vis = re.sub(r"(?<!\\)%.*", "", t)

    # --- format constraints that are desk-reject conditions -----------------
    a.check(sec, "acl.sty is loaded with [review], which supplies the line "
                 "numbers and anonymization ARR requires", True,
            "\\usepackage[review]{acl}" in vis, source="cot_faith_arr.tex")
    a.check(sec, "geometry is NOT loaded before acl.sty, which loads it "
                 "itself -- a prior load is a fatal Option Clash", True,
            not re.search(r"\\usepackage(\[[^\]]*\])?\{[^}]*geometry", vis),
            source="cot_faith_arr.tex")
    for pkg in ("authblk", "titlesec"):
        a.check(sec, f"{pkg} is not loaded (acl.sty owns the title and "
                     f"section formatting; overriding it fails the check)",
                True, f"{{{pkg}}}" not in vis, source="cot_faith_arr.tex")
    a.check(sec, "no \\baselinestretch override, which changes the page count "
                 "the limit is enforced on", True,
            "baselinestretch" not in vis, source="cot_faith_arr.tex")
    a.check(sec, "Limitations is an UNNUMBERED section, as ARR requires",
            True, "\\section*{Limitations}" in t, source="cot_faith_arr.tex")
    a.check(sec, "Ethics Statement is an unnumbered section", True,
            "\\section*{Ethics Statement}" in t, source="cot_faith_arr.tex")
    a.check(sec, "the ACL bibliography style is used", True,
            "\\bibliographystyle{acl_natbib}" in t, source="cot_faith_arr.tex")

    # \aclfinalcopy does not exist in current acl.sty -- the option system
    # replaced it. Leaving it in is a hard build failure, and it is the kind of
    # thing copied in from an older template.
    a.check(sec, "no \\aclfinalcopy, which current acl.sty does not define",
            True, "\\aclfinalcopy" not in t, source="cot_faith_arr.tex")

    # inconsolata ships in texlive-fonts-extra, which the build image does not
    # install. It cost bolt gg5sr9ndka an entire job for a \texttt font. Any
    # package outside the recommended set is the same bet, so the one that
    # actually bit is pinned here rather than left to be re-added.
    a.check(sec, "no inconsolata: it is not in the build image's TeX tree, "
                 "and a cosmetic font that fails the build is not a trade "
                 "worth making", True,
            "inconsolata" not in vis, source="cot_faith_arr.tex")

    # --- double-blind -------------------------------------------------------
    for needle in ("sharpguard", "ICLR 2026", "yudizhang"):
        # Skip LaTeX comments: they do not render, and the build provenance
        # header legitimately names the source file.
        visible = "\n".join(ln for ln in t.splitlines()
                            if not ln.lstrip().startswith("%"))
        a.check(sec, f"no '{needle}' in rendered text (double-blind)", True,
                needle.lower() not in visible.lower(),
                source="cot_faith_arr.tex, comments excluded")
    a.check(sec, "the appendix is anonymized too, since it ships in the same "
                 "PDF", True,
            not any(s in (root / "arr_appendix.tex").read_text().lower()
                    for s in ("sharpguard", "iclr 2026"))
            if (root / "arr_appendix.tex").exists() else None,
            source="arr_appendix.tex")

    # --- no label may be defined in both documents --------------------------
    # The body promotes two floats out of the appendix. If the generator ever
    # stops removing them, LaTeX defines the label twice and the number it
    # prints for \ref becomes whichever came last -- a WARNING, not an error,
    # so a paper that points readers at the wrong table builds cleanly.
    if (root / "arr_appendix.tex").exists():
        body_labels = set(re.findall(r"\\label\{([^}]*)\}", vis))
        ap_labels = set(re.findall(
            r"\\label\{([^}]*)\}", (root / "arr_appendix.tex").read_text()))
        dup = sorted(body_labels & ap_labels)
        a.check(sec, "no \\label is defined in both the body and the appendix, "
                     "which would make \\ref resolve unpredictably", 0,
                len(dup), source=f"duplicates: {dup}")

    # --- the body's floats must be able to land in the body -----------------
    # Six floats in eight two-column pages does not fit LaTeX's defaults, and
    # the way it fails is silent. \dbltopfraction caps a full-width float at
    # 0.7 x \textheight; fig:taxonomy is just over that, so it was refused, and
    # because floats are placed in order every later figure* queued behind it.
    # The queue drained past the bibliography and all four body figures printed
    # on pages 41-44 of the 49-page PDF -- numbered, cross-referenced, and
    # thirty-five pages from the text arguing from them.
    #
    # Nothing upstream caught it: the engine emits no warning for a deferred
    # float, and the 8-page metric is measured on the body-only build, whose
    # \end{document} flushes the queue at page 8. bolt/run_arr_build.sh now
    # reads float pages out of .aux and fails on this; these checks assert the
    # source-side conditions that let a build succeed in the first place, so
    # the two together cover both "did it regress" and "will it recur".
    floatp = {"topfraction": 0.9, "dbltopfraction": 0.9, "textfraction": 0.05}
    floatp_got = {}
    for cmd, want in floatp.items():
        m = re.search(r"\\renewcommand\{\\" + cmd + r"\}\{([\d.]+)\}", vis)
        got = float(m.group(1)) if m else None
        floatp_got[cmd] = got if got is not None else 0.7   # LaTeX's default
        # A float that must fit under \dbltopfraction gets no second chance,
        # so the direction of the inequality matters: top fractions must be at
        # least this generous, \textfraction at most this demanding.
        ok = got is not None and (got >= want if cmd != "textfraction"
                                  else got <= want)
        a.check(sec, f"\\{cmd} is relaxed to {'>=' if cmd != 'textfraction' else '<='}"
                     f" {want}, or the body's floats defer into the appendix",
                True, ok, source=f"cot_faith_arr.tex: {cmd}={got}")

    # \clearpage before the bibliography is the backstop: if a float is still
    # queued when the body ends, this is what keeps it out of the appendix.
    bib = vis.find(r"\bibliographystyle")
    a.check(sec, "\\clearpage precedes \\bibliographystyle, so any float still "
                 "queued at the end of the body flushes before the appendix "
                 "rather than into it", True,
            bib > 0 and r"\clearpage" in vis[max(0, bib - 400):bib],
            source="cot_faith_arr.tex")

    # Each body float must be small enough to be placeable at all. A figure*
    # taller than \dbltopfraction x \textheight can never be set as a top
    # float, and LaTeX will defer it forever without saying so. \textheight is
    # read from the build log rather than assumed, because acl.sty sets it via
    # geometry and a style update would move it.
    logp = Path("/tmp/arrbuild/arr/cot_faith_arr.log")
    th = None
    if logp.exists():
        m = re.search(r"\\textheight=([\d.]+)pt",
                      logp.read_text(errors="replace"))
        th = float(m.group(1)) if m else None
    figs = root / "figures"
    for block in re.findall(r"\\begin\{figure(\*?)\}(.*?)\\end\{figure\*?\}",
                            vis, re.S):
        star, body_ = block
        m = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body_)
        if not (m and th):
            continue
        pdf = figs / m.group(1)
        if not pdf.exists():
            a.check(sec, f"body figure {m.group(1)} exists", True, False,
                    source=str(pdf))
            continue
        box = re.search(rb"/MediaBox\s*\[([^\]]*)\]", pdf.read_bytes())
        x0, y0, x1, y1 = (float(v) for v in box.group(1).split())
        w, h = x1 - x0, y1 - y0
        # \textwidth in the ACL style; a \columnwidth figure gets half of it
        # less the 0.6cm gutter. Scaled height is what competes for the page.
        avail = 453.6 if star else (453.6 - 17.0) / 2
        scaled = h * avail / w
        # The governing fraction is the one this float's own environment obeys:
        # figure* competes for the double-column top area, figure for the
        # single-column one. Read from the source, not restated, so relaxing
        # one of them cannot leave this check asserting the other's value.
        cap = floatp_got["dbltopfraction" if star else "topfraction"] * th
        a.check(sec, f"{m.group(1)} at its include width is {scaled:.0f}pt "
                     f"tall, inside the {cap:.0f}pt a top float may occupy "
                     f"(caption excluded)", True, scaled < cap,
                source=f"{w:.0f}x{h:.0f}pt native, textheight {th:.0f}pt")
        # And the other direction: a figure authored wider than the slot it is
        # included in scales its type down with it. Matplotlib font sizes are
        # absolute points, so a figure authored w points wide and included at
        # `avail` sets every label on the page at avail/w of the size written in
        # its generator -- present in the PDF, unreadable on paper, and invisible
        # to the fits-on-the-page check above.
        #
        # This asserts the scale factor directly. It replaces an aspect-ratio
        # proxy that was both loose and wrong: the proxy waived fig14 at 2.2:1
        # on a note claiming it "renders legibly at 50% -- checked against the
        # built PDF", and it did not. Measured off the built page, fig14's tick
        # and value labels arrived at 3.5pt and its legend at 3.0pt, which is
        # smaller than the fig4 defect the proxy existed to catch. Aspect ratio
        # was never the quantity; it only correlated with it. Both figures are
        # now authored at their include width, so the real invariant is
        # available to state, and stating it is what stops the next figure from
        # being drawn at 7in and dropped into a 3in column.
        scale = avail / w
        a.check(sec, f"{m.group(1)} is included at {100 * scale:.0f}% of its "
                     f"authored width, so the font sizes in its generator are "
                     f"the sizes a reader gets on the page",
                True, 0.94 <= scale <= 1.06,
                source=f"{w:.1f}pt authored, {avail:.1f}pt available; the fix "
                       f"is to re-author at the include width, not to scale")

    # --- long \texttt paths must be breakable in two columns ----------------
    # \texttt is unbreakable and the ARR column is 3.1in. The first build put a
    # 347pt overfull box on a 54-character artifact filename -- text running
    # off the page, which single-column at 6.5in never showed. The generator
    # inserts \allowbreak at path separators; this asserts it stayed on, since
    # the failure is invisible in the .tex and only appears in the PDF.
    if (root / "arr_appendix.tex").exists():
        ap = (root / "arr_appendix.tex").read_text()
        long_unbroken = [m for m in re.findall(r"\\texttt\{([^{}]{24,})\}", ap)
                         if "allowbreak" not in m
                         and re.search(r"(/|\\_|,)", m)]
        a.check(sec, "every long \\texttt path in the appendix carries "
                     "\\allowbreak, so it can wrap instead of running off the "
                     "column", 0, len(long_unbroken),
                source=f"arr_appendix.tex: {long_unbroken[:2]}")

    # --- the appendix must be current --------------------------------------
    # A stale appendix ships numbers the manuscript no longer makes. It is
    # generated, so staleness is detectable rather than a matter of care.
    gen = root / "scripts" / "build_arr_appendix.py"
    a.check(sec, "the appendix is generated from the full-length source "
                 "rather than maintained as a second copy", True, gen.exists(),
            source=str(gen))
    if gen.exists():
        rc = subprocess.run([sys.executable, str(gen), "--check"],
                            cwd=root, capture_output=True, text=True)
        a.check(sec, "and the committed appendix is current (regenerating it "
                     "produces no diff)", 0, rc.returncode,
                source=f"{gen.name} --check: {rc.stdout.strip()}"
                       f"{rc.stderr.strip()}")

    # --- the numbers, re-asserted against the artifacts ---------------------
    fi = load(root / "results_v2/canonical_runs/floor_invariance/"
                     "floor_invariance.json") or {}
    cd = load(root / "results_v2/canonical_runs/collision_decomposition/"
                     "collision_decomposition.json") or {}
    fd = load(root / "results_v2/canonical_runs/fdir_null/fdir_null.json") or {}
    # tab:directional's cells (means and seed stds) come from here.
    d = load(root / "results_v2" / "derived_metrics.json") or {}

    a.check(sec, "the ARR abstract's 12/12 sign-flip claim matches the "
                 "artifact", 12, fi.get("n_sign_flips_between_floors"),
            source="floor_invariance.json")
    a.check(sec, "the ARR abstract's R^2 for the collision decomposition "
                 "rounds to the quoted 0.93", "0.93",
            f"{cd.get('r_squared', 0):.2f}", source="collision_decomposition.json")
    a.check(sec, "the ARR abstract's cell count", 324, cd.get("n_cells"),
            source="collision_decomposition.json")
    a.check(sec, "the ARR abstract's F_dir clearance rate", 7,
            fd.get("n_clearing_null"), source="fdir_null.json")

    # --- the two figures that carry text and rates, not just bars -----------
    # fig1 prints real generator strings and per-family judge rates, and fig13
    # prints its R^2 and counts in the panel titles. Both read their own JSON,
    # so a drifting artifact would silently redraw them while every number in
    # the prose stayed put. Assert the figure scripts still read the artifact
    # the caption describes, and that the captions' own numbers match it.
    figs = root / "figures"
    # Every artifact each script reads, not just its primary one: fig1 draws
    # its strings from edit_examples but its rates from the judge report, and
    # a check scoped to edit_examples alone passed a deliberately injected
    # hardcoded 0.975 because that value lives in the other file.
    for script, arts in [
            ("gen_fig1_task_examples.py",
             ["edit_examples/edit_examples.json",
              "judge_edit_families/judge_report.json"]),
            ("gen_fig13_collision.py",
             ["collision_decomposition/collision_decomposition.json"]),
            # fig14 reads derived_metrics.json, which is not under
            # canonical_runs/; the loop's path join handles the ../ escape.
            ("gen_fig14_noise_hierarchy.py", ["../derived_metrics.json"])]:
        srcf = figs / script
        body = srcf.read_text() if srcf.exists() else ""
        for artifact in arts:
            # Match on the filename, not the directory: fig14 reads
            # derived_metrics.json from one level up, where the directory
            # component is ".." and matching it asserted nothing.
            stem = artifact.rsplit("/", 1)[-1]
            a.check(sec, f"{script} reads {stem}", True, stem in body,
                    source=str(srcf))
        # R1 item 5b: no figure may hardcode a reported quantity. A precision
        # rule cannot express that -- these scripts are laid out in inches, so
        # 0.078 is a panel margin and 0.125 is a judge rate, and both have
        # three decimals. Check the property directly instead: collect every
        # number the artifact actually carries, and assert that no literal in
        # the drawing code is one of them. That flags a copied value and
        # ignores geometry, which is exactly the distinction R1 asked for.
        drawing = re.sub(r'"""[\s\S]*?"""', "", body)      # drop docstrings
        drawing = "\n".join(l for l in drawing.splitlines()
                             if not l.lstrip().startswith("#"))
        arts_json = []
        for artifact in arts:
            art = load(root / "results_v2" / "canonical_runs" / artifact)
            a.check(sec, f"{script}'s artifact {artifact} is present", True,
                    art is not None, source=artifact)
            arts_json.append(art)

        # Only values distinctive enough that a match implies copying. A bar
        # width of 0.72 and a percent conversion by 100 both appear in the
        # artifact by coincidence, and flagging them made the check noise. A
        # rate printed to 3dp, or a record count, does not collide by accident:
        # this paper reports every rate to 3dp and every count exactly.
        def _distinctive(x):
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                return False
            if isinstance(x, int) or float(x).is_integer():
                n = abs(int(x))
                # 100/1000 are unit conversions, not findings.
                return n >= 100 and str(n).strip("0") != "1"
            return len(f"{x!r}".partition(".")[2]) >= 3

        reported = set()

        def _walk(node):
            if isinstance(node, dict):
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for v in node:
                    _walk(v)
            elif _distinctive(node):
                reported.add(f"{float(node):.6g}")

        for art in arts_json:
            _walk(art)
        lits = {f"{float(x):.6g}"
                for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])",
                                    drawing)
                if _distinctive(float(x))}
        collisions = sorted(lits & reported)
        a.check(sec, f"{script} hardcodes no value its own artifact reports",
                [], collisions,
                source=f"{len(reported)} values across {len(arts)} artifact(s)")

    # fig1's caption quotes judge rates and fig13's quotes counts from the
    # collision artifact. Read the expectation out of the caption and the
    # observation out of the JSON -- never the other way round.
    jr = load(root / "results_v2/canonical_runs/judge_edit_families/"
                     "judge_report.json") or {}
    pf = jr.get("per_family", {})

    # The caption quotes a DIFFERENT judge field for each of the two families
    # it calls out -- syntactic_scramble is remarkable for being judged
    # meaning-PRESERVING, adversarial_plausible for being judged not
    # PLAUSIBLE. Reading both from meaning_preserved_rate would have compared
    # the caption's 0.125 against a 0.000 that is a true statement about a
    # different quantity.
    for fam, field, want in [
            ("syntactic_scramble", "meaning_preserved_rate", "1.000"),
            ("adversarial_plausible", "plausible_rate", "0.125")]:
        a.check(sec, f"fig:taxonomy's caption quotes the judge's {fam} "
                     f"{field}", want,
                f"{pf.get(fam, {}).get(field, -1):.3f}",
                source="judge_report.json")
        a.check(sec, f"and the ARR caption prints that {fam} rate",
                True, f"${want}$" in t, source="cot_faith_arr.tex")
    # fig:noise's caption states three derived ratios/counts in prose. Each is
    # recomputed here from derived_metrics.json rather than trusted, because a
    # caption is exactly where an arithmetic slip goes unnoticed.
    nh = d.get("noise_hierarchy", {})
    tr = d.get("training_replicate", {})
    a.check(sec, "fig:noise's caption: retraining is 7.5x the sampling seed",
            "7.5", f"{nh.get('training_run_diff_pp', 0) / nh.get('sampling_std_pp', 1):.1f}",
            source="noise_hierarchy.training_run_diff_pp / sampling_std_pp")
    a.check(sec, "and the between-variant spread is only 1.2x the retraining",
            "1.2", f"{nh.get('spread_over_training_run', 0):.1f}",
            source="noise_hierarchy.spread_over_training_run")
    wil = max([(w[1] - w[0]) / 2
               for mv in d.get("models", {}).values()
               for fv in mv.get("families", {}).values()
               for w in [fv.get("F_mag_wilson")] if w] or [0])
    per_max = tr.get("F_max_abs_diff_per_pair", {})
    a.check(sec, "and 6 of 7 replicate pairs move their worst family further "
                 "than the widest Wilson half-width in the release",
            (6, 7), (sum(1 for v in per_max.values() if v > wil),
                     len(per_max)),
            source=f"widest Wilson half-width = {wil:.4f}")

    # fig:dissociation's caption gained two numbers when the figure was
    # promoted to full width and its third panel became legible enough to be
    # worth describing. Both are recomputed, and both are asserted to appear
    # in the caption -- a value that is right in the JSON and absent from the
    # text is the same defect as one that is wrong.
    anf = d.get("attention_noise_floor", {})
    a.check(sec, "fig:dissociation's caption: the run-to-run noise band is "
                 "1.45 pp", "1.45", f"{anf.get('abs_diff_pp', 0):.2f}",
            source="attention_noise_floor.abs_diff_pp")
    nrm = [mv["F_bar_norm_ceiling"] for mv in d.get("models", {}).values()
           if mv.get("F_bar_norm_ceiling")]
    a.check(sec, "and the ceiling-normalized spread is 1.9x", "1.9",
            f"{max(nrm) / min(nrm):.1f}" if nrm else "n/a",
            source=f"max/min F_bar_norm_ceiling over {len(nrm)} models")
    for lit in ("$1.45$\\,pp", "$1.9\\times$"):
        a.check(sec, f"and the caption prints {lit}", True, lit in t,
                source="cot_faith_arr.tex")

    # The figure that made this caption necessary: 797x248pt at \columnwidth
    # renders 68pt tall and sets its tick labels at about 3pt. Aspect ratio,
    # not height, is what decides whether a figure can live in a 218pt column.
    m = re.search(r"\\includegraphics\[width=\\(\w+)\]"
                  r"\{fig4_dissociation\.pdf\}", vis)
    a.check(sec, "fig:dissociation is set at \\textwidth: at 3.2:1 it is "
                 "illegible in a single column", "textwidth",
            m.group(1) if m else None, source="cot_faith_arr.tex")

    for want, got, what in [
            ("28{,}443", f"{cd.get('n_scored_records', 0):,}".replace(",", "{,}"),
             "scored records"),
            ("324", str(cd.get("n_cells")), "cells"),
            ("0.926", f"{cd.get('r_squared', 0):.3f}", "R^2")]:
        a.check(sec, f"fig:collision's caption quotes the artifact's {what}",
                want, got, source="collision_decomposition.json")
        a.check(sec, f"and the ARR caption prints that {what}", True,
                want in t, source="cot_faith_arr.tex")

    # The ARR body states the audit's own claim count in three places
    # (abstract, contributions, release section). They must agree with each
    # other and with the full-length version, or the submission advertises a
    # number this script does not produce.
    # The ARR body states the audit's own claim count in three places
    # (abstract, contributions, release section), each wrapped differently --
    # bare, \textbf{...}, $...$. Strip the wrappers before matching, or the
    # check silently passes on the two it can parse and ignores the third.
    plain = re.sub(r"\\(?:textbf|emph|texttt)\{([^{}]*)\}", r"\1", vis)
    plain = plain.replace("$", "")
    counts = {int(re.sub(r"[^\d]", "", x)) for x in
              re.findall(r"(?:checks|asserting|asserts) ([\d{},]+) claims",
                         plain)}
    a.check(sec, "the claim count appears in all three places it is promised "
                 "(abstract, contributions, release)", 3,
            len(re.findall(r"(?:checks|asserting|asserts) [\d{},]+ claims",
                           plain)), source="cot_faith_arr.tex")
    m = re.search(r"checks \$([\d{},]+)\$ claims", TEX.read_text())
    iclr_n = int(re.sub(r"[^\d]", "", m.group(1))) if m else None
    a.check(sec, "the claim count is stated consistently across the ARR body "
                 "and matches the full-length version",
            {iclr_n}, counts, source="cot_faith_arr.tex vs cot_faith_iclr.tex")

    # Table tab:floors is retyped from the artifact, so every cell is checked.
    # This is the single most drift-prone thing in the ARR document.
    #
    # Scoped to the tab:floors environment, not matched against the whole file.
    # Both body tables now use \texttt row labels and they share label names
    # (`r=64`, `no-CoT`, `ECoT-bridge`), so a file-wide regex would silently
    # pick up tab:directional's rows and compare direction-aware numbers
    # against floor_invariance.json.
    def _table_body(label: str) -> str:
        i = t.find("\\label{" + label + "}")
        if i < 0:
            return ""
        j = t.rfind("\\begin{table}", 0, i)
        return t[j:i] if j >= 0 else ""

    floors_tex = _table_body("tab:floors")
    a.check(sec, "tab:floors is present in the ARR body", True,
            bool(floors_tex), source="cot_faith_arr.tex")
    rows = {c["config"]: c for c in (fi.get("per_config") or [])}
    tex_rows = dict(re.findall(
        r"\\texttt\{(no-CoT|r=8|r=16|r=32|r=64|data-50A|data-50B|ECoT-bridge|"
        r"DT base|DT SFT|DT RL|Bridge-4k)\}(?:\$\^\\dagger\$)?"
        r"\s*&([^\\]*)\\\\", floors_tex))
    key = {"no-CoT": "ours_no-cot", "r=8": "ours_lora-r8",
           "r=16": "ours_lora-r16", "r=32": "ours_lora-r32",
           "r=64": "ours_lora-r64", "data-50A": "ours_data-50A",
           "data-50B": "ours_data-50B", "ECoT-bridge": "ecot_bridge",
           "DT base": "deepthink_base", "DT SFT": "deepthink_sft",
           "DT RL": "deepthink_rl", "Bridge-4k": "bridge_subset_4k"}
    a.check(sec, "tab:floors has a row for every calibrated configuration",
            12, len(tex_rows), source="cot_faith_arr.tex")
    for label, cells in sorted(tex_rows.items()):
        r = rows.get(key.get(label, ""))
        if r is None:
            a.check(sec, f"tab:floors row '{label}' names a real "
                         f"configuration", True, False, source="cot_faith_arr.tex")
            continue
        got = [float(x) for x in re.findall(r"[-+]?\d*\.\d+", cells)]
        p = r["floor_paraphrase_null"]["F"]
        s = r["floor_syntactic_scramble"]["F"]
        dp, ds = r["f_diff_vs_paraphrase"], r["f_diff_vs_scramble"]
        # The trailing |para - scram| column is the one the caption calls
        # decisive, so it is derived here rather than copied from the row.
        want = [p, s, r["f_bar_semantic"], dp, ds, abs(p - s)]
        a.check(sec, f"tab:floors row '{label}' matches the artifact to 3dp",
                [round(x, 3) for x in want], [round(x, 3) for x in got],
                source="floor_invariance.json")
        a.check(sec, f"tab:floors row '{label}': the printed floor spread "
                     f"exceeds the larger |F_diff|, as the caption claims",
                True, abs(p - s) > max(abs(dp), abs(ds)),
                source=f"|{p:.3f}-{s:.3f}| vs max(|{dp:.3f}|,|{ds:.3f}|)")
    a.check(sec, "tab:floors' summary row states the 12/12 counts the caption "
                 "argues from", True,
            bool(re.search(r"12/12.*12/12.*12/12", floors_tex, re.S)),
            source="cot_faith_arr.tex")

    # tab:directional prints seed stds alongside every mean. They are quoted
    # to 3dp with the leading zero dropped ($.006$), which no other table does,
    # so they get their own parse rather than reusing the one above.
    dir_tex = _table_body("tab:directional")
    a.check(sec, "tab:directional is present in the ARR body", True,
            bool(dir_tex), source="cot_faith_arr.tex")
    dkey = {"ECoT-bridge": "ecot-bridge", "r=64": "ours-r64",
            "r=16": "ours-r16", "data-50A": "ours-data50A",
            "r=8": "ours-r8", "r=32": "ours-r32",
            "data-50B": "ours-data50B", "no-CoT": "ours-no-cot"}
    drows = re.findall(
        r"\\texttt\{([^}]+)\}\s*&(.*?)\\\\", dir_tex, re.S)
    a.check(sec, "tab:directional has all 8 leaderboard rows", 8, len(drows),
            source="cot_faith_arr.tex")
    for label, cells in drows:
        m = dkey.get(label)
        fam_ = dig(d, "models", m, "families", "direction_flip") if m else None
        if fam_ is None:
            a.check(sec, f"tab:directional row '{label}' names a real model",
                    True, False, source="cot_faith_arr.tex")
            continue
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+", cells)]
        # F_mag, sd, F_dir, sd, cos -- the integer rank is not matched by the
        # decimal pattern, so it does not appear here.
        want = [fam_["F_mag"], fam_["F_mag_std"], fam_["F_dir"],
                fam_["F_dir_std"], fam_["cos_xyz"]]
        a.check(sec, f"tab:directional row '{label}' matches the artifact "
                     f"(mean and seed std) to 3dp",
                [round(x, 3) for x in want], [round(x, 3) for x in nums],
                source=f"models['{m}'].families.direction_flip")
    a.check(sec, "tab:directional's error bars are 3-seed stds, as the caption "
                 "says", {3},
            {dig(d, "models", m, "families", "direction_flip", "n_runs")
             for m in dkey.values()},
            source="models[*].families.direction_flip.n_runs")

    # tab:fdirnull promotes the constructive result out of prose. Eleven rows
    # times five cells retyped by hand is the highest-drift thing in the
    # document, so every cell is read back out of the .tex and matched against
    # fdir_null.json -- the same treatment tab:floors gets, for the same reason.
    fdn_tex = _table_body("tab:fdirnull")
    a.check(sec, "tab:fdirnull is present in the ARR body", True,
            bool(fdn_tex), source="cot_faith_arr.tex")
    if fdn_tex:
        nice = {"ours_no-cot": "no-CoT", "ours_lora-r8": "r=8",
                "ours_lora-r16": "r=16", "ours_lora-r32": "r=32",
                "ours_lora-r64": "r=64", "ours_data-50A": "data-50A",
                "ours_data-50B": "data-50B", "ecot_bridge": "ECoT-bridge",
                "deepthink_sft": "DT SFT", "deepthink_rl": "DT RL",
                "deepthink_base": "DT base"}
        # The ceiling family is abbreviated in the table to fit the column, so
        # what is checked is the artifact key -> printed stem mapping rather
        # than an equality. An abbreviation that names the wrong family is
        # exactly as wrong as a wrong number, and harder to notice.
        abbrev = {"cross_task_swap": "cross_task", "verb_swap": "verb_swap",
                  "negation": "negation", "paraphrase_null": "paraphrase"}
        got_rows = {}
        for m in re.finditer(
                r"\\texttt\{([^}]+)\}\s*&\s*([\d.]+)\s*&\s*(\d+)\s*&\s*"
                r"([\d.]+)\s*&[^&]*?\\emph\{([A-Za-z\\_]+)\}\}\s*&\s*"
                r"(?:\\textbf\{)?([\d.]+|-+)", fdn_tex):
            got_rows[m.group(1)] = (m.group(2), int(m.group(3)), m.group(4),
                                    m.group(5).replace("\\_", "_"),
                                    m.group(6))
        a.check(sec, "all 11 calibrated configurations appear in tab:fdirnull",
                11, len(got_rows), source=f"parsed: {sorted(got_rows)}")
        fdn = load(ROOT / "results_v2/canonical_runs/fdir_null/"
                          "fdir_null.json") or {}
        for c in (fdn.get("per_config") or []):
            label = nice.get(c["config"], c["config"])
            r = c.get("ratio")
            want = (f"{c['treatment']['F_dir']:.3f}",
                    int(c["treatment"]["n"]),
                    f"{c['null_ceiling']:.3f}",
                    abbrev.get(c["null_ceiling_family"],
                               c["null_ceiling_family"]),
                    f"{r:.1f}" if r is not None else "---")
            a.check(sec, f"tab:fdirnull row {label} matches fdir_null.json "
                         f"(F_dir, N, ceiling, ceiling family, ratio)",
                    want, got_rows.get(label), source="fdir_null.json")
        # The rule inside the table is load-bearing: it separates clears from
        # fails, so it has to fall exactly where the artifact says. Seven above
        # it, four below, and the ablation among the four.
        order = [m.group(1) for m in re.finditer(r"\\texttt\{([^}]+)\}",
                                                 fdn_tex)]
        clears = {nice.get(c["config"], c["config"])
                  for c in (fdn.get("per_config") or []) if c["clears_null"]}
        a.check(sec, "tab:fdirnull's midrule separates the configurations that "
                     "clear their null from those that do not, in that order",
                [True] * len(clears) + [False] * (len(order) - len(clears)),
                [lab in clears for lab in order],
                source=f"clears per fdir_null.json: {sorted(clears)}")
        a.check(sec, "and the caption states that the no-CoT control failing "
                     "is the designed behaviour rather than missing coverage",
                True, "is the result, not a gap" in t,
                source="cot_faith_arr.tex")
        # The two aggregations differ, and saying so is what stops a reviewer
        # reading tab:directional's 0.120 against this table's 0.150 as drift.
        # Assert both that they do differ and that the caption explains why.
        ecot_pooled = next(
            (c["treatment"]["F_dir"] for c in (fdn.get("per_config") or [])
             if c["config"] == "ecot_bridge"), None)
        ecot_mean = dig(d, "models", "ecot-bridge", "families",
                        "direction_flip", "F_dir")
        a.check(sec, "the pooled rate and the 3-seed mean of F_dir really are "
                     "different numbers for ECoT-bridge, so the caption has to "
                     "explain the difference rather than leave it to be found",
                True,
                ecot_pooled is not None and ecot_mean is not None
                and f"{ecot_pooled:.3f}" != f"{ecot_mean:.3f}",
                source=f"pooled {ecot_pooled}, 3-seed mean {ecot_mean}")
        a.check(sec, "and the caption says N is pooled across the sampling "
                     "seeds", True,
                "pooled across the three sampling seeds" in t,
                source="cot_faith_arr.tex")
        a.check(sec, "the clearance count the prose states is the count this "
                     "table shows", (7, 11),
                (len(clears), len(fdn.get("per_config") or [])),
                source="fdir_null.json")

        # The 6.2--11.1x range describes six of the seven clearing configs, not
        # all seven: ECoT-bridge clears at 3.8x, well below the bottom of it.
        # An unscoped "7 of 11 clear at 6.2--11.1x" therefore overstates one
        # row by 1.6x, and it is exactly the row the rest of the paper leans on.
        # So: recompute which configs the range does cover, and require every
        # sentence that prints the range to name ECoT-bridge's ratio too.
        ratios = {c["config"]: c.get("ratio") for c in fdn["per_config"]
                  if c.get("clears_null")}
        lo, hi = 6.2, 11.1
        inside = sorted(k for k, v in ratios.items()
                        if v is not None and lo <= round(v, 1) <= hi)
        outside = sorted(k for k, v in ratios.items()
                         if v is None or not lo <= round(v, 1) <= hi)
        a.check(sec, "the 6.2--11.1x range covers our six LoRA/data variants "
                     "and no other clearing config",
                (6, ["ecot_bridge"]), (len(inside), outside),
                source="fdir_null.json per_config[*].ratio, clears_null only")
        a.check(sec, "ECoT-bridge's ratio really is outside that range, which "
                     "is why the range cannot be quoted for all 7", "3.8",
                f"{ratios.get('ecot_bridge'):.1f}", source="fdir_null.json")
        unscoped = []
        for m in re.finditer(r"\$6\.2\$?--\$?\\?mathbf\{?11\.1|6\.2.{0,12}11\.1",
                             t):
            near = t[m.start():m.end() + 260]
            if "3.8" not in near:
                unscoped.append(t[max(0, m.start() - 60):m.end() + 60])
        a.check(sec, "every sentence that quotes 6.2--11.1x also gives "
                     "ECoT-bridge's 3.8x, so the range is never read as "
                     "covering all 7 clearing configurations", [], unscoped,
                source="cot_faith_arr.tex")


def audit_rollout_insuite(a: Audit) -> None:
    """Limitation (v)'s 0/40: assert the bound, and assert it stays a bound.

    This is the one artifact in the release whose *headline* is a zero, and a
    zero is the easiest number in the world to misreport in the flattering
    direction. Two distinct misreadings are possible and both are guarded here.

    The first is upward drift: quoting a rollout SR anywhere, or letting the
    0/40 become a 0/N with a larger N than was run. The second is the more
    dangerous one, because it reads as a *finding*: reporting the edit-induced
    DSR as 0 and calling it evidence that editing the CoT does not change task
    completion. DSR is 0 by construction when the unedited control never
    succeeds -- it is undefined, not null -- and the harness records that as
    precondition_met=false. So this function asserts the flag is false and that
    the manuscript says "undefined" rather than reporting a difference.

    It also pins the two facts that make the zero attributable to the
    checkpoint rather than to us: the rollout suite is the suite the checkpoint
    was trained on (round 1's confound, released under
    rollout_edit_outofsuite_round1/ and explicitly not citable), and every
    generated CoT parsed, so the prompt-side harness was working throughout.
    """
    sec = "In-suite paired rollout (limitation v: a bound, not a measurement)"
    base = ROOT / "results_v2" / "canonical_runs" / "rollout_edit_insuite_sr"
    src = "results_v2/canonical_runs/rollout_edit_insuite_sr/"
    d = load(base / "rollout_edit_report.json")
    if not d:
        a.check(sec, "the in-suite rollout report is released (the manuscript "
                     "quotes 0/40 from it)", True, False, source=src)
        return
    arms = d.get("by_arm") or {}
    cfg = d.get("config") or {}

    for arm in ("nocot", "cot_clean"):
        v = arms.get(arm) or {}
        a.check(sec, f"{arm}: 0 successes over 40 episodes, as the manuscript "
                     f"states", [0, 40], [v.get("successes"), v.get("n")],
                source=src + "rollout_edit_report.json")
        ci = v.get("wilson95") or [None, None]
        a.check(sec, f"{arm}: Wilson 95% upper bound as printed ($0.088$)",
                [0.0, 0.088], [r3(ci[0]), r3(ci[1])], source=src)

    # The suite is the whole difference between this run and the cancelled
    # round 1. If it ever reads libero_spatial again the zero means nothing.
    a.check(sec, "the rollout ran on libero_90 -- the suite the checkpoint was "
                 "trained on, which is what makes the zero attributable to the "
                 "policy rather than to a suite mismatch",
            "libero_90", cfg.get("suite"), source=src)
    a.check(sec, "at upstream's own 400-step budget, not round 1's hard-coded "
                 "220, which would truncate every episode by construction",
            400, cfg.get("max_steps"), source=src)

    # DSR is undefined, and the harness must say so itself.
    a.check(sec, "the harness records its own precondition as FAILED rather "
                 "than reporting a DSR of zero", False,
            d.get("precondition_met"), source=src)
    a.check(sec, "and no per-family DSR is reported", {},
            d.get("delta_sr_vs_cot_clean"), source=src)

    # If the CoT machinery had failed, the zero would be about our harness.
    cc = arms.get("cot_clean") or {}
    a.check(sec, "every CoT the clean arm generated parsed as structured, so "
                 "the zero is not a prompt-side harness failure",
            [640, 0], [cc.get("n_cot_structured"), cc.get("n_cot_unstructured")],
            source=src)
    a.check(sec, "the no-CoT arm generated none, so the two arms differ in the "
                 "way they are supposed to", 0,
            (arms.get("nocot") or {}).get("n_cot_generated"), source=src)

    # The scale precondition, which is the ecot-bridge failure mode and would
    # make the zero uninterpretable for a reason unrelated to competence.
    probe = load(base / "rollout_edit_probe.json") or {}
    a.check(sec, "the action-scale precondition held (unlike ecot-bridge, "
                 "where a missing norm_stats entry pins SR at 0 for a reason "
                 "with nothing to do with the policy)", True,
            str(probe.get("scale_precondition", "")).startswith("ok"),
            source=src + "rollout_edit_probe.json")

    # The manuscript must state this as undefined, not as a null result.
    tex = TEX.read_text()
    a.check(sec, "the manuscript calls the edit-induced DSR undefined rather "
                 "than zero", True,
            bool(re.search(r"\$\\Delta\$SR is undefined", tex)), source="tex")
    a.check(sec, "and attributes the zero to checkpoint competence rather than "
                 "to compute or suite availability", True,
            "checkpoint-competence problem and not a compute" in tex,
            source="tex")
    a.check(sec, "no rollout-conditioned number is claimed anywhere", True,
            "No rollout-conditioned number appears anywhere in this paper"
            in tex, source="tex")

    # Round 1 is released as evidence for the confound and must never be cited
    # as the limitation-(v) row; its own README says so.
    r1 = ROOT / "results_v2" / "canonical_runs" / \
        "rollout_edit_outofsuite_round1" / "rollout_edit_report.json"
    d1 = load(r1) or {}
    a.check(sec, "the superseded out-of-suite round 1 is retained, and is "
                 "distinguishable from this run by its suite alone",
            "libero_spatial", (d1.get("config") or {}).get("suite"),
            source="results_v2/canonical_runs/rollout_edit_outofsuite_round1/")


def audit_rollout_edited_arm(a: Audit) -> None:
    """The one run that executed a CoT-EDITED rollout arm.

    r2kpkqsim4 ran the two control arms to 40 episodes and stopped before
    adding an edited one. nskmsunnpb added `cot_direction_flip` under the same
    wall-clock budget, which cost episodes (10/10/9 instead of 40/40) and
    bought the only closed-loop evidence in the release about an edit.

    The temptation this artifact creates is precise: three arms now exist, so
    a reader -- or a later draft -- can present 0/9 against 0/10 as a
    comparison. It is not one. The control never succeeds, so the difference
    is zero because both terms are, and the harness still reports
    precondition_met=false with an empty DSR map. Those two facts are asserted
    here exactly as they are for the two-arm run.

    What the run does add is two numbers the offline first-step protocol
    cannot produce, and both are unflattering, which is why they are pinned:
    381 edits were SKIPPED mid-rollout because the CoT carried no direction
    word to reverse, and the edited arm's next-generation parse failure rate
    is 10.2% against the clean arm's 0.2%.
    """
    sec = "In-suite paired rollout, edited arm (the only closed-loop edit)"
    base = ROOT / "results_v2" / "canonical_runs" / "rollout_edit_insuite_flip"
    src = "results_v2/canonical_runs/rollout_edit_insuite_flip/"
    d = load(base / "rollout_edit_report.json")
    if not d:
        a.check(sec, "the edited-arm rollout report is released", True, False,
                source=src)
        return
    arms = d.get("by_arm") or {}
    cfg = d.get("config") or {}

    a.check(sec, "the edited arm exists and names the family it applied",
            "direction_flip",
            (arms.get("cot_direction_flip") or {}).get("family"), source=src)
    for arm, n in (("nocot", 10), ("cot_clean", 10),
                   ("cot_direction_flip", 9)):
        v = arms.get(arm) or {}
        a.check(sec, f"{arm}: 0 successes over {n} episodes",
                [0, n], [v.get("successes"), v.get("n")], source=src)

    # Same suite and budget as the two-arm run: a different suite here would
    # make the two runs incomparable and the zero attributable to the mismatch.
    a.check(sec, "ran on libero_90, the suite the checkpoint trained on",
            "libero_90", cfg.get("suite"), source=src)
    a.check(sec, "at upstream's 400-step budget", 400, cfg.get("max_steps"),
            source=src)
    a.check(sec, "and applied direction_flip, the family with a signed "
                 "prediction", "direction_flip", cfg.get("families"),
            source=src)

    # The load-bearing pair. An edited arm does not make DSR definable.
    a.check(sec, "adding an edited arm does NOT make the comparison defined: "
                 "the harness still reports its precondition as failed",
            False, d.get("precondition_met"), source=src)
    a.check(sec, "and still reports no per-family DSR, because 0 against a "
                 "control that never succeeds is 0 by construction", {},
            d.get("delta_sr_vs_cot_clean"), source=src)

    # The two numbers only a rollout can produce.
    flip = arms.get("cot_direction_flip") or {}
    a.check(sec, "381 edits were skipped mid-rollout for want of a direction "
                 "word to reverse -- an applicability limit the first-step "
                 "protocol cannot see", 381, flip.get("n_edit_skipped"),
            source=src)
    tot = (flip.get("n_cot_structured") or 0) + (flip.get("n_cot_unstructured") or 0)
    cc = arms.get("cot_clean") or {}
    tot_c = (cc.get("n_cot_structured") or 0) + (cc.get("n_cot_unstructured") or 0)
    a.check(sec, "editing the CoT raises the next generation's parse-failure "
                 "rate to 10.2% from the clean arm's 0.2%", [10.2, 0.2],
            [round(100 * (flip.get("n_cot_unstructured") or 0) / tot, 1) if tot else None,
             round(100 * (cc.get("n_cot_unstructured") or 0) / tot_c, 1) if tot_c else None],
            source=src)

    # Same scale precondition as its companion; the ecot-bridge failure mode
    # would make this zero uninterpretable for an unrelated reason.
    probe = load(base / "rollout_edit_probe.json") or {}
    a.check(sec, "the action-scale precondition held here too", True,
            str(probe.get("scale_precondition", "")).startswith("ok"),
            source=src + "rollout_edit_probe.json")


def audit_dt_decode_equivalence(a: Audit) -> None:
    """The cross-family decode-equivalence bound, and the row it does not cover.

    This is the audit that keeps the DeepThinkVLA numbers from resting on our
    own transcription of someone else's decode. Two of the three checkpoints
    return EQUIVALENT and one returns UNDEFINED, and the interesting failure
    mode is that the second gets rounded to the first -- "the audit passed on
    DeepThinkVLA" is true of two thirds of the family and the paper must not
    say it of all of it. So the undefined row is asserted to STAY undefined:
    n_comparable must be 0 and the verdict must start with UNDEFINED.

    The base row's 768-token re-run is asserted separately, because it converts
    a hedge into a fact. At 320 tokens "generation did not terminate" is
    consistent with a budget we chose too small, which is our problem and
    raiseable. At 768 the diagnostic moved to the head of the sequence:
    think_start_first is 0/12, so the checkpoint never opens a think block and
    upstream's [</think>, <action>] criterion cannot fire at any budget. That
    distinction is the whole content of the disclosure, so both halves of it
    get a check -- the exhausted budget AND the absent think token.
    """
    sec = "DeepThinkVLA decode equivalence (2 of 3 defined, and the third named)"
    can = ROOT / "results_v2" / "canonical_runs"

    # --- the 320-token run over all three checkpoints ----------------------
    base = can / "dt_decode_equivalence"
    want = {
        "libero_cot_sft": ("EQUIVALENT", 12),
        "libero_cot_rl": ("EQUIVALENT", 12),
        "base": ("UNDEFINED", 0),
    }
    for tag, (verdict, n_cmp) in want.items():
        p = base / f"dt_decode_equivalence_yinchenghust_deepthinkvla_{tag}.json"
        d = load(p)
        if not d:
            a.check(sec, f"the {tag} report is released", True, False,
                    source=str(p))
            continue
        agg = d.get("aggregate") or {}
        src = f"results_v2/canonical_runs/dt_decode_equivalence/{p.name}"
        a.check(sec, f"{tag}: verdict is {verdict}, as the manuscript states",
                True, str(d.get("verdict", "")).startswith(verdict), source=src)
        a.check(sec, f"{tag}: {n_cmp} comparable frames", n_cmp,
                agg.get("n_comparable"), source=src)
        if n_cmp:
            # Every count that makes EQUIVALENT mean anything. Asserted
            # individually: a verdict string is one boolean and these are five.
            for k in ("n_ids_equal", "n_start_equal", "n_chunk_equal",
                      "n_segments_ok", "n_determinism_ok"):
                a.check(sec, f"{tag}: {k} on all {n_cmp} comparable frames",
                        n_cmp, agg.get(k), source=src)
            a.check(sec, f"{tag}: the (10,7) chunk matches exactly, not "
                         f"approximately", 0.0, agg.get("max_chunk_absdiff"),
                    source=src)
            a.check(sec, f"{tag}: bin centers identical to upstream's", True,
                    agg.get("bin_centers_identical"), source=src)
        a.check(sec, f"{tag}: the audit ran at the 320 tokens the edit runs "
                     f"used, so it audits the configuration that produced the "
                     f"published numbers", 320,
                (d.get("config") or {}).get("max_new_tokens"), source=src)

    # The mean CoT length the manuscript quotes, which is what made 320 look
    # generous and 768 look decisive. Read out of the tex rather than pinned
    # here: a hardcoded expectation on both sides checks nothing.
    sft = load(base / "dt_decode_equivalence_yinchenghust_"
                      "deepthinkvla_libero_cot_sft.json") or {}
    rl = load(base / "dt_decode_equivalence_yinchenghust_"
                     "deepthinkvla_libero_cot_rl.json") or {}
    means = [dig(x, "aggregate", "mean_cot_tokens") for x in (sft, rl)]
    m = re.search(r"a mean of \$([\d.]+)\$ CoT tokens on the two that do "
                  r"terminate", TEX.read_text())
    a.check(sec, "the mean CoT length the manuscript quotes is the mean over "
                 "the two checkpoints that do terminate",
            float(m.group(1)) if m else None,
            round(sum(m2 for m2 in means if m2) / 2, 1) if all(means) else None,
            source="dt_decode_equivalence/*_sft.json, *_rl.json")

    # --- the 768-token re-run: budget or checkpoint? -----------------------
    b7 = can / "dt_decode_equivalence_base768"
    d7 = load(b7 / "dt_decode_equivalence_yinchenghust_deepthinkvla_base.json")
    src7 = "results_v2/canonical_runs/dt_decode_equivalence_base768/"
    if not d7:
        a.check(sec, "the 768-token base re-run is released (the manuscript "
                     "cites it to rule out our own token budget)", True, False,
                source=src7)
        return
    agg7, cfg7 = d7.get("aggregate") or {}, d7.get("config") or {}
    a.check(sec, "the re-run doubled the budget past 320", 768,
            cfg7.get("max_new_tokens"), source=src7)
    a.check(sec, "and is still UNDEFINED, so the 320 result was not our budget",
            0, agg7.get("n_comparable"), source=src7)

    # The head-of-sequence diagnostic. This is the claim that makes the
    # disclosure a fact about the checkpoint rather than a to-do.
    a.check(sec, "the base checkpoint never opens a <think> block on any frame "
                 "-- which is why upstream's [</think>, <action>] criterion "
                 "cannot fire at ANY budget, not merely at this one",
            0, agg7.get("n_think_start_first"), source=src7)
    per = d7.get("per_sample") or []
    a.check(sec, "every frame exhausted the 768-token budget rather than "
                 "stopping early for some other reason",
            [768], sorted({p.get("n_generated") for p in per}), source=src7)
    a.check(sec, "the prompt survived generation on all 12, so the missing "
                 "<think> is not a prompt-assembly artifact of ours",
            12, agg7.get("n_prompt_preserved"), source=src7)
    a.check(sec, "and the run raised no errors, so UNDEFINED is a measurement "
                 "rather than a crash", 0, agg7.get("n_errors"), source=src7)

    # The manuscript must carry both halves: that it is the checkpoint, and
    # that the edit protocol does not depend on the capability it lacks.
    tex = TEX.read_text()
    for needle, what in (
            ("t57wgzya9a", "the 768-token re-run's task id"),
            ("never opens a \\texttt{<think>} block",
             "the head-of-sequence diagnostic"),
            ("undefinable by this method",
             "that no budget can define the comparison"),
            ("injects} a CoT and never asks a checkpoint to generate one",
             "why the base row's gap is not fatal"),
            ("four of five checkpoints tested, with the fifth named",
             "the scope of the equivalence claim")):
        a.check(sec, f"the manuscript states {what}", True, needle in tex,
                source="cot_faith_iclr.tex")


def audit_rollout_filmstrip(a: Audit) -> None:
    """The motion figure must be drawn from captured frames, not described.

    Fig 15 is the only figure in this paper whose content is pixels from a
    simulator rather than a plot of a JSON field, which makes it the easiest one
    to fake and the hardest one to check by reading the .tex. So the checks are
    on the chain: the generator reads the run report, the report carries
    trajectories, PNGs exist on disk, and every number in the caption is a field
    of fig15_facts.json.

    This is silent until the figure is actually in the manuscript. That branch is
    worth stating plainly: a figure not yet included is not a failed claim, but
    the moment `\\includegraphics{fig15...}` appears, every check below is live
    and a missing capture is a hard fail rather than a figure of unknown
    provenance.
    """
    sec = "Rollout filmstrip (fig15, WorldGym-style motion figure)"
    root = ROOT
    arr = root / "cot_faith_arr.tex"
    t = arr.read_text() if arr.exists() else ""
    if "fig15_rollout_filmstrip" not in t:
        return

    gen = root / "figures" / "gen_fig15_rollout_filmstrip.py"
    body = gen.read_text() if gen.exists() else ""
    a.check(sec, "the generator exists and reads the rollout run's own report "
                 "rather than a hand-made summary", True,
            "rollout_edit_report.json" in body, source=str(gen))
    # No offline fallback, asserted as a property of the source: a generator
    # that can draw something without frames is one that will, on the day the
    # capture is missing, and the figure would then illustrate nothing.
    a.check(sec, "and it refuses to draw a partial figure (die() on missing "
                 "data) instead of degrading quietly", True,
            "raise SystemExit(2)" in body, source=str(gen))

    cap = root / "results_v2" / "canonical_runs" / "rollout_filmstrip"
    facts = load(cap / "fig15_facts.json")
    a.check(sec, "the figure's fact sheet is released, so the caption's "
                 "numbers are checkable and not recalled", True,
            facts is not None, source=str(cap / "fig15_facts.json"))
    if not facts:
        return

    # The frames themselves. A facts file naming six columns beside a frames/
    # directory holding none would pass every check above.
    n_png = len(list((cap / "frames").rglob("*.png"))) if cap.exists() else 0
    a.check(sec, "captured frames are in the release, so a reader can redraw "
                 "the strip from the same pixels", True, n_png > 0,
            source=f"{n_png} PNG(s) under {cap / 'frames'}")

    # The figure's least visible and most load-bearing claim: its three rows are
    # one scene, so every row-to-row difference is the CoT edit. Two defects
    # reached a rendered strip before this was checked (limitation (v)), and both
    # were invisible in the report. The generator records its own step-0
    # measurement so this is assertable here without loading the PNGs.
    a.check(sec, "the released strip's three rows are BIT-IDENTICAL at the "
                 "first column, before any arm has acted", 0.0,
            facts.get("step0_pairing_max_mean_abs_pixel"),
            source="fig15_facts.json: step0_pairing_max_mean_abs_pixel")

    # Three arms, one init state. The argument of the figure is that the rows
    # differ only in the prompt, so a strip drawn across two episodes would be
    # comparing scenes rather than CoT conditions.
    arms = sorted((facts.get("steps_per_arm") or {}))
    a.check(sec, "the strip covers all three arms of one episode",
            ["cot_clean", "cot_direction_flip", "nocot"], arms,
            source="fig15_facts.json: steps_per_arm")
    a.check(sec, "and that episode is in-suite libero_90, matching the run "
                 "whose SR section 6 reports", "libero_90",
            facts.get("suite"), source="fig15_facts.json: suite")
    a.check(sec, "and the CoT is refreshed every step, as in the reported "
                 "protocol", 1, facts.get("cot_refresh_steps"),
            source="fig15_facts.json: cot_refresh_steps")

    # The figure exists because SR does not separate the arms. If some arm did
    # succeed, the caption's framing -- both arms fail, here is how differently
    # -- would be wrong, so the premise is checked rather than assumed.
    succ = facts.get("success_per_arm") or {}
    a.check(sec, "no arm succeeds in the filmed episode, which is why the "
                 "figure shows motion instead of a success rate",
            0, sum(1 for v in succ.values() if v),
            source=f"fig15_facts.json: success_per_arm={succ}")

    # Panels (b) and (c) are claims about measured pose. If the env logged none,
    # the generator drops them -- and the caption must then not describe them.
    # Checked in both directions, since either mismatch ships a caption
    # describing a figure the reader is not looking at.
    eef = bool(facts.get("eef_logged"))
    for lit, what in (("gripper distance", "panel (c)'s distance curve"),
                      ("top-down", "panel (b)'s path")):
        a.check(sec, f"the caption describes {what} exactly when a pose was "
                     f"logged (eef_logged={eef})", eef, lit in t,
                source=f"cot_faith_arr.tex mentions {lit!r}: {lit in t}")

    # Every number the caption quotes, read back out of the fact sheet, so a
    # caption edit that changes a digit fails here rather than in review.
    for val, what in ((len(facts.get("columns_at_steps") or []), "column count"),
                      (facts.get("n_edit_skipped_flip"), "edits skipped")):
        a.check(sec, f"the ARR caption prints the artifact's {what} ({val})",
                True, f"${val}$" in t, source="fig15_facts.json")


def audit_arm_pairing_defect(a: Audit) -> None:
    """The pairing defect the appendix discloses, checked against its pixels.

    Limitation (v) says the rollout arms were paired on the robot and mispaired
    on the furniture, and quotes five numbers for it. They are pixel
    measurements, so unlike every other number in this release they cannot be
    recomputed from a JSON of deltas -- the diagnostic artifact IS the record,
    and this asserts the manuscript against it rather than against a memory of
    running it.

    The two numbers doing the arguing are checked as a pair, because either one
    alone is weak. `outside_bbox_mean_abs == 0` says the arms were bit-identical
    everywhere the difference did not reach, which is what makes "paired on the
    robot" a measurement rather than an impression. The shift residual says the
    difference is a rigid translation, which is what distinguishes furniture
    placed differently from a policy that acted differently -- without it, a
    reader is entitled to read the same 8.84 as the arms simply diverging.

    Note what is NOT asserted: that the shift is small, or that the defect is
    minor. The check is that the released pixels say what the appendix says
    they say.

    Two defects, not one, and they are audited separately because their
    signatures are what tell them apart. Both have a bit-identical complement.
    The first is a rigid 3 px translation of a welded fixture; the second, found
    only because the first fix was checked by re-rendering, is a robot pose that
    no shift aligns (best shift (0, 0)) -- carried in through each arm's own
    settling loop from state set_init_state does not restore.

    Three, in fact. The third is audited from a different KIND of artifact,
    because it is a different kind of defect: it acts from step 1 onward, so no
    rendered first frame and no scalar in the report could show it. Its evidence
    is a probe that replays one scripted action sequence twice from one rewind
    (bolt gzv4nuhtfe) -- and the check that matters most there is the one
    asserting the two EARLIER channels were genuinely reset, since that is what
    stops the defect-2 paragraph from quietly covering for defect 3.
    """
    sec = "Rollout arm pairing defect (limitation v disclosure)"
    base = ROOT / "results_v2" / "canonical_runs" / "rollout_arm_pairing_defect"
    src = "results_v2/canonical_runs/rollout_arm_pairing_defect/"
    d = load(base / "pairing_defect.json")
    if not d:
        a.check(sec, "the pairing diagnostic is released, since the appendix "
                     "quotes it", True, False, source=src)
        return

    # The script that produced it ships too. A measurement no one else can
    # re-run is the thing this release exists not to publish.
    gen = ROOT / "scripts" / "diagnose_arm_pairing.py"
    a.check(sec, "the script that produces the diagnostic ships with it",
            True, gen.exists(), source="scripts/diagnose_arm_pairing.py")

    eps = d.get("episodes") or []
    a.check(sec, "the diagnostic covers at least one filmed episode",
            True, len(eps) >= 1, source=src)
    if not eps:
        return
    pairs = eps[0].get("pairs") or {}
    a.check(sec, "and at least one arm pair within it", True, len(pairs) >= 1,
            source=src)
    if not pairs:
        return
    rec = list(pairs.values())[0]

    a.check(sec, "the defect is present at step 0, before either arm acted "
                 "(so it cannot be behaviour)", False, rec.get("identical"),
            source=src)
    a.check(sec, "10.4% of pixels differ", 0.104,
            round(float(rec.get("frac_pixels_differing", -1)), 3), source=src)
    a.check(sec, "the arms are BIT-IDENTICAL outside the differing box, which "
                 "is what 'paired on the robot and the free objects' means",
            0.0, float(rec.get("outside_bbox_mean_abs", -1)), source=src)
    a.check(sec, "mean |dpix| inside the differing box is 8.8411",
            8.8411, float(rec.get("inside_bbox_mean_abs", -1)), source=src)

    sh = rec.get("best_shift") or {}
    a.check(sec, "a 3-pixel horizontal shift best aligns the two boxes, so the "
                 "difference is a rigid translation", 3,
            abs(int(sh.get("dx", 0))), source=src)
    a.check(sec, "and at that shift the residual falls to 2.7762",
            2.7762, float(sh.get("residual_mean_abs", -1)), source=src)

    # The disclosure's own claim about itself: the appendix says the harness now
    # refuses to proceed on a mispairing at three layers. Asserted as source
    # properties, because a disclosure that describes guards it does not have is
    # worse than no disclosure.
    for path, needle, what in (
        ("bolt/run_cotfaith_rollout_edit_s3.sh", "did not start from the",
         "the rollout script fails while the GPU is still allocated"),
        ("figures/gen_fig15_rollout_filmstrip.py", "def start_mismatch",
         "the figure generator refuses to draw mispaired rows"),
        ("tests/test_fig15_filmstrip.py", "test_real_defective_capture",
         "an offline test runs the detector against the real defective frames"),
    ):
        p = ROOT / path
        a.check(sec, what, True, p.exists() and needle in p.read_text(),
                source=path)

    # And that the fix is structural, not only a seed: the arms of one episode
    # must share an env with no reset() between them. A seed alone would leave
    # the pairing dependent on which RNG robosuite happens to draw from.
    h = ROOT / "experiments" / "cotfaith_rollout_edit.py"
    hb = h.read_text() if h.exists() else ""
    a.check(sec, "the harness seeds the placement sampler per episode",
            True, "_seed_scene" in hb and "--env-seed" in hb,
            source="experiments/cotfaith_rollout_edit.py")
    a.check(sec, "and shares one env across the arms of an episode, so the "
                 "pairing does not rest on an assumption about which RNG places "
                 "fixtures", True,
            "No reset() here" in hb,
            source="experiments/cotfaith_rollout_edit.py")

    # ---------------------------------------------------------------------
    # The SECOND pairing defect, found only because the first fix was checked
    # by re-rendering rather than by declaring victory. Audited separately
    # because its signature is the opposite of the first one's and that is the
    # whole content of the claim: same complement-is-zero, but no shift helps.
    # ---------------------------------------------------------------------
    d2 = load(base / "pairing_defect_settle.json")
    a.check(sec, "the second pairing diagnostic is released too, since the "
                 "appendix discloses two defects and not one", True, bool(d2),
            source=src)
    if d2:
        e2 = (d2.get("episodes") or [{}])[0]
        p2 = list((e2.get("pairs") or {}).values())
        a.check(sec, "the second diagnostic covers an arm pair", True,
                len(p2) >= 1, source=src)
    if d2 and p2:
        r2 = p2[0]
        a.check(sec, "defect 2 is also present at step 0, before either arm "
                     "acted", False, r2.get("identical"), source=src)
        a.check(sec, "defect 2: the fixture fix HELD -- the arms are still "
                     "bit-identical outside the differing box", 0.0,
                float(r2.get("outside_bbox_mean_abs", -1)), source=src)
        a.check(sec, "defect 2: 6.2% of pixels differ", 0.062,
                round(float(r2.get("frac_pixels_differing", -1)), 3),
                source=src)
        a.check(sec, "defect 2: mean |dpix| inside the box is 31.7356",
                31.7356, float(r2.get("inside_bbox_mean_abs", -1)), source=src)
        box = r2.get("diff_bbox") or {}
        a.check(sec, "defect 2: the box has moved OFF the furniture and onto "
                     "the robot (rows 0-134)", [0, 134],
                [int(box.get("row0", -1)), int(box.get("row1", -1))],
                source=src)
        sh2 = r2.get("best_shift") or {}
        a.check(sec, "defect 2: NO rigid shift aligns the two frames, which is "
                     "what makes it a different defect rather than an "
                     "incomplete fix of the first", [0, 0],
                [int(sh2.get("dy", 9)), int(sh2.get("dx", 9))], source=src)
        a.check(sec, "defect 2: and so the shift leaves the residual untouched",
                31.7356, float(sh2.get("residual_mean_abs", -1)), source=src)

    # Defect 2's fix, as source properties. The settle must happen once per
    # episode and each arm be rewound to that snapshot: a per-arm settle is what
    # let the second arm inherit the first arm's integrator state.
    for needle, what in (
        ("def _settle_once",
         "the settle runs once per episode and is snapshotted"),
        ("def _rewind_to",
         "each arm is rewound to that snapshot rather than re-settling"),
        ("qacc_warmstart",
         "the rewind zeroes MuJoCo's warm-start accelerations, which "
         "set_init_state does not restore"),
        ("reset_goal",
         "and resets the controller goal the previous arm left behind"),
        ("obs=rw[\"obs\"]",
         "the arm is handed the pre-settled observation, so it does not run a "
         "settling loop of its own"),
    ):
        a.check(sec, what, True, needle in hb,
                source="experiments/cotfaith_rollout_edit.py")
    gate = ROOT / "tests" / "test_rollout_arms_and_refresh.py"
    gt = gate.read_text() if gate.exists() else ""
    a.check(sec, "the pod's pre-budget gate covers the rewind, including the "
                 "case where a channel cannot be reached", True,
            "test_rewind_clears_carryover" in gt,
            source="tests/test_rollout_arms_and_refresh.py")

    # ---------------------------------------------------------------------
    # The THIRD defect, and the only one no rendered frame could have shown.
    # Defects 1 and 2 were visible at step 0. This one acts from step 1 onward,
    # where the arms are SUPPOSED to differ -- so the pixels, the report and the
    # step-0 guard above are all blind to it, and it took a probe that removes
    # the policy: replay one scripted action sequence twice from one rewind and
    # require the trajectories to match.
    # ---------------------------------------------------------------------
    d3 = load(base / "pairing_defect_gripper.json")
    a.check(sec, "the third pairing diagnostic -- the probe run that found the "
                 "gripper accumulator -- is released too", True, bool(d3),
            source=src + "pairing_defect_gripper.json")
    if d3:
        ch = d3.get("arm_rewind_channels") or {}
        # Audited in the direction that could embarrass us: the appendix says
        # defect 2's two channels WERE reset, and this artifact is the reason
        # that sentence is not an overstatement covering for defect 3.
        a.check(sec, "defect 3: the two channels defect 2's fix targets were "
                     "genuinely reset, so the paragraph above is not "
                     "overstating that fix", [1, True],
                [ch.get("controller"), ch.get("warmstart")], source=src)
        a.check(sec, "defect 3: and yet replaying ONE action sequence twice "
                     "from the same rewind did NOT give identical trajectories",
                False, d3.get("identical_qpos"), source=src)
        mx = d3.get("max_abs_qpos_diff_over_all_steps")
        a.check(sec, "defect 3: the two replays differ by 0.1876 in qpos",
                0.1876, round(float(mx if mx is not None else -1), 4),
                source=src)
        # The signature. An offset already at its maximum on the first step and
        # not growing is a stale value carried in; drift would grow. This one
        # equality is what identified the accumulator, so it is asserted rather
        # than described.
        a.check(sec, "defect 3: the maximum over all steps EQUALS the first-step "
                     "difference, i.e. a stale offset carried in rather than "
                     "drift that accumulates", True,
                mx is not None and mx == d3.get("first_step_qpos_diff"),
                source=src)
        a.check(sec, "defect 3: and 198 pixel levels at the worst step", 198.0,
                float(d3.get("max_abs_pixel_diff_over_all_steps") or -1),
                source=src)

    a.check(sec, "the probe that found defect 3 ships, since it is the only "
                 "check that can see a step-1-onward confound", True,
            (ROOT / "scripts" / "probe_rewind_pairing.py").exists(),
            source="scripts/probe_rewind_pairing.py")
    probe_t = (ROOT / "scripts" / "probe_rewind_pairing.py")
    pt = probe_t.read_text() if probe_t.exists() else ""
    a.check(sec, "and it gates on EVERY carry-over channel, not the two that "
                 "were known when it was written", True,
            "missing = [k for k, v in ch.items() if not v]" in pt,
            source="scripts/probe_rewind_pairing.py")

    # Defect 3's fix, as source properties: the accumulator must be snapshotted
    # WITH the state and restored on rewind. Zeroing it would pass a naive
    # equality check while putting the scene at a fresh-reset value instead of
    # where the settle left it.
    for needle, what in (
        ("def _grippers",
         "defect 3: the harness reaches the gripper across both robosuite "
         "spellings"),
        ("current_action",
         "defect 3: it is the rate-limited current_action accumulator that is "
         "handled, which is the state neither qpos nor reset_goal covers"),
        ('snap.get("grippers")',
         "defect 3: the rewind restores the SNAPSHOT's accumulator rather than "
         "zeroing it to a fresh-reset value"),
        ('"gripper": 0',
         "defect 3: the gripper is reported as its own channel, so one that a "
         "future robosuite renames is visible and not silent"),
    ):
        a.check(sec, what, True, needle in hb,
                source="experiments/cotfaith_rollout_edit.py")
    a.check(sec, "the pod's gate covers the accumulator restore too, against a "
                 "stub that actually accumulates", True,
            "restores the gripper accumulator to the snapshot" in gt,
            source="tests/test_rollout_arms_and_refresh.py")

    # Every number above is quoted in the appendix. Checked in that direction
    # too: an artifact that stops matching the prose is the failure this whole
    # script exists to catch, and it is silent unless someone looks.
    apx = ROOT / "arr_appendix.tex"
    at = apx.read_text() if apx.exists() else ""
    for val in ("10.4", "0.0000", "8.8411", "2.7762", "31.7356", "0.1876",
                "198"):
        a.check(sec, f"the appendix quotes ${val}$ from this artifact",
                True, val in at, source="arr_appendix.tex")


    # --- defect 4: identical state, different frame ------------------------
    # The one this release is least entitled to have missed, because every
    # check we had was on state and the state was bit-identical. Both halves
    # are asserted: the defective run (the arms differ in pixels while agreeing
    # to 0.0 in qpos) and the fixed run (bit-identical over every step), since
    # a disclosure that names a defect without releasing its repair is not
    # falsifiable.
    d4 = load(base / "pairing_defect_sampling_phase.json")
    a.check(sec, "defect 4's diagnostic is released", True, bool(d4),
            source=src)
    if d4:
        a.check(sec, "defect 4: the arms agree in qpos at EVERY step, which is "
                     "what made this defect invisible to every state-based "
                     "check", (True, 0.0),
                (bool(d4.get("identical_qpos")),
                 float(d4.get("max_abs_qpos_diff_over_all_steps", -1))),
                source=src)
        a.check(sec, "defect 4: and the frames differ anyway, at every one of "
                     "the 40 steps", (False, 40),
                (bool(d4.get("identical_frames")),
                 int(d4.get("n_pixel_steps_differing", -1))), source=src)
        a.check(sec, "defect 4: the worst step differs by 160 pixel levels",
                160.0, float(d4.get("max_abs_pixel_diff_over_all_steps", -1)),
                source=src)
        rg = d4.get("region_worst_step_1v2") or {}
        a.check(sec, "defect 4: and is bit-identical outside the differing "
                     "box, so this is a localized substep mismatch and not "
                     "noise across the frame", 0.0,
                float(rg.get("outside_mean_abs", -1)), source=src)
        # No rigid shift aligns it -- which is what separates defect 4 from
        # defect 1, where a 3-pixel translation cut the residual by 3x.
        al = d4.get("frame_alignment_1v2") or {}
        a.check(sec, "defect 4: no translation aligns the two frames at any "
                     "step, unlike defect 1's rigid 3-pixel offset", 0,
                int(al.get("n_steps_with_an_exact_match", -1)), source=src)
        # And the render itself is deterministic, so the difference is in what
        # was rendered rather than in the renderer.
        nf = d4.get("render_noise_floor") or {}
        a.check(sec, "defect 4: the renderer itself is bit-reproducible over "
                     "repeated renders, so the difference is in WHEN the frame "
                     "was taken", True, bool(nf.get("bit_identical")),
                source=src)

    d4f = load(base / "pairing_probe_all_four_fixed.json")
    a.check(sec, "the run that closes defect 4 is released too", True,
            bool(d4f), source=src)
    if d4f:
        a.check(sec, "defect 4 fixed: the arms are bit-identical over all 40 "
                     "steps -- the first run of this harness for which that is "
                     "true", (True, 0.0),
                (bool(d4f.get("identical_frames")),
                 float(d4f.get("max_abs_pixel_diff_over_all_steps", -1))),
                source=src)
        a.check(sec, "defect 4 fixed: on the second arm pair as well, since "
                     "two of three arms agreeing is not a paired harness",
                (True, 0.0),
                (bool(d4f.get("identical_frames_2v3")),
                 float(d4f.get("max_abs_pixel_diff_2v3", -1))), source=src)
        ch = d4f.get("arm_rewind_channels") or {}
        a.check(sec, "defect 4 fixed: all five carry-over channels are "
                     "restored, including the two the disclosure adds",
                ["clock", "controller", "gripper", "observables", "warmstart"],
                sorted(ch), source=src)
        a.check(sec, "defect 4 fixed: 3 clock fields and 29 observable caches "
                     "were actually restored, not merely gated on", (3, 29),
                (int(ch.get("clock", -1)), int(ch.get("observables", -1))),
                source=src)
        # The mechanism, from the artifact rather than from the prose: the
        # sampling phase differed BEFORE the rewind and agrees after it.
        pre = d4f.get("sampling_phase_before_rewind_1v2") or {}
        post = d4f.get("sampling_phase_after_rewind_1v2") or {}
        a.check(sec, "defect 4's mechanism: 31 of 206 observable fields "
                     "differed before the rewind", (31, 206),
                (int(pre.get("n_fields_differing", -1)),
                 int(pre.get("n_fields", -1))), source=src)
        a.check(sec, "and 0 of them differ after it", 0,
                int(post.get("n_fields_differing", -1)), source=src)
        # The clock difference the appendix quotes to 15 decimal places. It is
        # the whole point of the disclosure -- 1.1e-15 crossing a comparison
        # boundary became a 116-level image difference -- so it is asserted
        # rather than paraphrased.
        clocks = sorted({tuple(v) for k, v in
                         (pre.get("differing") or {}).items()
                         if k.endswith("._time_since_last_sample")})
        a.check(sec, "defect 4's mechanism: the two arms' observable clocks "
                     "differ in the last representable digit", 1,
                len(clocks), source=f"{src}: distinct clock pairs = {clocks}")
        if len(clocks) == 1:
            lo, hi = clocks[0]
            a.check(sec, "and the appendix quotes both clock values exactly, "
                         "since a rounded version of this number is the same "
                         "number", (True, True),
                    (repr(lo) in at, repr(hi) in at),
                    source="arr_appendix.tex")
            a.check(sec, "and their difference is at the 1e-15 scale the "
                         "appendix states", True,
                    0 < abs(hi - lo) < 1e-14,
                    source=f"|delta| = {abs(hi - lo):.3g}")
        sdf = d4f.get("state_determines_frame") or {}
        a.check(sec, "the probe records the implication defect 4 falsified: "
                     "writing the same state and re-rendering does NOT give "
                     "the same frame", False, bool(sdf.get("bit_identical")),
                source=src)
        a.check(sec, "and it differs by the 116 levels the appendix quotes",
                116.0, float((sdf.get("pixels") or {}).get("max", -1)),
                source=src)
        for val in ("160", "116", "206", "vu6yavsp4a", "e268dqs2t8"):
            a.check(sec, f"the appendix quotes '{val}' from the defect-4 "
                         f"artifacts", True, val in at,
                    source="arr_appendix.tex")
        a.check(sec, "the appendix says four defects rather than three, so the "
                     "count in the prose tracks the diagnostics released",
                True, "Four pairing defects" in at, source="arr_appendix.tex")
        a.check(sec, "and it retracts the state-implies-pixels inference it "
                     "made for defect 2 rather than quietly deleting it", True,
                "falsifies" in at and "the assumption that hid" in at,
                source="arr_appendix.tex")

    # Defect 4's fix, as source properties. The sampling phase must be
    # snapshotted with the settle and restored on rewind, and it must be
    # reported as its own channel -- a silently-skipped restore is how this
    # defect survived two fixes.
    for needle, what in (
        ("_update_observables",
         "defect 4: the harness names the robosuite call that samples "
         "observables inside the substep loop, which is the mechanism"),
        ("_restore_sampling",
         "defect 4: the rewind restores the sampling phase"),
        ('"clock": 0, "observables": 0',
         "defect 4: the clock and the observable caches are reported as their "
         "own channels, so a rename is visible rather than silent"),
    ):
        a.check(sec, what, True, needle in hb,
                source="experiments/cotfaith_rollout_edit.py")
    a.check(sec, "the probe measures state->pixel determinism directly rather "
                 "than assuming it, which is the check that would have caught "
                 "defect 4 two runs earlier", True,
            "def state_determines_frame" in pt,
            source="scripts/probe_rewind_pairing.py")


def audit_rank_correlation(a: Audit, d: dict) -> None:
    """One number for how much the two scoring rules disagree, recomputed.

    S6 used to say "the ranking inverts", and a reviewer asked for the summary
    statistic the two columns of tab:directional imply. It is +0.476 -- weakly
    POSITIVE -- so the honest claim is narrower than the one the section made,
    and the manuscript now makes the narrower one. That makes this check about
    an overclaim we removed rather than one we are defending, and the direction
    matters: a future edit that widens the wording back out to a global
    inversion has to get past a check that knows rho is positive.

    Both statistics are recomputed here by a DIFFERENT route than
    scripts/rank_correlation.py uses. That script computes Spearman as Pearson
    on midranks and tau as tau-b, both of which are the tie-corrected
    definitions; this one uses the no-ties closed forms, 1 - 6*sum(d^2)/n(n^2-1)
    and (con-dis)/(n(n-1)/2). The two agree only when there are no ties, so the
    artifact's own tie counts are asserted to be zero first. A shared bug in
    the midrank code would show up as a disagreement rather than cancelling.
    """
    sec = "Rank correlation between the two scoring rules (S6)"
    art = ROOT / "results_v2/canonical_runs/rank_correlation/" \
                 "rank_correlation.json"
    rc = load(art)
    a.check(sec, "the rank-correlation artifact is released", True, bool(rc),
            source=str(art.relative_to(ROOT)))
    if not rc:
        return
    src = "rank_correlation.json"
    fam_key = rc.get("family")
    a.check(sec, "it is computed on the family the section is about",
            "direction_flip", fam_key, source=src)

    mean = rc.get("on_3seed_mean") or {}
    for k, what in (("n_ties_magnitude_only", "magnitude"),
                    ("n_ties_direction_only", "direction"),
                    ("n_ties_both", "both")):
        a.check(sec, f"no ties on {what}, which is what licenses the "
                     f"no-ties closed form used to re-derive rho here", 0,
                mean.get(k), source=src)

    # --- the ranks, from derived_metrics rather than from the artifact ------
    models = rc.get("models") or []
    fams = {m: dig(d, "models", m, "families", "direction_flip") or {}
            for m in models}
    a.check(sec, "every cohort model still carries a direction_flip row in "
                 "derived_metrics.json", [],
            sorted(m for m in models
                   if fams[m].get("F_mag") is None
                   or fams[m].get("F_dir") is None),
            source="derived_metrics.json")
    if any(fams[m].get("F_dir") is None for m in models):
        return

    n = len(models)
    a.check(sec, "the cohort is the eight leaderboard configurations S6 walks "
                 "down", 8, n, source=src)

    def ranks(key):
        order = sorted(models, key=lambda m: -fams[m][key])
        return {m: i + 1 for i, m in enumerate(order)}

    rmag, rdir = ranks("F_mag"), ranks("F_dir")
    for key, got in (("rank_magnitude", rmag), ("rank_direction", rdir)):
        a.check(sec, f"the artifact's {key} is what derived_metrics.json "
                     f"implies", got,
                {k: int(v) for k, v in (mean.get(key) or {}).items()},
                source=f"{src} vs derived_metrics.json")

    # Spearman by the no-ties closed form.
    dsum = sum((rmag[m] - rdir[m]) ** 2 for m in models)
    rho = 1.0 - 6.0 * dsum / (n * (n * n - 1))
    a.check(sec, "Spearman rho, recomputed by the no-ties closed form, "
                 "matches the artifact to 3dp", f"{rho:.3f}",
            f"{mean.get('spearman_rho', 0):.3f}",
            source=f"sum d^2 = {dsum} over n = {n}")

    con = dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            mi, mj = models[i], models[j]
            same = ((rmag[mi] - rmag[mj]) > 0) == ((rdir[mi] - rdir[mj]) > 0)
            con, dis = (con + 1, dis) if same else (con, dis + 1)
    a.check(sec, "the concordant/discordant split S6 quotes", (21, 7),
            (con, dis), source="recomputed from the two rankings")
    a.check(sec, "and it is the split the artifact recorded",
            (con, dis), (mean.get("n_concordant_pairs"),
                         mean.get("n_discordant_pairs")), source=src)
    tau = (con - dis) / (n * (n - 1) / 2)
    a.check(sec, "Kendall tau, recomputed as (con-dis) over n(n-1)/2, matches "
                 "the artifact to 3dp", f"{tau:.3f}",
            f"{mean.get('kendall_tau_b', 0):.3f}", source=src)

    # --- the shape of the disagreement, which is the actual claim ----------
    shifts = sorted(((abs(rmag[m] - rdir[m]), m) for m in models),
                    reverse=True)
    a.check(sec, "the largest rank shift is 6 positions", 6.0,
            float(shifts[0][0]), source="recomputed")
    a.check(sec, "and it belongs to the configuration S6 names",
            "ecot-bridge", shifts[0][1], source="recomputed")
    a.check(sec, "S6's claim that no OTHER configuration shifts more than 2 "
                 "positions: the second-largest shift", True,
            float(shifts[1][0]) <= 2.0,
            source=f"second-largest = {shifts[1][0]:.0f} ({shifts[1][1]})")
    a.check(sec, "the hero configuration goes from rank 1 to rank 7, which is "
                 "what fig:overview panel (c) draws", (1, 7),
            (rmag["ecot-bridge"], rdir["ecot-bridge"]), source="recomputed")

    # --- per seed, because a statistic that needs the averaging is an artifact
    per = rc.get("per_seed") or []
    a.check(sec, "rho is reported per sampling seed as well as on the 3-seed "
                 "mean", 3, len(per), source=src)
    rhos = [p.get("spearman_rho") for p in per]
    a.check(sec, "and it is positive on every seed, so the weak agreement is "
                 "not produced by averaging the seeds", 0,
            sum(1 for r in rhos if r is not None and r < 0), source=src)
    a.check(sec, "the per-seed rho range S6 quotes", ("0.400", "0.500"),
            (f"{min(rhos):.3f}", f"{max(rhos):.3f}"), source=src)

    # --- and the prose, in every place it is stated ------------------------
    arr = ROOT / "cot_faith_arr.tex"
    t = arr.read_text() if arr.exists() else ""
    for lit, where in ((r"\rho = 0.48", "the abstract, rounded to 2dp"),
                       (r"\rho = 0.476", "S6 and the conclusion"),
                       (r"\tau_b = 0.500", "S6"),
                       ("21 concordant against 7 discordant", "S6"),
                       ("$0.400$--$0.500$", "S6's per-seed range")):
        a.check(sec, f"the ARR body states {lit!r} in {where}", True,
                lit in t, source="cot_faith_arr.tex")
    # rho is POSITIVE, so no sentence may claim the cohort ordering reverses
    # wholesale. This is the overclaim the statistic ruled out; it stays ruled
    # out only if something checks.
    #
    # The check is on the QUALIFIER, not on the phrase: "the top of the ranking
    # inverts" is the claim rho supports and it necessarily contains "the
    # ranking inverts" as a substring, and S6 also quotes the bare phrase in
    # scare quotes in order to disclaim it. So every occurrence must be either
    # scoped to the top of the ranking or quoted, and a bare one fails. The
    # qualifier may sit on either side -- the section title scopes it before
    # ("the top of the ranking inverts") and the paragraph lead-in after ("the
    # ranking reverses at the top") -- so both sides of the window are read.
    unqualified = []
    for m in re.finditer(r"ranking (?:inverts|reverses)", t):
        near = t[max(0, m.start() - 24):m.end() + 24]
        if not ("top of the " in near or "at the top" in near or "``" in near):
            unqualified.append(t[max(0, m.start() - 40):m.end() + 20])
    a.check(sec, "every claim that the ranking inverts is scoped to the TOP of "
                 "the ranking (or quoted in order to be disclaimed): a "
                 "positive rho does not support a wholesale reversal", [],
            unqualified, source="cot_faith_arr.tex")


def audit_overview_figure(a: Audit, d: dict) -> None:
    """Fig 1 says what the instrument does AND what it found, so it is checked.

    The overview figure prints five quantities on the canvas and its caption
    repeats them in prose. That is two copies of every number, in a float that
    a reader looks at before any table, and neither copy is derived from the
    other -- the panel reads derived_metrics.json, the caption was typed. So
    both are asserted against the artifacts here.

    The figure script is also checked for the specific values it displays, not
    by the general hardcoded-literal scan the other figures get. That scan
    compares every numeric literal in the drawing code against every value in
    the artifact, which works for a bar chart and does not work here: panel (a)
    is a hand-laid drawing whose box coordinates are three-decimal literals by
    the dozen, and against a 437-example export the scan collides by accident
    often enough to be waived rather than read. Naming the five reported
    quantities instead is narrower and does not go stale silently, because the
    same five are asserted against the artifacts two paragraphs down.
    """
    sec = "Overview figure (Fig 1)"
    arr = ROOT / "cot_faith_arr.tex"
    t = arr.read_text() if arr.exists() else ""
    gen = ROOT / "figures" / "gen_fig1_overview.py"
    pdf = ROOT / "figures" / "fig1_overview.pdf"

    a.check(sec, "the ARR body opens with an overview figure -- every "
                 "benchmark paper we compare against does, and until now this "
                 "one had a page of text on page 1", True,
            "\\label{fig:overview}" in t, source="cot_faith_arr.tex")
    a.check(sec, "it is full width, since three panels in one column is the "
                 "aspect ratio that made fig4 illegible", True,
            bool(re.search(r"\\includegraphics\[width=\\textwidth\]"
                           r"\{fig1_overview\.pdf\}", t)),
            source="cot_faith_arr.tex")
    a.check(sec, "the figure is generated by a released script", True,
            gen.exists(), source=str(gen.relative_to(ROOT)))
    a.check(sec, "and the PDF it produces is committed", True, pdf.exists(),
            source=str(pdf.relative_to(ROOT)))
    body = gen.read_text() if gen.exists() else ""
    for artifact in ("floor_invariance.json", "edit_examples.json",
                     "derived_metrics.json"):
        a.check(sec, f"the script reads {artifact} rather than restating it",
                True, artifact in body or "_data" in body,
                source=str(gen))

    # The three panel-(a) quantities, from the artifact the figure reads.
    hero = dig(d, "models", "ecot-bridge", "families", "direction_flip") or {}
    for field, want, what in (("cos_xyz", "+0.415", "mean translation cosine"),
                              ("F_mag", "0.963", "magnitude score"),
                              ("F_dir", "0.120", "direction-aware score")):
        v = hero.get(field)
        got = (f"{v:+.3f}" if field == "cos_xyz" else f"{v:.3f}") \
            if v is not None else None
        a.check(sec, f"panel (a)'s {what} is the released value", want, got,
                source=f"derived_metrics.json ecot-bridge.direction_flip."
                       f"{field}")
        a.check(sec, f"and the caption prints it", True, f"${want}$" in t,
                source="cot_faith_arr.tex")
        # It must reach the canvas from the artifact, not from a literal.
        a.check(sec, f"and the script does not hardcode {want}", True,
                want.lstrip("+") not in re.sub(r'"""[\s\S]*?"""', "", body),
                source=str(gen))

    # Panel (b)'s count, recomputed from the floor artifact the same way the
    # panel title computes it: the semantic mean between the two floors.
    fi = load(ROOT / "results_v2/canonical_runs/floor_invariance/"
                     "floor_invariance.json") or {}
    pc = fi.get("per_config") or []
    between = sum(1 for c in pc
                  if min(c["floor_paraphrase_null"]["F"],
                         c["floor_syntactic_scramble"]["F"])
                  <= c["f_bar_semantic"]
                  <= max(c["floor_paraphrase_null"]["F"],
                         c["floor_syntactic_scramble"]["F"]))
    a.check(sec, "panel (b): the semantic mean falls between the two floors "
                 "on every calibrated configuration", (12, 12),
            (between, len(pc)), source="floor_invariance.json per_config")
    a.check(sec, "and the caption says 12 of 12", True,
            "on 12 of 12" in t, source="cot_faith_arr.tex")

    # Panel (c) draws only one line as load-bearing. The caption says why, and
    # the reason is a number in another section -- if the pale-line disclaimer
    # ever goes away, the figure starts asserting an ordering S8 forbids.
    a.check(sec, "panel (c)'s caption says the pale reorderings are inside "
                 "the retraining error bar, so the figure does not assert an "
                 "ordering S8 refuses to publish", True,
            "should not be read as ordered" in t,
            source="cot_faith_arr.tex")
    a.check(sec, "and the script draws them pale for that stated reason "
                 "rather than for looks", True,
            "retraining error bar" in body, source=str(gen))


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
    audit_calibration_nine_models(a, d)
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
    audit_normstats_probe(a)
    audit_deepthink_provenance(a)
    audit_cited_environment(a)
    audit_dequant_convention(a, d)
    audit_deepthink_tau_units(a)
    audit_rollout_gate(a, d)
    audit_rollout_gate_winning(a, d)
    audit_p2_decode_equivalence(a)
    audit_resize_check(a)
    audit_citations(a)
    audit_tf_env_probe(a)
    audit_gripper_ab_null(a)
    audit_p3_frame_check(a)
    audit_judge_edit_families(a)
    audit_bridge_join_probe(a)
    audit_gate_factorial(a)
    audit_floor_invariance(a)
    audit_fdir_null(a)
    audit_collision_decomposition(a)
    audit_dt_decode_equivalence(a)
    audit_rollout_insuite(a)
    audit_rollout_edited_arm(a)
    audit_rollout_filmstrip(a)
    audit_arm_pairing_defect(a)
    audit_rank_correlation(a, d)
    audit_overview_figure(a, d)
    audit_arr_submission(a)
    audit_derived_paths_are_portable(a)

    # The manuscript states how many claims this script checks. Let the script
    # verify its own advertised size, so adding a check cannot silently make
    # the paper's description of the audit stale.
    #
    # Accept the LaTeX thousands separator: past 1,000 the count is typeset
    # $1{,}000$, and a \d+ pattern stopped matching it -- which surfaced as
    # "artifact missing", i.e. the check reporting itself unverifiable rather
    # than reporting a mismatch. Strip the separator before comparing.
    quoted = (re.search(r"checks \$([\d{},]+)\$ claims", TEX.read_text())
              if TEX.exists() else None)
    a.check("Release integrity (DATASHEET / LICENSE / artifact counts)",
            "the claim count the manuscript advertises matches this script",
            len(a.rows) + 1,
            int(re.sub(r"[^\d]", "", quoted.group(1))) if quoted else None,
            source="cot_faith_iclr.tex: 'checks $N$ claims'")

    rc = a.report()
    if args.json:
        Path(args.json).write_text(json.dumps(a.rows, indent=2))
        print(f"[json] wrote {args.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
