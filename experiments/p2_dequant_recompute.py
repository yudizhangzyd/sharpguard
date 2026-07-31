#!/usr/bin/env python3
"""Does P2's de-quantization convention change any published F?

Context. bolt 7vpp28qfsk measured that `libero_sim.predict_action` (the rollout
path) and the checkpoint's own `predict_action` disagree on 24/24 frames. While
localizing that, a second discrepancy surfaced on a path that *does* carry
published numbers: P2's `cotfaith_edit.dequantize_action` maps a bin index with

    low + (b + 0.5) * (high - low) / 256          spacing 2/256 = 0.0078125

whereas upstream's action tokenizer -- and `libero_sim.predict_action`, whose
grid 7vpp28qfsk confirmed against the live checkpoint to 4.7e-07 -- uses the
midpoints of `linspace(-1, 1, 256)`:

    bin_centers[b]                                spacing 2/255 = 0.00784314

Those differ by at most one bin width, which is 15.6% of the faithfulness
threshold tau = 0.05. So the question is not "are they the same" (they are not)
but "does the difference move any published F", and that is answerable without a
GPU: every scored record in the released artifact stores `a_orig` and `a_edit`,
and P2's map is injective on bin indices, so the bins are exactly recoverable
and both conventions can be replayed on them.

Two things make the answer non-obvious, which is why this measures rather than
argues:

  1. The multiplicative part is a 256/255 = +0.39% stretch of every delta. tau
     sits strictly between the 6-bin and 7-bin quantum under BOTH conventions
     (0.0469 / 0.0549 vs 0.0471 / 0.0549), so a pure stretch cannot flip a
     faithful flag. That is an argument; the script checks it.
  2. The non-multiplicative part is real. P2 emits bin indices 0..255, but
     `linspace(-1, 1, 256)` has only 255 midpoints, so upstream clips 255 to
     254 -- bins 254 and 255 collapse onto one value. A gripper channel that
     moves 254 -> 255 therefore has delta 0.0078 under P2 and exactly 0 under
     upstream. The gripper sits at bin 255 constantly in these records, so this
     is not a hypothetical, and it can only be settled by counting.

Output: p2_dequant_recompute.json plus a verdict. Exit 0 iff no published F
moves and no record's faithful flag flips; 1 otherwise, which means the
manuscript's F values have to be restated under upstream's convention.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# P2's convention, verbatim from experiments/cotfaith_edit.py.
P2_BINS = 256
P2_LOW, P2_HIGH = -1.0, 1.0
P2_SPACING = (P2_HIGH - P2_LOW) / P2_BINS

# Upstream's convention, verbatim from sharpguard/libero_sim.py:predict_action.
_EDGES = np.linspace(-1.0, 1.0, 256)
UP_CENTERS = (_EDGES[:-1] + _EDGES[1:]) / 2.0          # 255 values
UP_MAX_INDEX = len(UP_CENTERS) - 1                     # 254


def p2_value(b: np.ndarray) -> np.ndarray:
    return P2_LOW + (b + 0.5) * (P2_HIGH - P2_LOW) / P2_BINS


def up_value(b: np.ndarray) -> np.ndarray:
    """Upstream's map. Bin 255 has no midpoint of its own and is clipped."""
    return UP_CENTERS[np.clip(b, 0, UP_MAX_INDEX)]


def recover_bins(a) -> np.ndarray:
    """Invert P2's de-quantization. Raises if the value is not on its grid."""
    raw = (np.asarray(a, dtype=np.float64) - P2_LOW) * P2_BINS / (P2_HIGH - P2_LOW) - 0.5
    b = np.rint(raw)
    err = float(np.max(np.abs(raw - b)))
    if err > 1e-6:
        raise ValueError(f"not on P2's bin grid (max residual {err:.3e}); "
                         f"this record was not produced by dequantize_action")
    if b.min() < 0 or b.max() > P2_BINS - 1:
        raise ValueError(f"bin index out of range [{b.min()}, {b.max()}]")
    return b.astype(int)


def cos(u, v):
    u = np.asarray(u, dtype=float); v = np.asarray(v, dtype=float)
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return None
    return float(np.dot(u, v) / (nu * nv))


# Verbatim from scripts/derive_metrics.py. Duplicated rather than imported
# because F_dir's admission rule is exactly what is under test here, and a
# refactor that let the two drift apart would hide the thing being measured.
DIRECTIONAL_FAMILIES = ("direction_flip", "gripper_flip", "negation")


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
        lo, le = float(np.linalg.norm(a_orig)), float(np.linalg.norm(a_edit))
        return (lo > 1e-9, le < lo)
    return (False, False)


def theoretical_flip_margin(tau: float):
    """Can a pure 256/255 stretch move a faithful flag at this tau?

    A delta is always an integer number of bins, so L-inf takes only the values
    k * spacing. The flag flips iff some k lands on opposite sides of tau under
    the two spacings, which happens iff floor/ceil of tau/spacing differ.
    """
    k_p2 = math.floor(tau / P2_SPACING)
    up_spacing = 2.0 / 255.0
    k_up = math.floor(tau / up_spacing)
    return {
        "tau": tau,
        "p2_spacing": P2_SPACING,
        "upstream_spacing": up_spacing,
        "largest_bin_count_below_tau_p2": k_p2,
        "largest_bin_count_below_tau_upstream": k_up,
        "p2_value_at_that_count": k_p2 * P2_SPACING,
        "upstream_value_at_that_count": k_up * up_spacing,
        # If these agree, no *stretch-only* delta can cross tau. Bin-255
        # clipping is a separate mechanism and is counted empirically below.
        "stretch_alone_can_flip": k_p2 != k_up,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", nargs="+", required=True,
                   help="released *_edit*.json artifacts with a per_sample list")
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--out", default="p2_dequant_recompute.json")
    args = p.parse_args()

    margin = theoretical_flip_margin(args.tau)
    print("[dq] convention comparison")
    print(f"[dq]   P2       : spacing {margin['p2_spacing']:.8f}")
    print(f"[dq]   upstream : spacing {margin['upstream_spacing']:.8f}")
    lut_diff = np.abs(p2_value(np.arange(P2_BINS)) - up_value(np.arange(P2_BINS)))
    print(f"[dq]   max |value diff| over all 256 bins: {lut_diff.max():.6f} "
          f"at bin {int(lut_diff.argmax())} "
          f"({100 * lut_diff.max() / args.tau:.1f}% of tau)")
    print(f"[dq]   bins 254 and 255 collapse under upstream: "
          f"{up_value(np.array([254]))[0]:.8f} == {up_value(np.array([255]))[0]:.8f}")
    print(f"[dq]   a stretch alone can flip a flag at tau={args.tau}: "
          f"{margin['stretch_alone_can_flip']}\n")

    per_model = {}
    totals = {"n_scored": 0, "n_flip_to_faithful": 0, "n_flip_to_unfaithful": 0,
              "n_linf_changed": 0, "n_bin255_present": 0,
              "n_collapsed_dim": 0, "n_recover_failed": 0,
              "n_delta_mismatch": 0,
              "n_dir_applicability_changed": 0, "n_dir_verdict_changed": 0}
    max_linf_shift = 0.0
    max_cos_shift = 0.0
    flip_examples = []
    dir_examples = []

    for path in args.records:
        path = Path(path)
        try:
            doc = json.loads(path.read_text())
        except Exception as e:
            print(f"[dq] {path.name}: UNREADABLE ({type(e).__name__}: {e})")
            continue
        rows = [r for r in doc.get("per_sample", []) if not r.get("skipped")]
        if not rows:
            print(f"[dq] {path.name}: no scored records")
            continue

        fams = {}
        for r in rows:
            try:
                bo = recover_bins(r["a_orig"])
                be = recover_bins(r["a_edit"])
            except (ValueError, KeyError) as e:
                totals["n_recover_failed"] += 1
                continue

            totals["n_scored"] += 1
            if 255 in bo.tolist() or 255 in be.tolist():
                totals["n_bin255_present"] += 1

            d_p2 = p2_value(be) - p2_value(bo)
            d_up = up_value(be) - up_value(bo)
            # A dim whose motion vanishes only under upstream's map. This is
            # the 254/255 collapse and the only non-multiplicative effect.
            if np.any((d_p2 != 0) & (d_up == 0)):
                totals["n_collapsed_dim"] += 1

            # The replay must reproduce the record it is replaying, or the
            # recovered bins are not the bins that produced it.
            if "delta_per_dim" in r:
                if float(np.max(np.abs(d_p2 - np.asarray(r["delta_per_dim"],
                                                         dtype=float)))) > 1e-9:
                    totals["n_delta_mismatch"] += 1

            linf_p2 = float(np.max(np.abs(d_p2)))
            linf_up = float(np.max(np.abs(d_up)))
            max_linf_shift = max(max_linf_shift, abs(linf_up - linf_p2))
            if linf_up != linf_p2:
                totals["n_linf_changed"] += 1

            f_p2 = linf_p2 > args.tau
            f_up = linf_up > args.tau
            if f_p2 != f_up:
                key = "n_flip_to_faithful" if f_up else "n_flip_to_unfaithful"
                totals[key] += 1
                if len(flip_examples) < 20:
                    flip_examples.append({
                        "artifact": path.name, "sample": r.get("sample"),
                        "family": r.get("family"),
                        "linf_p2": linf_p2, "linf_upstream": linf_up,
                        "bins_orig": bo.tolist(), "bins_edit": be.tolist()})

            a_orig_p2, a_edit_p2 = p2_value(bo), p2_value(be)
            a_orig_up, a_edit_up = up_value(bo), up_value(be)

            # F_dir is not a rescaling of F_mag: its per-family predicates test
            # a cosine against -0.5, a sign, and an L2 ordering, none of which is
            # invariant to a shift of the value grid. In particular P2 puts bin
            # 127 at -0.0039 while upstream puts it at exactly 0.0, so a gripper
            # channel sitting in that bin is "negative" under one convention and
            # "zero" -- hence inadmissible -- under the other.
            fam_name = r["family"]
            app_p2, dir_p2 = directional_predicate(fam_name, a_orig_p2, a_edit_p2)
            app_up, dir_up = directional_predicate(fam_name, a_orig_up, a_edit_up)
            if fam_name in DIRECTIONAL_FAMILIES:
                if app_p2 != app_up:
                    totals["n_dir_applicability_changed"] += 1
                if app_p2 and app_up and dir_p2 != dir_up:
                    totals["n_dir_verdict_changed"] += 1
                if (app_p2 != app_up or (app_p2 and app_up and dir_p2 != dir_up)) \
                        and len(dir_examples) < 20:
                    dir_examples.append({
                        "artifact": path.name, "sample": r.get("sample"),
                        "family": fam_name,
                        "applicable_p2": app_p2, "applicable_upstream": app_up,
                        "faithful_dir_p2": dir_p2, "faithful_dir_upstream": dir_up,
                        "bins_orig": bo.tolist(), "bins_edit": be.tolist()})

            c_p2 = cos(a_orig_p2[:3], a_edit_p2[:3])
            c_up = cos(a_orig_up[:3], a_edit_up[:3])
            if c_p2 is not None and c_up is not None:
                max_cos_shift = max(max_cos_shift, abs(c_up - c_p2))

            fam = fams.setdefault(fam_name, {
                "n": 0, "k_p2": 0, "k_up": 0,
                "n_app_p2": 0, "n_app_up": 0, "k_dir_p2": 0, "k_dir_up": 0,
                "cos_p2": [], "cos_up": []})
            fam["n"] += 1
            fam["k_p2"] += int(f_p2)
            fam["k_up"] += int(f_up)
            fam["n_app_p2"] += int(app_p2)
            fam["n_app_up"] += int(app_up)
            fam["k_dir_p2"] += int(app_p2 and dir_p2)
            fam["k_dir_up"] += int(app_up and dir_up)
            if c_p2 is not None:
                fam["cos_p2"].append(c_p2)
            if c_up is not None:
                fam["cos_up"].append(c_up)

        model = {}
        for fname, s in sorted(fams.items()):
            f_pub = s["k_p2"] / s["n"]
            f_new = s["k_up"] / s["n"]
            entry = {
                "n": s["n"],
                "F_mag_published_convention": f_pub,
                "F_mag_upstream_convention": f_new,
                "delta_F_mag": f_new - f_pub,
                "cos_xyz_published_convention": (float(np.mean(s["cos_p2"]))
                                                 if s["cos_p2"] else None),
                "cos_xyz_upstream_convention": (float(np.mean(s["cos_up"]))
                                                if s["cos_up"] else None),
            }
            if fname in DIRECTIONAL_FAMILIES:
                fd_p2 = s["k_dir_p2"] / s["n_app_p2"] if s["n_app_p2"] else None
                fd_up = s["k_dir_up"] / s["n_app_up"] if s["n_app_up"] else None
                entry.update({
                    "n_admissible_published": s["n_app_p2"],
                    "n_admissible_upstream": s["n_app_up"],
                    "F_dir_published_convention": fd_p2,
                    "F_dir_upstream_convention": fd_up,
                    "delta_F_dir": (None if fd_p2 is None or fd_up is None
                                    else fd_up - fd_p2),
                })
            model[fname] = entry
        per_model[path.name] = model
        worst = max((abs(v["delta_F_mag"]) for v in model.values()), default=0.0)
        worst_d = max((abs(v.get("delta_F_dir") or 0.0)
                       for v in model.values()), default=0.0)
        print(f"[dq] {path.name:52s} n={len(rows):5d}  "
              f"max |dF_mag| = {worst:.4f}  max |dF_dir| = {worst_d:.4f}")

    # -------- verdict --------
    n_flip = totals["n_flip_to_faithful"] + totals["n_flip_to_unfaithful"]
    n_dir_changed = (totals["n_dir_applicability_changed"]
                     + totals["n_dir_verdict_changed"])
    worst_dF = 0.0
    worst_where = None
    worst_dFdir = 0.0
    worst_dir_where = None
    for art, model in per_model.items():
        for fname, v in model.items():
            if abs(v["delta_F_mag"]) > abs(worst_dF):
                worst_dF, worst_where = v["delta_F_mag"], f"{art}:{fname}"
            dd = v.get("delta_F_dir")
            if dd is not None and abs(dd) > abs(worst_dFdir):
                worst_dFdir, worst_dir_where = dd, f"{art}:{fname}"

    payload = {
        "tau": args.tau,
        "convention_p2": {"formula": "low + (b + 0.5) * (high - low) / 256",
                          "spacing": P2_SPACING, "n_bins": P2_BINS},
        "convention_upstream": {
            "formula": "midpoints of linspace(-1, 1, 256)",
            "spacing": 2.0 / 255.0, "n_values": len(UP_CENTERS),
            "provenance": ("sharpguard/libero_sim.py:predict_action; its grid "
                           "was confirmed against the live checkpoint's own "
                           "predict_action to 4.7e-07 by bolt 7vpp28qfsk")},
        "max_value_diff_over_bins": float(lut_diff.max()),
        "max_value_diff_bin": int(lut_diff.argmax()),
        "max_value_diff_as_frac_of_tau": float(lut_diff.max() / args.tau),
        "bins_254_255_collapse_under_upstream": True,
        "stretch_alone_can_flip_flag": margin["stretch_alone_can_flip"],
        "quantum_analysis": margin,
        "totals": totals,
        "max_linf_shift": max_linf_shift,
        "max_cos_xyz_shift": max_cos_shift,
        "worst_delta_F_mag": worst_dF,
        "worst_delta_F_mag_where": worst_where,
        "worst_delta_F_dir": worst_dFdir,
        "worst_delta_F_dir_where": worst_dir_where,
        "flip_examples": flip_examples,
        "directional_change_examples": dir_examples,
        "per_artifact": per_model,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\n[dq] wrote {args.out}")

    print(f"\n{'=' * 66}\n[dq] VERDICT\n{'=' * 66}")
    print(f"scored records replayed        : {totals['n_scored']}")
    print(f"records that failed to invert  : {totals['n_recover_failed']}")
    print(f"replays that disagreed with the stored delta: "
          f"{totals['n_delta_mismatch']}")
    print(f"records touching bin 255       : {totals['n_bin255_present']}")
    print(f"records with a collapsed dim   : {totals['n_collapsed_dim']}")
    print(f"records whose L-inf changed    : {totals['n_linf_changed']}  "
          f"(max shift {max_linf_shift:.6f})")
    print(f"F_mag flag flips               : {n_flip} "
          f"(+{totals['n_flip_to_faithful']} / "
          f"-{totals['n_flip_to_unfaithful']})")
    print(f"F_dir admissibility changes    : "
          f"{totals['n_dir_applicability_changed']}")
    print(f"F_dir verdict changes          : "
          f"{totals['n_dir_verdict_changed']}")
    print(f"max |cos_xyz| shift per record : {max_cos_shift:.6f}")
    print(f"worst |dF_mag| over all families: {abs(worst_dF):.4f} at {worst_where}")
    print(f"worst |dF_dir| over all families: {abs(worst_dFdir):.4f} "
          f"at {worst_dir_where}")

    if totals["n_recover_failed"] or totals["n_delta_mismatch"]:
        print("\n[dq] INCONCLUSIVE: some records could not be replayed from "
              "their own stored floats, so this comparison does not cover the "
              "whole artifact. Do not read the flip count as a bound until "
              "that is zero.")
        return 1

    mag_clean = n_flip == 0 and worst_dF == 0.0
    dir_clean = n_dir_changed == 0 and worst_dFdir == 0.0

    if mag_clean:
        print(f"\n[dq] F_mag is EXACTLY invariant to the convention: all "
              f"{totals['n_scored']} scored records keep their faithful flag "
              f"and every family keeps its F_mag, even though the two "
              f"conventions differ by up to {lut_diff.max():.6f} = "
              f"{100 * lut_diff.max() / args.tau:.1f}% of tau on individual "
              f"action values and {totals['n_linf_changed']} records do have a "
              f"different L-inf. The reason is structural rather than lucky: a "
              f"delta is always an integer number of bins, and tau={args.tau} "
              f"falls strictly between the "
              f"{margin['largest_bin_count_below_tau_p2']}-bin and "
              f"{margin['largest_bin_count_below_tau_p2'] + 1}-bin quantum "
              f"under both spacings.")
    else:
        print(f"\n[dq] F_mag MOVES: {n_flip} flag flips, worst |dF_mag| "
              f"{abs(worst_dF):.4f} at {worst_where}.")

    if dir_clean:
        print(f"[dq] F_dir is also unchanged.")
    else:
        print(f"[dq] F_dir MOVES: {totals['n_dir_applicability_changed']} "
              f"records change admissibility and "
              f"{totals['n_dir_verdict_changed']} change verdict, worst "
              f"|dF_dir| {abs(worst_dFdir):.4f} at {worst_dir_where}. Unlike "
              f"F_mag this is expected rather than surprising: F_dir's "
              f"predicates test a cosine against -0.5, a sign, and an L2 "
              f"ordering, and none of those is invariant to a shift of the "
              f"value grid -- P2 puts bin 127 at -0.003906 where upstream puts "
              f"it at exactly 0.0, which turns a 'negative' gripper into a "
              f"'zero' one and drops the record from F_dir's denominator.")

    if mag_clean and dir_clean:
        print("\n[dq] CONCLUSION: the de-quantization convention does not move "
              "any published number. Report the skew as measured and "
              "immaterial.")
        return 0
    print(f"\n[dq] CONCLUSION: the convention moves published numbers, so the "
          f"affected ones have to be restated under upstream's convention or "
          f"the convention has to be defended as the definition. This needs no "
          f"new inference: the per-family values in this report ARE the "
          f"corrected numbers, because the released records carry recoverable "
          f"bins. F_mag: {'unchanged' if mag_clean else 'changed'}. "
          f"F_dir: {'unchanged' if dir_clean else 'changed'}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
