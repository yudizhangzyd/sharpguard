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


def rel(path: str) -> str:
    """Make an in-repo path repo-relative for the `source` provenance fields.

    Absolute paths under ROOT used to be written verbatim, which baked the
    author's home directory into a released artifact and made the CI
    reproducibility check ("re-derive, then `git diff --exit-code`")
    unpassable anywhere but that one laptop -- the check failed on paths while
    every number matched. Paths outside ROOT (the /tmp/... run directories on
    the compute cluster) are left alone: they are provenance for where a run
    happened, they are constant strings, and rewriting them would claim the
    artifacts live in this repo when they do not.
    """
    try:
        ap = os.path.abspath(path)
    except TypeError:
        return path
    return (os.path.relpath(ap, ROOT).replace(os.sep, "/")
            if ap.startswith(ROOT + os.sep) else path)


SIG_DIGITS = 12


def quantize(node):
    """Round every float in the output to SIG_DIGITS significant digits.

    Not cosmetic. The CI gate re-derives this file on a clean x86-64 checkout
    and diffs it against the committed copy, which was produced on arm64, and
    that diff was never empty: values like cos_xyz differed in the last one or
    two bits (0.928937235617497 vs 0.9289372356174972, a relative difference
    near 1e-16). Chasing bit-identical doubles across two instruction sets is
    not a reproducibility property worth having, and the alternative -- letting
    the gate stay red -- means the one check that guards against
    manuscript-vs-artifact drift gets ignored.

    12 significant digits is three to four orders of magnitude finer than any
    value this paper quotes (nothing is reported beyond 3 significant digits)
    and still three digits short of a double's ~15-17, so it absorbs
    last-bit platform noise without touching anything a reader can see. Ints
    and bools pass through untouched, so counts and Ns stay exact.
    """
    if isinstance(node, bool) or isinstance(node, int):
        return node
    if isinstance(node, float):
        if node != node or node in (float("inf"), float("-inf")):
            return node          # NaN/inf: json.dump emits these as-is
        return float(f"{node:.{SIG_DIGITS}g}")
    if isinstance(node, dict):
        return {k: quantize(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [quantize(v) for v in node]
    return node

# ---------------------------------------------------------------------------
# ONE canonical run pinned per model.  (v6 mixed two r=32 runs: attention from
# /tmp/cf_sweep/ours-train, edits from /tmp/cf_done/bcihypv3gu.  We now pin
# bcihypv3gu for BOTH and record the other r=32 run as a noise-floor replicate.)
# ---------------------------------------------------------------------------
EDIT_RUNS = {
    # 3 sampling seeds x 13 families for every trained variant. What these
    # replace was one seed x 10 families, which is why seven of eight
    # leaderboard rows had no measured paraphrase floor and so no F_diff at
    # all. Seed 0 of each row reproduces the superseded single run
    # BIT-EXACTLY on 9 of its 10 families; the tenth is location_swap, which
    # moves because these runs postdate the C5 annotation fix and score it at
    # N~70 instead of N=12 (the same correction already applied to
    # ECoT-bridge). So the 3-seed means below are a strict extension of the
    # published numbers, not a re-measurement of them.
    "ours-r8":      ["/tmp/cf_edit13/q5apfb9zxv/cotfaith-edit/seed%d/cot_edit_report.json" % s for s in (0, 1, 2)],
    "ours-r16":     ["/tmp/cf_edit13/tf73jay9at/cotfaith-edit/seed%d/cot_edit_report.json" % s for s in (0, 1, 2)],
    "ours-r32":     ["/tmp/cf_edit13/58vkvnic4b/cotfaith-edit/seed%d/cot_edit_report.json" % s for s in (0, 1, 2)],
    "ours-r64":     ["/tmp/cf_edit13/7wtzrzeffk/cotfaith-edit/seed%d/cot_edit_report.json" % s for s in (0, 1, 2)],
    "ours-no-cot":  ["/tmp/cf_edit13/uq6q5w4vpz/cotfaith-edit/seed%d/cot_edit_report.json" % s for s in (0, 1, 2)],
    "ours-data50A": ["/tmp/cf_edit13/2kun548q37/cotfaith-edit/seed%d/cot_edit_report.json" % s for s in (0, 1, 2)],
    "ours-data50B": ["/tmp/cf_edit13/f8647i5u7s/cotfaith-edit/seed%d/cot_edit_report.json" % s for s in (0, 1, 2)],
    # 3 file-shuffle seeds, post-C5 location_swap fix, includes paraphrase_null
    "ecot-bridge":  ["/tmp/cf_r3_all/ai3sg9h568/cotfaith-edit/cot_edit_report.json",
                     "/tmp/cf_r3_all/xvz7z8eput/cotfaith-edit/cot_edit_report.json",
                     "/tmp/cf_r3_all/bznf3vq5yu/cotfaith-edit/cot_edit_report.json"],
}
# The superseded single-seed, 10-family runs. Kept pinned and mirrored: they are
# one half of the r=32 training-run replicate pair below, and they are what a
# reader comparing against the v6 manuscript will look for.
EDIT_RUNS_SUPERSEDED = {
    "ours-r8":      "/tmp/cf_full_sweep/lora-r8/cotfaith-edit/cot_edit_report.json",
    "ours-r16":     "/tmp/cf_full_sweep/lora-r16/cotfaith-edit/cot_edit_report.json",
    "ours-r32":     "/tmp/cf_done/bcihypv3gu/cotfaith-edit/cot_edit_report.json",
    "ours-r64":     "/tmp/cf_full_sweep/lora-r64/cotfaith-edit/cot_edit_report.json",
    "ours-no-cot":  "/tmp/cf_full_sweep/no-cot/cotfaith-edit/cot_edit_report.json",
    "ours-data50A": "/tmp/cf_full_sweep/data-50A/cotfaith-edit/cot_edit_report.json",
    "ours-data50B": "/tmp/cf_full_sweep/data-50B/cotfaith-edit/cot_edit_report.json",
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
    "/tmp/cf_edit13/q5apfb9zxv/cotfaith-edit/seed0/cot_edit_report.json": "ours_lora-r8_edit_13family_seed0.json",
    "/tmp/cf_edit13/q5apfb9zxv/cotfaith-edit/seed1/cot_edit_report.json": "ours_lora-r8_edit_13family_seed1.json",
    "/tmp/cf_edit13/q5apfb9zxv/cotfaith-edit/seed2/cot_edit_report.json": "ours_lora-r8_edit_13family_seed2.json",
    "/tmp/cf_edit13/tf73jay9at/cotfaith-edit/seed0/cot_edit_report.json": "ours_lora-r16_edit_13family_seed0.json",
    "/tmp/cf_edit13/tf73jay9at/cotfaith-edit/seed1/cot_edit_report.json": "ours_lora-r16_edit_13family_seed1.json",
    "/tmp/cf_edit13/tf73jay9at/cotfaith-edit/seed2/cot_edit_report.json": "ours_lora-r16_edit_13family_seed2.json",
    "/tmp/cf_edit13/58vkvnic4b/cotfaith-edit/seed0/cot_edit_report.json": "ours_lora-r32_edit_13family_seed0.json",
    "/tmp/cf_edit13/58vkvnic4b/cotfaith-edit/seed1/cot_edit_report.json": "ours_lora-r32_edit_13family_seed1.json",
    "/tmp/cf_edit13/58vkvnic4b/cotfaith-edit/seed2/cot_edit_report.json": "ours_lora-r32_edit_13family_seed2.json",
    "/tmp/cf_edit13/7wtzrzeffk/cotfaith-edit/seed0/cot_edit_report.json": "ours_lora-r64_edit_13family_seed0.json",
    "/tmp/cf_edit13/7wtzrzeffk/cotfaith-edit/seed1/cot_edit_report.json": "ours_lora-r64_edit_13family_seed1.json",
    "/tmp/cf_edit13/7wtzrzeffk/cotfaith-edit/seed2/cot_edit_report.json": "ours_lora-r64_edit_13family_seed2.json",
    "/tmp/cf_edit13/uq6q5w4vpz/cotfaith-edit/seed0/cot_edit_report.json": "ours_no-cot_edit_13family_seed0.json",
    "/tmp/cf_edit13/uq6q5w4vpz/cotfaith-edit/seed1/cot_edit_report.json": "ours_no-cot_edit_13family_seed1.json",
    "/tmp/cf_edit13/uq6q5w4vpz/cotfaith-edit/seed2/cot_edit_report.json": "ours_no-cot_edit_13family_seed2.json",
    "/tmp/cf_edit13/2kun548q37/cotfaith-edit/seed0/cot_edit_report.json": "ours_data-50A_edit_13family_seed0.json",
    "/tmp/cf_edit13/2kun548q37/cotfaith-edit/seed1/cot_edit_report.json": "ours_data-50A_edit_13family_seed1.json",
    "/tmp/cf_edit13/2kun548q37/cotfaith-edit/seed2/cot_edit_report.json": "ours_data-50A_edit_13family_seed2.json",
    "/tmp/cf_edit13/f8647i5u7s/cotfaith-edit/seed0/cot_edit_report.json": "ours_data-50B_edit_13family_seed0.json",
    "/tmp/cf_edit13/f8647i5u7s/cotfaith-edit/seed1/cot_edit_report.json": "ours_data-50B_edit_13family_seed1.json",
    "/tmp/cf_edit13/f8647i5u7s/cotfaith-edit/seed2/cot_edit_report.json": "ours_data-50B_edit_13family_seed2.json",
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
    "/tmp/dtfinal/xx2a3ztfgd/cotfaith-deepthink/deepthink_report.json": "deepthink_base.json",
    "/tmp/dtfinal/j8x3v6g4nh/cotfaith-deepthink/deepthink_report.json": "deepthink_sft.json",
    "/tmp/dtfinal/8t2n5ucjcd/cotfaith-deepthink/deepthink_report.json": "deepthink_rl.json",
    "/tmp/cf_r3_all/ik5n2thine/cotfaith-lerobot/bridge_report.json": "cross_corpus_bridge_v2_n30.json",
    "/tmp/cf_r3_all/c2saurubyx/cotfaith-lerobot/bridge_report.json": "cross_corpus_fractal_n30.json",
    "/tmp/cf_r3_all/kbb6g24nyg/cotfaith-lerobot/bridge_report.json": "cross_corpus_bcz_n30.json",
}
# Second model with all three floors: an independent retrain of the no-CoT
# variant (identical config, reasoning_mode=no_cot, r=32, 15k steps, seed 0)
# scored on all 13 families. This is what makes the F2 calibration claim a
# statement about more than one checkpoint.
CALIB_RUN_NO_COT = os.path.join(CANON, "ours_no-cot_edit_13family_calibration.json")

# Same config trained twice -> the run-to-run error bar the leaderboard needs.
# Sampling seeds cannot supply it: they redraw observations from one frozen
# checkpoint, so they bound sampling noise only.
#
# Two such pairs exist. The r=32 pair was previously written off as "two runs
# that differed in configuration"; their args.json disagree only in keys the
# older training script did not yet emit (reasoning_mode, data_fraction), whose
# defaults are the values the newer run records. Same base model, r, alpha, lr,
# steps, and seed -- an independent training run of one configuration, which is
# exactly the replicate the leaderboard needs and the reason its 1.45pp gap
# cannot be dismissed.
TRAIN_REPLICATE_PAIRS = [
    {"label": "ours-no-cot",
     "config": "r=32, alpha=16, lr=2e-5, 15k steps, no_cot, seed 0",
     "rvis": ("ours_no-cot_rvis.json", "ours_no-cot_rvis_retrain.json"),
     "edit": ("ours_no-cot_edit.json",
              "ours_no-cot_edit_13family_calibration.json")},
    {"label": "ours-r32",
     "config": "r=32, alpha=16, lr=2e-5, 15k steps, full CoT, seed 0",
     "rvis": ("ours_lora-r32_rvis.json", "ours_lora-r32_rvis_REPLICATE.json"),
     # The edit half of this pair was left uncomputed while the attention half
     # was published, which understated the error bar by an order of magnitude:
     # attention moves 1.45pp between these two trainings, F moves up to 0.180.
     "edit": ("ours_lora-r32_edit.json",
              "ours_lora-r32_edit_13family_RETRAIN.json")},
]

# 3 sampling seeds x 5 layer sets on the frozen public ECoT-bridge checkpoint.
# "early" (0-3) is what the submission reported; "full" is all 32. The three
# mid/late blocks were added because reporting only the two endpoints left the
# obvious question open -- is layers 0-3 unusual, or is the whole first half of
# the network CoT-leading? -- and the answer changes the claim.
ATTN_SEED_RUNS = {
    "early_layers_0_3": ["ecot_bridge_rvis_earlylayers_seed%d.json" % s for s in (0, 1, 2)],
    "layers_8_11":      ["ecot_bridge_rvis_layers8_11_seed%d.json" % s for s in (0, 1, 2)],
    "layers_16_19":     ["ecot_bridge_rvis_layers16_19_seed%d.json" % s for s in (0, 1, 2)],
    "layers_28_31":     ["ecot_bridge_rvis_layers28_31_seed%d.json" % s for s in (0, 1, 2)],
    "full_layers_0_31": ["ecot_bridge_rvis_fulllayers_seed%d.json" % s for s in (0, 1, 2)],
}
# Depth order for the sweep, endpoints last so the pre-existing two-set
# comparison below reads the same keys it always did.
ATTN_DEPTH_ORDER = ["early_layers_0_3", "layers_8_11", "layers_16_19",
                    "layers_28_31", "full_layers_0_31"]
ATTN_BUCKETS = ["action->cot", "action->visual", "action->instr",
                "action->action_prev"]


def load(p):
    """Resolve a pinned path, preferring the in-repo mirror.

    The mirror wins deliberately. If /tmp won, derived_metrics.json would be
    built from local scratch state on the authors' machine and from
    results_v2/canonical_runs/ on a reviewer's -- so a divergence between the
    two would pass our audit and fail theirs. Reading the released bytes
    everywhere makes the audit we run the audit they run."""
    alt = MIRROR.get(p)
    if alt:
        ap = os.path.join(CANON, alt)
        if os.path.exists(ap):
            with open(ap) as fh:
                return json.load(fh)
    if os.path.exists(p):
        with open(p) as fh:
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


# ---------------------------------------------------------------------------
# action grid: restate the stored actions on the checkpoint's own de-quantization
# ---------------------------------------------------------------------------
# experiments/cotfaith_edit.py de-quantizes a bin index b with
#     -1 + (b + 0.5) * 2/256                        spacing 2/256
# but the checkpoint's own action tokenizer -- and sharpguard/libero_sim.py,
# whose grid was confirmed against the live checkpoint's predict_action to
# 4.7e-07 by bolt 7vpp28qfsk -- uses the midpoints of linspace(-1, 1, 256):
#     bin_centers[b]                                spacing 2/255
# The two differ by up to 0.007797, which is 15.6% of tau = 0.05, so the
# published numbers are stated on the checkpoint's grid rather than on P2's.
#
# F_mag does not care: experiments/p2_dequant_recompute.py replayed all 10,780
# scored records and not one faithful flag moved, because a delta is always an
# integer number of bins and tau falls strictly between the 6-bin and 7-bin
# quantum under both spacings. F_dir and cos_xyz do care -- their predicates test
# a cosine against -0.5, a sign, and an L2 ordering, none of which survives a
# shift of the grid -- so this conversion is what makes those numbers the
# checkpoint's rather than our reimplementation's.
P2_BINS = 256
_GRID_EDGES = [-1.0 + i * (2.0 / 255.0) for i in range(256)]
UP_CENTERS = [(_GRID_EDGES[i] + _GRID_EDGES[i + 1]) / 2.0 for i in range(255)]

# Counted, not silenced: a report from a checkpoint with a different action
# tokenizer (e.g. DeepThinkVLA's FAST) would not sit on P2's bin grid, and
# forcing it onto this one would corrupt it. Such rows pass through untouched
# and the count is published in the artifact.
GRID_PASSTHROUGH = {"rows": 0, "converted": 0}


def _to_checkpoint_grid(a):
    """P2-de-quantized action -> the same bins on the checkpoint's grid.

    Returns the input unchanged if it is not on P2's grid, since then it was not
    produced by cotfaith_edit.dequantize_action and there is no bin to restate.
    """
    out = []
    for x in a:
        raw = (x + 1.0) * P2_BINS / 2.0 - 0.5
        b = int(round(raw))
        if abs(raw - b) > 1e-6 or b < 0 or b > P2_BINS - 1:
            GRID_PASSTHROUGH["rows"] += 1
            return list(a)
        out.append(UP_CENTERS[min(b, len(UP_CENTERS) - 1)])
    GRID_PASSTHROUGH["converted"] += 1
    return out


def _regrid_rows(rows):
    """Restate a list of edit records on the checkpoint's grid, in place-safe."""
    out = []
    for r in rows:
        if "a_orig" not in r or "a_edit" not in r:
            out.append(r)
            continue
        ao = _to_checkpoint_grid(r["a_orig"])
        ae = _to_checkpoint_grid(r["a_edit"])
        d = [e - o for o, e in zip(ao, ae)]
        r2 = dict(r)
        r2["a_orig"], r2["a_edit"] = ao, ae
        r2["delta_per_dim"] = d
        r2["delta_l1_mean"] = sum(abs(x) for x in d) / len(d)
        r2["delta_linf"] = max(abs(x) for x in d)
        out.append(r2)
    return out


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
    for s in _regrid_rows(rep.get("per_sample", [])):
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


def _canon(fname):
    fp = os.path.join(CANON, fname)
    if not os.path.exists(fp):
        return None
    with open(fp) as fh:
        return json.load(fh)


def derive_attention_seed_repeats():
    """Sampling error bars for the four attention buckets, and the depth
    dependence of the bucket ORDERING.

    Two separate questions that the submission conflated:
      (a) how much does alpha move when only the 100-observation draw changes?
      (b) does the reported bucket ordering survive using all 32 layers
          instead of the 4 the submission chose?
    (a) is a noise floor. (b) is a validity question, and the answer is no."""
    out = {}
    for tag, files in ATTN_SEED_RUNS.items():
        reps = [(f, _canon(f)) for f in files]
        reps = [(f, r) for f, r in reps if r]
        if not reps:
            continue
        entry = {"n_seeds": len(reps), "runs": [f for f, _ in reps],
                 "layers": reps[0][1].get("rvis_layers"),
                 "n_per_run": reps[0][1].get("n_samples")}
        for b in ATTN_BUCKETS:
            vals = [r["aggregate"][b]["mean"] for _, r in reps
                    if dig_agg(r, b) is not None]
            if not vals:
                continue
            entry[b.replace("action->", "")] = {
                "mean": mean(vals), "std": std(vals),
                "range_pp": (max(vals) - min(vals)) * 100,
                "per_seed": vals,
            }
        ranked = sorted(((k, v["mean"]) for k, v in entry.items()
                         if isinstance(v, dict) and "mean" in v),
                        key=lambda kv: -kv[1])
        entry["bucket_order"] = [k for k, _ in ranked]
        entry["top_bucket"] = ranked[0][0] if ranked else None
        entry["max_sampling_std_pp"] = max(
            (v["std"] * 100 for v in entry.values()
             if isinstance(v, dict) and "std" in v), default=None)
        out[tag] = entry

    e, f = out.get("early_layers_0_3"), out.get("full_layers_0_31")
    if e and f:
        out["depth_sensitivity"] = {
            "reported_layers_top_bucket": e["top_bucket"],
            "all_layers_top_bucket": f["top_bucket"],
            "ordering_is_preserved": e["bucket_order"] == f["bucket_order"],
            "cot_early": e["cot"]["mean"], "cot_full": f["cot"]["mean"],
            "cot_drop_pp": (e["cot"]["mean"] - f["cot"]["mean"]) * 100,
            "visual_early": e["visual"]["mean"], "visual_full": f["visual"]["mean"],
            "visual_rise_pp": (f["visual"]["mean"] - e["visual"]["mean"]) * 100,
            "note": ("the submitted alpha(cot) > alpha(visual) ordering holds only "
                     "in layers 0-3; over all 32 layers it reverses"),
        }

    # The full sweep. Reporting only the two endpoints left it possible that
    # "layers 0-3 vs all 32" was an averaging artifact rather than a depth
    # effect; with three intermediate blocks it is not. This block exists so
    # the paper can quote the sweep instead of the endpoints.
    depth = [t for t in ATTN_DEPTH_ORDER if t in out]
    blocks = [t for t in depth if t != "full_layers_0_31"]
    if len(blocks) >= 3:
        cot = {t: out[t]["cot"]["mean"] for t in blocks}
        lead = {t: out[t]["top_bucket"] for t in blocks}
        cot_leading = [t for t in blocks if lead[t] == "cot"]
        max_sig = max(out[t]["max_sampling_std_pp"] for t in depth)
        swing = (max(cot.values()) - min(cot.values())) * 100
        out["depth_sweep"] = {
            "layer_sets": depth,
            "n_seeds_each": 3,
            "blocks_probed": blocks,
            "cot_by_block": cot,
            "top_bucket_by_block": lead,
            "bucket_order_by_block": {t: out[t]["bucket_order"] for t in blocks},
            # The load-bearing number: only ONE of the four 4-layer blocks puts
            # CoT on top, and it is the block the submission happened to pick.
            "blocks_where_cot_leads": cot_leading,
            "n_blocks_where_cot_leads": len(cot_leading),
            "n_blocks_probed": len(blocks),
            "cot_swing_pp": swing,
            "cot_min_block": min(cot, key=cot.get),
            "cot_max_block": max(cot, key=cot.get),
            "max_sampling_std_pp": max_sig,
            # Depth variation vs the sampling floor. If this ratio is large the
            # layer set is not a free parameter, it is part of the claim.
            "swing_over_sampling_noise": (swing / max_sig) if max_sig else None,
            "cot_is_monotone_in_depth": (
                list(cot.values()) == sorted(cot.values())
                or list(cot.values()) == sorted(cot.values(), reverse=True)),
            "visual_leads_in_all_but": [t for t in blocks if lead[t] != "visual"],
            "note": ("alpha(cot) leads in exactly %d of %d four-layer blocks, "
                     "and it is the block the submission reported. Across depth "
                     "alpha(cot) swings %.1f pp against a %.2f pp sampling "
                     "floor, so the layer set is part of the claim, not a "
                     "presentation choice." % (len(cot_leading), len(blocks),
                                                swing, max_sig)),
        }
    return out or None


def dig_agg(rep, bucket):
    a = rep.get("aggregate") or rep.get("attention_aggregate") or {}
    return (a.get(bucket) or {}).get("mean")


def derive_training_replicate():
    """The error bar that actually matters for a leaderboard: the same training
    config run twice. Sampling seeds hold the checkpoint fixed and so cannot
    bound the quantity the leaderboard rows differ in."""
    pairs, out = [], {}
    for spec in TRAIN_REPLICATE_PAIRS:
        entry = {"label": spec["label"], "config": spec["config"]}
        ra, rb = (_canon(f) for f in spec["rvis"])
        if ra and rb:
            deltas = {}
            for b in ATTN_BUCKETS:
                va, vb = dig_agg(ra, b), dig_agg(rb, b)
                if va is None or vb is None:
                    continue
                deltas[b.replace("action->", "")] = {
                    "run_A": va, "run_B": vb, "abs_diff_pp": abs(vb - va) * 100}
            entry["attention"] = {
                "runs": list(spec["rvis"]), "layers": ra.get("rvis_layers"),
                "buckets": deltas,
                "max_abs_diff_pp": max((v["abs_diff_pp"] for v in deltas.values()),
                                       default=None),
                "cot_abs_diff_pp": (deltas.get("cot") or {}).get("abs_diff_pp"),
            }
        if spec.get("edit"):
            ea, eb = (_canon(f) for f in spec["edit"])
            if ea and eb:
                aa, ab = ea["aggregate"], eb["aggregate"]
                fams, per = sorted(set(aa) & set(ab)), {}
                for f in fams:
                    va, vb = aa[f]["faithful_rate"], ab[f]["faithful_rate"]
                    per[f] = {"run_A": va, "run_B": vb, "abs_diff": abs(vb - va),
                              "n_A": aa[f]["n"], "n_B": ab[f]["n"]}
                # location_swap is excluded from the headline spread: run A
                # predates the annotation fix and has n=12, so its gap measures
                # the fix, not training-run variation.
                cmp_fams = [f for f in fams
                            if min(per[f]["n_A"], per[f]["n_B"]) >= 50]
                entry["F_per_family"] = {
                    "runs": list(spec["edit"]),
                    "n_families_compared": len(cmp_fams), "families": per,
                    "excluded_low_n": [f for f in fams if f not in cmp_fams],
                    "max_abs_diff": max((per[f]["abs_diff"] for f in cmp_fams),
                                        default=None),
                    "max_abs_diff_family": max(cmp_fams,
                                               key=lambda f: per[f]["abs_diff"])
                                            if cmp_fams else None,
                    "mean_abs_diff": mean([per[f]["abs_diff"] for f in cmp_fams])
                                      if cmp_fams else None,
                }
                # The headline number the calibration section quotes as its
                # noise unit: how far F_bar itself (mean over the 7 non-control
                # families) moves when the SAME config is trained twice. This
                # is the yardstick every cot_specificity_ratio margin is
                # measured against, so it has to be derived, not eyeballed.
                nc = [f for f in NON_CONTROL if f in per]
                if len(nc) == len(NON_CONTROL):
                    fa = mean([per[f]["run_A"] for f in nc])
                    fb = mean([per[f]["run_B"] for f in nc])
                    entry["F_per_family"]["F_bar_run_A"] = fa
                    entry["F_per_family"]["F_bar_run_B"] = fb
                    entry["F_per_family"]["F_bar_abs_diff"] = abs(fb - fa)
        if len(entry) > 2:
            pairs.append(entry)
    if not pairs:
        return None
    cots = [p["attention"]["cot_abs_diff_pp"] for p in pairs
            if p.get("attention", {}).get("cot_abs_diff_pp") is not None]
    maxes = [p["attention"]["max_abs_diff_pp"] for p in pairs
             if p.get("attention", {}).get("max_abs_diff_pp") is not None]
    out = {
        "n_pairs": len(pairs), "pairs": pairs,
        "cot_abs_diff_pp_per_pair": cots,
        "cot_abs_diff_pp_max": max(cots) if cots else None,
        "cot_abs_diff_pp_mean": mean(cots) if cots else None,
        "any_bucket_abs_diff_pp_max": max(maxes) if maxes else None,
    }
    # The F rollup across pairs. Keyed by label rather than by position: the
    # audit used to read "the last pair that has an F block", which silently
    # re-pointed its assertions at a different checkpoint the moment a second
    # pair acquired one.
    out["by_label"] = {p["label"]: p for p in pairs}
    fpairs = [p for p in pairs if p.get("F_per_family")]
    if fpairs:
        worst = max(fpairs, key=lambda p: p["F_per_family"]["max_abs_diff"] or 0)
        wf = worst["F_per_family"]
        out["F_n_pairs"] = len(fpairs)
        out["F_max_abs_diff_over_pairs"] = wf["max_abs_diff"]
        out["F_max_abs_diff_where"] = "%s:%s" % (worst["label"],
                                                 wf["max_abs_diff_family"])
        out["F_mean_abs_diff_per_pair"] = {p["label"]: p["F_per_family"]["mean_abs_diff"]
                                           for p in fpairs}
        out["F_max_abs_diff_per_pair"] = {p["label"]: p["F_per_family"]["max_abs_diff"]
                                          for p in fpairs}
        fbar = {p["label"]: p["F_per_family"]["F_bar_abs_diff"] for p in fpairs
                if p["F_per_family"].get("F_bar_abs_diff") is not None}
        if fbar:
            out["F_bar_abs_diff_per_pair"] = fbar
            out["F_bar_abs_diff_max"] = max(fbar.values())
            out["F_bar_noise_note"] = (
                "how far F_bar moves when the same config is trained twice; the "
                "unit every cot_specificity_ratio margin is measured against")
    return out


def derive_calibration(path, label):
    """Two-sided calibration for one model from its 13-family run(s).

    Everything is compared only against families from the same run(s): mixing
    runs is what produced the 0.340-vs-0.354 discrepancy the audit catches.
    `path` may be a list of sampling-seed replicates of ONE checkpoint, in
    which case each family's F is the mean over seeds and its Wilson interval
    is computed on the pooled counts -- still a within-checkpoint comparison,
    just a less noisy one. It may not mix checkpoints, and nothing here checks
    that for you.
    """
    paths = [path] if isinstance(path, str) else list(path)
    creps = []
    for p in paths:
        r = load(p) if p.startswith("/tmp") else _canon(os.path.basename(p))
        if r:
            creps.append(r)
    if not creps:
        return None
    crep = creps[0]
    ags = [r["aggregate"] for r in creps]
    fams = sorted(set.intersection(*[set(a) for a in ags]))
    fr = {f: mean([a[f]["faithful_rate"] for a in ags]) for f in fams}
    fr_std = {f: std([a[f]["faithful_rate"] for a in ags]) for f in fams}
    n = {f: int(round(mean([a[f]["n"] for a in ags]))) for f in fams}
    n_pooled = {f: sum(a[f]["n"] for a in ags) for f in fams}
    k_pooled = {f: sum(round(a[f]["faithful_rate"] * a[f]["n"]) for a in ags)
                for f in fams}
    f_bar = mean([fr[f] for f in NON_CONTROL if f in fr])
    c = {
        "label": label, "source": [rel(p) for p in paths], "n_families": len(fams),
        "n_runs": len(creps), "seeds": [r.get("seed") for r in creps],
        "seed": crep.get("seed"), "F_bar_non_control": f_bar,
        "families": {f: {"F_mag": fr[f], "F_mag_std": fr_std[f], "n": n[f],
                         "n_pooled": n_pooled[f],
                         "wilson": wilson(k_pooled[f], n_pooled[f])}
                     for f in fams},
    }
    for floor in CALIB_FLOORS:
        if floor in fr:
            c[floor] = fr[floor]
            c[f"F_bar_diff_vs_{floor}"] = f_bar - fr[floor]
    if "instr_random_sub" in fr:
        c["cot_specificity_ratio"] = f_bar / fr["instr_random_sub"] if fr["instr_random_sub"] else None
        c["n_families_above_out_of_cot_control"] = sum(
            1 for f in NON_CONTROL if f in fr and fr[f] >= fr["instr_random_sub"])
    # Two-sided normalization, and whether it is even defined. On a saturated
    # model the floor rises to meet the ceiling and the denominator vanishes.
    if "paraphrase_null" in fr and "cross_task_swap" in fr:
        lo, hi = fr["paraphrase_null"], fr["cross_task_swap"]
        c["ceiling_cross_task_swap"] = hi
        c["dynamic_range"] = hi - lo
        c["calibration_is_degenerate"] = (hi - lo) < 0.05
        c["F_bar_two_sided"] = ((f_bar - lo) / (hi - lo)) if (hi - lo) >= 0.05 else None
    return c


def dig_fam(model_block, fam, key):
    return ((model_block.get("families") or {}).get(fam) or {}).get(key)


DT_RUNS = {
    "DT-base": "deepthink_base.json",
    "DT-SFT":  "deepthink_sft.json",
    "DT-RL":   "deepthink_rl.json",
}


def derive_deepthink_p2():
    """P2 on the SECOND architecture family, with its own floor beside it.

    Until the decode was fixed, the submission's answer to "does P2 generalize
    beyond ECoT?" was three rows of attention and no edit records at all. It now
    has both, and the point of deriving it here rather than quoting F alone is
    that these runs carry `paraphrase_null` (floor) and `cross_task_swap`
    (ceiling) in the SAME run as the semantic families -- so F_diff and the
    two-sided normalization are defined for DeepThinkVLA exactly as they are for
    ECoT-bridge, using the same code path (`per_run_stats`) and the same
    definitions (`derive_calibration`). A cross-family claim built from two
    different estimators would not be a cross-family claim.

    What is NOT available here: `bbox_jitter_null` and `instr_random_sub` are in
    the 13-family calibration set, not the 11-family protocol these runs use, so
    the out-of-CoT control and the CoT-specificity ratio are undefined for
    DeepThinkVLA. Stated, not silently omitted.
    """
    out = {}
    for label, fname in DT_RUNS.items():
        rep = _canon(fname)
        if not rep or not rep.get("per_sample_edit"):
            continue
        # per_run_stats reads rep["per_sample"]; the DeepThinkVLA harness names
        # the same records "per_sample_edit" because that report also carries
        # per_sample_attn. Shimming rather than duplicating keeps one estimator.
        pr = per_run_stats({"per_sample": rep["per_sample_edit"]})
        if not pr:
            continue
        fams = {}
        for fam, st in sorted(pr.items()):
            e = {"n": st["n"], "F_mag": st["F_mag"],
                 "F_mag_wilson": list(wilson(st["k_mag"], st["n"])),
                 "cos_xyz": st["cos_xyz"],
                 "cos_xyz_faithful_subset": st["cos_xyz_faithful_subset"]}
            if fam in DIRECTIONAL_FAMILIES:
                e["F_dir"] = st.get("F_dir")
                e["n_directional"] = st.get("n_directional")
                e["F_dir_wilson"] = list(wilson(st.get("k_dir", 0),
                                                st.get("n_directional", 0)))
            ag = (rep.get("edit_aggregate") or {}).get(fam) or {}
            e["n_skipped"] = ag.get("n_skipped")
            # Recorded because the leaderboard scores chunk step 0 while this
            # model emits 10 steps: if F were an artifact of scoring the first
            # step, the chunk-wide rate would disagree with it.
            e["F_mag_chunk"] = ag.get("faithful_rate_chunk")
            e["delta_linf_mean"] = ag.get("delta_linf_mean")
            e["delta_linf_chunk_mean"] = ag.get("delta_linf_chunk_mean")
            fams[fam] = e

        fr = {f: v["F_mag"] for f, v in fams.items()}
        f_bar = mean([fr[f] for f in NON_CONTROL if f in fr])
        m = {"label": label, "source": rel(os.path.join(CANON, fname)),
             "model": rep.get("model"), "seed": rep.get("seed"),
             "n_families_scored": len(fams),
             "n_attn_ok": rep.get("n_attn_ok"),
             "n_decode_failures": (rep.get("action_decode") or {}).get(
                 "n_sample_failures"),
             "F_bar_non_control": f_bar, "families": fams}
        if "paraphrase_null" in fr:
            m["paraphrase_null"] = fr["paraphrase_null"]
            m["F_bar_diff_vs_paraphrase_null"] = f_bar - fr["paraphrase_null"]
            m["n_families_above_floor"] = sum(
                1 for f in NON_CONTROL if f in fr and fr[f] > fr["paraphrase_null"])
        if "cross_task_swap" in fr and "paraphrase_null" in fr:
            lo, hi = fr["paraphrase_null"], fr["cross_task_swap"]
            m["ceiling_cross_task_swap"] = hi
            m["dynamic_range"] = hi - lo
            m["calibration_is_degenerate"] = (hi - lo) < 0.05
            m["F_bar_two_sided"] = ((f_bar - lo) / (hi - lo)) if (hi - lo) >= 0.05 else None
            # A two-sided score below 0 means F_bar sits BELOW the floor: the
            # semantic edits move the action less than a meaning-preserving
            # paraphrase does. The normalization still computes, but reporting
            # the number alone would read as "scored low on faithfulness" when
            # the actual finding is that the measurement has no signal to
            # normalize. Flagged so the paper cannot quote it unqualified.
            m["F_bar_is_below_floor"] = f_bar < lo
            m["F_bar_two_sided_is_negative"] = (
                m["F_bar_two_sided"] is not None and m["F_bar_two_sided"] < 0)
        # selfsplice_control replaces the CoT with itself, so a nonzero F here
        # would mean the harness manufactures deltas. It is the one family whose
        # expected value is exactly 0.
        if "selfsplice_control" in fr:
            m["selfsplice_control_F"] = fr["selfsplice_control"]
            m["identity_edit_is_exactly_zero"] = (fr["selfsplice_control"] == 0.0)
        out[label] = m

    if not out:
        return None
    # The cross-family claim itself, computed rather than asserted in prose.
    deg = [k for k, v in out.items() if v.get("calibration_is_degenerate")]
    floors = {k: v.get("paraphrase_null") for k, v in out.items()
              if v.get("paraphrase_null") is not None}
    summary = {
        "n_models": len(out),
        "n_with_measured_floor": len(floors),
        "paraphrase_null_by_model": floors,
        "n_degenerate": len(deg),
        "degenerate_models": sorted(deg),
        "F_bar_diff_by_model": {k: v.get("F_bar_diff_vs_paraphrase_null")
                                for k, v in out.items()},
        "any_model_with_negative_F_diff": any(
            (v.get("F_bar_diff_vs_paraphrase_null") or 0) < 0 for v in out.values()),
        "n_models_with_F_bar_below_floor": sum(
            1 for v in out.values() if v.get("F_bar_is_below_floor")),
        # The F6 finding, on the second family: magnitude F says the edit landed,
        # the direction-aware score says it landed the wrong way.
        "direction_flip_F_mag_by_model": {
            k: dig_fam(v, "direction_flip", "F_mag") for k, v in out.items()},
        "direction_flip_F_dir_by_model": {
            k: dig_fam(v, "direction_flip", "F_dir") for k, v in out.items()},
        "max_direction_flip_F_dir": max(
            (dig_fam(v, "direction_flip", "F_dir") or 0) for v in out.values()),
        "all_identity_edits_exactly_zero": all(
            v.get("identity_edit_is_exactly_zero") for v in out.values()),
    }
    summary["note"] = (
        "P2 now covers two architecture families. The finding replicates: on "
        "every DeepThinkVLA checkpoint a meaning-preserving paraphrase moves the "
        "action about as much as the semantic edits do, so magnitude F is no "
        "more interpretable here than on ECoT-bridge. This is the check that "
        "would have dissolved F2 had it come out the other way.")
    out["summary"] = summary
    return out


# Upstream's own per-suite episode budgets, read off OpenVLA's
# experiments/robot/libero/run_libero_eval.py by bolt task htrg4uchwi (a
# source-level diff, not from memory). Duplicated from
# sharpguard.libero_sim.UPSTREAM_MAX_STEPS rather than imported, because this
# script must stay runnable with numpy alone -- importing sharpguard pulls
# torch. tests/test_gripper_transform.py pins the values on the libero_sim
# side, and audit_rollout_gate() asserts the two copies agree.
GATE_UPSTREAM_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# Published Task SR for the openvla-7b-finetuned-libero-* checkpoints, from the
# OpenVLA paper's LIBERO table. The gate exists to reproduce these; the gap is
# the finding, so the target has to be recorded next to the measurement.
GATE_PUBLISHED_SR = {
    "libero_spatial": 0.844,
    "libero_object": 0.881,
    "libero_goal": 0.794,
    "libero_10": 0.539,
}


def derive_rollout_gate():
    """Read the four-suite decoder gate reports as run, with validity flags.

    Every number here is copied out of the per-suite sr.json that the bolt job
    wrote; nothing is recomputed and nothing is averaged across suites. The
    reason for the validity machinery: all four runs passed max_steps=400, which
    is above upstream's budget for three suites and *below* it for libero_10
    (520). A libero_10 episode can therefore be cut off mid-task and recorded as
    a failure, so its 0/50 is not evidence about the policy and must not be
    quoted or pooled as if it were. The flag is computed from the two numbers
    rather than asserted in prose, so a future re-run with a corrected budget
    flips it automatically instead of leaving a stale caveat behind.
    """
    base = os.path.join(ROOT, "results_v2", "canonical_runs", "gate_foursuite")
    suites = {}
    for suite in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        d = os.path.join(base, suite)
        if not os.path.isdir(d):
            continue
        sr = load(os.path.join(d, "sr.json"))
        args = load(os.path.join(d, "args.json"))
        tid_path = os.path.join(d, "bolt_task_id.txt")
        with open(tid_path) as fh:
            tid = fh.read().strip()
        steps = args.get("max_steps")
        budget = GATE_UPSTREAM_MAX_STEPS.get(suite)
        truncated = bool(budget and steps and steps < budget)
        n_total, n_succ = sr.get("n_total"), sr.get("n_success")
        suites[suite] = {
            "model": sr.get("model"),
            "unnorm_key": sr.get("unnorm_key"),
            "n_total": n_total,
            "n_success": n_succ,
            "SR": sr.get("SR"),
            "SR_wilson95": wilson(n_succ, n_total) if n_total else None,
            "published_SR": GATE_PUBLISHED_SR.get(suite),
            "max_steps_run": steps,
            "upstream_max_steps": budget,
            "step_budget_below_upstream": truncated,
            # The single field a reader needs: is this row evidence at all?
            "interpretable": not truncated,
            "n_episodes_canonical_init": sr.get("n_episodes_canonical_init"),
            "all_episodes_used_canonical_init": sr.get(
                "all_episodes_used_canonical_init"),
            "bolt_task": tid,
            "source": rel(os.path.join(d, "sr.json")),
        }

    valid = {k: v for k, v in suites.items() if v["interpretable"]}
    best = max(valid.values(), key=lambda v: v["SR"]) if valid else None
    summary = {
        "n_suites_run": len(suites),
        "n_episodes_total": sum(v["n_total"] or 0 for v in suites.values()),
        "n_suites_interpretable": len(valid),
        "suites_excluded_for_step_budget": sorted(
            k for k, v in suites.items() if not v["interpretable"]),
        "all_episodes_canonical_init": all(
            v["all_episodes_used_canonical_init"] for v in suites.values()),
        "best_interpretable_SR": best["SR"] if best else None,
        "best_interpretable_suite": (
            [k for k, v in valid.items() if v is best][0] if best else None),
        # The shape of the failure, which is what drove the diagnosis: a
        # nonzero-but-degraded SR means the harness is degraded, not that
        # success is impossible. That rules out a control channel which is
        # simply inverted and points at an input-distribution mismatch.
        "any_interpretable_suite_nonzero": any(v["SR"] > 0 for v in valid.values()),
        "gate_passed": bool(valid) and all(
            v["SR"] >= 0.5 * (v["published_SR"] or 1.0) for v in valid.values()),
    }
    summary["note"] = (
        "The gate does not pass. Three suites are interpretable and all three "
        "fall far below the published SR; libero_10 is excluded because it ran "
        "400 steps against upstream's 520. A source-level diff against "
        "openvla/openvla (bolt htrg4uchwi) found two differences that were live "
        "in these runs: the gripper channel was passed through where upstream "
        "applies -sign(2g-1), and every frame skipped upstream's JPEG "
        "round-trip plus lanczos3 resize to 224. Both are now implemented and "
        "under test; these numbers are the pre-fix baseline the corrected runs "
        "are measured against.")
    return {"suites": suites, "summary": summary}


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
        rec = {"n": len(rows), "source": rel(p), "mass": {}, "mass_std": {},
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
        "DT-base": "/tmp/dtfinal/xx2a3ztfgd/cotfaith-deepthink/deepthink_report.json",
        "DT-SFT":  "/tmp/dtfinal/j8x3v6g4nh/cotfaith-deepthink/deepthink_report.json",
        "DT-RL":   "/tmp/dtfinal/8t2n5ucjcd/cotfaith-deepthink/deepthink_report.json",
    }
    for name, dpath in DT_PATHS.items():
        rep = load(dpath)
        if not rep or not rep.get("n_attn_ok"):
            continue
        a = rep["attention_aggregate"]
        dt[name] = {"n": rep.get("n_attn_ok"), "source": rel(dpath), "model": rep.get("model"),
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
            "source": rel(path), "dataset": rep.get("dataset"),
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

    # -------- two-sided calibration floors, now on TWO models --------
    calib = derive_calibration(CALIB_RUN, "ecot-bridge")
    calib_no_cot = derive_calibration(CALIB_RUN_NO_COT, "ours-no-cot")
    calib_by_model = {c["label"]: c for c in (calib, calib_no_cot) if c}
    # Every trained variant now has all three floors from its own 3-seed
    # 13-family run, so the calibrated column is no longer 2 rows out of 11.
    # `ours-no-cot` appears twice by design and the two entries are NOT
    # interchangeable: `-retrain` is the independently trained checkpoint that
    # makes the F2 claim a statement about more than one model, and the entry
    # written here is the same checkpoint every other leaderboard row uses.
    # The retrain keeps its own key rather than overwriting or being
    # overwritten.
    if "ours-no-cot" in calib_by_model:
        calib_by_model["ours-no-cot-retrain"] = calib_by_model.pop("ours-no-cot")
        calib_no_cot = calib_by_model["ours-no-cot-retrain"]
        calib_no_cot["label"] = "ours-no-cot-retrain"
    for name in EDIT_RUNS:
        if name == "ecot-bridge":
            continue                       # already calibrated from CALIB_RUN
        c = derive_calibration(EDIT_RUNS[name], name)
        if c:
            calib_by_model[name] = c
    # The headline of the calibration story is the CONTRAST: on the saturated
    # model the floor rises to within 0.010 of the ceiling and F is
    # uninterpretable; on the low-F model the same protocol has a real dynamic
    # range. So the defect is a property of saturation, not of the protocol.
    calib_contrast = None
    if calib and calib_no_cot:
        calib_contrast = {
            "models": [calib["label"], calib_no_cot["label"]],
            "dynamic_range": {c["label"]: c.get("dynamic_range")
                              for c in (calib, calib_no_cot)},
            "paraphrase_floor": {c["label"]: c.get("paraphrase_null")
                                 for c in (calib, calib_no_cot)},
            "F_bar": {c["label"]: c["F_bar_non_control"]
                      for c in (calib, calib_no_cot)},
            "degenerate": {c["label"]: c.get("calibration_is_degenerate")
                           for c in (calib, calib_no_cot)},
            "n_degenerate": sum(1 for c in (calib, calib_no_cot)
                                if c.get("calibration_is_degenerate")),
            "note": ("both models were scored with the identical 13-family "
                     "protocol at n=100; only the high-F model loses its "
                     "dynamic range"),
        }

    attn_seeds = derive_attention_seed_repeats()
    train_rep = derive_training_replicate()

    # Noise hierarchy: three DIFFERENT quantities the submission treated as one.
    hierarchy = None
    if attn_seeds and train_rep and noise:
        samp = max(v["max_sampling_std_pp"] for k, v in attn_seeds.items()
                   if isinstance(v, dict) and v.get("max_sampling_std_pp") is not None)
        tr = train_rep.get("any_bucket_abs_diff_pp_max")
        tr_cot = train_rep.get("cot_abs_diff_pp_max")
        hierarchy = {
            "sampling_std_pp": samp,
            "sampling_note": "3 draws of n=100 from ONE frozen checkpoint",
            "training_run_diff_pp": tr,
            "training_run_cot_diff_pp": tr_cot,
            "training_run_diff_pp_per_pair": train_rep.get("cot_abs_diff_pp_per_pair"),
            "n_training_replicate_pairs": train_rep.get("n_pairs"),
            "training_run_note": ("independent trainings of the SAME config; "
                                  "this is the quantity leaderboard rows "
                                  "differ in"),
            "cross_variant_spread_pp": noise.get("cluster_spread_pp"),
            "cross_variant_note": "spread of alpha(cot) across the 7 ECoT variants",
            "spread_over_training_run": (noise.get("cluster_spread_pp") / tr)
                                        if tr else None,
            "spread_over_training_run_cot": (noise.get("cluster_spread_pp") / tr_cot)
                                            if tr_cot else None,
            "within_family_ordering_supported": bool(
                tr_cot and noise.get("cluster_spread_pp", 0) > 3 * tr_cot),
            "caveat": ("two replicate pairs give a magnitude, not a "
                       "distribution; a 2.30pp spread against a 1.45pp "
                       "same-config training difference does not establish "
                       "any within-family ordering"),
        }

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
        "deepthink_p2": derive_deepthink_p2(),
        "cross_corpus_n30": cross,
        "attention_noise_floor": noise,
        "calibration_floors": calib,
        "calibration_by_model": calib_by_model,
        "calibration_contrast": calib_contrast,
        "attention_seed_repeats": attn_seeds,
        "training_replicate": train_rep,
        "noise_hierarchy": hierarchy,
        "rollout_gate": derive_rollout_gate(),
    }
    dest = os.path.join(ROOT, "results_v2", "derived_metrics.json")
    with open(dest, "w") as fh:
        json.dump(quantize(out), fh, indent=1)
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
    if calib_contrast:
        print("\n=== calibration contrast across the two calibrated models ===")
        for m in calib_contrast["models"]:
            c = calib_by_model[m]
            print(f"  {m:14s} floor={c.get('paraphrase_null'):.3f} "
                  f"ceil={c.get('ceiling_cross_task_swap'):.3f} "
                  f"range={c.get('dynamic_range'):+.3f} "
                  f"F_bar={c['F_bar_non_control']:.3f} "
                  f"degenerate={c.get('calibration_is_degenerate')}")
    if attn_seeds:
        print("\n=== attention: sampling error bars and depth sensitivity ===")
        for tag, e in attn_seeds.items():
            if tag in ("depth_sensitivity", "depth_sweep"):
                continue
            print(f"  {tag:18s} n_seeds={e['n_seeds']} order={e['bucket_order']}")
            for b in ("cot", "visual", "instr", "action_prev"):
                if b in e:
                    print(f"       {b:12s} {e[b]['mean']:.4f} +-{e[b]['std']:.5f} "
                          f"range={e[b]['range_pp']:.3f}pp")
        ds = attn_seeds.get("depth_sensitivity")
        if ds:
            print(f"  ordering preserved across depth: {ds['ordering_is_preserved']} "
                  f"({ds['reported_layers_top_bucket']} -> {ds['all_layers_top_bucket']})")
        sw = attn_seeds.get("depth_sweep")
        if sw:
            print(f"  cot leads in {sw['n_blocks_where_cot_leads']}/"
                  f"{sw['n_blocks_probed']} 4-layer blocks "
                  f"({', '.join(sw['blocks_where_cot_leads'])}); "
                  f"cot swings {sw['cot_swing_pp']:.1f}pp across depth = "
                  f"{sw['swing_over_sampling_noise']:.0f}x the sampling floor")
    if train_rep:
        print("\n=== same-config training replicate (the leaderboard error bar) ===")
        for pr in train_rep["pairs"]:
            at = pr.get("attention", {})
            print(f"  {pr['label']:14s} |d alpha(cot)|={at.get('cot_abs_diff_pp'):.2f}pp "
                  f"max bucket |d|={at.get('max_abs_diff_pp'):.2f}pp")
            fp = pr.get("F_per_family")
            if fp:
                print(f"  {'':14s} F: max|d|={fp['max_abs_diff']:.3f} on "
                      f"{fp['max_abs_diff_family']}, mean|d|={fp['mean_abs_diff']:.3f} "
                      f"over {fp['n_families_compared']} families")
        print(f"  -> alpha(cot) training-run |delta| per pair: "
              f"{[round(v,2) for v in train_rep['cot_abs_diff_pp_per_pair']]} pp")
    if hierarchy:
        print("\n=== noise hierarchy ===")
        print(json.dumps(hierarchy, indent=1))
    if noise:
        print("\n=== attention noise floor ===")
        print(json.dumps(noise, indent=1))


if __name__ == "__main__":
    main()
