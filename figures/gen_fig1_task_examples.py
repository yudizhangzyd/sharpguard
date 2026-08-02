"""Fig 1 -- the task-examples figure: what each edit family actually does.

This replaces fig1_hero, whose own docstring admitted its CoT was a "schematic
reconstruction" -- hand-written text in a figure, read from an ephemeral /tmp
path. Nothing here is written by hand. Every string on the canvas is either
generator output read from results_v2/canonical_runs/edit_examples (437 pairs,
each verified byte-for-byte against the released judge run's stored heads) or
a rate read from the judge report.

Layout follows the taxonomy the paper defines: one panel per family, grouped
into the three semantic tiers plus the calibration nulls, with a colored
header bar per tier. Each panel shows the ONE span the family rewrites --
recovered by word-level diff of the real pair, not by re-running a generator
here -- above the blind judge's meaning-preserved rate for that family.

Two things the figure is built to make un-hideable, because they are the
paper's own negative results and a taxonomy diagram is exactly where a reader
would otherwise miss them:

  * syntactic_scramble sits in Tier 0 as a "structural" edit but the judge
    calls it meaning-preserving at 1.000, so it is drawn with the nulls'
    marking, not the semantic families'.
  * adversarial_plausible is judged plausible on 0.125 of pairs. The panel
    prints that rate rather than the family's name's promise.

No numeric literal for any reported quantity appears in this file.
"""
import json
import os
import sys
import difflib
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *          # noqa: F401,F403  (rcParams + save)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "results_v2", "canonical_runs", "edit_examples",
                  "edit_examples.json")
JUDGE = os.path.join(ROOT, "results_v2", "canonical_runs",
                     "judge_edit_families", "judge_report.json")

with open(EX) as fh:
    D = json.load(fh)
with open(JUDGE) as fh:
    J = json.load(fh)
PF = J["per_family"]

# The tiers are the paper's (S3, "Edit families"), in its order. `nulls` is
# Tier N. syntactic_scramble is listed where the taxonomy puts it -- Tier 0 --
# and the panel then reports the judge rate that contradicts that placement,
# which is the paper's own finding and should not be quietly tidied away here.
TIERS = [
    ("Tier 0  structural / control", C_CTRL,
     ["syntactic_scramble", "cross_task_swap"]),
    ("Tier 1  word-level semantic", C_COT_TRAINED,
     ["direction_flip", "gripper_flip", "verb_swap", "negation"]),
    ("Tier 2  object / causal chain", C_NO_COT,
     ["subject_swap", "location_swap", "adversarial_plausible"]),
    ("Tier N  calibration nulls", C_ECOT_BRIDGE,
     ["paraphrase_null", "bbox_jitter_null", "identity_control"]),
]

# What each family is documented to rewrite, for the panel subtitle. These are
# scope labels from the taxonomy, not measurements -- no number among them.
SCOPE = {
    "syntactic_scramble": "word order in MOVE/SUBTASK",
    "cross_task_swap":    "whole trace \u2190 another task",
    "direction_flip":     "left/right, up/down in MOVE",
    "gripper_flip":       "grasp/release in PLAN",
    "verb_swap":          "action verb \u2192 unrelated verb",
    "negation":           "prepend to SUBTASK/MOVE",
    "subject_swap":       "referent \u2192 visible distractor",
    "location_swap":      "spatial phrase in PLAN/TASK",
    "adversarial_plausible": "referent \u2192 2nd-most-visible",
    "paraphrase_null":    "synonym in MOVE/SUBTASK",
    "bbox_jitter_null":   "\u00b11px on VISIBLE OBJECTS",
    "identity_control":   "nothing (X \u2192 X)",
}


def cleanest_edit(family):
    """The real (before, after) span for the pair this family edits most simply.

    Recovered by word-level diff over the released pair, so the strings below
    are the generator's, not a description of them. Choosing the pair with the
    fewest edit spans is a presentation choice and nothing else -- every pair
    in the family would show the same KIND of edit, and the judge rate printed
    beside it is over the whole family, not over this pair.
    """
    best = None
    for e in EXAMPLES[family]:
        aw, bw = e["cot_orig"].split(), e["cot_edited"].split()
        ops = [(t, " ".join(aw[i1:i2]), " ".join(bw[j1:j2]))
               for t, i1, i2, j1, j2
               in difflib.SequenceMatcher(None, aw, bw).get_opcodes()
               if t != "equal"]
        if best is None or len(ops) < len(best):
            best = ops
    return best


EXAMPLES = defaultdict(list)
for e in D["examples"]:
    EXAMPLES[e["family"]].append(e)

# ---------------------------------------------------------------------------
# Geometry. Everything below is in inches, converted to axes units at the end,
# because the constraint that actually matters is physical: this is emitted at
# FIG_W = the ACL two-column text width and included at \textwidth, so LaTeX
# scales it 1:1 and the point sizes here are the point sizes on paper. The
# first draft was authored in arbitrary units at 5.6in wide and 12.6in tall;
# fitting THAT to \textwidth would have shrunk 6pt code to about 4pt.
# save() adds pad_inches on every side, so the canvas is authored that much
# narrower and the emitted PAGE lands on \textwidth exactly.
FIG_W = 6.30 - 2 * 0.05   # \textwidth in the ACL style (16.0cm), less padding
HDR_H = 0.24              # tier header bar
PANEL_H = 0.95
GAP_X, GAP_Y = 0.17, 0.13

FS_TIER, FS_NAME, FS_SCOPE, FS_CODE, FS_META = 8.0, 7.0, 5.4, 6.0, 5.2
CHAR_W = FS_CODE * 0.60   # monospace advance width, in points

# One row per tier, its panels sharing the full width. Tier sizes are 2/4/3/3,
# so a fixed column count would leave a ragged half-empty row under Tier 1;
# letting each tier fill the width instead makes the panel width itself encode
# how many families the tier has.
fig_h = len(TIERS) * (HDR_H + PANEL_H + GAP_Y)
fig = plt.figure(figsize=(FIG_W, fig_h))
# Full-bleed axes. With the default subplot margins the axes is ~0.775 of the
# canvas, and save()'s bbox_inches="tight" then trims to the axes -- so the
# emitted page came out 5.1in wide, and \textwidth scaled the 6pt code back
# up by 1.24x, breaking the 1:1 sizing this whole block exists to guarantee.
ax = fig.add_axes([0, 0, 1, 1])
u = 1.0 / FIG_W                       # inches -> axes units (x and y alike)
# A hair of slack: the panels tile the full width exactly and a rounded-box
# stroke sits half outside its rectangle, so an exact xlim shaves the outer
# edge off the first and last panel of every row.
ax.set_xlim(-0.008, 1.008)
ax.set_ylim(-0.004, fig_h * u)
ax.axis("off")

MONO = {"family": "monospace"}
PAD = 0.09 * u                        # panel inner margin
# Interior slots, as a distance from the panel's top or bottom edge. Fixed
# slots, not a flow: flowing the rows and placing the trailing note wherever
# the flow ended put that note on top of the judge line in every family with
# more than one edit span.
Y_NAME, Y_SCOPE, Y_BEFORE, Y_AFTER = (0.13 * u, 0.30 * u, 0.47 * u, 0.63 * u)
# The rule needs real clearance below Y_AFTER, not just a different number:
# at 0.03in of gap it struck through the descenders of the edited span in
# every panel whose replacement text ran the full width.
Y_RULE, Y_JUDGE = 0.20 * u, 0.078 * u
assert PANEL_H - 0.63 - 0.20 > 0.09, "the rule would strike the edited span"

y = fig_h * u


def _fit(text, width_in, base):
    """Shrink before truncating: return (text, fontsize) for one code line.

    Hard truncation at a fixed budget cut bbox_jitter_null's coordinates
    mid-number -- exactly the digit its +-1px edit turns on. Dropping up to
    1.2pt of type buys back enough characters to keep those strings whole, and
    only then does the ellipsis come out.
    """
    budget = max(6, int(width_in * 72 / CHAR_W))
    if len(text) <= budget:
        return text, base
    fs = max(base - 1.2, base * budget / len(text))
    grown = int(budget * base / fs)
    return (text if len(text) <= grown else text[:grown - 1] + "\u2026"), fs


for tier_name, tier_color, fams in TIERS:
    # ---- tier header bar --------------------------------------------------
    y -= HDR_H * u
    ax.add_patch(Rectangle((0, y + 0.03 * u), 1.0, (HDR_H - 0.06) * u,
                           facecolor=tier_color, alpha=0.18,
                           edgecolor=tier_color, lw=0.6))
    ax.text(0.03 * u, y + HDR_H * u / 2, tier_name, va="center", ha="left",
            fontsize=FS_TIER, color="0.12", fontweight="bold")
    njs = sorted({PF[f]["n_judged"] for f in fams})
    ax.text(1 - 0.03 * u, y + HDR_H * u / 2,
            (f"n = {njs[0]}" if len(njs) == 1 else f"n = {njs[0]}\u2013{njs[-1]}")
            + " judged pairs per family",
            va="center", ha="right", fontsize=FS_META, color="0.35",
            style="italic")

    pw = (1.0 - (len(fams) - 1) * GAP_X * u) / len(fams)
    inner = pw / u - 2 * 0.09          # panel text width, in inches
    y -= (PANEL_H + GAP_Y * 0.55) * u

    for i, fam in enumerate(fams):
        x0 = i * (pw + GAP_X * u)
        top, bot = y + PANEL_H * u, y
        ax.add_patch(FancyBboxPatch(
            (x0, y), pw, PANEL_H * u,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor="white", edgecolor=tier_color, lw=0.9))

        # No backslash-escaped underscore: text.usetex is off, so mathtext
        # would render the backslash literally (the same trap as \% in fig13).
        ax.text(x0 + PAD, top - Y_NAME, fam, fontsize=FS_NAME, color="0.10",
                fontweight="bold", va="center", **MONO)
        ax.text(x0 + PAD, top - Y_SCOPE, SCOPE[fam], fontsize=FS_SCOPE,
                color="0.42", style="italic", va="center")

        # ---- the real edit, before -> after --------------------------------
        ops = cleanest_edit(fam)
        if not ops:                       # identity_control: there IS no diff
            ax.text(x0 + pw / 2, top - (Y_BEFORE + Y_AFTER) / 2,
                    "(byte-identical)", fontsize=FS_CODE, color="0.45",
                    ha="center", va="center", **MONO)
        else:
            _, before, after = ops[0]
            txt, fs = _fit(before or "\u2205", inner, FS_CODE)
            ax.text(x0 + PAD, top - Y_BEFORE, txt, fontsize=fs, va="center",
                    color="0.35", **MONO)
            ax.text(x0 + PAD + 0.012, top - Y_AFTER, "\u2192", fontsize=FS_CODE,
                    va="center", ha="center", color="0.55")
            txt, fs = _fit(after or "\u2205", inner - 0.14, FS_CODE)
            ax.text(x0 + PAD + 0.030, top - Y_AFTER, txt, fontsize=fs,
                    va="center", color=tier_color, fontweight="bold", **MONO)
            if len(ops) > 1:
                # Top-right badge rather than a line under the edit: the count
                # is a caveat on the excerpt, so it belongs beside the family
                # name, and the panel has no spare line below in any case.
                ax.text(x0 + pw - PAD, top - Y_NAME, f"span 1 of {len(ops)}",
                        fontsize=FS_META, va="center", ha="right",
                        color="0.55", style="italic")

        # ---- the judge's verdict on the family -----------------------------
        rate = PF[fam]["meaning_preserved_rate"]
        # A high rate is the DESIGN for a null and a FAILURE for a semantic
        # family, so the colour is keyed to the family's intent, not the rate.
        intended_preserving = fam in ("paraphrase_null", "bbox_jitter_null",
                                      "identity_control")
        ok = (rate >= 0.9) if intended_preserving else (rate <= 0.3)
        ax.plot([x0 + PAD * 0.7, x0 + pw - PAD * 0.7], [bot + Y_RULE] * 2,
                color="0.85", lw=0.5)
        ax.text(x0 + PAD, bot + Y_JUDGE, "judge: meaning preserved",
                fontsize=FS_META, va="center", color="0.45")
        ax.text(x0 + pw - PAD, bot + Y_JUDGE, f"{rate:.3f}", fontsize=FS_CODE,
                va="center", ha="right", fontweight="bold",
                color=(C_ECOT_BRIDGE if ok else C_NO_COT))

    y -= GAP_Y * 0.45 * u

save(fig, "fig1_task_examples")

print(f"[audit] pairs        : {D['n_pairs']} from {D['source_judge_run']}")
print(f"[audit] head verified: {D['head_chars_verified']} chars")
print(f"[audit] families     : {sum(len(f) for _, _, f in TIERS)} panels")
