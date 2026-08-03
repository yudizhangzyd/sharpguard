"""Fig 12 (NEW, R1 item 3) -- the leaderboard INVERTS under a direction-aware
version of its own flagship edit.

(a) magnitude-F vs directional-F on direction_flip, per model.
(b) signed cosine of the xyz translation before vs after direction_flip.
    A CoT-faithful model must move the OPPOSITE way (cos -> -1).
(c) F_diff = F(f) - F(paraphrase_null): the differential leaderboard for
    ours-no-cot. Every model now carries a measured paraphrase floor, but this
    is the only one whose floor (0.19) is far enough below its ceiling for the
    per-family differential to be legible; on the full-CoT variants the floors
    sit at 0.45-0.66 and the bars collapse toward zero.

All values from results_v2/derived_metrics.json.

AUTHORED AT THE INCLUDE WIDTH, for the reason spelled out in
gen_fig4_dissociation.py: matplotlib font sizes are absolute points, so a canvas
drawn 11.09in wide and included at 0.8\\textwidth (5.06in) prints every label at
46% of its authored size -- this figure's tick and value labels were 7pt on the
canvas and 3.2pt on the page, the smallest type in the document by a wide
margin. Nothing about the content changed in the rewrite; the sizes below are
the sizes a reader gets.

Two panels carry 16 bars between 8 model slots, which leaves ~8pt per bar --
narrower than a horizontal "0.96". Panel (a)'s value labels are therefore
rotated to run along the bar rather than across it, and panel (b) keeps the
original's no-labels treatment: its claim is the SIGN, which the zero line and
the faithful-target rule already carry.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *
from _data import MODELS, ORDER, LABELS, OURS, NON_CONTROL, fam
import numpy as np
import matplotlib.pyplot as plt

MS = [m for m in ORDER if m in MODELS]
# Only panels (a) and (b) carry a model axis; (c) is a family axis. See
# paper_plot_style.ours_bracket for why the "Ours" prefix left the labels.
N_OURS = sum(1 for m in MS if m in OURS)
assert all(m in OURS for m in MS[:N_OURS]), \
    "ORDER no longer puts our fine-tunes first; the bracket would mislabel"
COL = {m: (C_NO_COT if m == "ours-no-cot" else
           C_ECOT_BRIDGE if m == "ecot-bridge" else C_COT_TRAINED) for m in MS}

# Every size below is the size on the page, not on a canvas that will be
# shrunk. Eight model slots in a 1.9in panel is the binding constraint.
TITLE, TICK, VAL, YLAB, LEG = 6.6, 5.3, 4.8, 6.0, 5.2

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.28, 3.05))
# Explicit margins rather than the defaults: savefig(bbox_inches="tight") crops
# to the artists, so default 10% side margins would emit a page narrower than
# 6.28in and LaTeX would scale it back up -- reintroducing the mismatch this
# rewrite removes. Filling the canvas makes the crop a no-op.
fig.subplots_adjust(left=0.070, right=0.998, top=0.867, bottom=0.258,
                    wspace=0.30)
for ax in (ax1, ax2, ax3):
    ax.tick_params(axis="y", labelsize=TICK + 0.4, length=2.2, pad=1.5)
    ax.tick_params(axis="x", length=0, pad=1.5)
# One line per model: the two-line forms in LABELS are wider than an 8pt slot
# at any font size, and a rotated label needs a single baseline in any case.
FLAT = [LABELS[m].replace("\n", "") for m in MS]
x = np.arange(len(MS))
w = 0.38

# ---- (a) magnitude-F vs directional-F on direction_flip --------------------
mag = [fam(m, "direction_flip", "F_mag") for m in MS]
dr = [fam(m, "direction_flip", "F_dir") for m in MS]
ax1.bar(x - w / 2, mag, w, color=[COL[m] for m in MS], edgecolor="black", lw=0.4,
        label=r"magnitude-$\mathcal{F}$ (Eq. 1)")
ax1.bar(x + w / 2, dr, w, color=[COL[m] for m in MS], edgecolor="black", lw=0.4,
        hatch="////", label=r"directional-$\mathcal{F}$ ($\cos<-0.5$)")
for i, (a, b) in enumerate(zip(mag, dr)):
    # Rotated: a horizontal "0.96" at 4.8pt is 11pt wide and the bar slot is 8pt,
    # so the labels of adjacent bars overlapped outright at this width.
    ax1.text(i - w / 2, a + 0.03, f"{a:.2f}", ha="center", va="bottom",
             fontsize=VAL, rotation=90)
    ax1.text(i + w / 2, b + 0.03, f"{b:.2f}", ha="center", va="bottom",
             fontsize=VAL, rotation=90)
ax1.set_xticks(x)
ax1.set_xticklabels(FLAT, fontsize=TICK, rotation=45, ha="right",
                    rotation_mode="anchor")
ax1.set_ylabel(r"$\mathcal{F}(m,\mathrm{direction\_flip})$", fontsize=YLAB,
               labelpad=1.5)
# Headroom for the rotated labels: a 4-character label at 4.8pt runs ~11pt,
# which is 0.13 of this axis, so a bar at 1.00 needs the ceiling at 1.16.
ax1.set_ylim(0, 1.30)
from matplotlib.patches import Patch
_h = [Patch(facecolor="0.6", edgecolor="black", lw=0.4,
            label=r"magnitude-$\mathcal{F}$ (Eq. 1)"),
      Patch(facecolor="0.6", edgecolor="black", lw=0.4, hatch="////",
            label=r"directional-$\mathcal{F}$ ($\cos<-0.5$)")]
# A tight legend box: at 5.2pt the default padding is a third of the legend's
# height, and this panel has 0.14 of axis height to spare above the tallest
# rotated value label.
ax1.legend(handles=_h, frameon=False, fontsize=LEG, loc="upper left",
           handlelength=1.1, handletextpad=0.5, labelspacing=0.25,
           borderpad=0.1, borderaxespad=0.2)
ax1.set_title("(a) The flagship edit:\nmagnitude vs direction",
              loc="left", fontsize=TITLE, style="italic", pad=2.5,
              linespacing=1.25)
ax1.set_axisbelow(True); ax1.yaxis.grid(True, ls=":", lw=0.4, alpha=0.5)
# y is an axes fraction: this panel grew from 1.43in to 1.86in of axes when
# panel (c) went horizontal, so the old -0.31 put the bracket 0.45in below
# the labels instead of 0.34in.
ours_bracket(ax1, N_OURS, fontsize=TICK, y=-0.205)

# ---- (b) signed cosine ----------------------------------------------------
cs = [fam(m, "direction_flip", "cos_xyz") for m in MS]
csf = [fam(m, "direction_flip", "cos_xyz_faithful_subset") for m in MS]
ax2.bar(x - w / 2, cs, w, color=[COL[m] for m in MS], edgecolor="black", lw=0.4,
        label="all samples")
ax2.bar(x + w / 2, csf, w, color=[COL[m] for m in MS], edgecolor="black", lw=0.4,
        hatch="////", label=r"samples Eq. 1 calls faithful")
ax2.axhline(0.0, color="black", lw=0.7)
ax2.axhline(-1.0, color=C_ECOT_BRIDGE, ls="--", lw=0.8)
ax2.text(len(MS) - 0.4, -0.90, "faithful\ntarget", fontsize=VAL,
         ha="right", va="bottom", color=C_ECOT_BRIDGE, linespacing=1.2)
ax2.set_xticks(x)
ax2.set_xticklabels(FLAT, fontsize=TICK, rotation=45, ha="right",
                    rotation_mode="anchor")
ax2.set_ylabel(r"$\cos(a_{orig}[0{:}3],\ a_{edit}[0{:}3])$", fontsize=YLAB,
               labelpad=1.5)
ax2.set_ylim(-1.1, 1.20)
_h2 = [Patch(facecolor="0.6", edgecolor="black", lw=0.4, label="all samples"),
       Patch(facecolor="0.6", edgecolor="black", lw=0.4, hatch="////",
             label=r"samples Eq. 1 calls faithful")]
ax2.legend(handles=_h2, frameon=False, fontsize=LEG, loc="upper left",
           handlelength=1.1, handletextpad=0.5, labelspacing=0.25,
           borderpad=0.1, borderaxespad=0.2)
ax2.set_title("(b) Sign of the response after\nleft$\\leftrightarrow$right is reversed",
              loc="left", fontsize=TITLE, style="italic", pad=2.5,
              linespacing=1.25)
ax2.set_axisbelow(True); ax2.yaxis.grid(True, ls=":", lw=0.4, alpha=0.5)
# The bracket hangs off the bottom of the axes, and this axes' bottom is at
# cos = -1.1 rather than at zero, so it needs the same offset as the others
# measured in axes fraction -- which is what ours_bracket's y already is.
ours_bracket(ax2, N_OURS, fontsize=TICK, y=-0.205)

# ---- (c) F_diff for the model with a measured paraphrase floor -------------
# ALL 12 non-reference families, not the 9 semantic-plus-two ones this panel used
# to draw. The three it dropped are the three that fix the scale: selfsplice
# (-0.193, the identity null, which is minus the floor by construction),
# bbox_jitter_null (-0.147), and instr_random_sub (+0.070) -- the random
# instruction substitution, i.e. the CEILING. Once the ceiling is on the plot,
# direction_flip's +0.081 is ABOVE it, and the panel's real content is that the
# whole floor-to-ceiling band is 0.07 wide.
#
# Horizontal and sorted: 12 rotated tick labels do not fit in a 1.9in panel, and
# "differential leaderboard" is a ranking, so a ranked axis is the honest layout.
# All eight models carry a measured floor, so the selection rule has to be the
# one the caption states -- lowest floor -- and not HAVE[0], which picked this
# model only because ORDER happens to list it first.
HAVE = [m for m in MS if MODELS[m].get("paraphrase_null_floor") is not None]
FLOORS = sorted(HAVE, key=lambda m: MODELS[m]["paraphrase_null_floor"])
m0 = FLOORS[0]
assert MODELS[FLOORS[1]]["paraphrase_null_floor"] \
    - MODELS[m0]["paraphrase_null_floor"] > 0.2, \
    "another model's floor is now comparable to this one's; 'the model whose " \
    "floor is low enough for the differential to be readable' no longer " \
    "picks out one model"
REFERENCE = "paraphrase_null"        # F_diff is defined as 0 here
SHORT = {"syntactic_scramble": "scram", "cross_task_swap": "cross-task",
         "direction_flip": "dir flip", "gripper_flip": "grip flip",
         "verb_swap": "verb", "negation": "negation", "subject_swap": "subj",
         "location_swap": "loc", "adversarial_plausible": "adv plaus",
         "selfsplice_control": "selfsplice", "bbox_jitter_null": "bbox jitter",
         "instr_random_sub": "instr random"}
CEILING = "instr_random_sub"
_fams = MODELS[m0]["families"]
FAMS = [f for f in SHORT if f in _fams]
assert set(FAMS) | {REFERENCE} == set(_fams), \
    f"this panel is missing a family the release measures: " \
    f"{set(_fams) - set(FAMS) - {REFERENCE}}"
assert set(NON_CONTROL) <= set(FAMS), "a semantic family dropped out of panel (c)"
pairs = sorted(((f, fam(m0, f, "F_diff")) for f in FAMS),
               key=lambda t: t[1] or 0.0)
ys = np.arange(len(pairs))
vals = [v for _, v in pairs]
# Semantic families in the trained colour, controls and calibrators in grey: the
# point of drawing all 12 is that they interleave.
cols = [C_COT_TRAINED if f in NON_CONTROL else C_CTRL for f, _ in pairs]
ax3.barh(ys, vals, 0.66, color=cols, edgecolor="black", lw=0.4)
ax3.axvline(0.0, color="black", lw=0.8)
ceil_v = fam(m0, CEILING, "F_diff")
ax3.axvline(ceil_v, color="0.15", lw=0.7, ls=(0, (3, 2)))
for i, v in enumerate(vals):
    ax3.text(v + (0.006 if v >= 0 else -0.006), i, f"{v:+.2f}",
             ha="left" if v >= 0 else "right", va="center", fontsize=VAL)
ax3.set_yticks(ys)
ax3.set_yticklabels([SHORT[f] for f, _ in pairs], fontsize=TICK)
ax3.tick_params(axis="y", length=0, pad=1.5)
ax3.set_ylim(-0.7, len(pairs) - 0.3)
ax3.set_xlim(min(vals) - 0.055, max(max(vals), ceil_v) + 0.062)
ax3.set_xlabel(r"$\mathcal{F}_{diff}=\mathcal{F}(f)-\mathcal{F}(para)$",
               fontsize=YLAB, labelpad=1.5)
floor = MODELS[m0]["paraphrase_null_floor"]
# Short enough to stay inside the canvas: at 6.6pt the earlier three-clause form
# ran past the right edge, and savefig's crop then emitted a page WIDER than the
# canvas, which LaTeX scales back down -- shrinking every label on the figure.
ax3.set_title(f"(c) {m0}, all {len(pairs)} families\n"
              f"(floor {floor:.2f}, ceiling {ceil_v:+.2f})",
              loc="left", fontsize=TITLE, style="italic", pad=2.5,
              linespacing=1.25)
ax3.text(ceil_v + 0.004, -0.62, "ceiling", fontsize=VAL, style="italic",
         color="0.15", ha="left", va="bottom")
ax3.set_axisbelow(True); ax3.xaxis.grid(True, ls=":", lw=0.4, alpha=0.5)

save(fig, "fig12_directional_inversion")

print(f"[audit] magnitude on direction_flip : {min(mag):.3f}-{max(mag):.3f}")
print(f"[audit] directional on the same     : {min(dr):.3f}-{max(dr):.3f}")
print(f"[audit] F_diff for {m0:<16}: {min(vals):+.3f} to {max(vals):+.3f}"
      f"  ({len(pairs)} families)")
_pos = [(f, v) for f, v in pairs if (v or 0) > 0]
print(f"[audit] below their own floor      : "
      f"{sum(1 for _, v in pairs if (v or 0) < 0)} of {len(pairs)}")
print(f"[audit] above it                   : "
      f"{[(f, round(v, 3)) for f, v in _pos]}")
print(f"[audit] ceiling (instr_random_sub) : {ceil_v:+.3f}; families above the "
      f"ceiling: {[f for f, v in pairs if (v or 0) > ceil_v]}")
