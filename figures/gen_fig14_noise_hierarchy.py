"""Fig 14 -- the error bar that matters is the retraining, not the seed (S6).

Section 6 makes the single most actionable claim in the paper for anyone
reading the leaderboard -- that a single-family difference below ~0.26 is not
a model difference -- and carries no exhibit. It is a claim about three noise
sources of very different size, which is a comparison of magnitudes, and a bar
chart on a shared axis states it in one glance where the prose needs a
paragraph.

(a) the hierarchy. Three ways of asking "how much does this number move if I
    do nothing interesting?", drawn on one axis so the ordering is the
    argument: resampling the decode from a FROZEN checkpoint, retraining the
    SAME config, and the between-variant spread a reader would want to
    interpret. The third is smaller than the second, which is why S6 says the
    within-family ordering is not supported.
(b) the same thing per retrained pair, in F rather than in attention points,
    because that is the unit the leaderboard is read in. Each pair is one
    config trained twice at a byte-identical recipe; the bar is how far its
    worst family moved, the dot is how far its F_bar moved. The widest Wilson
    half-width over every leaderboard cell is drawn as a line, and the point
    of the panel is that almost every bar clears it.

Everything is read from derived_metrics.json -- including the axis annotations
and the Wilson reference, which is recomputed here by scanning every released
cell rather than copied from the text. No number is written into this file.

AUTHORED AT THE INCLUDE WIDTH, like Figure 4. This was drawn 7.1in wide and
included at \\columnwidth (3.03in), i.e. downscaled to 50%, which put its tick
and value labels on the page at 3.5pt and its legend at 3.0pt -- smaller than
the Figure 4 defect that prompted this pass, and an earlier note in
verify_paper_numbers.py claiming it "renders legibly at 50%" was simply wrong.
Matplotlib sizes are absolute points, so every size below is the size a reader
gets. The panels are STACKED rather than side by side because two panels cannot
share 3.03in: at that width panel (b)'s seven row labels and its axis label are
each wider than the half they would get.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *          # noqa: F401,F403  (rcParams + save)
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "results_v2", "derived_metrics.json")) as fh:
    D = json.load(fh)

NH = D["noise_hierarchy"]
TR = D["training_replicate"]

# The sizes a reader actually gets, since the figure is drawn at its include
# width. Stacked at 3.03in, panel (b) has ~10pt of vertical room per row, so
# 5.5pt row labels are what the slot holds rather than a preference.
TITLE, TICK, VAL, LAB, LEG = 7.0, 5.5, 5.5, 6.0, 5.0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.035, 2.10),
                               gridspec_kw={"height_ratios": [1.0, 1.5]})
# Explicit margins, not the defaults: savefig(bbox_inches="tight") crops to the
# artists, so a canvas on the default margins comes out narrower than
# \columnwidth and LaTeX scales it back up -- reintroducing the very
# authored-size/include-size mismatch this rewrite removes.
#
# The height is set by the page, not by taste. Stacking costs column space the
# side-by-side version did not use, and at 2.32in the Conclusion's last line
# ("score.") was pushed onto page 9 -- one line over the 8-page body limit. This
# is the tallest the figure can be and leave the body inside 8 pages, with about
# a line to spare.
fig.subplots_adjust(left=0.117, right=0.985, top=0.940, bottom=0.112,
                    hspace=0.34)
for ax in (ax1, ax2):
    ax.tick_params(labelsize=TICK, length=2.2, pad=1.5)

# ---- (a) the three noise sources, in attention points ---------------------
# Ordered smallest-first so the reader meets them in the order the argument
# needs: the error bar the paper used to report, the one it should have, and
# the effect both were meant to license.
#
# One line per label, not two: stacked, each slot is ~0.9in wide and the longest
# name sets in 0.57in, so the "(frozen ckpt)" qualifier the two-line version
# carried is left to the caption, which already says it.
BARS = [("sampling seed", NH["sampling_std_pp"], C_CTRL),
        ("same-config", NH["training_run_diff_pp"], C_NO_COT),
        ("between-variant", NH["cross_variant_spread_pp"], C_COT_TRAINED)]
xs = np.arange(len(BARS))
ax1.bar(xs, [b[1] for b in BARS], color=[b[2] for b in BARS],
        edgecolor="black", lw=0.4, width=0.62)
for x, (_, v, _) in zip(xs, BARS):
    ax1.text(x, v + 0.04, f"{v:.2f}", ha="center", fontsize=VAL,
             fontweight="bold")
ax1.set_xticks(xs)
ax1.set_xticklabels([b[0] for b in BARS], fontsize=TICK)
# Broken across two lines: rotated, this label's vertical extent is its text
# width, and on one line it is taller than the 0.67in panel it labels.
ax1.set_ylabel("$|\\Delta\\,\\alpha(\\mathrm{cot})|$\n(pp)", fontsize=LAB,
               labelpad=1.5, linespacing=1.2)
ax1.set_ylim(0, max(b[1] for b in BARS) * 1.42)
# Ticks every whole point, derived from the limit rather than chosen: at this
# panel height the automatic locator drops to two labels (0 and 2), which leaves
# the axis top unlabelled and the scale hard to read off.
ax1.set_yticks(np.arange(0, np.floor(ax1.get_ylim()[1]) + 1, 1.0))
# The ratio is the claim: an effect only 1.2x its own null is not an effect.
ratio = NH["spread_over_training_run"]
ax1.set_title(f"(a) The effect is only {ratio:.1f}$\\times$ its own null",
              loc="left", fontsize=TITLE, style="italic", pad=2.5)
# A reference at the null's own height, carried across to the effect, rather than
# a double-headed arrow between the two bar tops: stacked, the panel is wide
# enough that such an arrow spans most of it and crosses both value labels. The
# line makes the claim visible instead of measured -- how little of the third bar
# clears it IS the 1.2x.
ax1.hlines(BARS[1][1], 0.66, len(BARS) - 0.55, color="0.35", ls=(0, (2.5, 1.8)),
           lw=0.7, zorder=4)
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, ls=":", lw=0.4, alpha=0.5)

# ---- (b) per retrained pair, in F -----------------------------------------
# The widest Wilson half-width anywhere in the release, recomputed here: it is
# the error bar the submitted version reported, so it is the right reference
# for showing what that error bar missed.
wilson = max((w[1] - w[0]) / 2
             for mv in D["models"].values()
             for fv in mv.get("families", {}).values()
             for w in [fv.get("F_mag_wilson")] if w)

per_max = TR["F_max_abs_diff_per_pair"]
per_bar = TR["F_bar_abs_diff_per_pair"]
# Sorted by the quantity the panel is about, so the reader can find the worst
# pair without hunting; the labels carry the identity.
labels = sorted(per_max, key=per_max.get, reverse=True)
ys = np.arange(len(labels))
ax2.barh(ys, [per_max[m] for m in labels], color=C_COT_TRAINED,
         edgecolor="black", lw=0.4, height=0.6,
         label=r"worst family ($\max_f |\Delta\mathcal{F}|$)")
ax2.plot([per_bar[m] for m in labels], ys, "o", ms=4,
         color=C_NO_COT, mec="black", mew=0.4, ls="none", zorder=3,
         label=r"$|\Delta\bar{\mathcal{F}}|$ (7-family mean)")
ax2.axvline(wilson, color="0.35", ls="--", lw=0.8, zorder=2,
            label=f"widest Wilson half-width ({wilson:.3f})")
ax2.set_yticks(ys)
# Spell the configs the way every table and the other four figures spell them.
# The artifact keys are ours-r8 / ours-data50A, which strip to "r8" / "data50A"
# and read as different models from the "r=8" / "data-50A" rows they are.
NICE = {"r8": "r=8", "r16": "r=16", "r32": "r=32", "r64": "r=64",
        "data50A": "data-50A", "data50B": "data-50B", "no-cot": "no-CoT"}
short = [m.replace("ours-", "") for m in labels]
missing = [s for s in short if s not in NICE]
assert not missing, f"no display name for replicate pair(s) {missing}"
ax2.set_yticklabels([NICE[s] for s in short], fontsize=TICK)
# No headroom above the top row. The Wilson reference used to be annotated into
# 1.35 rows of empty axis, which at this height would cost every row a sixth of
# its space; as a legend entry it costs none, and the dashed line then explains
# itself in the same place the other two marks do.
ax2.set_ylim(len(labels) - 0.5, -0.7)
ax2.set_xlabel(r"$|\Delta\mathcal{F}|$ between two trainings of the same config",
               fontsize=LAB, labelpad=1.5)
ax2.set_xlim(0, max(per_max.values()) * 1.30)
worst = TR["F_max_abs_diff_where"]
ax2.set_title(f"(b) Up to {TR['F_max_abs_diff_over_pairs']:.2f} on "
              f"{worst.split(':')[1].replace('_', ' ')}",
              loc="left", fontsize=TITLE, style="italic", pad=2.5)
# Lower right, which the two shortest bars leave free.
ax2.legend(fontsize=LEG, loc="lower right", frameon=True,
           framealpha=0.9, borderpad=0.35, handlelength=1.4,
           labelspacing=0.25, handletextpad=0.45)
ax2.set_axisbelow(True)
ax2.xaxis.grid(True, ls=":", lw=0.4, alpha=0.5)

save(fig, "fig14_noise_hierarchy")

n_over = sum(1 for m in labels if per_max[m] > wilson)
print(f"[audit] replicate pairs : {TR['n_pairs']}")
print(f"[audit] widest Wilson   : {wilson:.4f}")
print(f"[audit] pairs over it   : {n_over}/{len(labels)}")
print(f"[audit] worst cell      : {worst} at "
      f"{TR['F_max_abs_diff_over_pairs']:.2f}")
