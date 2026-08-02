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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.65),
                               gridspec_kw={"wspace": 0.30,
                                            "width_ratios": [1.0, 1.35]})

# ---- (a) the three noise sources, in attention points ---------------------
# Ordered smallest-first so the reader meets them in the order the argument
# needs: the error bar the paper used to report, the one it should have, and
# the effect both were meant to license.
BARS = [("sampling seed\n(frozen ckpt)", NH["sampling_std_pp"], C_CTRL),
        ("same-config\nretraining", NH["training_run_diff_pp"], C_NO_COT),
        ("between-variant\nspread", NH["cross_variant_spread_pp"],
         C_COT_TRAINED)]
xs = np.arange(len(BARS))
ax1.bar(xs, [b[1] for b in BARS], color=[b[2] for b in BARS],
        edgecolor="black", lw=0.4, width=0.62)
for x, (_, v, _) in zip(xs, BARS):
    ax1.text(x, v + 0.04, f"{v:.2f}", ha="center", fontsize=FONT_SIZE - 3,
             fontweight="bold")
ax1.set_xticks(xs)
ax1.set_xticklabels([b[0] for b in BARS], fontsize=FONT_SIZE - 3)
ax1.set_ylabel(r"$|\Delta\,\alpha(\mathrm{cot})|$  (pp)")
ax1.set_ylim(0, max(b[1] for b in BARS) * 1.42)
# The ratio is the claim: an effect only 1.2x its own null is not an effect.
ratio = NH["spread_over_training_run"]
ax1.set_title(f"(a) The effect is only {ratio:.1f}$\\times$ its own null",
              loc="left", fontsize=FONT_SIZE, style="italic")
ax1.annotate("", xy=(2, BARS[2][1]), xytext=(1, BARS[1][1]),
             arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.35",
                             shrinkA=2, shrinkB=2))
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
ax2.axvline(wilson, color="0.35", ls="--", lw=0.8, zorder=2)
# Above the axis, not beside the line: at the bottom the label ran straight
# through the x tick labels, and inside the plot it landed on the r16 bar.
# The top-left is the only region both free and adjacent to the line.
ax2.annotate(f"widest Wilson half-width ({wilson:.3f})",
             xy=(wilson, -0.55), xytext=(wilson + 0.018, -1.05),
             fontsize=FONT_SIZE - 4, color="0.30", va="center",
             annotation_clip=False,
             arrowprops=dict(arrowstyle="->", lw=0.6, color="0.35"))
ax2.set_yticks(ys)
ax2.set_yticklabels([m.replace("ours-", "") for m in labels],
                    fontsize=FONT_SIZE - 3)
ax2.set_ylim(len(labels) - 0.5, -1.35)   # headroom for the Wilson label
ax2.set_xlabel(r"$|\Delta\mathcal{F}|$ between two trainings of the same config")
ax2.set_xlim(0, max(per_max.values()) * 1.30)
worst = TR["F_max_abs_diff_where"]
ax2.set_title(f"(b) Up to {TR['F_max_abs_diff_over_pairs']:.2f} on "
              f"{worst.split(':')[1].replace('_', ' ')}",
              loc="left", fontsize=FONT_SIZE, style="italic")
ax2.legend(fontsize=FONT_SIZE - 4, loc="lower right", frameon=True,
           framealpha=0.9, borderpad=0.4)
ax2.set_axisbelow(True)
ax2.xaxis.grid(True, ls=":", lw=0.4, alpha=0.5)

save(fig, "fig14_noise_hierarchy")

n_over = sum(1 for m in labels if per_max[m] > wilson)
print(f"[audit] replicate pairs : {TR['n_pairs']}")
print(f"[audit] widest Wilson   : {wilson:.4f}")
print(f"[audit] pairs over it   : {n_over}/{len(labels)}")
print(f"[audit] worst cell      : {worst} at "
      f"{TR['F_max_abs_diff_over_pairs']:.2f}")
