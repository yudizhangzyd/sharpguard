"""Fig 8: LoRA rank & data-fraction ablation.

Panel (a): faithful rate vs LoRA rank, all four trained ranks.
Panel (b): the two 50%-data seeds.

Three things about the previous version of this figure were wrong, and all three
fed the caption, so they are recorded here:

  * it read /tmp/cf_full_sweep, which is a pinned scratch directory no reader
    has. That also broke _data.py's stated hard rule -- every reported quantity
    comes from results_v2/derived_metrics.json -- by going around it.
  * those /tmp reports are the SUPERSEDED single-run point estimates, not the
    3-sampling-seed means the rest of the paper reports. The gap is not
    cosmetic: cross_task_swap at r=64 reads 0.940 there and 0.887 in the
    release, and the manuscript was quoting the 0.940.
  * /tmp/cf_full_sweep has no lora-r32 directory, so `ranks` was [8, 16, 64]
    while the docstring still said (8, 16, 32, 64). Dropping the paper's own
    canonical rank is what produced the "monotonic on 4 of 5 families" claim:
    with r=32 restored, ZERO of the 5 families are monotonic in rank, because
    r=32 falls below r=16 on all five and below r=8 on four of them. Same shape
    as the threshold figure's dropped location_swap -- a subset chosen by what
    a scratch directory happened to contain, reported as a finding.

So the finding is the opposite of the old one, and it is the more useful
direction: rank does not order causal effect. That is what a metric dominated
by argmax collisions should look like, and it is consistent with the
same-config retraining spread of the noise-hierarchy section.

AUTHORED AT THE INCLUDE WIDTH (see gen_fig4_dissociation.py): at 9.2in wide and
included at 0.95\\textwidth this printed its 8pt tick labels at 6.2pt.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *
from _data import fam
import numpy as np
import matplotlib.pyplot as plt

FAMS = ["direction_flip", "gripper_flip", "verb_swap", "negation",
        "cross_task_swap"]
# All four trained ranks. r=32 is the paper's canonical fine-tune; a rank sweep
# that omits it is not a rank sweep.
RANKS = [(8, "ours-r8"), (16, "ours-r16"), (32, "ours-r32"), (64, "ours-r64")]
SEEDS = [("50% seed A", "ours-data50A"), ("50% seed B", "ours-data50B")]

rank_data = {f: [fam(m, f, "F_mag") for _, m in RANKS] for f in FAMS}
rank_std = {f: [fam(m, f, "F_mag_std") or 0.0 for _, m in RANKS] for f in FAMS}
seed_data = {f: [fam(m, f, "F_mag") for _, m in SEEDS] for f in FAMS}
seed_std = {f: [fam(m, f, "F_mag_std") or 0.0 for _, m in SEEDS] for f in FAMS}
assert not any(v is None for f in FAMS for v in rank_data[f] + seed_data[f]), \
    "a family is missing from the release; the figure would draw a gap as a value"

xs_r = np.arange(len(RANKS))          # categorical: 8/16/32/64 is not linear
n_mono = sum(1 for f in FAMS
             if all(rank_data[f][i] <= rank_data[f][i + 1]
                    for i in range(len(RANKS) - 1)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.67, 2.08))
# Explicit margins because savefig crops to the artists, so default margins emit
# a narrower page that LaTeX scales back up.
fig.subplots_adjust(left=0.078, right=0.996, top=0.885, bottom=0.245,
                    wspace=0.26)
# Sizes on the page. Panel (b) puts 5 two-line family labels in 2.9in.
TITLE, TICK, LAB, LEG, ANNOT = 7.0, 5.6, 6.4, 5.4, 5.2
for ax in (ax1, ax2):
    ax.tick_params(axis="both", labelsize=TICK, length=2.2, pad=1.5)

# Panel (a): rank. Bars carry the 3-seed std, because the dip at r=32 is the
# panel's content and a reader has to be able to see it is larger than the
# sampling noise on the points either side of it.
palette = ["#4477AA", "#EE6677", "#228833", "#AA3377", "#CCBB44"]
for i, f in enumerate(FAMS):
    ax1.errorbar(xs_r, rank_data[f], yerr=rank_std[f], marker="o",
                 label=f.replace("_", " "), color=palette[i], linewidth=1.1,
                 markersize=3.0, capsize=1.3, elinewidth=0.5)
# Name the dip on the canvas: it is the reason this panel is in the paper, and
# leaving it to the caption is how the old "monotonic" reading survived.
ax1.axvspan(2 - 0.30, 2 + 0.30, color="#EE6677", alpha=0.055, lw=0, zorder=0)
ax1.text(2, 0.985, "r=32 dips on\nall 5 families", fontsize=ANNOT,
         color="#B03A2E", ha="center", va="top", linespacing=1.25,
         transform=ax1.get_xaxis_transform())
ax1.set_xticks(xs_r)
ax1.set_xticklabels([str(r) for r, _ in RANKS])
ax1.set_xlabel("LoRA rank", fontsize=LAB, labelpad=1.5)
ax1.set_ylabel("Faithful rate", fontsize=LAB, labelpad=1.5)
ax1.set_ylim(0, 1.0)
ax1.set_title(f"(a) LoRA rank: monotonic on {n_mono} of {len(FAMS)}",
              loc="left", fontsize=TITLE, style="italic", pad=2.5)
ax1.legend(fontsize=LEG, frameon=False, loc="center left", ncol=1,
           bbox_to_anchor=(0.01, 0.40), handlelength=1.4, handletextpad=0.5,
           labelspacing=0.22, borderpad=0.1)
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

# Panel (b): the two 50% data seeds.
xs = np.arange(len(FAMS))
w = 0.35
for off, (lab, _), color in ((-w / 2, SEEDS[0], "#4477AA"),
                             (+w / 2, SEEDS[1], "#EE6677")):
    j = 0 if off < 0 else 1
    ax2.bar(xs + off, [seed_data[f][j] for f in FAMS], w,
            yerr=[seed_std[f][j] for f in FAMS], capsize=1.3,
            error_kw={"lw": 0.5}, label=lab, color=color,
            edgecolor="black", linewidth=0.4)
ax2.set_xticks(xs)
ax2.set_xticklabels([f.replace("_", "\n") for f in FAMS], fontsize=TICK)
ax2.set_ylabel("Faithful rate", fontsize=LAB, labelpad=1.5)
ax2.set_ylim(0, 1.0)
spread = max(abs(seed_data[f][0] - seed_data[f][1]) for f in FAMS) * 100
# One decimal, not zero: at .0f this printed "<= 11 pp" against a caption that
# says 10.7 pp, and a reader comparing the two has to wonder which is the number.
ax2.set_title(f"(b) Data-seed spread $\\leq {spread:.1f}$ pp", loc="left",
              fontsize=TITLE, style="italic", pad=2.5)
ax2.legend(fontsize=LEG, frameon=False, loc="upper right", handlelength=1.1,
           handletextpad=0.5, labelspacing=0.22, borderpad=0.1,
           borderaxespad=0.2)
ax2.set_axisbelow(True)
ax2.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

save(fig, "fig8_ablation")

print(f"[audit] ranks {[r for r, _ in RANKS]}, {len(FAMS)} families, "
      f"{len(SEEDS)} data seeds")
print(f"[audit] monotonic in rank: {n_mono} of {len(FAMS)} families")
for f in FAMS:
    print(f"[audit]   {f:17} " + " ".join(f"{v:.3f}" for v in rank_data[f]))
_below16 = sum(1 for f in FAMS if rank_data[f][2] < rank_data[f][1])
_below8 = sum(1 for f in FAMS if rank_data[f][2] < rank_data[f][0])
print(f"[audit] r=32 below r=16 on {_below16}/5, below r=8 on {_below8}/5")
print(f"[audit] max data-seed spread: {spread:.1f} pp")
