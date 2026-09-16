"""Fig 16 -- the paper's central calibration failure, in one picture (S3.1).

Section 3.1's headline is a sign count: on every one of 11 calibrated
configurations, the mean score over semantic edits sits below that model's
own paraphrase floor. A plain point of F_diff = F_bar(semantic) -
F(paraphrase floor) only shows the gap, not the two raw quantities it is a
difference of. This "calibration ladder" plots both ends directly, per
model: a filled dot at the semantic mean, an open circle at the paraphrase
floor, joined by a segment. The floor sits to the right of the mean on
every single row -- that is the 11/11 claim, readable without doing any
subtraction.

Colored and grouped by base-checkpoint lineage (ECoT/LoRA vs. DeepThinkVLA),
since S4.2 (pseudoreplication) says the 11 rows are two independent
confirmations, not eleven -- the grouping is not decorative, it is the
caveat the section itself makes.

Source: results_v2/canonical_runs/floor_convention_robustness/
floor_convention_robustness.json, the same artifact Table 2 (tab:floors) is
built from. bridge_subset_4k is excluded, matching tab:floors' own caption
("calibrates but is excluded"). Both the floor and the semantic mean, and
their difference, are asserted against Table 2's own printed F_diff column
before plotting, so this figure cannot silently drift from the table it
visualizes.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *          # noqa: F401,F403  (rcParams + save)
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results_v2", "canonical_runs",
                   "floor_convention_robustness",
                   "floor_convention_robustness.json")
with open(SRC) as fh:
    D = json.load(fh)

PC = D["per_config"]

# (json key, display label, lineage) -- lineage in tab:floors' own row order,
# grouped so the lineage split reads as two blocks, not an interleaving.
ROWS = [
    ("ours_no-cot",   "no-CoT",      "ecot"),
    ("ours_lora-r8",  "r=8",         "ecot"),
    ("ours_lora-r16", "r=16",        "ecot"),
    ("ours_lora-r32", "r=32",        "ecot"),
    ("ours_lora-r64", "r=64",        "ecot"),
    ("ours_data-50A", "data-50A",    "ecot"),
    ("ours_data-50B", "data-50B",    "ecot"),
    ("ecot_bridge",   "ECoT-bridge", "ecot"),
    ("deepthink_base", "DT base",    "deepthink"),
    ("deepthink_sft",  "DT SFT",     "deepthink"),
    ("deepthink_rl",   "DT RL",      "deepthink"),
]
# Table 2's own printed F_diff (vs. paraphrase) column, to 3dp -- this figure
# is not allowed to disagree with the table it visualizes.
TABLE2_PRINTED = {
    "no-CoT": -0.027, "r=8": -0.180, "r=16": -0.100, "r=32": -0.172,
    "r=64": -0.142, "data-50A": -0.144, "data-50B": -0.095,
    "ECoT-bridge": -0.088, "DT base": -0.013, "DT SFT": -0.109,
    "DT RL": -0.095,
}

labels, means, floors, diffs, lineages = [], [], [], [], []
for key, label, lineage in ROWS:
    mean = PC[key]["fbar_B"]
    floor = PC[key]["floors"]["paraphrase_null"]
    diff = PC[key]["diff_B"]["paraphrase_null"]
    if round(mean - floor, 3) != round(diff, 3):
        raise SystemExit(f"[fig16] {label}: fbar_B - floor != diff_B in the "
                         f"artifact itself -- read the wrong pair of fields.")
    printed = TABLE2_PRINTED[label]
    if round(diff, 3) != printed:
        raise SystemExit(f"[fig16] {label}: recomputed F_diff {diff:.3f} "
                         f"disagrees with Table 2's printed {printed} -- "
                         f"the table and this figure have drifted apart.")
    labels.append(label)
    means.append(mean)
    floors.append(floor)
    diffs.append(diff)
    lineages.append(lineage)

if not all(d < 0 for d in diffs):
    raise SystemExit("[fig16] not every configuration is below its own "
                     "floor -- the '11 of 11' claim this figure draws no "
                     "longer holds against the artifact, fix before "
                     "plotting a claim the data does not support.")

LINEAGE_COLOR = {"ecot": C_COT_TRAINED, "deepthink": C_DEEPTHINK}
LINEAGE_LABEL = {"ecot": "ECoT / LoRA lineage", "deepthink": "DeepThinkVLA lineage"}

n = len(labels)
# One extra row of vertical gap between the two lineage blocks, so the split
# S4.2 argues for is visible as whitespace, not just as color. gap_y is where
# that blank row lands -- used below to place the headline count in empty
# space rather than guessing axes-fraction coordinates.
ys, y, gap_y = [], n, None
for i, lineage in enumerate(lineages):
    if i > 0 and lineage != lineages[i - 1]:
        gap_y = y
        y -= 1
    ys.append(y)
    y -= 1
ys = np.array(ys)

fig, ax = plt.subplots(figsize=(3.35, 2.0))
for y, mean, floor, lineage in zip(ys, means, floors, lineages):
    color = LINEAGE_COLOR[lineage]
    ax.plot([mean, floor], [y, y], color=color, lw=1.3, alpha=0.6, zorder=1)
    ax.scatter([mean], [y], s=24, color=color, zorder=3,
              edgecolor="white", linewidth=0.4)
    ax.scatter([floor], [y], s=30, facecolor="white", edgecolor=color,
              linewidth=1.2, zorder=3)

ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=FONT_SIZE - 2)
ax.set_xlabel(r"$\mathcal{F}$ (semantic mean $\bullet$ vs.\ paraphrase floor $\circ$)",
             fontsize=FONT_SIZE - 1.5)
ax.set_xlim(0.0, 1.03)
ax.set_ylim(ys.min() - 0.9, n + 1.55)
ax.set_axisbelow(True)
ax.xaxis.grid(True, ls=":", lw=0.4, alpha=0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# The headline claim, as a visual object rather than a caption sentence: every
# row's segment points the same way (mean below floor), so put the count in
# the blank divider row between the two lineages, where no data lands.
ax.text(1.0, gap_y, "11/11 below floor", ha="right", va="center",
        fontsize=FONT_SIZE - 1, fontweight="bold", color="0.2")

handles = [
    mlines.Line2D([0], [0], marker="o", ls="", color="0.35",
                 markersize=5, label="semantic mean"),
    mlines.Line2D([0], [0], marker="o", ls="", markerfacecolor="white",
                 markeredgecolor="0.35", markeredgewidth=1.2,
                 markersize=5.5, label="paraphrase floor"),
] + [mlines.Line2D([0], [0], marker="s", ls="", color=LINEAGE_COLOR[k],
                   markersize=5, label=LINEAGE_LABEL[k])
     for k in ("ecot", "deepthink")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.12),
         frameon=False, fontsize=FONT_SIZE - 3, handletextpad=0.4,
         ncol=2, columnspacing=1.0, labelspacing=0.5)

save(fig, "fig16_calibration_gap")

print(f"[audit] source: {os.path.relpath(SRC, ROOT)}")
print(f"[audit] n configs: {n}, all below floor: {all(d < 0 for d in diffs)}")
print(f"[audit] range: [{min(diffs):.3f}, {max(diffs):.3f}]")
