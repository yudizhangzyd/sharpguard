"""Fig 3 — causal-edit faithful-rate heatmap (models × edit families).

Rows: 8 CoT-VLA models (7 ours variants + ECoT-bridge).
Cols: 10 edit families.
Cells: faithful rate (0-1), colored via a sequential blue-to-red map.
Highlights the no-CoT collapse and selfsplice-control null.

AUTHORED AT THE INCLUDE WIDTH (see gen_fig4_dissociation.py). Drawn 9.4in wide
and included at 0.95\\textwidth, every label printed at 78% of its authored size.
An 88-cell grid has no room to waste on a scale factor, so the canvas is the
include width and the sizes below are the sizes on the page.
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *
from _data import MODELS as DM, fam
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


MODELS = [
    ("Ours r=8", "ours-r8"), ("Ours r=16", "ours-r16"),
    ("Ours r=32", "ours-r32"), ("Ours r=64", "ours-r64"),
    ("Ours no-CoT", "ours-no-cot"), ("Ours data-50A", "ours-data50A"),
    ("Ours data-50B", "ours-data50B"), ("ECoT-bridge", "ecot-bridge"),
]

FAMILIES = [
    ("subj\nswap",     "subject_swap"),
    ("dir\nflip",      "direction_flip"),
    ("grip\nflip",     "gripper_flip"),
    ("loc\nswap",      "location_swap"),
    ("verb\nswap",     "verb_swap"),
    ("negation",       "negation"),
    ("adv\nplaus",     "adversarial_plausible"),
    ("selfspl\n(null)", "selfsplice_control"),
    ("syntactic\nscramble", "syntactic_scramble"),
    ("cross-task\nswap", "cross_task_swap"),
    ("paraphrase\n(null)", "paraphrase_null"),
]

grid = np.full((len(MODELS), len(FAMILIES)), np.nan)
for i, (name, key) in enumerate(MODELS):
    for j, (_, f) in enumerate(FAMILIES):
        v = fam(key, f, "F_mag")
        if v is not None:
            grid[i, j] = v

# Sizes on the page. Eleven columns share 5.0in, so each cell is 33pt wide and
# 18pt tall -- a 4-character value at 6pt is 14pt, which is what sets CELL.
CELL, XTICK, YTICK, XLAB, CBAR = 6.0, 6.0, 6.4, 6.6, 5.8

fig, ax = plt.subplots(figsize=(6.28, 2.55))
# Explicit margins, and the colorbar in an axes of its own: fig.colorbar(ax=ax)
# steals width from the heatmap AFTER subplots_adjust has run, so the emitted
# page came out narrower than the canvas and LaTeX scaled it back up.
fig.subplots_adjust(left=0.108, right=0.900, top=0.905, bottom=0.230)
# Use a diverging cmap centered at 0.5 — low = decorative, high = causal.
cmap = plt.get_cmap("RdYlBu_r")
im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)
ax.tick_params(axis="both", length=2.0, pad=1.5)

# Annotate cells
for i in range(len(MODELS)):
    for j in range(len(FAMILIES)):
        v = grid[i, j]
        if np.isnan(v):
            ax.text(j, i, "—", ha="center", va="center",
                     fontsize=CELL, color="gray")
        else:
            color = "white" if (v < 0.35 or v > 0.75) else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=CELL, color=color)

ax.set_xticks(range(len(FAMILIES)))
ax.set_xticklabels([f[0] for f in FAMILIES], fontsize=XTICK)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels([m[0] for m in MODELS], fontsize=YTICK)
# Mark no-CoT row
for lbl in ax.get_yticklabels():
    if "no-CoT" in lbl.get_text():
        lbl.set_color(C_NO_COT); lbl.set_fontweight("bold")
    elif "bridge" in lbl.get_text():
        lbl.set_color(C_ECOT_BRIDGE); lbl.set_fontweight("bold")
# Mark selfsplice column
for i, lbl in enumerate(ax.get_xticklabels()):
    fam = FAMILIES[i][1]
    if fam == "selfsplice_control":
        lbl.set_color(C_CTRL); lbl.set_fontweight("bold")
    if fam == "cross_task_swap":
        lbl.set_color("black"); lbl.set_fontweight("bold")

# Colorbar, in an axes placed by hand so the heatmap keeps the width the
# margins gave it.
cax = fig.add_axes([0.916, 0.230, 0.017, 0.675])
cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=CBAR - 0.6, length=1.8, pad=1.2)
cbar.outline.set_linewidth(0.5)
cbar.set_label("Faithful rate  ($\\Delta_\\infty > 0.05$)",
                 fontsize=CBAR, labelpad=2.0)

ax.set_xlabel("Edit family (semantic → controls)", fontsize=XLAB, labelpad=2.0)
save(fig, "fig3_edit_heatmap")

print(f"[audit] grid: {grid.shape[0]} models x {grid.shape[1]} families, "
      f"{int(np.isnan(grid).sum())} missing")
