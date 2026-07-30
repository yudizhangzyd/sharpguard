"""Fig 3 — causal-edit faithful-rate heatmap (models × edit families).

Rows: 8 CoT-VLA models (7 ours variants + ECoT-bridge).
Cols: 10 edit families.
Cells: faithful rate (0-1), colored via a sequential blue-to-red map.
Highlights the no-CoT collapse and selfsplice-control null.
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

fig, ax = plt.subplots(figsize=(9.4, 3.9))
# Use a diverging cmap centered at 0.5 — low = decorative, high = causal.
cmap = plt.get_cmap("RdYlBu_r")
im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)

# Annotate cells
for i in range(len(MODELS)):
    for j in range(len(FAMILIES)):
        v = grid[i, j]
        if np.isnan(v):
            ax.text(j, i, "—", ha="center", va="center",
                     fontsize=FONT_SIZE-2, color="gray")
        else:
            color = "white" if (v < 0.35 or v > 0.75) else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=FONT_SIZE-2, color=color)

ax.set_xticks(range(len(FAMILIES)))
ax.set_xticklabels([f[0] for f in FAMILIES], fontsize=FONT_SIZE-2)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels([m[0] for m in MODELS])
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

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label("Faithful rate  ($\\Delta_\\infty > 0.05$)",
                 fontsize=FONT_SIZE-1)

ax.set_xlabel("Edit family (semantic → controls)")
save(fig, "fig3_edit_heatmap")
