"""Fig 3 — causal-edit faithful-rate heatmap (models × edit families).

Rows: 8 CoT-VLA models (7 ours variants + ECoT-bridge).
Cols: 10 edit families.
Cells: faithful rate (0-1), colored via a sequential blue-to-red map.
Highlights the no-CoT collapse and selfsplice-control null.
"""
import json, sys
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


BASE = "/tmp/cf_full_sweep"
OLD = "/tmp/cf_sweep"


MODELS = [
    ("Ours r=8",     f"{BASE}/lora-r8/cotfaith-edit/cot_edit_report.json"),
    ("Ours r=16",    f"{BASE}/lora-r16/cotfaith-edit/cot_edit_report.json"),
    ("Ours r=32",    "/tmp/cf_done/bcihypv3gu/cotfaith-edit/cot_edit_report.json"),
    ("Ours r=64",    f"{BASE}/lora-r64/cotfaith-edit/cot_edit_report.json"),
    ("Ours no-CoT",  f"{BASE}/no-cot/cotfaith-edit/cot_edit_report.json"),
    ("Ours data-50A",f"{BASE}/data-50A/cotfaith-edit/cot_edit_report.json"),
    ("Ours data-50B",f"{BASE}/data-50B/cotfaith-edit/cot_edit_report.json"),
    ("ECoT-bridge",  "/tmp/cf_done/8rcgy9kukj/cotfaith-edit/cot_edit_report.json"),
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
]

grid = np.full((len(MODELS), len(FAMILIES)), np.nan)
for i, (name, path) in enumerate(MODELS):
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        continue
    a = d.get("aggregate", {})
    for j, (_, key) in enumerate(FAMILIES):
        v = a.get(key, {})
        if v.get("n", 0) > 0:
            grid[i, j] = v["faithful_rate"]

fig, ax = plt.subplots(figsize=(8.6, 3.9))
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
