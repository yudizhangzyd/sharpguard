"""Fig 3 — causal-edit faithful-rate heatmap (models × edit families).

Rows: 8 CoT-VLA models (7 ours variants + ECoT-bridge).
Cols: ALL 13 edit families, grouped, with the group boundaries drawn.

Two things were wrong with the previous version and both are the same defect:

  * it drew 11 of the 13 families, and the two it dropped -- bbox_jitter_null
    and instr_random_sub -- are measured on all seven "ours" rows and absent
    only on ecot-bridge. They are also the two most load-bearing columns the
    figure could carry: bbox_jitter_null is the tightest floor in the paper
    (0.05-0.09) and instr_random_sub is the out-of-CoT CEILING that Section 4
    compares the CoT families against. Dropping them left a grid whose columns
    are all mid-to-high and removed the figure's own evidence that the score is
    not simply always-high. The docstring meanwhile said "10 edit families"
    while the list held 11, so nothing in the file agreed with anything else.
  * the axis read "semantic -> controls" over an order that interleaved them:
    the identity null sat in column 8, between adversarial_plausible and
    syntactic_scramble. The grouping the label promised is now real, drawn, and
    read from _data.NON_CONTROL rather than retyped here, so the figure cannot
    drift from the derivation's own definition of "non-control".

The two ecot-bridge cells that genuinely have no measurement print as "—",
which is what the absence of a run should look like next to a value.
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
from _data import MODELS as DM, fam, NON_CONTROL
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


MODELS = [
    ("Ours r=8", "ours-r8"), ("Ours r=16", "ours-r16"),
    ("Ours r=32", "ours-r32"), ("Ours r=64", "ours-r64"),
    ("Ours no-CoT", "ours-no-cot"), ("Ours data-50A", "ours-data50A"),
    ("Ours data-50B", "ours-data50B"), ("ECoT-bridge", "ecot-bridge"),
]

# Grouped, and the groups are the paper's own: the 7 NON_CONTROL families are
# imported from _data so this figure and derive_metrics.py cannot disagree about
# which families are semantic.
_SEMANTIC = [("dir\nflip", "direction_flip"), ("negation", "negation"),
             ("verb\nswap", "verb_swap"), ("subj\nswap", "subject_swap"),
             ("loc\nswap", "location_swap"), ("grip\nflip", "gripper_flip"),
             ("adv\nplaus", "adversarial_plausible")]
assert {f for _, f in _SEMANTIC} == set(NON_CONTROL), \
    "this figure's 'semantic' group has drifted from derive_metrics.NON_CONTROL"
_TIER0 = [("selfspl\n(null)", "selfsplice_control"),
          ("syntactic\nscramble", "syntactic_scramble"),
          ("cross-task\nswap", "cross_task_swap")]
_CALIB = [("bbox\njitter", "bbox_jitter_null"),
          ("paraphrase\n(null)", "paraphrase_null"),
          ("instr rand\n(ceiling)", "instr_random_sub")]
FAMILIES = _SEMANTIC + _TIER0 + _CALIB
GROUPS = [("semantic (7)", 0, len(_SEMANTIC)),
          ("Tier-0 controls", len(_SEMANTIC), len(_SEMANTIC) + len(_TIER0)),
          ("calibration", len(_SEMANTIC) + len(_TIER0), len(FAMILIES))]

grid = np.full((len(MODELS), len(FAMILIES)), np.nan)
for i, (name, key) in enumerate(MODELS):
    for j, (_, f) in enumerate(FAMILIES):
        v = fam(key, f, "F_mag")
        if v is not None:
            grid[i, j] = v

# Sizes on the page. Thirteen columns share 5.0in, so each cell is 27.7pt wide
# and 18pt tall -- a 4-character value at 5.4pt is 12.6pt, which is what sets
# CELL. The x tick labels are two lines each and get 27.7pt of slot, so
# "paraphrase" at 5.4pt (25pt) is what sets XTICK.
CELL, XTICK, YTICK, XLAB, CBAR = 5.4, 5.4, 6.4, 6.6, 5.8

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

# The group boundaries the axis label used to only promise.
for _, _, end in GROUPS[:-1]:
    ax.axvline(end - 0.5, color="white", lw=1.6)
    ax.axvline(end - 0.5, color="0.25", lw=0.6)
for name, lo, hi in GROUPS:
    ax.text((lo + hi - 1) / 2, -0.68, name, ha="center", va="bottom",
            fontsize=XTICK, style="italic", color="0.25")
ax.set_ylim(len(MODELS) - 0.5, -0.85)
# En dash: this is a matplotlib string, so "---" prints as three hyphens.
ax.set_xlabel("Edit family \u2014 all 13, grouped", fontsize=XLAB, labelpad=2.0)
save(fig, "fig3_edit_heatmap")

print(f"[audit] grid: {grid.shape[0]} models x {grid.shape[1]} families, "
      f"{int(np.isnan(grid).sum())} missing")
for name, lo, hi in GROUPS:
    print(f"[audit]   {name:16} cols {lo}-{hi - 1}: "
          + " ".join(f[1] for f in FAMILIES[lo:hi]))
_miss = [(MODELS[i][0], FAMILIES[j][1]) for i in range(len(MODELS))
         for j in range(len(FAMILIES)) if np.isnan(grid[i, j])]
print(f"[audit] drawn as absent: {_miss}")
