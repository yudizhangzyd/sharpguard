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
Cells: faithful rate (0-1) on a sequential map whose LUMINANCE is monotonic in
the value (Blues: low = light, high = dark).
Highlights the no-CoT collapse and selfsplice-control null.

Two further defects, both about whether the figure survives a printer:

  * it used RdYlBu_r, a DIVERGING map. Diverging maps are light in the middle
    and dark at both ends, so 0.00 (dark red-brown, L* 28) and 0.96 (dark blue,
    L* 34) printed at essentially the same grey. The figure's entire gestalt --
    the no-CoT row collapsing while the ECoT row stays uniformly high --
    therefore INVERTED in greyscale, which is how this appendix reaches anyone
    reading a printout. Blues is monotonic in L* from 97 down to 21, so the
    ranking a reader sees in colour is the ranking they see in grey.
  * the cell-text colour switched on the VALUE (white outside 0.35--0.75),
    which under any map is a guess about what the map returns. Under RdYlBu_r it
    was wrong for about ten cells -- 0.36-0.45 and 0.76-0.80 are the palest
    fills in that map and they got white text on near-yellow. The switch now
    reads the rendered rgba the colormap actually returns and picks the colour
    with the higher contrast ratio against it, so it cannot disagree with the
    fill by construction.

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
          # "paraphr." not "paraphrase": at the 6.5pt the cells now carry, the
          # full word is 28.6pt wide in a 27.6pt column slot and its box runs
          # into "instr rand" next door (0.5pt of air between them). The rest of
          # this axis is already abbreviated -- "subj swap", "grip flip",
          # "selfspl" -- so the short form is the axis's own convention, and
          # the caption spells the family out.
          ("paraphr.\n(null)", "paraphrase_null"),
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

# Sizes on the page. Thirteen columns share 5.0in, so each cell is 27.6pt wide
# and 14.8pt tall; a 4-character value at 6.5pt is 11.8pt, so the cell text is
# set by the ROW height rather than the column width and 6.5pt clears it. The x
# tick labels are two lines each and get 27.6pt of slot, and 6.5pt is what the
# widest of them fits in (measured: "cross-task" 26.6pt, "paraphr." 23.0pt).
# 5.4pt -- what all of these were -- is below every legibility floor in the
# style guides this venue's reviewers print by.
CELL, XTICK, YTICK, XLAB, CBAR = 6.5, 6.5, 6.5, 7.0, 8.6


def _cell_text_colour(rgba):
    """white or black, whichever contrasts more with the RENDERED fill.

    The old switch was `"white" if (v < 0.35 or v > 0.75) else "black"` -- a
    hand-typed guess about where the colormap goes light, which the colormap
    was under no obligation to honour. Read the rgba the map actually returns
    instead: convert it to CIE L* and put white on anything below L* 50, which
    is the point where white-on-fill and black-on-fill have equal WCAG contrast
    ratio. A cell can no longer print pale text on a pale fill whatever cmap
    this figure is drawn with next.
    """
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    y = (0.2126 * lin(rgba[0]) + 0.7152 * lin(rgba[1]) + 0.0722 * lin(rgba[2]))
    lstar = 116.0 * y ** (1 / 3) - 16.0 if y > 0.008856 else 903.3 * y
    return "white" if lstar < 50.0 else "black"


fig, ax = plt.subplots(figsize=(5.51, 2.24))
# Explicit margins, and the colorbar in an axes of its own: fig.colorbar(ax=ax)
# steals width from the heatmap AFTER subplots_adjust has run, so the emitted
# page came out narrower than the canvas and LaTeX scaled it back up.
fig.subplots_adjust(left=0.108, right=0.900, top=0.905, bottom=0.230)
# Sequential and monotonic in luminance, NOT diverging: this figure is read in
# greyscale by anyone with a printer, and RdYlBu_r sent 0.00 and 0.96 to the
# same grey (see the module docstring). Blues runs L* 97 -> 21 with no reversal,
# so low = light and high = dark on paper as well as on screen.
cmap = plt.get_cmap("Blues")
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
            # (v - vmin) / (vmax - vmin) with vmin=0, vmax=1 -- i.e. exactly the
            # rgba imshow put in that cell, not a proxy for it.
            color = _cell_text_colour(cmap(im.norm(v)))
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
cbar.ax.tick_params(labelsize=CBAR - 2.1, length=1.8, pad=1.2)
cbar.outline.set_linewidth(0.5)
# Two lines, and 8.6pt rather than the 5.8pt this label used to be set at.
# Mathtext renders a subscript at 0.7x the base size, so the "inf" of
# $\Delta_\infty$ printed at 4.1pt -- the smallest mark in the paper, and part
# of the definition of the quantity the whole colorbar is about. 8.6pt puts it
# at 6.0pt. One line of it at that size is 105pt against the colorbar's 124pt
# of height, which leaves no air at either end, so it is broken at the paren.
cbar.set_label("Faithful rate\n($\\Delta_\\infty > 0.05$)",
                 fontsize=CBAR, labelpad=2.0, linespacing=1.15)

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
