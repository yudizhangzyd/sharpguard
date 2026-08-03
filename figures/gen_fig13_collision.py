"""Fig 13 -- what the magnitude score actually counts (S5).

Section 5 makes the paper's sharpest claim about the instrument -- that
F_mag is close to a decode-collision counter rather than a measure of how far
the action moved -- and carries no exhibit at all. It is a claim about the
shape of a distribution and about agreement across 324 cells, which is
precisely the thing prose is worst at and a figure is best at.

(a) the per-record Delta_inf distribution, which is bimodal: nearly half the
    records are EXACTLY zero and most of the rest are far above tau. The bar
    at tau is the whole argument -- almost no mass lies near the threshold, so
    moving tau cannot change much, and F is nearly 1 - P(Delta = 0).
(b) that identity drawn directly: F at tau against 1 - P(Delta = 0), one point
    per (run, family) cell, against the diagonal.

Everything is read from the released collision_decomposition.json. No number
is written into this file -- including the R^2 and the counts in the panel
titles, which are read from the artifact so the figure cannot disagree with
the audit.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *          # noqa: F401,F403  (rcParams + save)
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results_v2", "canonical_runs",
                   "collision_decomposition", "collision_decomposition.json")
with open(SRC) as fh:
    D = json.load(fh)

TAU = D["tau"]
DIST = D["delta_distribution"]
CELLS = D["cells"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.7),
                               gridspec_kw={"wspace": 0.32})

# ---- (a) the Delta_inf distribution ---------------------------------------
# Ordered by the bins the artifact defines, so relabelling a bin upstream
# cannot silently reorder the bars here.
# Leading zeros are kept on every decimal: mathtext treats a bare "." as
# punctuation and inserts a thin space after it, so "$.005$" sets as ". 005".
BINS = [("exactly_zero",  r"$0$"),
        ("zero_to_0.005", "$(0,$\n$0.005]$"),
        ("0.005_to_tau",  f"$(0.005,$\n${TAU:.2f}]$"),
        ("tau_to_0.10",   f"$({TAU:.2f},$\n$0.10]$"),
        ("above_0.10",    r"$>0.10$")]
vals = [DIST[k] for k, _ in BINS]
total = sum(vals)
xs = np.arange(len(BINS))

# The two bins that decide the argument are the ends; the middle three are the
# mass that WOULD have to be there for tau to matter, so they are the ones
# worth colouring as the exception.
colors = [C_NO_COT, C_CTRL, C_CTRL, C_COT_TRAINED, C_COT_TRAINED]
ax1.bar(xs, vals, color=colors, edgecolor="black", lw=0.4, width=0.72)
for x, v in zip(xs, vals):
    ax1.text(x, v + total * 0.018, f"{100*v/total:.1f}%",
             ha="center", fontsize=FONT_SIZE - 4)

# The threshold sits between the third and fourth bin. Drawing it makes the
# point visually: it is placed in a valley, not on a slope.
ax1.axvline(2.5, color="0.25", lw=0.8, ls="--")
ax1.text(2.44, max(vals) * 0.93, rf"$\tau={TAU:g}$", ha="right",
         fontsize=FONT_SIZE - 3, color="0.25", style="italic")

ax1.set_xticks(xs)
ax1.set_xticklabels([lab for _, lab in BINS], fontsize=FONT_SIZE - 3)
ax1.set_xlabel(r"$\Delta_\infty$", fontsize=FONT_SIZE - 1)
ax1.set_ylabel("scored records")
ax1.set_ylim(0, max(vals) * 1.16)
# No \% here: text.usetex is off, so mathtext renders a backslash literally.
ax1.set_title(f"(a) {D['n_scored_records']:,} records: "
              f"{100*DIST['exactly_zero']/total:.0f}% are exactly zero",
              loc="left", fontsize=FONT_SIZE, style="italic")
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, ls=":", lw=0.4, alpha=0.5)

# ---- (b) F at tau vs the collision rate -----------------------------------
f = np.array([c["F_at_tau"] for c in CELLS])
g = np.array([c["one_minus_collision"] for c in CELLS])

ax2.plot([0, 1], [0, 1], color="0.35", lw=0.8, ls="--", zorder=1)
ax2.scatter(g, f, s=9, alpha=0.55, color=C_COT_TRAINED,
            edgecolor="none", zorder=2)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.set_xlabel(r"$1 - P(\Delta_\infty = 0)$", fontsize=FONT_SIZE - 1)
ax2.set_ylabel(rf"$\mathcal{{F}}$ at $\tau={TAU:g}$", fontsize=FONT_SIZE - 1)
ax2.set_title(rf"(b) {D['n_cells']} cells, $R^2={D['r_squared']:.2f}$",
              loc="left", fontsize=FONT_SIZE, style="italic")
ax2.set_axisbelow(True)
ax2.grid(True, ls=":", lw=0.4, alpha=0.5)

# The identical-to-the-last-digit cells are the strongest single number in the
# section, so it is stated on the panel rather than left to the caption.
ax2.text(0.97, 0.06, f"{D['n_cells_exactly_equal']} cells identical\n"
                     f"to the last digit",
         transform=ax2.transAxes, ha="right", va="bottom",
         fontsize=FONT_SIZE - 4, color="0.25")

save(fig, "fig13_collision")

print(f"[audit] source     : {os.path.relpath(SRC, ROOT)}")
print(f"[audit] records    : {D['n_scored_records']}")
print(f"[audit] cells      : {D['n_cells']}  R^2 = {D['r_squared']:.4f}")
print(f"[audit] exact zero : {DIST['exactly_zero']} "
      f"({100*DIST['exactly_zero']/total:.1f}%)")
