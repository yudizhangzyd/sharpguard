"""Fig 11 (M3): does the magnitude leaderboard's ORDERING survive the threshold?

Reads results_v2/canonical_runs/threshold_sweep/threshold_sweep.json, which
scripts/threshold_sweep.py derives from the released per-sample records. There
are no /tmp paths and no numeric literals in this file.

Two things about the previous version of this figure were wrong, and both are
worth recording because the caption was drawn from it:

  * it read the pre-C5-fix /tmp run directories, where location_swap carries
    N=12, so it averaged over a family set that dropped location_swap and
    substituted cross_task_swap -- a Tier-0 CONTROL -- while the caption called
    the set "7 non-control families". The sweep artifact uses the canonical
    NON_CONTROL seven, every one at its full post-fix N.
  * it drew "Ours r=8", "Ours r=16" and "Ours r=64" in one identical blue, so
    the three curves whose crossings are the whole point were indistinguishable.
    The four LoRA ranks now share a blue SEQUENCE, light to dark, and carry
    distinct markers, so they separate in grayscale too.

AUTHORED AT THE INCLUDE WIDTH (see gen_fig4_dissociation.py): matplotlib font
sizes are absolute points, so a 6.4in canvas included at 0.87\\textwidth printed
every label at 75% of the size asked for here. The canvas below IS the include
width.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = json.load(open(os.path.join(
    ROOT, "results_v2/canonical_runs/threshold_sweep/threshold_sweep.json")))

TAUS = SWEEP["taus"]
TAU_DEF = SWEEP["tau_default"]
BY = SWEEP["by_tau"]
# Last tau whose ordering is still the tau=0.05 ordering. Everything to the
# right of it is shaded: that is the figure's actual finding.
TAU_STABLE = SWEEP["max_tau_with_identical_ordering"]

# The four LoRA ranks as one blue sequence so they read as a family and still
# separate; the two data-fraction seeds and the two anchors keep the palette
# they have everywhere else in the paper.
STYLE = {
    "Ours r=8":    ("#A6C8E8", "o", "-"),
    "Ours r=16":   ("#7BA7D0", "s", "-"),
    "Ours r=32":   ("#4477AA", "^", "-"),
    "Ours r=64":   ("#1F4E79", "D", "-"),
    "data-50A":    ("#AA3377", "v", "--"),
    "data-50B":    ("#CCBB44", "P", "--"),
    "no-CoT":      (C_NO_COT, "X", ":"),
    "ECoT-bridge": (C_ECOT_BRIDGE, "*", "-"),
}

# Sizes on the page. Eight legend entries in two columns inside a 4.9in axes is
# what sets LEG.
TICK, LAB, LEG, ANNOT = 6.0, 6.6, 5.8, 5.6

fig, ax = plt.subplots(1, 1, figsize=(5.06, 2.90))
# Explicit margins: savefig crops to the artists, so default margins emit a
# narrower page that LaTeX scales back up, undoing the point of authoring here.
fig.subplots_adjust(left=0.108, right=0.995, top=0.878, bottom=0.155)
ax.tick_params(axis="both", labelsize=TICK, length=2.2, pad=1.5)

# Shade the region where the ordering is no longer the tau=0.05 ordering.
ax.axvspan(TAU_STABLE, TAUS[-1], color="#EE6677", alpha=0.055, lw=0, zorder=0)

for name in SWEEP["configurations"]:
    color, marker, ls = STYLE[name]
    ax.plot(TAUS, [BY[f"{t:g}"]["rate"][name] for t in TAUS],
            marker=marker, label=name, color=color, linestyle=ls,
            linewidth=1.0, markersize=2.8, markeredgewidth=0.0, zorder=3)

ax.axvline(TAU_DEF, color="gray", linestyle=":", linewidth=0.7, alpha=0.7)
ax.text(TAU_DEF * 0.93, 0.985, rf"$\tau={TAU_DEF:g}$ (default)", rotation=90,
        fontsize=ANNOT, color="gray", ha="right", va="top",
        transform=ax.get_xaxis_transform())
# Name the finding on the canvas rather than leaving it to the caption: the
# shaded band is the only reason this figure is in the paper. The band starts at
# the last tau whose ordering is still identical, so it contains tau=0.10, where
# the ordering differs by a single adjacent swap -- hence rho is quoted at both
# ends rather than only at its worst, which would misread the band's left edge.
_rhos = {t: BY[f"{t:g}"]["spearman_vs_default"] for t in TAUS}
_inside = [t for t in TAUS if t > TAU_STABLE]
# Upper RIGHT inside the band: the only corner no curve enters. Every other
# placement tried collided with either the ECoT-bridge curve or the legend.
ax.text(TAUS[-1] * 0.97, 0.980,
        "ordering no longer identical\n"
        rf"($\rho={_rhos[_inside[0]]:.2f}$ at $\tau={_inside[0]:g}$, "
        rf"${min(_rhos.values()):.2f}$ at $\tau={min(_rhos, key=_rhos.get):g}$)",
        fontsize=ANNOT, color="#B03A2E", ha="right", va="top",
        linespacing=1.3, transform=ax.get_xaxis_transform())

ax.set_xscale("log")
ax.set_xlabel(r"Faithfulness threshold  $\tau$", fontsize=LAB, labelpad=1.5)
ax.set_ylabel(f"Mean magnitude faithful rate\n(over {SWEEP['n_families']} "
              "non-control families)", fontsize=LAB, labelpad=1.5,
              linespacing=1.3)
ax.set_ylim(0, 1.0)
# Eight entries, and nowhere inside the axes to put them: the no-CoT curve runs
# along y=0.08-0.17 for the full width (killing lower left), and ECoT-bridge runs
# along y=0.87-0.90 (killing upper left). So the legend goes outside, two rows of
# four above the axes, and `top` below leaves the room for it.
_hl, _ll = ax.get_legend_handles_labels()
fig.legend(_hl, _ll, ncol=4, frameon=False, fontsize=LEG, loc="upper center",
           bbox_to_anchor=(0.54, 1.000), handlelength=1.6, handletextpad=0.5,
           labelspacing=0.25, columnspacing=1.6, borderpad=0.1)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

save(fig, "fig11_threshold_sensitivity")

print(f"[audit] {SWEEP['n_configurations']} configurations, "
      f"{SWEEP['n_families']} families, taus {TAUS}")
print(f"[audit] ordering identical to tau={TAU_DEF:g} for tau <= {TAU_STABLE:g}")
for t in TAUS:
    b = BY[f"{t:g}"]
    print(f"[audit] tau={t:<5.2f} rho={b['spearman_vs_default']:5.3f} "
          f"max_rank_move={b['max_rank_move']} "
          f"minCoT/noCoT={b['min_cot_over_nocot']:.2f}x")
