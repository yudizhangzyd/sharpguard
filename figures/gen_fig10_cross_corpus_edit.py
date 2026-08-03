"""Fig 10 -- cross-corpus causal-edit response (F5), N=100 per non-LIBERO corpus.

Loaded from results_v2/derived_metrics.json.  No hardcoded literals.
NOTE: these are magnitude-F values; the directional caveat of Fig. 12 applies
to the direction_flip column here too (self-decoded CoT logs on the lerobot
corpora do not store a_orig/a_edit, so directional-F cannot yet be computed
cross-corpus -- stated as a limitation).

AUTHORED AT THE INCLUDE WIDTH (see gen_fig4_dissociation.py): the previous canvas
was 6.8in and the figure was included at 0.92\\textwidth, so its 10pt labels
printed at 7.6pt. The canvas below IS the include width.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *
from _data import MODELS, CROSS, fam
import numpy as np, matplotlib.pyplot as plt

rows = [("LIBERO",
         fam("ecot-bridge", "direction_flip", "F_mag"),
         MODELS["ecot-bridge"]["families"]["direction_flip"]["n_total"],
         fam("ecot-bridge", "gripper_flip", "F_mag"),
         MODELS["ecot-bridge"]["families"]["gripper_flip"]["n_total"])]
for tag, name in (("bridge_v2", "Bridge V2"), ("fractal", "Fractal"), ("bcz", "BC-Z")):
    e = CROSS[tag]["edit"]
    rows.append((name, e["direction_flip"]["faithful_rate"], e["direction_flip"]["n"],
                 e["gripper_flip"]["faithful_rate"], e["gripper_flip"]["n"]))

labels = [f"{r[0]}\n(dir N={r[2]}; grip N={r[4]})" for r in rows]
direction = [r[1] for r in rows]
gripper = [r[3] for r in rows]
x = np.arange(len(labels)); w = 0.35

# Sizes on the page.
TICK, LAB, VAL, LEG, TITLE = 6.2, 6.6, 6.0, 6.2, 7.0

fig, ax = plt.subplots(1, 1, figsize=(5.18, 2.45))
# Explicit margins: savefig crops to the artists, so default margins emit a page
# narrower than the canvas and LaTeX scales it back up.
fig.subplots_adjust(left=0.108, right=0.995, top=0.845, bottom=0.155)
ax.tick_params(axis="both", labelsize=TICK, length=2.2, pad=1.5)
b1 = ax.bar(x - w / 2, direction, w, label="direction_flip", color=C_COT_TRAINED,
            edgecolor="black", linewidth=0.4)
b2 = ax.bar(x + w / 2, gripper, w, label="gripper_flip", color=C_NO_COT,
            edgecolor="black", linewidth=0.4)
for bars, vals in ((b1, direction), (b2, gripper)):
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=VAL)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=TICK)
ax.set_ylabel(r"magnitude-$\mathcal{F}$  ($\Delta_\infty > 0.05$)",
              fontsize=LAB, labelpad=1.5)
ax.set_ylim(0, 1.12)
# The legend goes ABOVE the axes, next to the title. Inside at upper right it
# printed through the 0.95 value label of the BC-Z direction_flip bar.
_hl, _ll = ax.get_legend_handles_labels()
fig.legend(_hl, _ll, ncol=2, frameon=False, fontsize=LEG, loc="upper right",
           bbox_to_anchor=(0.995, 1.000), handlelength=1.2, handletextpad=0.5,
           columnspacing=1.6, borderpad=0.1)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
ax.set_title("ECoT-bridge magnitude-$\\mathcal{F}$ across 4 corpora (F5)",
             fontsize=TITLE, loc="left", style="italic", pad=2.5)
ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.4, alpha=0.5)
save(fig, "fig10_cross_corpus_edit")

print("[audit] " + "  ".join(f"{r[0]}: dir={r[1]:.2f} (N={r[2]}), "
                             f"grip={r[3]:.2f} (N={r[4]})" for r in rows))
