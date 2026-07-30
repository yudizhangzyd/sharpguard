"""Fig 10 -- cross-corpus causal-edit response (F5), N=30 per non-LIBERO corpus.

Loaded from results_v2/derived_metrics.json.  No hardcoded literals.
NOTE: these are magnitude-F values; the directional caveat of Fig. 12 applies
to the direction_flip column here too (self-decoded CoT logs on the lerobot
corpora do not store a_orig/a_edit, so directional-F cannot yet be computed
cross-corpus -- stated as a limitation).
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
fig, ax = plt.subplots(1, 1, figsize=(6.8, 3.4))
b1 = ax.bar(x - w / 2, direction, w, label="direction_flip", color=C_COT_TRAINED,
            edgecolor="black", linewidth=0.4)
b2 = ax.bar(x + w / 2, gripper, w, label="gripper_flip", color=C_NO_COT,
            edgecolor="black", linewidth=0.4)
for bars, vals in ((b1, direction), (b2, gripper)):
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=FONT_SIZE - 2)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=FONT_SIZE - 1)
ax.set_ylabel(r"magnitude-$\mathcal{F}$  ($\Delta_\infty > 0.05$)")
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right", frameon=False, fontsize=FONT_SIZE - 1)
ax.set_title("ECoT-bridge magnitude-$\\mathcal{F}$ across 4 corpora (F5)",
             fontsize=FONT_SIZE, loc="left", style="italic")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.4, alpha=0.5)
save(fig, "fig10_cross_corpus_edit")
