"""Fig 10: Cross-corpus causal-edit (F5 causal counterpart, N=15-100).
ECoT-bridge direction_flip / gripper_flip on LIBERO + 3 external corpora.
"""
import sys
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import numpy as np, matplotlib.pyplot as plt

# Real from v13 aws_6 (ik5n2thine, c2saurubyx, kbb6g24nyg) + LIBERO ECoT-bridge run
DATA = [
    ("LIBERO",    0.96, 100, 0.70, 100),
    ("Bridge V2", 1.00,  26, 0.80,  15),
    ("Fractal",   0.92,  25, 0.80,  20),
    ("BC-Z",      0.92,  25, 0.67,  15),
]

labels = [f"{d[0]}\n(dir N={d[2]}; grip N={d[4]})" for d in DATA]
direction = [d[1] for d in DATA]
gripper   = [d[3] for d in DATA]
x = np.arange(len(labels)); w = 0.35
fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.4))
b1 = ax.bar(x - w/2, direction, w, label="direction_flip", color=C_COT_TRAINED, edgecolor="black", linewidth=0.4)
b2 = ax.bar(x + w/2, gripper,   w, label="gripper_flip",   color="#EE6677",     edgecolor="black", linewidth=0.4)
for bar, v in zip(b1, direction):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=FONT_SIZE-2)
for bar, v in zip(b2, gripper):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=FONT_SIZE-2)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=FONT_SIZE-1)
ax.set_ylabel(r"Faithful rate  ($\Delta_\infty > 0.05$)")
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right", frameon=False, fontsize=FONT_SIZE-1)
ax.set_title("ECoT-bridge causal-edit response across 4 corpora (F5)",
              fontsize=FONT_SIZE, loc="left", style="italic")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.4, alpha=0.5)
save(fig, "fig10_cross_corpus_edit")
