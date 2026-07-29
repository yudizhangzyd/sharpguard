"""Fig 6: Cross-corpus attention profile — LIBERO vs Bridge V2 vs Fractal vs BC-Z.
Shows that CoT-VLA attention shape is consistent across corpora,
supporting F5 (cross-corpus generalization).
"""
import sys
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import numpy as np, matplotlib.pyplot as plt

DATA = [
    ("LIBERO",     0.349, 0.287, 0.292, 0.071, 100),
    ("Bridge V2",  0.330, 0.301, 0.309, 0.062,   1),
    ("Fractal",    0.352, 0.287, 0.291, 0.068,   1),
    ("BC-Z",       0.322, 0.285, 0.316, 0.076,   1),
]

labels  = [d[0] for d in DATA]
cot     = [d[1] for d in DATA]
vis     = [d[2] for d in DATA]
instr   = [d[3] for d in DATA]
prev    = [d[4] for d in DATA]
n_sizes = [d[5] for d in DATA]

x = np.arange(len(labels))
w = 0.20

fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.5))
ax.bar(x - 1.5*w, vis,   w, label="visual",       color=BUCKET_COLORS["visual"])
ax.bar(x - 0.5*w, instr, w, label="instruction",  color=BUCKET_COLORS["instr"])
ax.bar(x + 0.5*w, cot,   w, label="CoT",          color=BUCKET_COLORS["cot"])
ax.bar(x + 1.5*w, prev,  w, label="action-prev",  color=BUCKET_COLORS["action_prev"])

ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(N={n})" for l, n in zip(labels, n_sizes)],
                     fontsize=FONT_SIZE - 1)
ax.set_ylabel(r"Attention mass $\alpha(m,B)$")
ax.set_ylim(0, 0.45)
ax.legend(ncol=2, frameon=False, loc="upper right",
           bbox_to_anchor=(1.0, 1.02), fontsize=FONT_SIZE-1)
ax.set_title("ECoT-bridge model, evaluation across 4 manipulation corpora",
              fontsize=FONT_SIZE, loc="left", style="italic")

save(fig, "fig7_cross_corpus")
