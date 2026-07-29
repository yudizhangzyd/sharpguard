"""Fig 7: Cross-corpus attention profile — LIBERO vs Bridge V2 vs Fractal vs BC-Z.
n=30 for each non-LIBERO corpus (n=100 LIBERO). Shows CoT-VLA attention
shape is consistent across corpora (F5).
"""
import sys
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import numpy as np, matplotlib.pyplot as plt

# (label, cot, visual, instr, prev, N, cot_std, vis_std, instr_std, prev_std)
DATA = [
    ("LIBERO",     0.349, 0.287, 0.292, 0.071, 100, 0.009, 0.008, 0.007, 0.002),
    ("Bridge V2",  0.338, 0.295, 0.301, 0.066,  30, 0.016, 0.010, 0.010, 0.004),
    ("Fractal",    0.336, 0.291, 0.301, 0.072,  30, 0.013, 0.008, 0.009, 0.004),
    ("BC-Z",       0.322, 0.296, 0.311, 0.071,  30, 0.024, 0.010, 0.015, 0.003),
]

labels    = [d[0] for d in DATA]
cot       = [d[1] for d in DATA]
vis       = [d[2] for d in DATA]
instr     = [d[3] for d in DATA]
prev      = [d[4] for d in DATA]
n_sizes   = [d[5] for d in DATA]
cot_std, vis_std, instr_std, prev_std = ([d[i] for d in DATA] for i in (6,7,8,9))

x = np.arange(len(labels))
w = 0.20

fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.5))
ax.bar(x - 1.5*w, vis,   w, yerr=vis_std,   label="visual",      color=BUCKET_COLORS["visual"],      capsize=2)
ax.bar(x - 0.5*w, instr, w, yerr=instr_std, label="instruction", color=BUCKET_COLORS["instr"],       capsize=2)
ax.bar(x + 0.5*w, cot,   w, yerr=cot_std,   label="CoT",         color=BUCKET_COLORS["cot"],         capsize=2)
ax.bar(x + 1.5*w, prev,  w, yerr=prev_std,  label="action-prev", color=BUCKET_COLORS["action_prev"], capsize=2)

ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(N={n})" for l, n in zip(labels, n_sizes)],
                     fontsize=FONT_SIZE - 1)
ax.set_ylabel(r"Attention mass $\alpha(m,B)$  (mean $\pm$ std)")
ax.set_ylim(0, 0.42)
ax.legend(ncol=2, frameon=False, loc="upper right",
           bbox_to_anchor=(1.0, 1.02), fontsize=FONT_SIZE-1)
ax.set_title("ECoT-bridge model, evaluation across 4 manipulation corpora",
              fontsize=FONT_SIZE, loc="left", style="italic")

save(fig, "fig7_cross_corpus")

