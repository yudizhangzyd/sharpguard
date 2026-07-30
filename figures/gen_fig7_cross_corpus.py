"""Fig 7: Cross-corpus attention (F5, N=30 per non-LIBERO corpus).
ECoT-bridge on LIBERO (N=100) + Bridge V2 / Fractal / BC-Z (N=30 each).
Real mean±std bars from v13 aws_6 runs (ik5n2thine/c2saurubyx/kbb6g24nyg).
"""
import sys, json
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import numpy as np, matplotlib.pyplot as plt

# Real means/stds from v13 runs
DATA = [
    ("LIBERO",    (0.287, 0.008, 100), (0.292, 0.007, 100), (0.349, 0.009, 100), (0.071, 0.002, 100)),
    ("Bridge V2", (0.295, 0.010,  30), (0.301, 0.010,  30), (0.338, 0.016,  30), (0.066, 0.004,  30)),
    ("Fractal",   (0.291, 0.008,  30), (0.301, 0.009,  30), (0.336, 0.013,  30), (0.072, 0.004,  30)),
    ("BC-Z",      (0.296, 0.010,  30), (0.311, 0.015,  30), (0.322, 0.024,  30), (0.071, 0.003,  30)),
]

labels  = [d[0] for d in DATA]
vis_m   = [d[1][0] for d in DATA]; vis_s   = [d[1][1] for d in DATA]
instr_m = [d[2][0] for d in DATA]; instr_s = [d[2][1] for d in DATA]
cot_m   = [d[3][0] for d in DATA]; cot_s   = [d[3][1] for d in DATA]
prev_m  = [d[4][0] for d in DATA]; prev_s  = [d[4][1] for d in DATA]
n_sizes = [d[3][2] for d in DATA]

x = np.arange(len(labels)); w = 0.20
fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.5))
ax.bar(x - 1.5*w, vis_m,   w, yerr=vis_s,   label="visual",      color=BUCKET_COLORS["visual"],      capsize=2)
ax.bar(x - 0.5*w, instr_m, w, yerr=instr_s, label="instruction", color=BUCKET_COLORS["instr"],       capsize=2)
ax.bar(x + 0.5*w, cot_m,   w, yerr=cot_s,   label="CoT",         color=BUCKET_COLORS["cot"],         capsize=2)
ax.bar(x + 1.5*w, prev_m,  w, yerr=prev_s,  label="action-prev", color=BUCKET_COLORS["action_prev"], capsize=2)
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(N={n})" for l, n in zip(labels, n_sizes)], fontsize=FONT_SIZE - 1)
ax.set_ylabel(r"Attention mass $\alpha(m,B)$  (mean$\pm$std)")
ax.set_ylim(0, 0.42)
ax.legend(ncol=2, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.02), fontsize=FONT_SIZE-1)
ax.set_title("ECoT-bridge attention across 4 corpora (F5, N=30 per non-LIBERO)",
              fontsize=FONT_SIZE, loc="left", style="italic")
save(fig, "fig7_cross_corpus")
