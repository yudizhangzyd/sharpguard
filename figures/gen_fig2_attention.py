"""Fig 2: Multi-family attention distribution — 13 models × 4 buckets.
Anchor figure for finding F1 (attention architecture-determined).
"""
import sys
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import numpy as np, matplotlib.pyplot as plt

MODELS = [
    ("OpenVLA\nspatial", 0.000, 0.524, 0.423, 0.053),
    ("OpenVLA\nobject",  0.000, 0.519, 0.423, 0.059),
    ("OpenVLA\ngoal",    0.000, 0.511, 0.432, 0.056),
    ("OpenVLA\n10",      0.000, 0.535, 0.411, 0.054),
    ("DT-base",          0.229, 0.000, 0.510, 0.253),
    ("DT-SFT",           0.305, 0.000, 0.433, 0.236),
    ("DT-RL",            0.305, 0.000, 0.433, 0.236),
    ("Ours\nr=8",        0.343, 0.288, 0.293, 0.075),
    ("Ours\nr=16",       0.345, 0.290, 0.291, 0.073),
    ("Ours\nr=32",       0.349, 0.287, 0.292, 0.071),
    ("Ours\nr=64",       0.348, 0.284, 0.294, 0.074),
    ("Ours\nno-CoT",     0.340, 0.286, 0.301, 0.073),
    ("ECoT\nbridge",     0.352, 0.287, 0.291, 0.070),
]

labels = [m[0] for m in MODELS]
cot    = [m[1] for m in MODELS]
vis    = [m[2] for m in MODELS]
instr  = [m[3] for m in MODELS]
prev   = [m[4] for m in MODELS]

x = np.arange(len(labels))
w = 0.20

fig, ax = plt.subplots(1, 1, figsize=(11, 4))
ax.bar(x - 1.5*w, vis,   w, label="visual",      color=BUCKET_COLORS["visual"])
ax.bar(x - 0.5*w, instr, w, label="instruction", color=BUCKET_COLORS["instr"])
ax.bar(x + 0.5*w, cot,   w, label="CoT",         color=BUCKET_COLORS["cot"])
ax.bar(x + 1.5*w, prev,  w, label="action-prev", color=BUCKET_COLORS["action_prev"])

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=FONT_SIZE - 2)
ax.set_ylabel(r"Attention mass $\alpha(m,B)$")
ax.set_ylim(0, 0.66)
ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))

for xpos in [3.5, 6.5, 11.5]:
    ax.axvline(xpos, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax.text(1.5, 0.62, "OpenVLA (non-CoT)",             ha="center", fontsize=FONT_SIZE-1, style="italic")
ax.text(5,   0.62, "DeepThinkVLA",                   ha="center", fontsize=FONT_SIZE-1, style="italic")
ax.text(9,   0.62, "Ours (ECoT-LIBERO variants)",    ha="center", fontsize=FONT_SIZE-1, style="italic")
ax.text(12.5,0.62, "Public",                         ha="center", fontsize=FONT_SIZE-1, style="italic")

save(fig, "fig2_attention_distribution")
