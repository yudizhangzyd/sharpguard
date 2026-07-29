"""Fig 2 (M2 fix): Multi-family attention distribution — 13 models × 4 buckets.
DeepThinkVLA (PaliGemma) shown in separate panel because its image tokens
live outside the 4-bucket decomposition (visual=0 is a schema artifact,
not a training effect).
"""
import sys
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import numpy as np, matplotlib.pyplot as plt

# Panel A: OpenVLA/ECoT families (visual bucket comparable across these 10 models)
MAIN = [
    ("OpenVLA\nspatial", 0.000, 0.524, 0.423, 0.053),
    ("OpenVLA\nobject",  0.000, 0.519, 0.423, 0.059),
    ("OpenVLA\ngoal",    0.000, 0.511, 0.432, 0.056),
    ("OpenVLA\n10",      0.000, 0.535, 0.411, 0.054),
    ("Ours\nr=8",        0.343, 0.288, 0.293, 0.075),
    ("Ours\nr=16",       0.345, 0.290, 0.291, 0.073),
    ("Ours\nr=32",       0.349, 0.287, 0.292, 0.071),
    ("Ours\nr=64",       0.348, 0.284, 0.294, 0.074),
    ("Ours\nno-CoT",     0.340, 0.286, 0.301, 0.073),
    ("ECoT\nbridge",     0.352, 0.287, 0.291, 0.070),
]

# Panel B: DeepThinkVLA (PaliGemma) — visual bucket=0 is schema artifact (image
# tokens don't map to our 4-bucket decomposition on PaliGemma).
DT = [
    ("DT-base", 0.229, 0.000, 0.510, 0.253),
    ("DT-SFT",  0.305, 0.000, 0.433, 0.236),
    ("DT-RL",   0.305, 0.000, 0.433, 0.236),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.0),
                                  gridspec_kw={"width_ratios": [10, 3], "wspace": 0.18})

def _plot(ax, rows, hide_visual=False):
    labels = [r[0] for r in rows]
    cot    = [r[1] for r in rows]
    vis    = [r[2] for r in rows]
    instr  = [r[3] for r in rows]
    prev   = [r[4] for r in rows]
    x = np.arange(len(labels))
    w = 0.20
    if not hide_visual:
        ax.bar(x - 1.5*w, vis, w, color=BUCKET_COLORS["visual"], label="visual")
    ax.bar(x - 0.5*w, instr, w, color=BUCKET_COLORS["instr"],  label="instruction")
    ax.bar(x + 0.5*w, cot,   w, color=BUCKET_COLORS["cot"],    label="CoT")
    ax.bar(x + 1.5*w, prev,  w, color=BUCKET_COLORS["action_prev"], label="action-prev")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE - 2)
    ax.set_ylim(0, 0.66)

_plot(ax1, MAIN)
ax1.set_ylabel(r"Attention mass $\alpha(m,B)$")
ax1.axvline(3.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax1.axvline(8.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax1.text(1.5, 0.62, "OpenVLA (non-CoT)",         ha="center", fontsize=FONT_SIZE-1, style="italic")
ax1.text(6,   0.62, "Ours (ECoT LoRA variants)",  ha="center", fontsize=FONT_SIZE-1, style="italic")
ax1.text(9,   0.62, "Public ECoT",                ha="center", fontsize=FONT_SIZE-1, style="italic")
ax1.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=FONT_SIZE-1)
ax1.set_title("(a) OpenVLA + ECoT families  —  visual bucket comparable",
                fontsize=FONT_SIZE, loc="left", style="italic")

_plot(ax2, DT, hide_visual=True)
ax2.text(1, 0.62, "DeepThinkVLA (PaliGemma)", ha="center", fontsize=FONT_SIZE-1, style="italic")
ax2.set_title("(b) DT: image tokens outside 4-bucket",
                fontsize=FONT_SIZE, loc="left", style="italic")

save(fig, "fig2_attention_distribution")
