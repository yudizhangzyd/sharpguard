"""Fig 1 hero: CoT-Faith workflow (matplotlib version — no TikZ overlap).

Left→right pipeline: (image, instr, GT-CoT) → CoT-VLA → 3 probes → findings.
Boxes are placed with explicit coords to avoid overlap.
"""
import sys
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


fig, ax = plt.subplots(figsize=(11.5, 4.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 40)
ax.axis("off")

def _box(x, y, w, h, text, color, edge="black", fs=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.5,rounding_size=1.2",
        linewidth=1.0, edgecolor=edge, facecolor=color))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
             fontsize=fs or FONT_SIZE-1, wrap=True)

def _arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=15,
        linewidth=1.0, color="black"))

# --- Column 1: inputs ---
_box(2,  30, 15, 5, "LIBERO scene\n(image)", "#EAF2F8")
_box(2,  22, 15, 5, "Task\ninstruction", "#EAF2F8")
_box(2,  14, 15, 5, "Ground-truth\n9-tag CoT", "#EAF2F8")

# --- Column 2: model ---
_box(24, 20, 16, 10,
      "CoT-VLA\n(ECoT family / ablation variant)",
      "#D5F5E3", fs=FONT_SIZE)

# arrows: inputs -> model
_arrow(17, 32.5, 24, 27)
_arrow(17, 24.5, 24, 25)
_arrow(17, 16.5, 24, 23)

# --- Column 3: three probes ---
_box(48, 30, 20, 6,
      "Probe 1: 4-bucket\nattention decomposition",
      "#FDEBD0")
_box(48, 21, 20, 6,
      "Probe 2: 10 causal-\nedit families (Δ-action)",
      "#FDEBD0")
_box(48, 12, 20, 6,
      "Probe 3: attention →\naction-error AUROC",
      "#FDEBD0")

_arrow(40, 27, 48, 33)
_arrow(40, 25, 48, 24)
_arrow(40, 23, 48, 15)

# --- Column 4: 4 findings ---
_box(75, 32, 22, 4.3,
      "F1: attention on CoT is\narchitecture-determined",
      "#FADBD8", fs=FONT_SIZE-1)
_box(75, 26, 22, 4.3,
      "F2: causal effect requires\nreasoning-target training",
      "#FADBD8", fs=FONT_SIZE-1)
_box(75, 20, 22, 4.3,
      "F3: attention $\\ne$ causation\n(selfsplice=0, AUROC=0.5)",
      "#FADBD8", fs=FONT_SIZE-1)
_box(75, 14, 22, 4.3,
      "F4: Bridge-trained CoT-VLAs\n2$\\times$ more faithful",
      "#FADBD8", fs=FONT_SIZE-1)

_arrow(68, 33, 75, 34)
_arrow(68, 24, 75, 22)
_arrow(68, 15, 75, 16)

# Fake extra arrow probe 2 -> F2
_arrow(68, 24, 75, 28)

# Column labels
ax.text(9.5, 38, "Inputs (per sample)",  ha="center", fontsize=FONT_SIZE, fontweight="bold")
ax.text(32,  38, "Model",                ha="center", fontsize=FONT_SIZE, fontweight="bold")
ax.text(58,  38, "3 Probes",             ha="center", fontsize=FONT_SIZE, fontweight="bold")
ax.text(86,  38, "Findings",             ha="center", fontsize=FONT_SIZE, fontweight="bold")

save(fig, "fig1_workflow_matplotlib")
