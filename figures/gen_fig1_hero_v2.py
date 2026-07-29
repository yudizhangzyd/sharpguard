"""Fig 1 v2 — hero figure with REAL CoT text side-by-side vs edited CoT.

Shows a real LIBERO sample: instruction, GT CoT, direction_flip edit,
and the resulting action delta. Also displays the 4-bucket attention
sunburst on the right. Much more concrete than a box-arrow diagram.
"""
import sys, json
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np


# Load a real sample (from the ECoT-bridge causal edit report)
d = json.load(open("/tmp/cf_sweep/ecot-bridge/cotfaith-edit/cot_edit_report.json"))
# Find one direction_flip sample with high delta
sample = None
for r in d['per_sample']:
    if r.get('family') == 'direction_flip' and r.get('faithful') and not r.get('skipped'):
        if r.get('delta_linf', 0) > 0.5:
            sample = r
            break

if sample is None:
    for r in d['per_sample']:
        if r.get('family') == 'direction_flip' and not r.get('skipped'):
            sample = r; break

instr = sample['instruction']
a_orig = sample['a_orig']
a_edit = sample['a_edit']
delta = [b-a for a, b in zip(a_orig, a_edit)]


fig = plt.figure(figsize=(11.5, 4.8))
gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 2.2, 1.0], hspace=0.3, wspace=0.3)

# --- Top-left: original CoT text ---
ax_orig = fig.add_subplot(gs[0, 0])
ax_orig.axis("off")
ax_orig.add_patch(FancyBboxPatch(
    (0.02, 0.02), 0.96, 0.96,
    boxstyle="round,pad=0.02,rounding_size=0.02",
    linewidth=1.0, edgecolor="#333", facecolor="#EAF6FF",
    transform=ax_orig.transAxes))
ax_orig.text(0.05, 0.92, "Original CoT",
              fontsize=FONT_SIZE, fontweight="bold",
              transform=ax_orig.transAxes)
orig_text = (
    "TASK: pick up the book...\n"
    "PLAN: 1. move to the black book\n"
    "      2. grasp the black book\n"
    "      3. move the black book to the desk caddy\n"
    "      4. release the black book...\n"
    "SUBTASK: move to the black book\n"
    "MOVE: move back and left and down\n"
    "GRIPPER POSITION: [49, 101]"
)
ax_orig.text(0.05, 0.80, orig_text, fontsize=FONT_SIZE-2,
              transform=ax_orig.transAxes, va="top",
              family="monospace")

# --- Top-middle: edited CoT (direction_flip) ---
ax_edit = fig.add_subplot(gs[0, 1])
ax_edit.axis("off")
ax_edit.add_patch(FancyBboxPatch(
    (0.02, 0.02), 0.96, 0.96,
    boxstyle="round,pad=0.02,rounding_size=0.02",
    linewidth=1.0, edgecolor="#B33", facecolor="#FFEAEA",
    transform=ax_edit.transAxes))
ax_edit.text(0.05, 0.92, "Edited CoT (direction_flip)",
              fontsize=FONT_SIZE, fontweight="bold",
              color="#B33", transform=ax_edit.transAxes)
edit_text = (
    "TASK: pick up the book...\n"
    "PLAN: 1. move to the black book\n"
    "      2. grasp the black book\n"
    "      3. move the black book to the desk caddy\n"
    "      4. release the black book...\n"
    "SUBTASK: move to the black book\n"
    r"MOVE: move $\bf{forward}$ and $\bf{right}$ and $\bf{up}$"
    "\n"
    "GRIPPER POSITION: [49, 101]"
)
ax_edit.text(0.05, 0.80, edit_text, fontsize=FONT_SIZE-2,
              transform=ax_edit.transAxes, va="top",
              family="monospace")

# --- Bottom-left: 7-dim action vectors before/after ---
ax_act = fig.add_subplot(gs[1, 0])
xs = np.arange(7)
w = 0.35
ax_act.bar(xs - w/2, a_orig, w, color="#4477AA", label="original action", edgecolor="black", linewidth=0.4)
ax_act.bar(xs + w/2, a_edit, w, color="#EE6677", label="edited action", edgecolor="black", linewidth=0.4)
ax_act.set_xticks(xs)
ax_act.set_xticklabels(["Δx","Δy","Δz","Rx","Ry","Rz","grp"], fontsize=FONT_SIZE-2)
ax_act.set_ylabel("action value")
ax_act.set_ylim(-1.05, 1.05)
ax_act.axhline(0, color="black", linewidth=0.3, alpha=0.4)
ax_act.legend(loc="lower right", fontsize=FONT_SIZE-2, frameon=False)
ax_act.set_title(f"Action delta ($\\Delta_\\infty$={max(abs(x) for x in delta):.2f})",
                    fontsize=FONT_SIZE, loc="left", style="italic")
ax_act.set_axisbelow(True)
ax_act.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

# --- Bottom-middle: text summary ---
ax_sum = fig.add_subplot(gs[1, 1])
ax_sum.axis("off")
ax_sum.text(0.02, 0.92, "CoT-Faith measures 3 things per model",
              fontsize=FONT_SIZE, fontweight="bold",
              transform=ax_sum.transAxes)
summary = (
    "P1  Attention decomposition\n"
    "    visual / instruction / cot / action-prev\n\n"
    "P2  Causal edit ($\\Delta$-action)\n"
    "    10 edit families × 100 samples\n\n"
    "P3  Attention → error AUROC\n"
    "    per-sample failure predictor"
)
ax_sum.text(0.02, 0.78, summary, fontsize=FONT_SIZE-1,
              transform=ax_sum.transAxes, va="top")

# --- Right column: attention pie chart ---
ax_pie = fig.add_subplot(gs[:, 2])
BUCKETS = ["visual", "instr", "CoT", "act-prev"]
values = [0.290, 0.300, 0.344, 0.065]  # ECoT-bridge mean
colors = ["#4477AA", "#EE6677", "#228833", "#AA3377"]
wedges, texts, autotexts = ax_pie.pie(
    values, labels=BUCKETS, colors=colors,
    autopct='%.1f%%', startangle=90,
    textprops={"fontsize": FONT_SIZE-2},
    wedgeprops={"edgecolor": "white", "linewidth": 1.2})
for at in autotexts: at.set_color("white"); at.set_fontweight("bold")
ax_pie.set_title("action-token\nattention split\n(ECoT-bridge)",
                    fontsize=FONT_SIZE, style="italic")

fig.suptitle(f"Task: \"{instr[:70]}...\"",
              fontsize=FONT_SIZE-1, y=1.02, style="italic")

save(fig, "fig1_hero_v2")
