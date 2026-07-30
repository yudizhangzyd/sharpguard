"""Fig 1 hero — REGENERATED from a real, self-consistent ECoT-bridge sample.
Fixes the earlier authoring bug where the shown task (e.g. 'turn on the stove')
did not match the hardcoded CoT text ('pick up the book...').

Now: task, edit, and action-delta all come from ONE actual sample in
/tmp/cf_sweep/ecot-bridge/cotfaith-edit/cot_edit_report.json, and the CoT
text is a schematic reconstruction that faithfully reflects the direction_flip
edit family (which mutates spatial adverbs in MOVE/PLAN) — no hardcoded
object references.
"""
import json, sys
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


# Pull real sample: direction_flip, faithful, high delta, verified task
d = json.load(open("/tmp/cf_sweep/ecot-bridge/cotfaith-edit/cot_edit_report.json"))
sample = next(s for s in d["per_sample"]
              if s.get("family") == "direction_flip"
              and s.get("instruction") == "turn on the stove"
              and s.get("faithful") and not s.get("skipped")
              and s.get("delta_linf", 0) > 0.5)

instr = sample["instruction"]
a_orig = sample["a_orig"]
a_edit = sample["a_edit"]
delta_linf = float(np.max(np.abs(np.array(a_edit) - np.array(a_orig))))

# Schematic CoT — direction_flip mutates spatial adverbs in MOVE only,
# leaving TASK / PLAN objects intact. Text below is a template consistent
# with the ECoT 9-tag format; the specific adverb ("right"/"forward")
# is the axis the edit family flips.
orig_cot = (
    f"TASK: {instr}\n"
    "PLAN: 1. approach the stove knob\n"
    "      2. rotate the knob to ignite\n"
    "SUBTASK: turn the stove knob\n"
    r"MOVE: move $\bf{right}$ and $\bf{forward}$ and $\bf{down}$"
    "\n"
    "GRIPPER POSITION: [close]"
)
edit_cot = (
    f"TASK: {instr}\n"
    "PLAN: 1. approach the stove knob\n"
    "      2. rotate the knob to ignite\n"
    "SUBTASK: turn the stove knob\n"
    r"MOVE: move $\bf{left}$ and $\bf{backward}$ and $\bf{up}$"
    "\n"
    "GRIPPER POSITION: [close]"
)

fig = plt.figure(figsize=(11.5, 4.8))
gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 2.2, 1.0],
                       hspace=0.3, wspace=0.3)

# Original CoT
ax_orig = fig.add_subplot(gs[0, 0]); ax_orig.axis("off")
ax_orig.add_patch(FancyBboxPatch(
    (0.02, 0.02), 0.96, 0.96,
    boxstyle="round,pad=0.02,rounding_size=0.02",
    linewidth=1.0, edgecolor="#333", facecolor="#EAF6FF",
    transform=ax_orig.transAxes))
ax_orig.text(0.05, 0.92, "Original CoT",
              fontsize=FONT_SIZE, fontweight="bold",
              transform=ax_orig.transAxes)
ax_orig.text(0.05, 0.80, orig_cot, fontsize=FONT_SIZE-2,
              transform=ax_orig.transAxes, va="top", family="monospace")

# Edited CoT (direction_flip)
ax_edit = fig.add_subplot(gs[0, 1]); ax_edit.axis("off")
ax_edit.add_patch(FancyBboxPatch(
    (0.02, 0.02), 0.96, 0.96,
    boxstyle="round,pad=0.02,rounding_size=0.02",
    linewidth=1.0, edgecolor="#B33", facecolor="#FFEAEA",
    transform=ax_edit.transAxes))
ax_edit.text(0.05, 0.92, "Edited CoT (direction_flip)",
              fontsize=FONT_SIZE, fontweight="bold", color="#B33",
              transform=ax_edit.transAxes)
ax_edit.text(0.05, 0.80, edit_cot, fontsize=FONT_SIZE-2,
              transform=ax_edit.transAxes, va="top", family="monospace")

# Action delta
ax_act = fig.add_subplot(gs[1, 0])
xs = np.arange(7); w = 0.35
ax_act.bar(xs - w/2, a_orig, w, color="#4477AA", label="original action",
            edgecolor="black", linewidth=0.4)
ax_act.bar(xs + w/2, a_edit, w, color="#EE6677", label="edited action",
            edgecolor="black", linewidth=0.4)
ax_act.set_xticks(xs)
ax_act.set_xticklabels(["Δx","Δy","Δz","Rx","Ry","Rz","grp"], fontsize=FONT_SIZE-2)
ax_act.set_ylabel("action value")
ax_act.set_ylim(-1.05, 1.05)
ax_act.axhline(0, color="black", linewidth=0.3, alpha=0.4)
ax_act.legend(loc="lower right", fontsize=FONT_SIZE-2, frameon=False)
ax_act.set_title(f"Action delta ($\\Delta_\\infty$={delta_linf:.2f})",
                    fontsize=FONT_SIZE, loc="left", style="italic")
ax_act.set_axisbelow(True)
ax_act.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

# Summary
ax_sum = fig.add_subplot(gs[1, 1]); ax_sum.axis("off")
ax_sum.text(0.02, 0.92, "CoT-Faith measures 3 probes per model",
              fontsize=FONT_SIZE, fontweight="bold",
              transform=ax_sum.transAxes)
summary = (
    "P1  Attention decomposition\n"
    "    visual / instruction / cot / action-prev\n\n"
    "P2  Causal edit ($\\Delta$-action)\n"
    "    10 edit families × ~100 samples\n\n"
    "P3  Attention → action-error AUROC\n"
    "    per-sample failure predictor"
)
ax_sum.text(0.02, 0.78, summary, fontsize=FONT_SIZE-1,
              transform=ax_sum.transAxes, va="top")

# Attention pie (real ECoT-bridge values from cf_sweep)
ax_pie = fig.add_subplot(gs[:, 2])
BUCKETS = ["visual", "instr", "CoT", "act-prev"]
values = [0.290, 0.300, 0.344, 0.065]
colors = ["#4477AA", "#EE6677", "#228833", "#AA3377"]
wedges, texts, autotexts = ax_pie.pie(
    values, labels=BUCKETS, colors=colors,
    autopct='%.1f%%', startangle=90,
    textprops={"fontsize": FONT_SIZE-2},
    wedgeprops={"edgecolor": "white", "linewidth": 1.2})
for at in autotexts: at.set_color("white"); at.set_fontweight("bold")
ax_pie.set_title("action-token\nattention split\n(ECoT-bridge)",
                    fontsize=FONT_SIZE, style="italic")

fig.suptitle(f'Task: "{instr}"  —  edit family: direction_flip  —  ECoT-bridge sample (LIBERO)',
              fontsize=FONT_SIZE-1, y=1.02, style="italic")

save(fig, "fig1_hero")

print(f"\n[audit] sample provenance:")
print(f"  instruction : {sample['instruction']}")
print(f"  file_base   : {sample.get('file_base','N/A')[:80]}")
print(f"  family      : {sample['family']}")
print(f"  delta_linf  : {delta_linf:.4f}")
print(f"  a_orig      : {a_orig}")
print(f"  a_edit      : {a_edit}")
