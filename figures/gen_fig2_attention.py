"""Fig 2 -- attention distribution, 15 models x 4 buckets, PLUS the
per-token-normalized view (R1 reviewer item 5f).

Every number is read from results_v2/derived_metrics.json.  There are no
hardcoded numeric literals in this file.

AUTHORED AT THE INCLUDE WIDTH, for the reason gen_fig4_dissociation.py spells
out: matplotlib sizes are absolute points, so a 14.4in canvas included at
0.98\\textwidth (6.2in) printed every tick label at 53% of its authored size --
7pt on the canvas, 3.7pt on the page. Redrawing at the include width is the only
fix; scaling a figure up cannot add the points back.

The rewrite changes no value. It does change three presentation decisions that
the old canvas width was hiding:

  * model labels rotate to one line. Twelve two-line labels in panel (a) get
    18pt of slot each, and "spatial" at 5.3pt is 19pt wide.
  * panel (b) is 0.9in wide and its title is 40 characters, so the title breaks
    over three short lines instead of running a full inch into panel (c).
  * the legends lose their default padding, which at 5pt is a third of the box.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *
from _data import ATTN, ATTN_BASE, ATTN_DT, ORDER
import numpy as np
import matplotlib.pyplot as plt

# ---- Panel (a): OpenVLA non-CoT + ECoT family, raw bucket mass -------------
MAIN = []
for suite in ("spatial", "object", "goal", "10"):
    k = f"openvla-libero-{suite}"
    if k in ATTN_BASE:
        MAIN.append((f"OVLA {suite}", ATTN_BASE[k]["mass"], ATTN_BASE[k]["mass_std"]))
_LBL = {"ours-r8": "Ours r=8", "ours-r16": "Ours r=16", "ours-r32": "Ours r=32",
        "ours-r64": "Ours r=64", "ours-no-cot": "Ours no-CoT",
        "ours-data50A": "Ours d-50A", "ours-data50B": "Ours d-50B",
        "ecot-bridge": "ECoT bridge"}
for m in ["ours-r8", "ours-r16", "ours-r32", "ours-r64", "ours-no-cot",
          "ours-data50A", "ours-data50B", "ecot-bridge"]:
    if m in ATTN:
        MAIN.append((_LBL[m], ATTN[m]["mass"], ATTN[m]["mass_std"]))

DT = [(n, ATTN_DT[n]["mass"], ATTN_DT[n]["mass_std"]) for n in
      ("DT-base", "DT-SFT", "DT-RL") if n in ATTN_DT]

# per-token normalized (mass / segment length), ECoT family only -- the
# baselines have |cot| = 0 so the ratio is undefined there.
PT = [(_LBL[m], ATTN[m]["per_token"]) for m in
      ["ours-r8", "ours-r16", "ours-r32", "ours-r64", "ours-no-cot",
       "ours-data50A", "ours-data50B", "ecot-bridge"] if m in ATTN]

# Sizes on the page, not on a canvas that will be shrunk. 6.5pt is the floor
# every surviving figure in the submission is held to; panel (a) fits 12 model
# slots into 3.0in, and at 45 degrees of rotation the perpendicular gap between
# adjacent labels is 0.71 of the 16pt slot, so 6.5pt clears it (measured).
TITLE, TICK, YLAB, LEG, GRP = 6.5, 6.5, 6.6, 6.5, 6.5

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(6.28, 2.80),
    gridspec_kw={"width_ratios": [12, 3.6, 8.4]})
# Explicit margins: savefig crops to the artists, so the default side margins
# would emit a page narrower than 6.28in and LaTeX would scale it back up.
fig.subplots_adjust(left=0.072, right=0.998, top=0.815, bottom=0.215,
                    wspace=0.30)
for ax in (ax1, ax2, ax3):
    ax.tick_params(axis="y", labelsize=TICK + 0.4, length=2.2, pad=1.5)
    ax.tick_params(axis="x", length=0, pad=1.5)

BKEY = {"visual": "visual", "instruction": "instr", "cot": "cot",
        "action_prev": "action_prev"}


def _bars(ax, rows, keys, ylim, err=True, dt=False):
    labels = [r[0] for r in rows]
    x = np.arange(len(labels))
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        vals = [r[1].get(k if not dt else ("instr" if k == "instruction" else k), 0.0) or 0.0
                for r in rows]
        errs = None
        if err and len(r := rows) and len(rows[0]) > 2:
            errs = [rows[j][2].get(k if not dt else ("instr" if k == "instruction" else k), 0.0) or 0.0
                    for j in range(len(rows))]
        ax.bar(x + (i - (len(keys) - 1) / 2) * w, vals, w,
               yerr=errs, capsize=1.0, error_kw={"lw": 0.45},
               color=BUCKET_COLORS[BKEY[k]], label=k)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_ylim(0, ylim)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)


KEYS = ["visual", "instruction", "cot", "action_prev"]
_bars(ax1, MAIN, KEYS, 0.70)
ax1.set_ylabel(r"Attention mass $\alpha(m,B)$", fontsize=YLAB, labelpad=1.5)
ax1.axvline(3.5, color="gray", lw=0.5, ls="--", alpha=0.5)
ax1.axvline(10.5, color="gray", lw=0.5, ls="--", alpha=0.5)
ax1.text(1.5, 0.655, "OpenVLA (non-CoT)", ha="center", fontsize=GRP, style="italic")
ax1.text(7, 0.655, "Ours (ECoT LoRA)", ha="center", fontsize=GRP, style="italic")
ax1.text(11, 0.655, "ECoT", ha="center", fontsize=GRP, style="italic")
# Two lines: at 6.5pt this claim is 260pt wide and the panel is 216pt. It is
# also the panel's whole point, so it is not a candidate for the caption.
ax1.set_title(r"(a) Raw bucket mass. $\alpha(\mathrm{cot}){=}0$ for OpenVLA"
              "\n" r"is DEFINITIONAL (no CoT segment exists).",
              fontsize=TITLE, loc="left", style="italic", pad=2.0,
              linespacing=1.25)

# All four buckets, including visual. The earlier version of this panel dropped
# the visual bar because our harness reported it as identically 0.0 -- which was a
# prompt-format bug on our side, not a property of these checkpoints. The
# corrected runs segment on token ids and measure visual at 0.18-0.19.
_bars(ax2, DT, KEYS, 0.70, dt=True)
# Three short lines, not one long one: this panel is 0.9in wide and the
# single-line form ran a full inch into panel (c)'s territory.
# Three lines because the panel is 0.9in wide, but broken at word boundaries:
# the earlier form hyphenated "DeepThink-/VLA" across two of them.
ax2.set_title("(b) DT family:\n" + r"$\alpha(\mathrm{cot})$" + "\nnever largest",
              fontsize=TITLE, loc="left", style="italic", pad=2.0,
              linespacing=1.25)

# Panel (c): per-token attention -- the headline "CoT bucket is largest" is a
# token-count artifact.
labels = [r[0] for r in PT]
x = np.arange(len(labels))
w = 0.8 / 4
for i, k in enumerate(KEYS):
    vals = [r[1][k] for r in PT]
    ax3.bar(x + (i - 1.5) * w, vals, w, color=BUCKET_COLORS[BKEY[k]], label=k)
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=TICK, rotation=45, ha="right",
                    rotation_mode="anchor")
ax3.set_ylabel(r"$\alpha(m,B)\,/\,|B|$  (per key token)", fontsize=YLAB,
               labelpad=1.5)
ax3.set_yscale("log")
ax3.set_ylim(8e-4, 3e-2)
ax3.set_axisbelow(True)
ax3.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
ratios = [r[1]["instruction"] / r[1]["cot"] for r in PT]
# A RANGE over the 8 CoT-VLAs, not their mean. The mean printed "4.0x" here
# while Section 5's per-token paragraph quotes "3.9x" for r=32 -- two correct
# numbers for two different quantities, rounded apart, with nothing on either
# to say which was which. The range covers the r=32 value the prose quotes and
# cannot be mistaken for a single-model figure.
ratio_lo, ratio_hi = min(ratios), max(ratios)
# En dash, not "--": this is a matplotlib string, not LaTeX, so "--" prints as
# two hyphens.
ax3.set_title(f"(c) Per-token: instruction beats CoT by {ratio_lo:.1f}"
              f"\u2013{ratio_hi:.1f}" r"$\times$"
              "\non every one of the 8 CoT-VLAs",
              fontsize=TITLE, loc="left", style="italic", pad=2.0,
              linespacing=1.25)
# ONE legend for the whole figure, above the panel titles. Per-axes legends put
# it inside the axes at bbox (0.5, 1.15), which is where the title now is: the
# titles grew from one line to two when the canvas narrowed, and the legend
# printed straight through them. Four buckets are the same four buckets in every
# panel, so one legend is also the correct number of legends.
_hl, _ll = ax1.get_legend_handles_labels()
fig.legend(_hl, _ll, ncol=4, frameon=False, fontsize=LEG, loc="upper center",
           bbox_to_anchor=(0.5, 1.000), handlelength=1.1, handletextpad=0.4,
           columnspacing=1.4, borderpad=0.1)

save(fig, "fig2_attention_distribution")

print(f"[audit] panel (a) models : {len(MAIN)}")
print(f"[audit] panel (b) models : {len(DT)}")
print(f"[audit] per-token ratio  : {ratio_lo:.2f}x--{ratio_hi:.2f}x "
      f"(r=32 alone: {PT[2][1]['instruction'] / PT[2][1]['cot']:.2f}x)")
print(f"[audit] action_prev/cot  : "
      f"{PT[2][1]['action_prev'] / PT[2][1]['cot']:.2f}x at r=32")