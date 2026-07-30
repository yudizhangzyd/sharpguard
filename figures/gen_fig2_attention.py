"""Fig 2 -- attention distribution, 15 models x 4 buckets, PLUS the
per-token-normalized view (R1 reviewer item 5f).

Every number is read from results_v2/derived_metrics.json.  There are no
hardcoded numeric literals in this file.
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
        MAIN.append((f"OVLA\n{suite}", ATTN_BASE[k]["mass"], ATTN_BASE[k]["mass_std"]))
_LBL = {"ours-r8": "Ours\nr=8", "ours-r16": "Ours\nr=16", "ours-r32": "Ours\nr=32",
        "ours-r64": "Ours\nr=64", "ours-no-cot": "Ours\nno-CoT",
        "ours-data50A": "Ours\nd-50A", "ours-data50B": "Ours\nd-50B",
        "ecot-bridge": "ECoT\nbridge"}
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

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(14.4, 3.9),
    gridspec_kw={"width_ratios": [12, 3, 9], "wspace": 0.22})

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
               yerr=errs, capsize=1.5, error_kw={"lw": 0.6},
               color=BUCKET_COLORS[BKEY[k]], label=k)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE - 3)
    ax.set_ylim(0, ylim)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)


KEYS = ["visual", "instruction", "cot", "action_prev"]
_bars(ax1, MAIN, KEYS, 0.70)
ax1.set_ylabel(r"Attention mass $\alpha(m,B)$")
ax1.axvline(3.5, color="gray", lw=0.5, ls="--", alpha=0.5)
ax1.axvline(10.5, color="gray", lw=0.5, ls="--", alpha=0.5)
ax1.text(1.5, 0.655, "OpenVLA (non-CoT)", ha="center", fontsize=FONT_SIZE - 2, style="italic")
ax1.text(7, 0.655, "Ours (ECoT LoRA)", ha="center", fontsize=FONT_SIZE - 2, style="italic")
ax1.text(11, 0.655, "ECoT", ha="center", fontsize=FONT_SIZE - 2, style="italic")
ax1.legend(ncol=4, frameon=False, loc="upper center",
           bbox_to_anchor=(0.5, 1.17), fontsize=FONT_SIZE - 2)
ax1.set_title(r"(a) Raw bucket mass. $\alpha(\mathrm{cot}){=}0$ for OpenVLA is "
              r"DEFINITIONAL (no CoT segment exists).",
              fontsize=FONT_SIZE - 1, loc="left", style="italic")

# All four buckets, including visual. The earlier version of this panel dropped
# the visual bar because our harness reported it as identically 0.0 -- which was a
# prompt-format bug on our side, not a property of these checkpoints. The
# corrected runs segment on token ids and measure visual at 0.18-0.19.
_bars(ax2, DT, KEYS, 0.70, dt=True)
ax2.set_title("(b) DeepThinkVLA\n" + r"$\alpha(\mathrm{cot})$ is never the largest bucket",
              fontsize=FONT_SIZE - 1, loc="left", style="italic")

# Panel (c): per-token attention -- the headline "CoT bucket is largest" is a
# token-count artifact.
labels = [r[0] for r in PT]
x = np.arange(len(labels))
w = 0.8 / 4
for i, k in enumerate(KEYS):
    vals = [r[1][k] for r in PT]
    ax3.bar(x + (i - 1.5) * w, vals, w, color=BUCKET_COLORS[BKEY[k]], label=k)
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=FONT_SIZE - 3)
ax3.set_ylabel(r"$\alpha(m,B)\,/\,|B|$  (per key token)")
ax3.set_yscale("log")
ax3.set_ylim(8e-4, 3e-2)
ax3.set_axisbelow(True)
ax3.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
ratio = np.mean([r[1]["instruction"] / r[1]["cot"] for r in PT])
ax3.set_title(f"(c) Per-token: instruction gets {ratio:.1f}"
              r"$\times$ MORE attention per token than CoT",
              fontsize=FONT_SIZE - 1, loc="left", style="italic")
ax3.legend(ncol=4, frameon=False, fontsize=FONT_SIZE - 3,
           loc="upper center", bbox_to_anchor=(0.5, 1.17))

save(fig, "fig2_attention_distribution")
