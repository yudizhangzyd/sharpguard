"""Fig 4 -- attention/causation dissociation (F3), with the ceiling-normalized
panel demanded by R1 item 4.

(a) alpha(m, cot) per model, with the measured run-to-run noise floor drawn as
    a band so the 2.3 pp cluster spread can be read against it.
(b) raw mean magnitude-F over the 7 non-control families.
(c) the SAME quantity normalized by each model's own cross_task_swap ceiling.
    Under normalization the no-CoT collapse largely disappears -- F2 has to be
    restated.

All numbers from results_v2/derived_metrics.json.  No hardcoded literals.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *
from _data import MODELS, ATTN, NOISE, ORDER, LABELS, OURS
import numpy as np
import matplotlib.pyplot as plt

MS = [m for m in ORDER if m in MODELS and m in ATTN]
# Count from MS, not from OURS: if a variant is missing from the artifact it
# drops out of the axis, and a bracket sized from the full list would then
# reach past the bars it is supposed to group.
N_OURS = sum(1 for m in MS if m in OURS)
# The bracket spans positions 0..N_OURS-1, so it is only truthful if our
# variants are the leading contiguous block of the axis.
assert all(m in OURS for m in MS[:N_OURS]), \
    "ORDER no longer puts our fine-tunes first; the bracket would mislabel"
COL = [C_NO_COT if m == "ours-no-cot" else
       C_ECOT_BRIDGE if m == "ecot-bridge" else C_COT_TRAINED for m in MS]
xs = np.arange(len(MS))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.6, 3.3),
                                    gridspec_kw={"wspace": 0.28})

# ---- (a) attention on CoT + measured noise floor --------------------------
att = [ATTN[m]["mass"]["cot"] for m in MS]
astd = [ATTN[m]["mass_std"]["cot"] for m in MS]
ax1.bar(xs, att, color=COL, edgecolor="black", lw=0.4, yerr=astd, capsize=2,
        error_kw={"lw": 0.6})
mid = float(np.mean(att))
half = NOISE["abs_diff_pp"] / 200.0          # +/- half of the observed run-to-run gap
ax1.axhspan(mid - half, mid + half, color="gray", alpha=0.22, lw=0)
# Anchored top-RIGHT. Two earlier placements both collided: bottom-right ran
# back under the value labels of the final two models, and bottom-left ran
# across the first bar. The band sits at ~0.35 and the tallest label reaches
# ~0.39, so everything above that is free -- which is the only region of this
# axes that is. The leader line carries the reference down to the band.
ax1.text(len(MS) - 0.45, 0.487,
         f"run-to-run noise: {NOISE['abs_diff_pp']:.2f} pp\n"
         f"({100*NOISE['noise_as_frac_of_spread']:.0f}% of the "
         f"{NOISE['cluster_spread_pp']:.2f} pp spread)",
         ha="right", va="top", fontsize=FONT_SIZE - 4, color="0.25",
         linespacing=1.35)
ax1.annotate("", xy=(len(MS) - 0.62, mid + half), xytext=(len(MS) - 0.62, 0.428),
             arrowprops=dict(arrowstyle="->", lw=0.6, color="0.35"))
for i, v in enumerate(att):
    ax1.text(i, v + 0.016, f"{v:.3f}", ha="center", fontsize=FONT_SIZE - 4)
ax1.set_xticks(xs); ax1.set_xticklabels([LABELS[m] for m in MS], fontsize=FONT_SIZE - 3)
ax1.set_ylabel(r"$\alpha(m,\mathrm{cot})$")
ax1.set_ylim(0, 0.50)
ax1.set_title("(a) Attention on CoT: spread is inside the noise floor",
              loc="left", fontsize=FONT_SIZE, style="italic")
ax1.set_axisbelow(True); ax1.yaxis.grid(True, ls=":", lw=0.4, alpha=0.5)
ours_bracket(ax1, N_OURS)

# ---- (b) raw mean magnitude-F -------------------------------------------
raw = [MODELS[m]["F_bar_mag"] for m in MS]
ax2.bar(xs, raw, color=COL, edgecolor="black", lw=0.4)
for i, v in enumerate(raw):
    ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=FONT_SIZE - 4)
ax2.set_xticks(xs); ax2.set_xticklabels([LABELS[m] for m in MS], fontsize=FONT_SIZE - 3)
ax2.set_ylabel(r"$\bar{\mathcal{F}}$ (raw, 7 families)")
ax2.set_ylim(0, 1.05)
ax2.set_title(f"(b) Raw: {max(raw)/min(raw):.1f}$\\times$ spread",
              loc="left", fontsize=FONT_SIZE, style="italic")
ax2.set_axisbelow(True); ax2.yaxis.grid(True, ls=":", lw=0.4, alpha=0.5)
ours_bracket(ax2, N_OURS)

# ---- (c) normalized by each model's own cross_task_swap ceiling ------------
nrm = [MODELS[m]["F_bar_norm_ceiling"] for m in MS]
ax3.bar(xs, nrm, color=COL, edgecolor="black", lw=0.4)
for i, v in enumerate(nrm):
    ax3.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=FONT_SIZE - 4)
ax3.set_xticks(xs); ax3.set_xticklabels([LABELS[m] for m in MS], fontsize=FONT_SIZE - 3)
ax3.set_ylabel(r"$\bar{\mathcal{F}}\,/\,\mathcal{F}(\mathrm{cross\_task\_swap})$")
ax3.set_ylim(0, 1.05)
ax3.set_title(f"(c) Ceiling-normalized: only {max(nrm)/min(nrm):.1f}$\\times$ spread",
              loc="left", fontsize=FONT_SIZE, style="italic")
ax3.set_axisbelow(True); ax3.yaxis.grid(True, ls=":", lw=0.4, alpha=0.5)
ours_bracket(ax3, N_OURS)

save(fig, "fig4_dissociation")
