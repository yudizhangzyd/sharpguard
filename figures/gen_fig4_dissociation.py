"""Fig 4 -- attention/causation dissociation (F3), with the ceiling-normalized
panel demanded by R1 item 4.

(a) alpha(m, cot) per model, with the measured run-to-run noise floor drawn as
    a band so the 2.3 pp cluster spread can be read against it.
(b) raw mean magnitude-F over the 7 non-control families.
(c) the SAME quantity normalized by each model's own cross_task_swap ceiling.
    Under normalization the no-CoT collapse largely disappears -- F2 has to be
    restated.

All numbers from results_v2/derived_metrics.json.  No hardcoded literals.

AUTHORED AT THE INCLUDE WIDTH. This figure used to be drawn 13.6in wide and
included at \\textwidth (6.30in), i.e. downscaled to 57%, which put its bar
value labels on the page at 3.4pt and its tick labels at 4.0pt -- unreadable in
print, and the smallest type anywhere in the paper. Matplotlib font sizes are
absolute points, so the only fix is to draw at the size the page will show.
Every size below is therefore the size a reader actually gets; the on-page
footprint is held at what the 57% version occupied so the 8-page body budget
does not move.
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
# One line each: at 5.3pt in a 1.5in panel the two-line forms were still
# wider than their slot, and rotation needs a single baseline anyway.
FLAT = [LABELS[m].replace("\n", "") for m in MS]

# Eight bars in a 1.6in panel leave ~14pt per slot, so these are the sizes the
# slot can hold, not a preference.
TITLE, TICK, VAL, YLAB, NOTE = 7.0, 5.3, 5.0, 6.4, 5.0

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.31, 2.15))
# Explicit margins, not the defaults: savefig(bbox_inches="tight") crops to the
# artists, so a canvas with the default 10% side margins comes out ~380pt wide
# and then LaTeX scales it back up to \textwidth -- reintroducing exactly the
# authored-size/include-size mismatch this rewrite exists to remove. Filling the
# canvas means the crop is a no-op and 1pt here is 1pt on the page.
fig.subplots_adjust(left=0.062, right=0.997, top=0.865, bottom=0.40,
                    wspace=0.34)
for ax in (ax1, ax2, ax3):
    ax.tick_params(axis="y", labelsize=TICK + 0.4, length=2.2, pad=1.5)
    ax.tick_params(axis="x", length=0, pad=1.5)

# ---- (a) attention on CoT + measured noise floor --------------------------
att = [ATTN[m]["mass"]["cot"] for m in MS]
astd = [ATTN[m]["mass_std"]["cot"] for m in MS]
ax1.bar(xs, att, color=COL, edgecolor="black", lw=0.4, yerr=astd, capsize=1.5,
        error_kw={"lw": 0.5})
mid = float(np.mean(att))
half = NOISE["abs_diff_pp"] / 200.0          # +/- half of the observed run-to-run gap
ax1.axhspan(mid - half, mid + half, color="gray", alpha=0.22, lw=0)
# Anchored top-RIGHT. Two earlier placements both collided: bottom-right ran
# back under the value labels of the final two models, and bottom-left ran
# across the first bar. The band sits at ~0.35 and the tallest label reaches
# ~0.39, so everything above that is free -- which is the only region of this
# axes that is. The leader line carries the reference down to the band.
ax1.text(len(MS) - 0.45, 0.495,
         f"run-to-run noise: {NOISE['abs_diff_pp']:.2f} pp\n"
         f"({100*NOISE['noise_as_frac_of_spread']:.0f}% of the "
         f"{NOISE['cluster_spread_pp']:.2f} pp spread)",
         ha="right", va="top", fontsize=NOTE, color="0.25",
         linespacing=1.30)
# No leader line down to the band. At this size every vertical path from the
# note to the band crosses the row of value labels at ~0.37, and there is only
# one grey band in the panel for the note to be about.
for i, v in enumerate(att):
    ax1.text(i, v + astd[i] + 0.012, f"{v:.3f}", ha="center", fontsize=VAL)
ax1.set_xticks(xs)
ax1.set_xticklabels(FLAT, fontsize=TICK, rotation=45, ha="right",
                     rotation_mode="anchor")
ax1.set_ylabel(r"$\alpha(m,\mathrm{cot})$", fontsize=YLAB, labelpad=1.5)
ax1.set_ylim(0, 0.50)
# Two lines rather than one: at 7pt a single line of this claim is wider than
# the panel, and the claim is the reason the panel is here.
ax1.set_title("(a) Attention on CoT:\nspread is inside the noise floor",
              loc="left", fontsize=TITLE, style="italic", pad=2.5,
              linespacing=1.25)
ax1.set_axisbelow(True); ax1.yaxis.grid(True, ls=":", lw=0.4, alpha=0.5)
ours_bracket(ax1, N_OURS, fontsize=TICK, y=-0.42)

# ---- (b) raw mean magnitude-F -------------------------------------------
raw = [MODELS[m]["F_bar_mag"] for m in MS]
ax2.bar(xs, raw, color=COL, edgecolor="black", lw=0.4)
for i, v in enumerate(raw):
    ax2.text(i, v + 0.018, f"{v:.2f}", ha="center", fontsize=VAL)
ax2.set_xticks(xs)
ax2.set_xticklabels(FLAT, fontsize=TICK, rotation=45, ha="right",
                     rotation_mode="anchor")
ax2.set_ylabel(r"$\bar{\mathcal{F}}$ (raw, 7 families)", fontsize=YLAB,
               labelpad=1.5)
ax2.set_ylim(0, 1.05)
ax2.set_title(f"(b) Causal effect, raw:\n{max(raw)/min(raw):.1f}$\\times$ spread",
              loc="left", fontsize=TITLE, style="italic", pad=2.5,
              linespacing=1.25)
ax2.set_axisbelow(True); ax2.yaxis.grid(True, ls=":", lw=0.4, alpha=0.5)
ours_bracket(ax2, N_OURS, fontsize=TICK, y=-0.42)

# ---- (c) normalized by each model's own cross_task_swap ceiling ------------
nrm = [MODELS[m]["F_bar_norm_ceiling"] for m in MS]
ax3.bar(xs, nrm, color=COL, edgecolor="black", lw=0.4)
for i, v in enumerate(nrm):
    ax3.text(i, v + 0.018, f"{v:.2f}", ha="center", fontsize=VAL)
ax3.set_xticks(xs)
ax3.set_xticklabels(FLAT, fontsize=TICK, rotation=45, ha="right",
                     rotation_mode="anchor")
ax3.set_ylabel(r"$\bar{\mathcal{F}}/\mathcal{F}(\mathrm{cross\_task\_swap})$",
               fontsize=YLAB, labelpad=1.5)
ax3.set_ylim(0, 1.05)
ax3.set_title(f"(c) Ceiling-normalized:\n"
              f"only {max(nrm)/min(nrm):.1f}$\\times$ spread",
              loc="left", fontsize=TITLE, style="italic", pad=2.5,
              linespacing=1.25)
ax3.set_axisbelow(True); ax3.yaxis.grid(True, ls=":", lw=0.4, alpha=0.5)
ours_bracket(ax3, N_OURS, fontsize=TICK, y=-0.42)

save(fig, "fig4_dissociation")

print(f"[audit] attention on cot : {min(att):.3f}-{max(att):.3f} "
      f"({100*(max(att)-min(att)):.2f} pp spread)")
print(f"[audit] raw spread       : {max(raw)/min(raw):.2f}x")
print(f"[audit] normalized spread: {max(nrm)/min(nrm):.2f}x")
