"""Fig 4 -- attention/causation dissociation (F3), with the ceiling-normalized
panel demanded by R1 item 4.

(a) alpha(m, cot) per model, with the worst same-config RETRAINING difference
    drawn as a band so the cluster spread can be read against it. The spread is
    the larger of the two -- by 1.2x, far short of the 3x an ordering would
    need -- and the panel says so rather than claiming containment.
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
from _data import MODELS, ATTN, NH, ORDER, LABELS, OURS
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

# Sizes on the page. 6.5pt is the floor this paper's figures are held to (a
# reviewer prints the body at 100%); the earlier 5.0/5.3pt run text was below
# every venue's legibility guidance, and panel (a) in particular was READABLE
# ONLY through its 5.0pt value labels -- see the panel's own comment block.
TITLE, TICK, VAL, YLAB, NOTE = 7.2, 6.5, 6.5, 6.6, 6.5

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.50, 1.87))
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

# ---- (a) attention on CoT + the retraining band ---------------------------
# A STRIP PLOT, not bars, and on a 0.32--0.37 axis rather than 0-0.5.
#
# The eight values span 0.335--0.358. Drawn as bars from zero on a 0-0.5 axis
# they were eight rectangles differing by 4.6% of the panel height -- visually
# identical, so the only way to read the panel was the row of 5.0pt numerals
# above them, and the grey retraining band (1.95 pp) was 3.9% of the panel
# height and effectively invisible. That is the exact opposite of the panel's
# claim: a reader who cannot see the band cannot see that the spread does not
# clear it, and a reader who reads the numerals instead sees a RANKING, which is
# what noise_hierarchy says the data does not support.
#
# On a tight axis the band is ~30% of the panel height and the points sit inside
# it. The zero baseline is what bars need and is not information here -- nothing
# in this panel is a proportion of the axis -- so dropping it costs nothing and
# buys a 10x expansion of the only interval that matters. Markers carry the same
# per-model colours the other two panels use, and the sampling std stays as a
# bar so the ties are visibly ties at the level of the measurement too.
att = [ATTN[m]["mass"]["cot"] for m in MS]
astd = [ATTN[m]["mass_std"]["cot"] for m in MS]
mid = float(np.mean(att))
# The band is the WORST same-config retraining difference over the seven
# replicate pairs (1.95 pp), not the single r=32 pair the submission used
# (1.45 pp): what a reader compares between two bars is two training runs, so
# that is the noise the comparison has to clear. The pair estimate also let this
# panel say the 2.30 pp spread sat "inside" a 1.45 pp floor, which is backwards
# -- the spread is the LARGER of the two, and the honest reading is that it
# misses the 3x an ordering needs, not that it vanishes into the band.
half = NH["training_run_cot_diff_pp"] / 200.0
ax1.axhspan(mid - half, mid + half, color="gray", alpha=0.30, lw=0, zorder=0)
ax1.axhline(mid - half, color="0.55", lw=0.5, zorder=0.5)
ax1.axhline(mid + half, color="0.55", lw=0.5, zorder=0.5)
ax1.errorbar(xs, att, yerr=astd, fmt="none", ecolor="0.30", elinewidth=0.6,
             capsize=1.8, capthick=0.6, zorder=2)
ax1.scatter(xs, att, s=20, c=COL, edgecolors="black", linewidths=0.45,
            zorder=3, clip_on=False)
# The axis is set from the data plus the band, not typed: whichever of the two
# reaches further decides the limit, so a re-derivation that moves either one
# cannot silently push a marker off the panel.
_lo = min(min(a - s for a, s in zip(att, astd)), mid - half)
_hi = max(max(a + s for a, s in zip(att, astd)), mid + half)
# The note lives in the strip below the lowest error bar, which is the only
# clear region once the band spans the full width. It is two lines of NOTE, so
# the axis is opened by that much underneath rather than by a round number.
ax1.set_ylim(_lo - 0.0135, _hi + 0.0015)
# No per-point numerals. They were the panel's only readable content when the
# bars were indistinguishable, and printing three decimals for eight values
# whose differences are inside the retraining band invites exactly the ordering
# the caption says the data cannot support. The axis carries the scale now.
ax1.text(-0.45, _lo - 0.0125,
         f"grey band: same-config retraining, "
         f"{NH['training_run_cot_diff_pp']:.2f} pp\n"
         f"spread across variants: "
         f"{NH['cross_variant_spread_pp']:.2f} pp "
         f"= {NH['spread_over_training_run_cot']:.1f}× it",
         ha="left", va="bottom", fontsize=NOTE, color="0.25",
         linespacing=1.25)
ax1.set_xticks(xs)
ax1.set_xticklabels(FLAT, fontsize=TICK, rotation=45, ha="right",
                     rotation_mode="anchor")
ax1.set_ylabel(r"$\alpha(m,\mathrm{cot})$", fontsize=YLAB, labelpad=1.5)
ax1.set_xlim(-0.6, len(MS) - 0.4)
# Two lines rather than one: at 7pt a single line of this claim is wider than
# the panel, and the claim is the reason the panel is here.
ax1.set_title("(a) Attention on CoT:\nspread ≈ retraining noise",
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
