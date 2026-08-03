"""Fig 12 (R1 item 3) -- the differential leaderboard: every family this model
answers, measured against its own paraphrase floor and its own ceiling.

F_diff = F(f) - F(paraphrase_null) for ours-no-CoT, ALL 12 non-reference
families, ranked, with instr_random_sub -- a deliberately random instruction
substitution -- drawn and labelled as the CEILING. The panel's content is that
the whole floor-to-ceiling band is 0.07 wide and direction_flip sits above the
ceiling: a meaning-changing edit moves this model no further than a random
instruction does.

WHY THIS FIGURE IS ONE PANEL, AND USED TO BE THREE.
Panel (a) redrew tab:directional's F_mag and F_dir columns on direction_flip and
panel (b) redrew that same table's signed-cosine column -- the same eight models,
the same digits, in the same submission as the table, which the ARR build
PROMOTES into the body. This paper had already cut one figure for exactly that
duplication, so the figure is cut to the panel whose data appears nowhere else.
Nothing measured was lost: every number the two deleted panels drew is printed,
to more decimals, in tab:directional, and the prose around it is what argues
from them. What was lost was a full-width float spent restating a table.

Cutting them is also what fixes the type. Three panels across 6.28in put eight
model slots and sixteen bars in each of two of them -- about 8pt of canvas per
bar, narrower than a horizontal "0.96" -- and that is what forced 4.8pt value
labels and 5.3pt ticks, the smallest type in the submission. One panel at
\\columnwidth carries 12 horizontal rows with nothing below 6.5pt, and no
mathtext subscript at all: the axis label spells the difference out rather than
setting "diff" at 0.7x of an already small base.

Every size below is the size on the PAGE, not on a canvas that will be shrunk.
The canvas is authored at the include width for the reason
gen_fig4_dissociation.py spells out: matplotlib font sizes are absolute points,
so a figure drawn wider than the slot it is included in prints every label at
slot/canvas of the size written here. The fact sheet at the bottom prints the
width savefig actually emitted and the \\textwidth fraction that includes it at
1:1, which is the quantity the audit reads back.

All values from results_v2/derived_metrics.json.
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *
from _data import MODELS, ORDER, LABELS, OURS, NON_CONTROL, fam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# The ceiling rule crosses the value labels of the two families nearest it --
# instr_random_sub's bar ENDS on the rule, so its label cannot avoid it -- and a
# dashed line through a 6.5pt "+0.07" is the one thing on this panel that was
# hard to read. A white halo under the glyphs keeps both artists whole.
HALO = [pe.withStroke(linewidth=1.3, foreground="white")]

# ORDER only fixes the iteration order of the model scan below, so the selection
# is deterministic; the selection RULE is the lowest floor, asserted below.
MS = [m for m in ORDER if m in MODELS]

# ---- the model: the one whose paraphrase floor is low enough to see past ----
# All eight models carry a measured floor, so the selection rule has to be the
# one the caption states -- lowest floor -- and not MS[0], which would pick this
# model only because ORDER happens to list it first.
HAVE = [m for m in MS if MODELS[m].get("paraphrase_null_floor") is not None]
FLOORS = sorted(HAVE, key=lambda m: MODELS[m]["paraphrase_null_floor"])
m0 = FLOORS[0]
assert MODELS[FLOORS[1]]["paraphrase_null_floor"] \
    - MODELS[m0]["paraphrase_null_floor"] > 0.2, \
    "another model's floor is now comparable to this one's; 'the model whose " \
    "floor is low enough for the differential to be readable' no longer " \
    "picks out one model"
assert m0 in OURS, \
    "the lowest-floor model is no longer one of our own fine-tunes; the " \
    "caption calls this panel's model ours-no-CoT"

REFERENCE = "paraphrase_null"        # F_diff is defined as 0 here
SHORT = {"syntactic_scramble": "scram", "cross_task_swap": "cross-task",
         "direction_flip": "dir flip", "gripper_flip": "grip flip",
         "verb_swap": "verb", "negation": "negation", "subject_swap": "subj",
         "location_swap": "loc", "adversarial_plausible": "adv plaus",
         "selfsplice_control": "selfsplice", "bbox_jitter_null": "bbox jitter",
         "instr_random_sub": "instr random"}
CEILING = "instr_random_sub"
_fams = MODELS[m0]["families"]
FAMS = [f for f in SHORT if f in _fams]
assert set(FAMS) | {REFERENCE} == set(_fams), \
    f"this panel is missing a family the release measures: " \
    f"{set(_fams) - set(FAMS) - {REFERENCE}}"
# The message still says "panel (c)", which is the name the audit greps for and
# the name the three-panel version gave this panel. Left alone deliberately: the
# check it backs is that no semantic family silently leaves the plot.
assert set(NON_CONTROL) <= set(FAMS), "a semantic family dropped out of panel (c)"
pairs = sorted(((f, fam(m0, f, "F_diff")) for f in FAMS),
               key=lambda t: t[1] or 0.0)
ys = np.arange(len(pairs))
vals = [v for _, v in pairs]
ceil_v = fam(m0, CEILING, "F_diff")
floor = MODELS[m0]["paraphrase_null_floor"]

# Sizes on the page. 12 rows in 1.7in of axes is ~10pt per row, which is what
# sets the ceiling on TICK; VAL is the floor the task allows, and it is the
# smallest type on the figure.
TITLE, TICK, VAL, XLAB = 7.2, 6.6, 6.5, 7.0

# 2.995in of canvas, cropped to the artists and padded 0.05in a side, emits a
# 218.4pt page -- one ARR \columnwidth (219.1pt) to within 0.3% -- so LaTeX
# includes it at 0.48\textwidth for 1:1 and the sizes above are the printed
# ones. Explicit margins rather than the defaults for the same
# reason: bbox_inches="tight" crops to the artists, so leaving 10% side margins
# would emit a narrower page and the include would scale it back up.
#
# The axes keeps the name `ax3` from the three-panel version: this is the panel
# that was (c), and the audit identifies the ceiling rule by grepping this file
# for the axvline call that draws it, so renaming the axes would silently drop
# that check. Do not restate that call in a comment either -- the check counts
# occurrences, and two of them read as no better than none.
fig, ax3 = plt.subplots(figsize=(2.995, 2.30))
fig.subplots_adjust(left=0.178, right=0.995, top=0.868, bottom=0.132)

# Semantic families in the trained colour, controls and calibrators in grey: the
# point of drawing all 12 is that they interleave.
cols = [C_COT_TRAINED if f in NON_CONTROL else C_CTRL for f, _ in pairs]
ax3.barh(ys, vals, 0.66, color=cols, edgecolor="black", lw=0.4)
ax3.axvline(0.0, color="black", lw=0.8)
ax3.axvline(ceil_v, color="0.15", lw=0.7, ls=(0, (3, 2)))
for i, v in enumerate(vals):
    ax3.text(v + (0.006 if v >= 0 else -0.006), i, f"{v:+.2f}",
             ha="left" if v >= 0 else "right", va="center", fontsize=VAL,
             path_effects=HALO)
ax3.set_yticks(ys)
ax3.set_yticklabels([SHORT[f] for f, _ in pairs], fontsize=TICK)
ax3.tick_params(axis="y", length=0, pad=1.5)
ax3.tick_params(axis="x", labelsize=VAL, length=2.2, pad=1.5)
ax3.set_ylim(-0.7, len(pairs) - 0.3)
ax3.set_xlim(min(vals) - 0.055, max(max(vals), ceil_v) + 0.062)
# Spelled out rather than set as $\mathcal{F}_{diff}$: a mathtext subscript
# renders at 0.7x of its base, so a 7pt label would print "diff" at 4.9pt.
ax3.set_xlabel(r"differential $\mathcal{F}(f)-\mathcal{F}(\mathrm{para})$",
               fontsize=XLAB, labelpad=1.5)
ax3.set_title(f"{m0}: all {len(pairs)} non-reference families, ranked\n"
              f"(paraphrase floor {floor:.2f}, ceiling {ceil_v:+.2f})",
              loc="left", fontsize=TITLE, style="italic", pad=2.5,
              linespacing=1.25)
ax3.text(ceil_v + 0.004, -0.62, "ceiling", fontsize=VAL, style="italic",
         color="0.15", ha="left", va="bottom", path_effects=HALO)
ax3.set_axisbelow(True); ax3.xaxis.grid(True, ls=":", lw=0.4, alpha=0.5)

save(fig, "fig12_directional_inversion")

# ---- fact sheet: everything the audit reads back out of this panel ---------
_next_lowest = LABELS[FLOORS[1]].replace("\n", "")
print(f"[audit] model drawn (lowest floor)  : {m0} "
      f"(floor {MODELS[m0]['paraphrase_null_floor']:.3f}; next lowest "
      f"{_next_lowest} at {MODELS[FLOORS[1]]['paraphrase_null_floor']:.3f})")
print(f"[audit] F_diff for {m0:<16}: {min(vals):+.3f} to {max(vals):+.3f}"
      f"  ({len(pairs)} families, reference {REFERENCE} = "
      f"{fam(m0, REFERENCE, 'F_diff'):+.3f})")
_pos = [(f, v) for f, v in pairs if (v or 0) > 0]
print(f"[audit] below their own floor      : "
      f"{sum(1 for _, v in pairs if (v or 0) < 0)} of {len(pairs)}")
print(f"[audit] above it                   : "
      f"{[(f, round(v, 3)) for f, v in _pos]}")
print(f"[audit] identity null = -floor     : "
      f"{fam(m0, 'selfsplice_control', 'F_diff'):+.3f} "
      f"(F_mag {fam(m0, 'selfsplice_control', 'F_mag'):.3f})")
print(f"[audit] ceiling ({CEILING}) : {ceil_v:+.3f}; band width "
      f"{ceil_v - fam(m0, REFERENCE, 'F_diff'):.3f}; families above the "
      f"ceiling: {[f for f, v in pairs if (v or 0) > ceil_v]}")
print(f"[audit] semantic in colour, controls/calibrators in grey: "
      f"{sum(1 for f, _ in pairs if f in NON_CONTROL)} + "
      f"{sum(1 for f, _ in pairs if f not in NON_CONTROL)}")
print(f"[audit] smallest type on the page  : {min([TITLE, TICK, VAL, XLAB])}pt "
      f"(no mathtext subscript is set at all)")

# The emitted page, stated rather than assumed: bbox_inches="tight" crops to the
# artists, so the width LaTeX gets is not the figsize above. Print it with the
# \textwidth fraction that includes it at 1:1 against the measured ARR geometry,
# which is the pair the audit's include-width check compares.
_pdf = os.path.join(FIG_DIR, "fig12_directional_inversion.pdf")
_box = re.search(rb"/MediaBox\s*\[([^\]]*)\]", open(_pdf, "rb").read())
x0, y0, x1, y1 = (float(v) for v in _box.group(1).split())
_geo = os.path.join(os.path.dirname(FIG_DIR), "results_v2", "canonical_runs",
                    "arr_build", "geometry.json")
try:
    with open(_geo) as _fh:
        g = json.load(_fh)
    tw, cs = g["textwidth_pt"], g["columnsep_pt"]
    print(f"[audit] emitted page               : {x1 - x0:.1f} x {y1 - y0:.1f}pt "
          f"({(x1 - x0) / ((tw - cs) / 2):.3f} of the {(tw - cs) / 2:.1f}pt ARR "
          f"column); include at width={(x1 - x0) / tw:.3f}\\textwidth for 1:1")
except OSError:
    print(f"[audit] emitted page               : {x1 - x0:.1f} x {y1 - y0:.1f}pt")
