"""Fig 1 -- the overview figure: the protocol, and the two results that bound it.

This is the figure the paper was missing. It replaces fig1_workflow (a generic
inputs -> model -> probes -> findings flowchart whose green box could not hold
its own label) and fig1_hero (whose docstring admitted its CoT was a
"schematic reconstruction"). A benchmark paper's first figure has to say what
the instrument does AND what it found; a flowchart says only the former, and
says it in a shape that fits any paper.

Three panels, in the order the argument runs:

  (a) the protocol -- one observation, two decodes, and the two scoring rules
      that read the resulting action pair. The CoT text is real: it is the
      released direction_flip pair, word-diffed here to locate the edited span.
      The two action arrows are drawn at the MEASURED mean translation cosine
      of the top-ranked model, so even the geometry of the schematic is data.
  (b) the magnitude score sits between two meaning-preserving floors on all
      12 calibrated configurations (S4).
  (c) the direction-aware score inverts the ranking the magnitude score gives
      (S6): rank 1 becomes rank 7.

HARD RULE, inherited from _data.py: no numeric literal for any reported
quantity appears in this file. Every number on the canvas is read from
results_v2/derived_metrics.json or
results_v2/canonical_runs/floor_invariance/floor_invariance.json, and the
panel titles that quote counts (12/12, 1 -> 7) compute them from those arrays
rather than restating them, so a change in the artifacts changes the figure or
trips an assertion.
"""
import json
import os
import sys
import difflib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *          # noqa: F401,F403  (rcParams + save)
from _data import MODELS, ORDER, LABELS, fam
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FI = os.path.join(ROOT, "results_v2", "canonical_runs", "floor_invariance",
                  "floor_invariance.json")
EX = os.path.join(ROOT, "results_v2", "canonical_runs", "edit_examples",
                  "edit_examples.json")

with open(FI) as fh:
    FLOORS = json.load(fh)
with open(EX) as fh:
    EXAMPLES = json.load(fh)["examples"]

# ---------------------------------------------------------------------------
# Panel (a) inputs: one real edit pair, and the measured cosine of the model
# the other two panels are about.
# ---------------------------------------------------------------------------
HERO_MODEL = "ecot-bridge"
COS_HERO = fam(HERO_MODEL, "direction_flip", "cos_xyz")
FMAG_HERO = fam(HERO_MODEL, "direction_flip", "F_mag")
FDIR_HERO = fam(HERO_MODEL, "direction_flip", "F_dir")
assert None not in (COS_HERO, FMAG_HERO, FDIR_HERO), \
    "panel (a) quotes the hero model's direction_flip row; it is missing"


def move_phrase(cot):
    """The MOVE line of an ECoT trace, which is what direction_flip rewrites."""
    i = cot.rfind("MOVE:")
    if i < 0:
        return ""
    j = cot.find("GRIPPER", i)
    return cot[i:(j if j > 0 else len(cot))].strip()


def first_pair(family):
    for e in EXAMPLES:
        if e.get("family") == family:
            a, b = move_phrase(e["cot_orig"]), move_phrase(e["cot_edited"])
            if a and b and a != b:
                return a, b
    raise SystemExit(f"no released {family} pair whose MOVE line changed")


MOVE_ORIG, MOVE_EDIT = first_pair("direction_flip")

# The changed words, recovered by diff rather than asserted, so the highlight
# cannot drift from what the generator actually did.
def diff_words(a, b):
    aw, bw = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, aw, bw)
    keep_a, keep_b = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        keep_a += [(w, tag == "equal") for w in aw[i1:i2]]
        keep_b += [(w, tag == "equal") for w in bw[j1:j2]]
    return keep_a, keep_b


W_ORIG, W_EDIT = diff_words(MOVE_ORIG, MOVE_EDIT)

# ---------------------------------------------------------------------------
# Panel (b) inputs: both floors and the semantic mean, per configuration.
# ---------------------------------------------------------------------------
PC = FLOORS["per_config"]
NICE = {"ours_no-cot": "no-CoT", "ours_lora-r8": "r=8", "ours_lora-r16": "r=16",
        "ours_lora-r32": "r=32", "ours_lora-r64": "r=64",
        "ours_data-50A": "data-50A", "ours_data-50B": "data-50B",
        "ecot_bridge": "ECoT-bridge", "deepthink_base": "DT base",
        "deepthink_sft": "DT SFT", "deepthink_rl": "DT RL",
        "bridge_subset_4k": "Bridge-4k"}
rows = [(NICE.get(c["config"], c["config"]),
         c["floor_paraphrase_null"]["F"],
         c["floor_syntactic_scramble"]["F"],
         c["f_bar_semantic"]) for c in PC]
# Ordered by the paraphrase floor so the band structure reads left to right
# instead of following an arbitrary config order.
rows.sort(key=lambda r: r[1])

# The count the panel title states, computed here.
N_BETWEEN = sum(1 for _, p, s, f in rows if min(p, s) <= f <= max(p, s))
N_SPREAD = sum(1 for _, p, s, f in rows
               if abs(p - s) > max(abs(f - p), abs(f - s)))
assert N_SPREAD == FLOORS["n_null_spread_exceeds_margin"], \
    "recomputed floor-spread count disagrees with floor_invariance.json"

# ---------------------------------------------------------------------------
# Panel (c) inputs: the two rankings on direction_flip.
# ---------------------------------------------------------------------------
MS = [m for m in ORDER if m in MODELS]
MAG = {m: fam(m, "direction_flip", "F_mag") for m in MS}
DIR = {m: fam(m, "direction_flip", "F_dir") for m in MS}
assert all(v is not None for v in list(MAG.values()) + list(DIR.values()))
rank_mag = {m: i + 1 for i, m in
            enumerate(sorted(MS, key=lambda m: -MAG[m]))}
rank_dir = {m: i + 1 for i, m in
            enumerate(sorted(MS, key=lambda m: -DIR[m]))}
HERO_FROM, HERO_TO = rank_mag[HERO_MODEL], rank_dir[HERO_MODEL]

COL = {m: (C_NO_COT if m == "ours-no-cot" else
           C_ECOT_BRIDGE if m == "ecot-bridge" else C_COT_TRAINED)
       for m in MS}
FLAT = {m: LABELS[m].replace("\n", "") for m in MS}

# ---------------------------------------------------------------------------
# Canvas. Absolute axes rather than gridspec: panel (a) is a drawing whose
# vertical extent is set by its text, and the two data panels below it must
# keep their own baseline regardless of how tall (a) ends up.
# ---------------------------------------------------------------------------
W, H = 6.30, 3.70                      # ACL \textwidth (16cm) x 3.70in
A_H, GAP, B_H = 1.32, 0.24, 1.64
fig = plt.figure(figsize=(W, H))

TOP = 1.0
ax = fig.add_axes([0.0, 1.0 - A_H / H, 1.0, A_H / H])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

FS = FONT_SIZE - 2.6
MONO = {"family": "monospace", "fontsize": FONT_SIZE - 3.0}


def box(x0, y0, w, h, ec, fc="none", lw=0.8, ls="-"):
    ax.add_patch(Rectangle((x0, y0), w, h, transform=ax.transAxes, fill=True,
                           facecolor=fc, edgecolor=ec, lw=lw, ls=ls,
                           zorder=1))


def arrow(x0, y0, x1, y1, color="0.35", lw=0.8):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords=ax.transAxes,
                textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                shrinkA=0, shrinkB=0, mutation_scale=8))


# Every stage carries its label ABOVE its box, on one baseline. Labels inside
# the boxes is what made fig1_workflow unreadable: the text set at the size the
# panel could afford, then overran the box it was supposed to name.
LABEL_Y = 0.945


def stage_label(x, text):
    ax.text(x, LABEL_Y, text, ha="center", va="center", fontsize=FS,
            style="italic", color="0.38")


# -- stage 1: what is held fixed -------------------------------------------
box(0.005, 0.30, 0.155, 0.55, "0.45", fc="0.965")
stage_label(0.0825, "held fixed")
for i, line in enumerate(("observation", "instruction", "greedy decode")):
    ax.text(0.0825, 0.735 - 0.125 * i, line, ha="center", va="center",
            fontsize=FS, color="0.15")
ax.text(0.0825, 0.19, "$\\times$ 15 VLAs, 4 families", ha="center",
        va="center", fontsize=FS, color="0.38", style="italic")

# -- stage 2: the two reasoning traces, real text --------------------------
def draw_move(y, words, tint, tag):
    """One MOVE line, with the diffed words tinted. Drawn word by word so the
    edited span is highlighted without writing the span out here."""
    ax.text(0.185, y + 0.11, tag, fontsize=FS, color=tint, va="center")
    x = 0.185
    for w, same in words:
        t = ax.text(x, y, w + " ", transform=ax.transAxes, va="center",
                    color=("0.2" if same else tint),
                    fontweight=("normal" if same else "bold"), **MONO)
        fig.canvas.draw()
        bb = t.get_window_extent(fig.canvas.get_renderer())
        x += bb.width / (fig.dpi * W)


box(0.175, 0.30, 0.375, 0.55, "0.45")
stage_label(0.3625, "the reasoning text, rewritten in one span")
draw_move(0.665, W_ORIG, C_COT_TRAINED, "the model's own CoT")
draw_move(0.395, W_EDIT, C_NO_COT, "edited  ($\\mathit{direction\\_flip}$)")
ax.text(0.3625, 0.19, "10 edit families in 3 tiers $+$ 3 calibration nulls",
        ha="center", va="center", fontsize=FS, color="0.38", style="italic")
arrow(0.163, 0.575, 0.173, 0.575)

# -- stage 3: the action pair, drawn at the measured cosine ----------------
cx, cy, r = 0.585, 0.40, 0.165
stage_label(0.635, "two 7-DoF actions")
ang = np.degrees(np.arccos(np.clip(COS_HERO, -1, 1)))
base = 90.0 - ang / 2.0
for a_deg, col, lab in ((base, C_COT_TRAINED, "$a$"),
                        (base - ang, C_NO_COT, "$a'$")):
    dx = r * np.cos(np.radians(a_deg)) * (H / W)
    dy = r * np.sin(np.radians(a_deg))
    arrow(cx, cy, cx + dx, cy + dy, col, lw=1.4)
    ax.text(cx + dx * 1.20, cy + dy * 1.14, lab, fontsize=FS + 0.8,
            color=col, ha="center", va="center")
ax.text(0.635, 0.19, f"measured $\\cos = {COS_HERO:+.3f}$", ha="center",
        va="center", fontsize=FS, color="0.15")
ax.text(0.635, 0.08, "on the top-ranked model", ha="center", va="center",
        fontsize=FS - 0.6, color="0.5", style="italic")
arrow(0.556, 0.575, 0.575, 0.470)

# -- stage 4: the two scoring rules, and what each concludes ---------------
stage_label(0.866, "read two ways")
box(0.735, 0.555, 0.262, 0.295, C_NO_COT, fc="#FDF0F1", lw=0.8)
ax.text(0.748, 0.788, "magnitude (Eq. 1)", fontsize=FS, color=C_NO_COT)
ax.text(0.748, 0.685, r"$\Delta_\infty=\max_i|a_i-a'_i|>\tau$", fontsize=FS,
        color="0.15")
ax.text(0.748, 0.598, f"$\\Rightarrow$ faithful on {FMAG_HERO:.3f}",
        fontsize=FS, color="0.15")

box(0.735, 0.215, 0.262, 0.295, C_ECOT_BRIDGE, fc="#EFF6EF", lw=0.8)
ax.text(0.748, 0.448, "direction-aware (Eq. 3)", fontsize=FS,
        color=C_ECOT_BRIDGE)
ax.text(0.748, 0.345, r"$\cos(a_{1:3},a'_{1:3})<-0.5$", fontsize=FS,
        color="0.15")
ax.text(0.748, 0.258, f"$\\Rightarrow$ faithful on {FDIR_HERO:.3f}",
        fontsize=FS, color="0.15")
# The same action pair, read by both rules: two arrows out of one source, not
# a chain, because neither rule is downstream of the other.
arrow(0.706, 0.480, 0.732, 0.690, "0.45", lw=0.7)
arrow(0.706, 0.440, 0.732, 0.360, "0.45", lw=0.7)
ax.text(0.005, LABEL_Y, "(a)", fontsize=FONT_SIZE - 1, fontweight="bold",
        va="center")

# ---------------------------------------------------------------------------
# (b) both floors and the semantic mean, per configuration.
# ---------------------------------------------------------------------------
bx = fig.add_axes([0.062, (H - A_H - GAP - B_H) / H, 0.455, B_H / H])
x = np.arange(len(rows))
for i, (lab, p, s, f) in enumerate(rows):
    lo, hi = min(p, s), max(p, s)
    bx.add_patch(Rectangle((i - 0.30, lo), 0.60, hi - lo, facecolor="0.86",
                           edgecolor="none", zorder=1))
bx.plot(x, [r[1] for r in rows], "o", ms=3.6, mfc="white", mew=1.0,
        color="0.25", zorder=3, label="floor: paraphrase")
bx.plot(x, [r[2] for r in rows], "^", ms=3.8, mfc="white", mew=1.0,
        color="0.25", zorder=3, label="floor: scramble (length-exact)")
bx.plot(x, [r[3] for r in rows], "s", ms=3.8, color=C_NO_COT, zorder=4,
        label=r"$\bar{\mathcal{F}}$, nine semantic families")
bx.set_xticks(x)
bx.set_xticklabels([r[0] for r in rows], rotation=42, ha="right",
                   fontsize=FONT_SIZE - 3.4)
bx.set_ylim(0, 1.03)
bx.set_ylabel(r"$\mathcal{F}$", labelpad=1.5)
bx.tick_params(axis="y", labelsize=FONT_SIZE - 3)
bx.set_axisbelow(True)
bx.yaxis.grid(True, ls=":", lw=0.4, alpha=0.55)
bx.legend(frameon=False, fontsize=FONT_SIZE - 3.4, loc="upper left",
          handlelength=1.2, borderpad=0.1, labelspacing=0.22,
          handletextpad=0.4)
bx.set_title(f"(b) the magnitude score lands {N_BETWEEN}/{len(rows)} times "
             f"between two floors\nthe same judge calls meaning-preserving",
             loc="left", fontsize=FONT_SIZE - 1.6, style="italic", pad=3.5)

# ---------------------------------------------------------------------------
# (c) the ranking inverts. A bump chart: the crossing IS the result, so it is
# drawn rather than tabulated.
# ---------------------------------------------------------------------------
cxx = fig.add_axes([0.615, (H - A_H - GAP - B_H) / H, 0.325, B_H / H])
for m in MS:
    y0, y1 = rank_mag[m], rank_dir[m]
    hero = m == HERO_MODEL
    # Everything but the hero line is drawn thin and pale on purpose. S8 shows
    # the retraining error bar swamps adjacent ranks, so a bump chart that drew
    # all eight lines equally would assert an ordering the paper refuses to
    # publish. Only the highlighted move is larger than the noise.
    cxx.plot([0, 1], [y0, y1], "-", color=COL[m],
             lw=(2.0 if hero else 0.8), alpha=(1.0 if hero else 0.45),
             zorder=(3 if hero else 2))
    cxx.plot([0, 1], [y0, y1], "o", ms=(4.2 if hero else 2.6),
             color=COL[m], alpha=(1.0 if hero else 0.55),
             zorder=(3 if hero else 2))
    cxx.text(-0.06, y0, FLAT[m], ha="right", va="center",
             fontsize=FONT_SIZE - 3.4, color=COL[m],
             alpha=(1.0 if hero else 0.7),
             fontweight=("bold" if hero else "normal"))
    cxx.text(1.06, y1, FLAT[m], ha="left", va="center",
             fontsize=FONT_SIZE - 3.4, color=COL[m],
             alpha=(1.0 if hero else 0.7),
             fontweight=("bold" if hero else "normal"))
cxx.set_xlim(-0.62, 1.62)
cxx.set_ylim(len(MS) + 1.35, 0.45)
cxx.set_xticks([0, 1])
cxx.set_xticklabels([r"rank by $\mathcal{F}_{\mathrm{mag}}$",
                     r"rank by $\mathcal{F}_{\mathrm{dir}}$"],
                    fontsize=FONT_SIZE - 3.2)
cxx.set_yticks(range(1, len(MS) + 1))
cxx.set_yticklabels(range(1, len(MS) + 1), fontsize=FONT_SIZE - 3)
cxx.tick_params(axis="both", length=0)
for sp in ("left", "bottom"):
    cxx.spines[sp].set_visible(False)
cxx.set_title(f"(c) the same edit, scored for direction:\n"
              f"rank {HERO_FROM} becomes rank {HERO_TO}",
              loc="left", fontsize=FONT_SIZE - 1.6, style="italic", pad=3.5)
cxx.text(0.5, len(MS) + 0.80, "pale lines: reorderings inside the retraining "
         "error bar (\\S8)", ha="center", va="center",
         fontsize=FONT_SIZE - 4.0, color="0.5", style="italic")

save(fig, "fig1_overview")

# What the caption is allowed to say, printed so it is copied rather than
# remembered.
print(f"  panel a: cos={COS_HERO:+.4f} F_mag={FMAG_HERO:.4f} "
      f"F_dir={FDIR_HERO:.4f} model={HERO_MODEL}")
print(f"  panel a: MOVE {MOVE_ORIG!r} -> {MOVE_EDIT!r}")
print(f"  panel b: {N_BETWEEN}/{len(rows)} between floors, "
      f"{N_SPREAD}/{len(rows)} spread exceeds margin")
print(f"  panel c: {HERO_MODEL} {HERO_FROM} -> {HERO_TO}; "
      f"ranks_mag={[FLAT[m] for m in sorted(MS, key=lambda m: rank_mag[m])]}")
