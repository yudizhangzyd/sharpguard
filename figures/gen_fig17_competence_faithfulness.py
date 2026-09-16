"""Fig 17 -- competence (action-space richness) x faithfulness (F_dir).

Appendix B.3's confound paragraph already states the finding in prose: the
six LoRA/data variants that clear the F_dir null all have a collapsed
action space, and the one rich-action-space row that clears (ECoT-bridge)
does so by a smaller margin attributed to genuine signal rather than the
collapse artifact. This plots that same fact as a 2D scatter instead of a
sentence. The pattern is not a clean split on this axis alone: the no-CoT
control also has a collapsed action space and does not clear (it is the
null control behaving correctly), and the three DeepThinkVLA checkpoints
have a rich action space and do not clear either -- so CoT-training status,
not action-space richness by itself, is doing real work too, and the
annotation says so rather than overstating a clean separation.

x = distinct_frac, the fraction of scored samples with a distinct original
action (higher = richer, less-collapsed action space; the same proxy
Appendix B.3 uses, not a rollout-conditioned competence measure -- this
benchmark reports none). y = F_dir on direction_flip. Both are read from
floor_convention_robustness.json, the same artifact Table 2 and Figure 3
are built from, so this figure cannot silently drift from either.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *          # noqa: F401,F403  (rcParams + save)
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results_v2", "canonical_runs",
                   "floor_convention_robustness",
                   "floor_convention_robustness.json")
with open(SRC) as fh:
    D = json.load(fh)
PC = D["per_config"]

ROWS = [
    ("ours_no-cot",   "no-CoT",      "nocot"),
    ("ours_lora-r8",  "r=8",         "ecot"),
    ("ours_lora-r16", "r=16",        "ecot"),
    ("ours_lora-r32", "r=32",        "ecot"),
    ("ours_lora-r64", "r=64",        "ecot"),
    ("ours_data-50A", "data-50A",    "ecot"),
    ("ours_data-50B", "data-50B",    "ecot"),
    ("ecot_bridge",   "ECoT-bridge", "ecot"),
    ("deepthink_base", "DT base",    "deepthink"),
    ("deepthink_sft",  "DT SFT",     "deepthink"),
    ("deepthink_rl",   "DT RL",      "deepthink"),
]
COLOR = {"ecot": C_COT_TRAINED, "deepthink": C_DEEPTHINK, "nocot": C_NO_COT}
LABEL = {"ecot": "ECoT / LoRA lineage", "deepthink": "DeepThinkVLA lineage",
        "nocot": "no-CoT (action-only) control"}

xs, ys, cs, labels = [], [], [], []
for key, label, group in ROWS:
    r = PC[key]
    x = r["action_space"]["distinct_frac"]
    y = r["fdir_direction_flip"]["F_dir"]
    xs.append(x); ys.append(y); cs.append(group); labels.append(label)

# Independently re-derive the pattern this figure draws, rather than trusting
# the prose: the six collapsed-action-space LoRA/data variants all clear
# their own F_dir null; the one rich-action-space CoT-trained row
# (ECoT-bridge) also clears, by a smaller margin; the collapsed-action-space
# no-CoT control and the three rich-action-space DeepThinkVLA rows do not.
# Distinct_frac alone does not separate clearing from non-clearing (no-CoT
# is collapsed but does not clear) -- CoT-training status also matters, and
# the figure's annotation says so rather than overstating a clean split.
fdn = json.load(open(os.path.join(ROOT, "results_v2", "canonical_runs",
                                  "fdir_null", "fdir_null.json")))
CLEARS = {r["config"] for r in fdn["per_config"] if r["clears_null"]}
KEY_OF = {lab: key for key, lab, _ in ROWS}
LORA_SIX = {"r=8", "r=16", "r=32", "r=64", "data-50A", "data-50B"}
checks = {
    "the six LoRA/data variants all clear": LORA_SIX <= {
        lab for lab in labels if KEY_OF[lab] in CLEARS},
    "ECoT-bridge clears": KEY_OF["ECoT-bridge"] in CLEARS,
    "no-CoT does not clear despite a collapsed action space":
        KEY_OF["no-CoT"] not in CLEARS,
    "all three DeepThinkVLA rows do not clear despite a rich action space":
        all(KEY_OF[f"DT {s}"] not in CLEARS for s in ("base", "SFT", "RL")),
}
failed = [k for k, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"[fig17] no longer holds against the artifact: {failed}")

fig, ax = plt.subplots(figsize=(3.4, 2.7))
OFFSETS = {  # (dx, dy, ha) hand-tuned only to dodge overlap; positions are data
    "r=8": (0.025, -0.028, "left"),
    "r=16": (0.025, 0.035, "left"),
    "data-50A": (0.025, -0.025, "left"),
    "DT base": (0.03, -0.028, "left"),
    "DT SFT": (0.03, 0.022, "left"),
    "DT RL": (-0.03, 0.05, "right"),
}
for x, y, g, lab in zip(xs, ys, cs, labels):
    ax.scatter([x], [y], s=32, color=COLOR[g], zorder=3,
              edgecolor="white", linewidth=0.5)
    dx, dy, ha = OFFSETS.get(lab, (0.025, 0.0, "left"))
    ax.annotate(lab, (x, y), xytext=(x + dx, y + dy), fontsize=FONT_SIZE - 4,
               ha=ha, va="center", color="0.25")

ax.set_xlabel("action-space richness (fraction of samples with a\n"
             r"distinct original action, $\times$ = 1$-$collapse)",
             fontsize=FONT_SIZE - 1.5)
ax.set_ylabel(r"$\mathcal{F}_{\text{dir}}$ (direction\_flip)",
             fontsize=FONT_SIZE - 1.5)
ax.set_xlim(-0.02, 1.05)
ax.set_ylim(-0.05, 1.0)
ax.set_axisbelow(True)
ax.grid(True, ls=":", lw=0.4, alpha=0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax.axvspan(-0.02, 0.40, color=C_COT_TRAINED, alpha=0.05, zorder=0)
ax.text(0.19, 0.94, "collapsed action space:\n6/7 CoT-trained clear\n(no-CoT does not)",
       fontsize=FONT_SIZE - 4.5, ha="center", va="top", color=C_COT_TRAINED,
       style="italic")
ax.text(0.72, 0.94, "rich action space:\nonly ECoT-bridge clears\n(DeepThinkVLA does not)",
       fontsize=FONT_SIZE - 4.5, ha="center", va="top", color=C_DEEPTHINK,
       style="italic")

handles = [mlines.Line2D([0], [0], marker="o", ls="", color=COLOR[k],
                         markersize=5, label=LABEL[k])
          for k in ("ecot", "deepthink", "nocot")]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.44, 0.42),
         frameon=False,
         fontsize=FONT_SIZE - 4, handletextpad=0.4)

save(fig, "fig17_competence_faithfulness")

print(f"[audit] source: {os.path.relpath(SRC, ROOT)}")
print(f"[audit] all pattern checks hold: {not failed}")
