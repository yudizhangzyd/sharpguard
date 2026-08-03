"""Fig 5 — Bridge- vs LIBERO-trained CoT-VLA on the 3 shared families.

Grouped bar chart: 3 edit families × 2 models (Ours r=32 LIBERO fine-tune
vs public ECoT-bridge Bridge-V2). Highlights the 2× gap in causal effect
despite identical architecture and near-identical attention distributions.
Supports Finding 4: CoT-Faith is a training-domain effect.

Data source: the SAME cot_edit_report.json files used to populate Table 1
(lora-r32 row and ECoT-bridge row). This ensures Fig 5 exactly agrees with
Table 1 and with the numeric values quoted in Section 5 (F4). Paths:
  - lora-r32       : /tmp/cf_done/bcihypv3gu/cotfaith-edit/cot_edit_report.json
  - ECoT-bridge    : /tmp/cf_done/8rcgy9kukj/cotfaith-edit/cot_edit_report.json

AUTHORED AT THE INCLUDE WIDTH (see gen_fig4_dissociation.py): the previous
canvas was 5.2in and the figure was included at 0.72\\textwidth, so its 10pt
labels printed at 7.2pt while the multi-panel figures either side of it printed
at 6pt. The canvas below IS the include width, and the sizes below are the sizes
on the page.

Both series now carry their 3-sampling-seed std, which this file used to compute
and then discard -- drawing a 3-seed mean as a bare bar states a precision the
run does not have, and the gap being argued for here survives the bars easily.
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *
from _data import fam
import matplotlib.pyplot as plt
import numpy as np

# Both rows now come from results_v2/derived_metrics.json, which pins ONE
# canonical run per model (ECoT-bridge = 3-seed mean).  No hardcoded literals.
FAMS = [("subject swap",   "subject_swap"),
        ("direction flip", "direction_flip"),
        ("gripper flip",   "gripper_flip")]
ours   = [fam("ours-r32", k, "F_mag")    for _, k in FAMS]
bridge = [fam("ecot-bridge", k, "F_mag") for _, k in FAMS]
ours_std   = [fam("ours-r32", k, "F_mag_std")    for _, k in FAMS]
bridge_std = [fam("ecot-bridge", k, "F_mag_std") for _, k in FAMS]

xs = np.arange(len(FAMS))
w = 0.35

# Sizes on the page.
TICK, LAB, VAL, LEG = 6.4, 6.8, 6.0, 6.2

fig, ax = plt.subplots(figsize=(4.43, 2.30))
# Explicit margins: savefig crops to the artists, so default margins emit a page
# narrower than the canvas and LaTeX scales it back up.
fig.subplots_adjust(left=0.108, right=0.995, top=0.870, bottom=0.115)
ax.tick_params(axis="both", labelsize=TICK, length=2.2, pad=1.5)

bars1 = ax.bar(xs - w/2, ours,  w, yerr=ours_std, capsize=1.4,
                error_kw={"lw": 0.5}, label="Ours r=32 (LIBERO fine-tune)",
                color=C_COT_TRAINED, edgecolor="black", linewidth=0.4)
bars2 = ax.bar(xs + w/2, bridge, w, yerr=bridge_std, capsize=1.4,
                error_kw={"lw": 0.5}, label="ECoT-bridge (Bridge-V2)",
                color=C_ECOT_BRIDGE, edgecolor="black", linewidth=0.4)

for bars, vals, errs in ((bars1, ours, ours_std), (bars2, bridge, bridge_std)):
    for bar, v, e in zip(bars, vals, errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (e or 0) + 0.03,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=VAL)

ax.set_xticks(xs)
ax.set_xticklabels([f[0] for f in FAMS], fontsize=TICK + 0.4)
ax.set_ylabel("Faithful rate", fontsize=LAB, labelpad=1.5)
ax.set_ylim(0, 1.12)
# The legend goes ABOVE the axes. Inside at upper left it printed straight
# through the 0.93 value label of the first ECoT-bridge bar, which is the tallest
# thing in the left third of the panel.
_hl, _ll = ax.get_legend_handles_labels()
fig.legend(_hl, _ll, ncol=2, frameon=False, fontsize=LEG, loc="upper center",
           bbox_to_anchor=(0.55, 1.000), handlelength=1.3, handletextpad=0.5,
           columnspacing=1.8, borderpad=0.1)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

save(fig, "fig5_bridge_vs_libero")

print("[audit] ours   " + " ".join(f"{v:.3f}+-{e:.3f}" for v, e in zip(ours, ours_std)))
print("[audit] bridge " + " ".join(f"{v:.3f}+-{e:.3f}" for v, e in zip(bridge, bridge_std)))
print("[audit] ratio  " + " ".join(f"{b/o:.2f}x" for o, b in zip(ours, bridge)))
