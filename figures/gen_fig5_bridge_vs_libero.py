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
bridge_std = [fam("ecot-bridge", k, "F_mag_std") for _, k in FAMS]

xs = np.arange(len(FAMS))
w = 0.35

fig, ax = plt.subplots(figsize=(5.2, 3.0))
bars1 = ax.bar(xs - w/2, ours,  w, label="Ours r=32 (LIBERO fine-tune)",
                color=C_COT_TRAINED, edgecolor="black", linewidth=0.4)
bars2 = ax.bar(xs + w/2, bridge, w, label="ECoT-bridge (Bridge-V2)",
                color=C_ECOT_BRIDGE, edgecolor="black", linewidth=0.4)

for bar, v in zip(bars1, ours):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{v:.2f}", ha="center", va="bottom", fontsize=FONT_SIZE-2)
for bar, v in zip(bars2, bridge):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{v:.2f}", ha="center", va="bottom", fontsize=FONT_SIZE-2)

ax.set_xticks(xs)
ax.set_xticklabels([f[0] for f in FAMS])
ax.set_ylabel("Faithful rate")
ax.set_ylim(0, 1.15)
ax.legend(loc="upper left", frameon=False, fontsize=FONT_SIZE-1)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

save(fig, "fig5_bridge_vs_libero")
