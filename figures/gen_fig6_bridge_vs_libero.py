"""Fig 6 — Bridge- vs LIBERO-trained CoT-VLA on the 3 shared families.

Grouped bar chart: 3 edit families × 2 models (Ours r=32 LIBERO fine-tune
vs public ECoT-bridge Bridge-V2). Highlights the 2× gap in causal effect
despite identical architecture and near-identical attention distributions.
Supports Finding 4: CoT-Faith is a training-domain effect.
"""
import json, sys
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import matplotlib.pyplot as plt
import numpy as np

OLD = "/tmp/cf_sweep"

# Load
def _faithful(path, fam):
    a = json.load(open(path))["aggregate"].get(fam, {})
    return a.get("faithful_rate", 0.0)

FAMS = [("subject swap",    "subject_swap"),
        ("direction flip",  "direction_flip"),
        ("gripper flip",    "gripper_flip")]
ours  = [_faithful(f"{OLD}/ours-train/cotfaith-edit/cot_edit_report.json", k) for _, k in FAMS]
bridge = [_faithful(f"{OLD}/ecot-bridge/cotfaith-edit/cot_edit_report.json", k) for _, k in FAMS]

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

save(fig, "fig6_bridge_vs_libero")
