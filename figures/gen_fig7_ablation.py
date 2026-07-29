"""Fig 7: LoRA rank & data-fraction ablation.
Panel A: faithful rate vs LoRA rank (8, 16, 32, 64).
Panel B: data-fraction seed variance (50A/50B/full).
"""
import sys, json
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import numpy as np, matplotlib.pyplot as plt

BASE = "/tmp/cf_full_sweep"

def _fam(path, fam):
    a = json.load(open(path))["aggregate"].get(fam, {})
    return a.get("faithful_rate", None), a.get("n", 0)

FAMS = ["direction_flip", "gripper_flip", "verb_swap", "negation", "cross_task_swap"]

ranks   = [8, 16, 64]
rank_data = {fam: [] for fam in FAMS}
for r in ranks:
    p = f"{BASE}/lora-r{r}/cotfaith-edit/cot_edit_report.json"
    for f in FAMS:
        v, _ = _fam(p, f)
        rank_data[f].append(v if v is not None else np.nan)

seeds = ["data-50A", "data-50B"]
seed_data = {fam: [] for fam in FAMS}
for s in seeds:
    p = f"{BASE}/{s}/cotfaith-edit/cot_edit_report.json"
    for f in FAMS:
        v, _ = _fam(p, f)
        seed_data[f].append(v if v is not None else np.nan)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.0),
                                  gridspec_kw={"wspace": 0.28})

# Panel A: LoRA rank
palette = ["#4477AA","#EE6677","#228833","#AA3377","#CCBB44"]
for i, f in enumerate(FAMS):
    ax1.plot(ranks, rank_data[f], marker="o", label=f.replace("_"," "),
              color=palette[i], linewidth=1.5, markersize=6)
ax1.set_xticks(ranks)
ax1.set_xlabel("LoRA rank")
ax1.set_ylabel("Faithful rate")
ax1.set_ylim(0, 1.0)
ax1.set_title("(a) LoRA rank ablation", loc="left",
                fontsize=FONT_SIZE, style="italic")
ax1.legend(fontsize=FONT_SIZE-2, frameon=False, loc="lower right", ncol=1)
ax1.set_axisbelow(True); ax1.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

# Panel B: data-fraction 50A / 50B
xs = np.arange(len(FAMS))
w = 0.35
ax2.bar(xs - w/2, [seed_data[f][0] for f in FAMS], w, label="50% seed A",
         color="#4477AA", edgecolor="black", linewidth=0.4)
ax2.bar(xs + w/2, [seed_data[f][1] for f in FAMS], w, label="50% seed B",
         color="#EE6677", edgecolor="black", linewidth=0.4)
ax2.set_xticks(xs)
ax2.set_xticklabels([f.replace("_","\n") for f in FAMS], fontsize=FONT_SIZE-2)
ax2.set_ylabel("Faithful rate")
ax2.set_ylim(0, 1.0)
ax2.set_title("(b) Data-seed variance (50%)", loc="left",
                fontsize=FONT_SIZE, style="italic")
ax2.legend(fontsize=FONT_SIZE-2, frameon=False, loc="upper right")
ax2.set_axisbelow(True); ax2.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

save(fig, "fig7_ablation")
