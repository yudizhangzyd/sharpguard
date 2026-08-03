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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.28, 2.30))
# AUTHORED AT THE INCLUDE WIDTH (see gen_fig4_dissociation.py): at 9.2in wide
# and included at 0.95\textwidth this printed its 8pt tick labels at 6.2pt and
# its legends at the same. Explicit margins because savefig crops to the
# artists, so default margins emit a narrower page that LaTeX scales back up.
fig.subplots_adjust(left=0.078, right=0.996, top=0.885, bottom=0.245,
                    wspace=0.26)
# Sizes on the page. Panel (b) puts 5 two-line family labels in 2.9in.
TITLE, TICK, LAB, LEG = 7.0, 5.6, 6.4, 5.4
for ax in (ax1, ax2):
    ax.tick_params(axis="both", labelsize=TICK, length=2.2, pad=1.5)

# Panel A: LoRA rank
palette = ["#4477AA","#EE6677","#228833","#AA3377","#CCBB44"]
for i, f in enumerate(FAMS):
    ax1.plot(ranks, rank_data[f], marker="o", label=f.replace("_"," "),
              color=palette[i], linewidth=1.1, markersize=3.0)
ax1.set_xticks(ranks)
ax1.set_xlabel("LoRA rank", fontsize=LAB, labelpad=1.5)
ax1.set_ylabel("Faithful rate", fontsize=LAB, labelpad=1.5)
ax1.set_ylim(0, 1.0)
ax1.set_title("(a) LoRA rank ablation", loc="left",
                fontsize=TITLE, style="italic", pad=2.5)
ax1.legend(fontsize=LEG, frameon=False, loc="center left", ncol=1,
           handlelength=1.4, handletextpad=0.5, labelspacing=0.22,
           borderpad=0.1, borderaxespad=0.2)
ax1.set_axisbelow(True); ax1.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

# Panel B: data-fraction 50A / 50B
xs = np.arange(len(FAMS))
w = 0.35
ax2.bar(xs - w/2, [seed_data[f][0] for f in FAMS], w, label="50% seed A",
         color="#4477AA", edgecolor="black", linewidth=0.4)
ax2.bar(xs + w/2, [seed_data[f][1] for f in FAMS], w, label="50% seed B",
         color="#EE6677", edgecolor="black", linewidth=0.4)
ax2.set_xticks(xs)
ax2.set_xticklabels([f.replace("_","\n") for f in FAMS], fontsize=TICK)
ax2.set_ylabel("Faithful rate", fontsize=LAB, labelpad=1.5)
ax2.set_ylim(0, 1.0)
ax2.set_title("(b) Data-seed variance (50%)", loc="left",
                fontsize=TITLE, style="italic", pad=2.5)
ax2.legend(fontsize=LEG, frameon=False, loc="upper right", handlelength=1.1,
           handletextpad=0.5, labelspacing=0.22, borderpad=0.1,
           borderaxespad=0.2)
ax2.set_axisbelow(True); ax2.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

save(fig, "fig8_ablation")

print(f"[audit] ranks {ranks}, families {len(FAMS)}, seeds {seeds}")
