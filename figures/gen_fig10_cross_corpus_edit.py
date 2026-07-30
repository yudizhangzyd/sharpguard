"""Fig 10: Cross-corpus causal-edit pilot (F5, preliminary N=1).
ECoT-bridge direction_flip / gripper_flip response on LIBERO (N=100)
and single pilot episodes on Bridge V2, Fractal, BC-Z. At N=1 per
non-LIBERO corpus these are point observations only; we make no
quantitative cross-corpus consistency claim in the submission.

Numbers read from:
  /tmp/cf_done/8rcgy9kukj/cotfaith-edit/cot_edit_report.json  (LIBERO N=100)
  /Users/yudizhang/Documents/sharpguard/results_v2/{bridge_v2,fractal,bcz}.json  (n<=1)
"""
import json, sys
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import numpy as np, matplotlib.pyplot as plt

# LIBERO ECoT-bridge (N=100 both families)
libero = json.load(open("/tmp/cf_done/8rcgy9kukj/cotfaith-edit/cot_edit_report.json"))["aggregate"]

def get_cross(name):
    d = json.load(open(f"/Users/yudizhang/Documents/sharpguard/results_v2/{name}.json"))["edit_aggregate"]
    def cell(fam):
        v = d.get(fam, {})
        return v.get("faithful_rate", None), v.get("n", 0)
    return cell("direction_flip"), cell("gripper_flip")

br_dir, br_grip = get_cross("bridge_v2")
fr_dir, fr_grip = get_cross("fractal")
bz_dir, bz_grip = get_cross("bcz")

DATA = [
    ("LIBERO",    libero["direction_flip"]["faithful_rate"], libero["direction_flip"]["n"],
                  libero["gripper_flip"]["faithful_rate"],   libero["gripper_flip"]["n"]),
    ("Bridge V2", br_dir[0], br_dir[1], br_grip[0], br_grip[1]),
    ("Fractal",   fr_dir[0], fr_dir[1], fr_grip[0], fr_grip[1]),
    ("BC-Z",      bz_dir[0], bz_dir[1], bz_grip[0], bz_grip[1]),
]

labels     = [f"{d[0]}\n(dir N={d[2]}; grip N={d[4]})" for d in DATA]
direction  = [d[1] if d[1] is not None else float("nan") for d in DATA]
gripper    = [d[3] if d[3] is not None else float("nan") for d in DATA]

x = np.arange(len(labels))
w = 0.35

fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.4))
b1 = ax.bar(x - w/2, direction, w, label="direction_flip",
              color=C_COT_TRAINED, edgecolor="black", linewidth=0.4)
b2 = ax.bar(x + w/2, gripper,   w, label="gripper_flip",
              color="#EE6677", edgecolor="black", linewidth=0.4)

for bar, v in zip(b1, direction):
    if v == v:  # not nan
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=FONT_SIZE-2)
for bar, v in zip(b2, gripper):
    if v == v:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=FONT_SIZE-2)
    else:
        ax.text(bar.get_x() + bar.get_width()/2, 0.05,
                "N=0", ha="center", va="bottom", fontsize=FONT_SIZE-2, style="italic")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=FONT_SIZE-1)
ax.set_ylabel("Faithful rate  ($\\Delta_\\infty > 0.05$)")
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right", frameon=False, fontsize=FONT_SIZE-1)
ax.set_title("ECoT-bridge causal-edit pilot (N=1 per non-LIBERO corpus)",
              fontsize=FONT_SIZE, loc="left", style="italic")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.4, alpha=0.5)

save(fig, "fig10_cross_corpus_edit")
