"""Fig 5 — prompt-format ablation: which parts of CoT are load-bearing?

Bar chart of faithful rate under 5 truncation variants (vs full CoT).
Shows that removing ANY portion of the CoT causes ~100% action change,
supporting the "model is strongly conditioned on the full 9-tag CoT
structure" finding.

AUTHORED AT THE INCLUDE WIDTH (see gen_fig4_dissociation.py): the previous canvas
was 6.2in and the figure was included at 0.85\\textwidth, so its 10pt labels
printed at 6.5pt. The canvas below IS the include width.
"""
import json, sys
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import matplotlib.pyplot as plt
import numpy as np

BASE = "/tmp/cf_full_sweep"

d = json.load(open(f"{BASE}/prompt/cotfaith-prompt/cot_prompt_report.json"))
agg = d["aggregate"]

VARIANTS = [
    ("full CoT",         "full",              C_CTRL),
    ("task only",        "task_only",         C_COT_TRAINED),
    ("plan only",        "plan_only",         C_COT_TRAINED),
    ("task+plan+subtask", "task_plan_subtask", C_COT_TRAINED),
    ("shuffled",         "shuffled",          "#AA3377"),
    ("empty",            "empty",             C_NO_COT),
]

xs = np.arange(len(VARIANTS))
fr = [agg[k]["faithful_rate"] for _, k, _ in VARIANTS]
med = [agg[k]["delta_linf_median"] for _, k, _ in VARIANTS]

# Sizes on the page.
TICK, LAB, VAL = 6.2, 6.6, 6.0

fig, ax = plt.subplots(figsize=(4.93, 2.30))
# Explicit margins: savefig crops to the artists, so default margins emit a page
# narrower than the canvas and LaTeX scales it back up.
fig.subplots_adjust(left=0.118, right=0.995, top=0.975, bottom=0.215)
ax.tick_params(axis="both", labelsize=TICK, length=2.2, pad=1.5)
colors = [v[2] for v in VARIANTS]
bars = ax.bar(xs, fr, color=colors, edgecolor="black", linewidth=0.4)
for i, (bar, v, m) in enumerate(zip(bars, fr, med)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{v:.2f}", ha="center", va="bottom",
             fontsize=VAL)
ax.set_xticks(xs)
ax.set_xticklabels([v[0] for v in VARIANTS], rotation=15, ha="right",
                     fontsize=TICK, rotation_mode="anchor")
ax.set_ylabel("Faithful rate  ($\\Delta_\\infty > 0.05$)", fontsize=LAB,
              labelpad=1.5)
ax.set_ylim(0, 1.15)
ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

save(fig, "fig9_prompt_ablation")

print("[audit] " + "  ".join(f"{n}={v:.2f}" for (n, _, _), v in zip(VARIANTS, fr)))
