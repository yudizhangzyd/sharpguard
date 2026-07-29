"""Fig 2 — action-token attention distribution across 11 models.

Grouped bar chart: 11 rows on x-axis, 4 buckets stacked/grouped.
Highlights the clean separation between CoT-VLA cluster (visual 0.29,
instr 0.30, cot 0.34) and non-CoT baseline cluster (visual 0.52,
instr 0.42, cot 0.00) — the anchor visual for Finding 1.
"""
import json, sys
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import matplotlib.pyplot as plt
import numpy as np


BASE = "/tmp/cf_full_sweep"
OLD = "/tmp/cf_sweep"
SEED = "/tmp/cf_sweep2"


def _agg(paths):
    aggs = []
    for p in paths:
        try:
            d = json.load(open(p))
            aggs.append(d.get("aggregate", {}))
        except FileNotFoundError:
            pass
    def stat(k):
        vs = [a.get(k, {}).get("mean") for a in aggs]
        vs = [x for x in vs if x is not None]
        if not vs: return None, None
        return float(np.mean(vs)), float(np.std(vs))
    return {b: stat(f"action->{b if b != 'action_prev' else 'action_prev'}")
             for b in BUCKETS}, len(aggs)


rows = [
    ("Ours r=32",       [f"{OLD}/ours-train/cotfaith-rvis/rvis_cot_report.json"]),
    ("Ours r=8",        [f"{BASE}/lora-r8/cotfaith-rvis/rvis_cot_report.json"]),
    ("Ours r=16",       [f"{BASE}/lora-r16/cotfaith-rvis/rvis_cot_report.json"]),
    ("Ours r=64",       [f"{BASE}/lora-r64/cotfaith-rvis/rvis_cot_report.json"]),
    ("Ours no-CoT",     [f"{BASE}/no-cot/cotfaith-rvis/rvis_cot_report.json"]),
    ("Ours data-50 A",  [f"{BASE}/data-50A/cotfaith-rvis/rvis_cot_report.json"]),
    ("Ours data-50 B",  [f"{BASE}/data-50B/cotfaith-rvis/rvis_cot_report.json"]),
    ("ECoT-bridge",     [f"{OLD}/ecot-bridge/cotfaith-rvis/rvis_cot_report.json",
                          f"{SEED}/ecot-bridge-s1/cotfaith-rvis/rvis_cot_report.json",
                          f"{SEED}/ecot-bridge-s2/cotfaith-rvis/rvis_cot_report.json"]),
    ("OpenVLA-spatial", [f"{OLD}/baseline-spatial/cotfaith-rvis-baseline/rvis_baseline_report.json",
                          f"{SEED}/baseline-spatial-s1/cotfaith-rvis-baseline/rvis_baseline_report.json",
                          f"{SEED}/baseline-spatial-s2/cotfaith-rvis-baseline/rvis_baseline_report.json"]),
    ("OpenVLA-object",  [f"{OLD}/baseline-object/cotfaith-rvis-baseline/rvis_baseline_report.json",
                          f"{SEED}/baseline-object-s1/cotfaith-rvis-baseline/rvis_baseline_report.json",
                          f"{SEED}/baseline-object-s2/cotfaith-rvis-baseline/rvis_baseline_report.json"]),
    ("OpenVLA-goal",    [f"{OLD}/baseline-goal/cotfaith-rvis-baseline/rvis_baseline_report.json",
                          f"{SEED}/baseline-goal-s1/cotfaith-rvis-baseline/rvis_baseline_report.json",
                          f"{SEED}/baseline-goal-s2/cotfaith-rvis-baseline/rvis_baseline_report.json"]),
    ("OpenVLA-10",      [f"{OLD}/baseline-10/cotfaith-rvis-baseline/rvis_baseline_report.json",
                          f"{SEED}/baseline-10-s1/cotfaith-rvis-baseline/rvis_baseline_report.json",
                          f"{SEED}/baseline-10-s2/cotfaith-rvis-baseline/rvis_baseline_report.json"]),
]

data = []
for name, paths in rows:
    stats, n = _agg(paths)
    means = [stats[b][0] or 0.0 for b in BUCKETS]
    stds  = [stats[b][1] or 0.0 for b in BUCKETS]
    data.append((name, means, stds, n))

# Draw grouped bar
n_models = len(data)
x = np.arange(n_models)
width = 0.20

fig, ax = plt.subplots(figsize=(9, 3.4))
for i, b in enumerate(BUCKETS):
    means = [d[1][i] for d in data]
    stds  = [d[2][i] for d in data]
    ax.bar(x + (i - 1.5) * width, means, width,
            yerr=stds, capsize=1.5,
            label=b.replace("_", "\\_"), color=BUCKET_COLORS[b],
            edgecolor="black", linewidth=0.4)

# Divider between CoT and non-CoT
ax.axvline(x=7.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
ax.text(3.5, 0.63, "CoT-VLAs", ha="center", fontsize=FONT_SIZE-1,
         style="italic", color="dimgray")
ax.text(9.5, 0.63, "Non-CoT baselines", ha="center", fontsize=FONT_SIZE-1,
         style="italic", color="dimgray")

ax.set_xticks(x)
ax.set_xticklabels([d[0] for d in data], rotation=35, ha="right")
ax.set_ylabel("Action-token attention mass")
ax.set_ylim(0, 0.68)
ax.legend(loc="upper right", ncol=4, frameon=False,
            bbox_to_anchor=(1.0, 1.14))
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

save(fig, "fig2_attention_distribution")
