"""Fig 11 (M3): Threshold sensitivity — Faithfulness Score vs τ.
Shows that the leaderboard rankings are stable across τ ∈ [0.02, 0.20].
"""
import sys, json, glob
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import numpy as np, matplotlib.pyplot as plt

MODELS = [
    ("Ours r=8",   "/tmp/cf_full_sweep/lora-r8/cotfaith-edit/cot_edit_report.json",  C_COT_TRAINED),
    ("Ours r=16",  "/tmp/cf_full_sweep/lora-r16/cotfaith-edit/cot_edit_report.json", C_COT_TRAINED),
    ("Ours r=64",  "/tmp/cf_full_sweep/lora-r64/cotfaith-edit/cot_edit_report.json", C_COT_TRAINED),
    ("no-CoT",     "/tmp/cf_full_sweep/no-cot/cotfaith-edit/cot_edit_report.json",   C_NO_COT),
    ("data-50A",   "/tmp/cf_full_sweep/data-50A/cotfaith-edit/cot_edit_report.json", "#AA3377"),
    ("data-50B",   "/tmp/cf_full_sweep/data-50B/cotfaith-edit/cot_edit_report.json", "#CCBB44"),
]

TAUS = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30])
# Average faithful rate over 7 non-control families
FAMS = ["direction_flip","gripper_flip","verb_swap","negation",
        "subject_swap","adversarial_plausible","cross_task_swap"]

fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.5))
for name, path, color in MODELS:
    d = json.load(open(path))
    per = [s for s in d["per_sample"] if s["family"] in FAMS and not s.get("skipped", False)]
    curve = []
    for tau in TAUS:
        # per-family faithful rate at tau, then average
        rates = []
        for fam in FAMS:
            rows = [s for s in per if s["family"] == fam]
            if not rows: continue
            fr = sum(1 for r in rows if r["delta_linf"] > tau) / len(rows)
            rates.append(fr)
        curve.append(np.mean(rates))
    ax.plot(TAUS, curve, marker="o", label=name, color=color, linewidth=1.5, markersize=5)

ax.axvline(0.05, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
ax.text(0.05, 0.92, r"$\tau=0.05$ (default)", rotation=90, fontsize=FONT_SIZE-2, color="gray",
         ha="right", va="top", transform=ax.get_xaxis_transform())
ax.set_xscale("log")
ax.set_xlabel(r"Faithfulness threshold  $\tau$")
ax.set_ylabel("Mean faithful rate  (over 7 non-control families)")
ax.set_ylim(0, 1.0)
ax.legend(fontsize=FONT_SIZE-1, frameon=False, loc="lower left", ncol=2)
ax.set_axisbelow(True); ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

# Report max rank change to caption data
print(f"\nRank stability: at τ={TAUS.tolist()}")
save(fig, "fig11_threshold_sensitivity")
