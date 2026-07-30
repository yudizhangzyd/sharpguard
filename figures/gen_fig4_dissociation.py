"""Fig 4 — attention vs causation DISSOCIATION plot.

Two-panel figure: (a) attn->cot per model (all cluster ~0.34), (b) avg
faithful rate across shared families per model (huge spread). Same
horizontal axis order, same model set → shows attention shape does
NOT predict causal effect. This is the paper's central story figure.
"""
import json, sys
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import matplotlib.pyplot as plt
import numpy as np


BASE = "/tmp/cf_full_sweep"
OLD = "/tmp/cf_sweep"


MODELS = [
    ("Ours\nno-CoT",   f"{BASE}/no-cot",         C_NO_COT),
    ("Ours\ndata-50 A", f"{BASE}/data-50A",      C_COT_TRAINED),
    ("Ours\ndata-50 B", f"{BASE}/data-50B",      C_COT_TRAINED),
    ("Ours\nr=8",      f"{BASE}/lora-r8",        C_COT_TRAINED),
    ("Ours\nr=16",     f"{BASE}/lora-r16",       C_COT_TRAINED),
    ("Ours\nr=32",     "/tmp/cf_done/bcihypv3gu", C_COT_TRAINED),
    ("Ours\nr=64",     f"{BASE}/lora-r64",       C_COT_TRAINED),
    ("ECoT-\nbridge",  "/tmp/cf_done/8rcgy9kukj/edit_only", C_ECOT_BRIDGE),  # attention shared with old
]

# ECoT-bridge attention: reuse old (bridge model didn't change)
# but pull edit from new 10-family run
def _load_edit(path):
    if path.endswith("edit_only"):
        return json.load(open(f"{path.replace('/edit_only','')}/cotfaith-edit/cot_edit_report.json"))
    return json.load(open(f"{path}/cotfaith-edit/cot_edit_report.json"))

def _load_rvis(path):
    if "cf_done/bcihypv3gu" in path:
        return json.load(open(f"{path}/cotfaith-rvis/rvis_cot_report.json"))
    if "cf_done/8rcgy9kukj" in path:
        # ECoT-bridge attention from cf_sweep
        return json.load(open(f"{OLD}/ecot-bridge/cotfaith-rvis/rvis_cot_report.json"))
    return json.load(open(f"{path}/cotfaith-rvis/rvis_cot_report.json"))

# 7 non-control families
SHARED_FAMS = ["direction_flip", "gripper_flip",
                "verb_swap", "negation",
                "subject_swap", "location_swap", "adversarial_plausible"]

attn_cot, faithful_avg = [], []
for name, path, _ in MODELS:
    a = _load_rvis(path).get("aggregate", {})
    attn_cot.append(a.get("action->cot", {}).get("mean", 0.0))
    e = _load_edit(path).get("aggregate", {})
    fr = [e[f]["faithful_rate"] for f in SHARED_FAMS if e.get(f, {}).get("n", 0) > 0]
    faithful_avg.append(np.mean(fr) if fr else 0.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.2),
                                  gridspec_kw={"wspace": 0.24})
xs = np.arange(len(MODELS))
colors = [m[2] for m in MODELS]

# Panel A: attention-to-CoT
ax1.bar(xs, attn_cot, color=colors, edgecolor="black", linewidth=0.4)
ax1.set_xticks(xs)
ax1.set_xticklabels([m[0] for m in MODELS], fontsize=FONT_SIZE-2)
ax1.set_ylabel("action$\\to$cot attention")
ax1.set_ylim(0, 0.5)
ax1.axhline(0.34, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
ax1.text(6.3, 0.36, "$\\sim$0.34", fontsize=FONT_SIZE-2, color="gray")
ax1.set_title("(a) Attention on CoT segment  —  all similar",
                loc="left", fontsize=FONT_SIZE, style="italic")
for i, v in enumerate(attn_cot):
    ax1.text(i, v + 0.01, f"{v:.2f}", ha="center",
              fontsize=FONT_SIZE-2)
ax1.set_axisbelow(True)
ax1.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

# Panel B: average faithful rate
ax2.bar(xs, faithful_avg, color=colors, edgecolor="black", linewidth=0.4)
ax2.set_xticks(xs)
ax2.set_xticklabels([m[0] for m in MODELS], fontsize=FONT_SIZE-2)
ax2.set_ylabel("avg faithful rate across 7 families")
ax2.set_ylim(0, 1.0)
ax2.set_title("(b) Causal effect  —  huge spread",
                loc="left", fontsize=FONT_SIZE, style="italic")
for i, v in enumerate(faithful_avg):
    ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=FONT_SIZE-2)
ax2.set_axisbelow(True)
ax2.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

save(fig, "fig4_dissociation")
