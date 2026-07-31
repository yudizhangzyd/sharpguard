"""Fig 12 (NEW, R1 item 3) -- the leaderboard INVERTS under a direction-aware
version of its own flagship edit.

(a) magnitude-F vs directional-F on direction_flip, per model.
(b) signed cosine of the xyz translation before vs after direction_flip.
    A CoT-faithful model must move the OPPOSITE way (cos -> -1).
(c) F_diff = F(f) - F(paraphrase_null): the differential leaderboard for
    ours-no-cot. Every model now carries a measured paraphrase floor, but this
    is the only one whose floor (0.19) is far enough below its ceiling for the
    per-family differential to be legible; on the full-CoT variants the floors
    sit at 0.45-0.66 and the bars collapse toward zero.

All values from results_v2/derived_metrics.json.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *
from _data import MODELS, ORDER, LABELS, fam
import numpy as np
import matplotlib.pyplot as plt

MS = [m for m in ORDER if m in MODELS]
COL = {m: (C_NO_COT if m == "ours-no-cot" else
           C_ECOT_BRIDGE if m == "ecot-bridge" else C_COT_TRAINED) for m in MS}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.6, 3.5),
                                    gridspec_kw={"wspace": 0.30})
x = np.arange(len(MS))
w = 0.38

# ---- (a) magnitude-F vs directional-F on direction_flip --------------------
mag = [fam(m, "direction_flip", "F_mag") for m in MS]
dr = [fam(m, "direction_flip", "F_dir") for m in MS]
ax1.bar(x - w / 2, mag, w, color=[COL[m] for m in MS], edgecolor="black", lw=0.4,
        label=r"magnitude-$\mathcal{F}$ (Eq. 1)")
ax1.bar(x + w / 2, dr, w, color=[COL[m] for m in MS], edgecolor="black", lw=0.4,
        hatch="////", label=r"directional-$\mathcal{F}$ ($\cos<-0.5$)")
for i, (a, b) in enumerate(zip(mag, dr)):
    ax1.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=FONT_SIZE - 3)
    ax1.text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=FONT_SIZE - 3)
ax1.set_xticks(x); ax1.set_xticklabels([LABELS[m] for m in MS], fontsize=FONT_SIZE - 3)
ax1.set_ylabel(r"$\mathcal{F}(m,\mathrm{direction\_flip})$")
ax1.set_ylim(0, 1.14)
from matplotlib.patches import Patch
_h = [Patch(facecolor="0.6", edgecolor="black", lw=0.4,
            label=r"magnitude-$\mathcal{F}$ (Eq. 1)"),
      Patch(facecolor="0.6", edgecolor="black", lw=0.4, hatch="////",
            label=r"directional-$\mathcal{F}$ ($\cos<-0.5$)")]
ax1.legend(handles=_h, frameon=False, fontsize=FONT_SIZE - 3, loc="upper left")
ax1.set_title("(a) The flagship edit: magnitude vs direction",
              loc="left", fontsize=FONT_SIZE, style="italic")
ax1.set_axisbelow(True); ax1.yaxis.grid(True, ls=":", lw=0.4, alpha=0.5)

# ---- (b) signed cosine ----------------------------------------------------
cs = [fam(m, "direction_flip", "cos_xyz") for m in MS]
csf = [fam(m, "direction_flip", "cos_xyz_faithful_subset") for m in MS]
ax2.bar(x - w / 2, cs, w, color=[COL[m] for m in MS], edgecolor="black", lw=0.4,
        label="all samples")
ax2.bar(x + w / 2, csf, w, color=[COL[m] for m in MS], edgecolor="black", lw=0.4,
        hatch="////", label=r"samples Eq. 1 calls faithful")
ax2.axhline(0.0, color="black", lw=0.7)
ax2.axhline(-1.0, color=C_ECOT_BRIDGE, ls="--", lw=0.8)
ax2.text(len(MS) - 0.4, -0.94, "faithful\ntarget", fontsize=FONT_SIZE - 3,
         ha="right", color=C_ECOT_BRIDGE)
ax2.set_xticks(x); ax2.set_xticklabels([LABELS[m] for m in MS], fontsize=FONT_SIZE - 3)
ax2.set_ylabel(r"$\cos(a_{orig}[0{:}3],\ a_{edit}[0{:}3])$")
ax2.set_ylim(-1.1, 1.0)
_h2 = [Patch(facecolor="0.6", edgecolor="black", lw=0.4, label="all samples"),
       Patch(facecolor="0.6", edgecolor="black", lw=0.4, hatch="////",
             label=r"samples Eq. 1 calls faithful")]
ax2.legend(handles=_h2, frameon=False, fontsize=FONT_SIZE - 3, loc="upper left")
ax2.set_title("(b) Sign of the response after left$\\leftrightarrow$right is reversed",
              loc="left", fontsize=FONT_SIZE, style="italic")
ax2.set_axisbelow(True); ax2.yaxis.grid(True, ls=":", lw=0.4, alpha=0.5)

# ---- (c) F_diff for the model with a measured paraphrase floor -------------
HAVE = [m for m in MS if MODELS[m].get("paraphrase_null_floor") is not None]
FAMS = ["syntactic_scramble", "cross_task_swap", "direction_flip", "gripper_flip",
        "verb_swap", "negation", "subject_swap", "location_swap",
        "adversarial_plausible"]
SHORT = {"syntactic_scramble": "scram", "cross_task_swap": "cross",
         "direction_flip": "dir", "gripper_flip": "grip", "verb_swap": "verb",
         "negation": "neg", "subject_swap": "subj", "location_swap": "loc",
         "adversarial_plausible": "adv"}
m0 = HAVE[0]
xs = np.arange(len(FAMS))
vals = [fam(m0, f, "F_diff") for f in FAMS]
ax3.bar(xs, vals, 0.62,
        color=[C_ECOT_BRIDGE if (v or 0) > 0.05 else C_NO_COT for v in vals],
        edgecolor="black", lw=0.4)
ax3.axhline(0.0, color="black", lw=0.8)
for i, v in enumerate(vals):
    ax3.text(i, v + (0.015 if v >= 0 else -0.055), f"{v:+.2f}", ha="center",
             fontsize=FONT_SIZE - 3)
ax3.set_xticks(xs); ax3.set_xticklabels([SHORT[f] for f in FAMS],
                                        fontsize=FONT_SIZE - 3, rotation=40, ha="right")
floor = MODELS[m0]["paraphrase_null_floor"]
ax3.set_ylabel(r"$\mathcal{F}_{diff}=\mathcal{F}(f)-\mathcal{F}(para)$")
ax3.set_ylim(min(vals) - 0.14, max(max(vals), 0.05) + 0.10)
ax3.set_title(f"(c) {m0}: differential leaderboard\n"
              f"(paraphrase floor $=$ {floor:.2f})",
              loc="left", fontsize=FONT_SIZE, style="italic")
ax3.set_axisbelow(True); ax3.yaxis.grid(True, ls=":", lw=0.4, alpha=0.5)

save(fig, "fig12_directional_inversion")
