"""Fig 7: Cross-corpus attention profile (F5, preliminary N=1 pilot).
ECoT-bridge on LIBERO (N=100, mean +/- std) and one episode each on
Bridge V2, RT-1/Fractal, BC-Z (self-decoded CoT). At N=1 per non-LIBERO
corpus these are point observations only; error bars are undefined and
we show a marker instead of a bar with std. Full N=30 sweep in progress.

Numbers read directly from:
  /Users/yudizhang/Documents/sharpguard/results_v2/{bridge_v2,fractal,bcz}.json  (n=1)
  /Users/yudizhang/Documents/sharpguard/results_v2/SUMMARY.json  (LIBERO n=100)
"""
import json, os, sys
sys.path.insert(0, "/Users/yudizhang/Documents/sharpguard/figures")
from paper_plot_style import *
import numpy as np, matplotlib.pyplot as plt

RESULTS = "/Users/yudizhang/Documents/sharpguard/results_v2"

# LIBERO reference from ECoT-bridge on LIBERO (N=100)
libero = json.load(open("/tmp/cf_sweep/ecot-bridge/cotfaith-rvis/rvis_cot_report.json"))
lib_ag = libero["aggregate"]

def get(d, k):
    x = d[k]
    return x["mean"], x["std"], x["n"]

lib_cot = get(lib_ag, "action->cot")
lib_vis = get(lib_ag, "action->visual")
lib_ins = get(lib_ag, "action->instr")
lib_prv = get(lib_ag, "action->action_prev")

# Non-LIBERO N=1 pilot points
def loadone(name):
    d = json.load(open(f"{RESULTS}/{name}.json"))["attention_aggregate"]
    return {
        "cot":   (d["action->cot"]["mean"],       d["action->cot"]["std"],       d["action->cot"]["n"]),
        "vis":   (d["action->visual"]["mean"],    d["action->visual"]["std"],    d["action->visual"]["n"]),
        "ins":   (d["action->instr"]["mean"],     d["action->instr"]["std"],     d["action->instr"]["n"]),
        "prv":   (d["action->action_prev"]["mean"], d["action->action_prev"]["std"], d["action->action_prev"]["n"]),
    }
bridge  = loadone("bridge_v2")
fractal = loadone("fractal")
bcz     = loadone("bcz")

DATA = [
    ("LIBERO",     lib_cot, lib_vis, lib_ins, lib_prv),
    ("Bridge V2",  bridge["cot"], bridge["vis"], bridge["ins"], bridge["prv"]),
    ("Fractal",    fractal["cot"], fractal["vis"], fractal["ins"], fractal["prv"]),
    ("BC-Z",       bcz["cot"], bcz["vis"], bcz["ins"], bcz["prv"]),
]

labels    = [d[0] for d in DATA]
cot_m     = [d[1][0] for d in DATA]
vis_m     = [d[2][0] for d in DATA]
instr_m   = [d[3][0] for d in DATA]
prev_m    = [d[4][0] for d in DATA]
cot_s     = [d[1][1] if d[1][2] > 1 else 0 for d in DATA]
vis_s     = [d[2][1] if d[2][2] > 1 else 0 for d in DATA]
instr_s   = [d[3][1] if d[3][2] > 1 else 0 for d in DATA]
prev_s    = [d[4][1] if d[4][2] > 1 else 0 for d in DATA]
n_sizes   = [d[1][2] for d in DATA]

x = np.arange(len(labels))
w = 0.20

fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.5))
ax.bar(x - 1.5*w, vis_m,   w, yerr=vis_s,   label="visual",      color=BUCKET_COLORS["visual"],      capsize=2)
ax.bar(x - 0.5*w, instr_m, w, yerr=instr_s, label="instruction", color=BUCKET_COLORS["instr"],       capsize=2)
ax.bar(x + 0.5*w, cot_m,   w, yerr=cot_s,   label="CoT",         color=BUCKET_COLORS["cot"],         capsize=2)
ax.bar(x + 1.5*w, prev_m,  w, yerr=prev_s,  label="action-prev", color=BUCKET_COLORS["action_prev"], capsize=2)

# Overlay open-circle markers on non-LIBERO bars to signal "single point observation"
for xi, n in enumerate(n_sizes):
    if n == 1:
        for dx, y in [(-1.5*w, vis_m[xi]), (-0.5*w, instr_m[xi]),
                      (0.5*w, cot_m[xi]), (1.5*w, prev_m[xi])]:
            ax.plot(xi + dx, y, marker="o", markerfacecolor="white",
                    markeredgecolor="black", markersize=4, zorder=5)

ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(N={n})" for l, n in zip(labels, n_sizes)],
                    fontsize=FONT_SIZE - 1)
ax.set_ylabel(r"Attention mass $\alpha(m,B)$")
ax.set_ylim(0, 0.42)
ax.legend(ncol=2, frameon=False, loc="upper right",
          bbox_to_anchor=(1.0, 1.02), fontsize=FONT_SIZE-1)
ax.set_title("ECoT-bridge attention across 4 corpora "
             "(LIBERO mean$\\pm$std; N=1 pilot elsewhere, open circles)",
             fontsize=FONT_SIZE-1, loc="left", style="italic")

save(fig, "fig7_cross_corpus")
