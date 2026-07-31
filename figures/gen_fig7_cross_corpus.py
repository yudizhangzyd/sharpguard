"""Fig 7 -- cross-corpus attention (F5), N=100 per non-LIBERO corpus.

Loaded from results_v2/derived_metrics.json (which pins the aws_6 lerobot runs
qzvywaxg6u / ae8ikp2zv7 / q27nbyr3w8 and the 3-seed LIBERO ECoT-bridge profile).
No hardcoded literals.

The N=30 pilot these replaced is retained under results_v2/superseded/; every
bucket mean here is within 0.3 pp of it, which is checked by audit_f5 rather
than asserted in the caption.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *
from _data import ATTN, CROSS
import numpy as np, matplotlib.pyplot as plt

KEYS = ["visual", "instruction", "cot", "action_prev"]
CKEY = {"visual": "visual", "instruction": "instr", "cot": "cot",
        "action_prev": "action_prev"}

rows = [("LIBERO", ATTN["ecot-bridge"]["mass"], ATTN["ecot-bridge"]["mass_std"],
         ATTN["ecot-bridge"]["n"], False)]
for tag, name in (("bridge_v2", "Bridge V2"), ("fractal", "Fractal"), ("bcz", "BC-Z")):
    c = CROSS[tag]
    rows.append((name, c["mass"], c["mass_std"], c["n_attn"], True))

x = np.arange(len(rows)); w = 0.20
fig, ax = plt.subplots(1, 1, figsize=(6.8, 3.5))
for i, k in enumerate(KEYS):
    m = [r[1][CKEY[k] if r[4] else k] for r in rows]
    s = [r[2][CKEY[k] if r[4] else k] for r in rows]
    ax.bar(x + (i - 1.5) * w, m, w, yerr=s, capsize=2,
           label=k, color=BUCKET_COLORS[CKEY[k]])
ax.set_xticks(x)
ax.set_xticklabels([f"{r[0]}\n(N={r[3]})" for r in rows], fontsize=FONT_SIZE - 1)
ax.set_ylabel(r"Attention mass $\alpha(m,B)$  (mean$\pm$std)")
ax.set_ylim(0, 0.44)
ax.legend(ncol=2, frameon=False, loc="upper right", fontsize=FONT_SIZE - 1)
ax.set_title("ECoT-bridge attention across 4 corpora (F5)",
             fontsize=FONT_SIZE, loc="left", style="italic")
save(fig, "fig7_cross_corpus")
