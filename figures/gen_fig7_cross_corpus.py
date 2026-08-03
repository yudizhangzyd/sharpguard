"""Fig 7 -- cross-corpus attention (F5), N=100 per non-LIBERO corpus.

Loaded from results_v2/derived_metrics.json (which pins the aws_6 lerobot runs
qzvywaxg6u / ae8ikp2zv7 / q27nbyr3w8 and the 3-seed LIBERO ECoT-bridge profile).
No hardcoded literals.

AUTHORED AT THE INCLUDE WIDTH (see gen_fig4_dissociation.py): the previous canvas
was 6.8in and the figure was included at 0.93\\textwidth, so its 10pt labels
printed at 7.5pt. The canvas below IS the include width and the sizes below are
the sizes on the page.

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

# Sizes on the page. 6.5pt is the floor for run text in the submission's
# figures; four corpus slots share 4.6in here, so nothing is close to tight.
TICK, LAB, LEG, TITLE = 6.5, 6.6, 6.5, 7.0

fig, ax = plt.subplots(1, 1, figsize=(5.18, 2.45))
# Explicit margins: savefig crops to the artists, so default margins emit a page
# narrower than the canvas and LaTeX scales it back up.
fig.subplots_adjust(left=0.112, right=0.995, top=0.895, bottom=0.145)
ax.tick_params(axis="both", labelsize=TICK, length=2.2, pad=1.5)
for i, k in enumerate(KEYS):
    m = [r[1][CKEY[k] if r[4] else k] for r in rows]
    s = [r[2][CKEY[k] if r[4] else k] for r in rows]
    ax.bar(x + (i - 1.5) * w, m, w, yerr=s, capsize=2,
           label=k, color=BUCKET_COLORS[CKEY[k]])
ax.set_xticks(x)
ax.set_xticklabels([f"{r[0]}\n(N={r[3]})" for r in rows], fontsize=TICK + 0.2)
ax.set_ylabel(r"Attention mass $\alpha(m,B)$  (mean$\pm$std)", fontsize=LAB,
              labelpad=1.5)
ax.set_ylim(0, 0.46)
ax.legend(ncol=4, frameon=False, loc="upper center", fontsize=LEG,
          handlelength=1.1, handletextpad=0.4, columnspacing=1.4,
          borderpad=0.1, borderaxespad=0.15)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
ax.set_title("ECoT-bridge attention across 4 corpora (F5)",
             fontsize=TITLE, loc="left", style="italic", pad=2.5)
save(fig, "fig7_cross_corpus")

_dev = max(abs(r[1][CKEY[k] if r[4] else k] - rows[0][1][k])
           for r in rows[1:] for k in KEYS)
print(f"[audit] {len(rows)} corpora x {len(KEYS)} buckets; "
      f"max deviation from LIBERO on any bucket = {100 * _dev:.2f} pp")
