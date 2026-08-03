"""Fig 9 -- prompt-format ablation: which parts of the CoT are load-bearing?

Faithful rate ($\\Delta_\\infty > \\tau$, $\\tau = 0.05$) for ECoT-bridge under the
full CoT and five perturbations of it. Read from the released report, not from a
scratch directory.

Three things the earlier version of this figure got wrong, all of the same kind:
it drew numbers without drawing what they have to be compared against.

  * it read /tmp/cf_full_sweep, which no reader has. The report is now released
    under results_v2/canonical_runs/prompt_ablation/ and this file reads THAT, so
    the figure is reproducible from the artifact and the audit can check it.

  * the "full CoT" bar at 0.00 is DEFINITIONAL -- that variant is the reference
    the other five are differenced against, so its delta is identically zero and
    the bar carries no measurement. It is labelled as such on the plot, the way
    Figure 2(a) labels alpha(cot)=0 for the non-CoT baselines. (Hatching it does
    nothing: a zero-height bar has no interior to hatch.)

  * the ~0.95-1.00 heights were presented as "the model is strongly conditioned
    on the full nine-tag target". They have to be read against this model's OWN
    floor: ECoT-bridge scores 0.947 on the paraphrase null at the same tau. The
    null is drawn as a line. Once it is on the plot, the shuffled bar -- 0.95,
    from a perturbation the harness documents as "grammar destroyed, content
    preserved" -- is at the floor, and the figure's honest reading is that this
    is sensitivity to the CoT token string rather than to its content.

The per-variant denominators are not all 100: a variant is dropped for a sample
when its continuation did not decode 7 action bins (4 samples for plan_only, 5
for task_plan_subtask). That is printed under each bar, because a rate whose
denominator moved is not comparable to one whose denominator did not.

AUTHORED AT THE INCLUDE WIDTH (see gen_fig4_dissociation.py): matplotlib font
sizes are absolute points, so a figure authored wide and included narrow prints
its labels smaller than authored.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *
from _data import fam
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "results_v2", "canonical_runs", "prompt_ablation",
                      "cot_prompt_report.json")

d = json.load(open(REPORT))
agg = d["aggregate"]
TAU = d["threshold"]
N_REQ = d["n_samples"]

# The floor this figure has to be read against: the same model's rate under a
# meaning-preserving instruction paraphrase, at the same tau. Read from the
# derivation so it cannot drift from Table 1.
FLOOR = fam("ecot-bridge", "paraphrase_null", "F_mag")

VARIANTS = [
    ("full CoT",          "full",              C_CTRL),
    ("task only",         "task_only",         C_COT_TRAINED),
    ("plan only",         "plan_only",         C_COT_TRAINED),
    ("task+plan+subtask", "task_plan_subtask", C_COT_TRAINED),
    ("shuffled",          "shuffled",          "#AA3377"),
    ("empty",             "empty",             C_NO_COT),
]

xs = np.arange(len(VARIANTS))
fr = [agg[k]["faithful_rate"] for _, k, _ in VARIANTS]
ns = [agg[k]["n"] for _, k, _ in VARIANTS]

# Sizes on the page.
TICK, LAB, VAL, NOTE = 6.2, 6.6, 6.0, 5.4

fig, ax = plt.subplots(figsize=(4.93, 2.52))
# Explicit margins: savefig crops to the artists, so default margins emit a page
# narrower than the canvas and LaTeX scales it back up.
fig.subplots_adjust(left=0.118, right=0.995, top=0.985, bottom=0.205)
ax.tick_params(axis="both", labelsize=TICK, length=2.2, pad=1.5)

bars = ax.bar(xs, fr, color=[v[2] for v in VARIANTS], edgecolor="black",
              linewidth=0.4)
for bar, v in zip(bars, fr):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{v:.2f}", ha="center", va="bottom", fontsize=VAL)
ax.text(0, 0.085, "definitional:\nthis IS the\nreference", ha="center",
        va="bottom", fontsize=NOTE, style="italic", color="0.3",
        linespacing=1.2)

# The floor. Without it the four ~0.97 bars read as a strong effect; with it,
# only the gap ABOVE it is an effect at all.
ax.axhline(FLOOR, color="0.15", lw=0.7, ls=(0, (4, 2)), zorder=3)
# Annotated from the empty band above the bars: printed on the line itself it
# overlapped the shuffled bar's own value label, which is the one number the
# annotation exists to be compared against. The leader lands at x=0.5, the gap
# between bars: at x=0.62 it was inside the task_only bar and drawn behind it.
ax.annotate("paraphrase null, same model, same " r"$\tau$" f": {FLOOR:.3f}",
            xy=(0.50, FLOOR), xytext=(-0.34, 1.235),
            fontsize=NOTE, style="italic", color="0.15", ha="left",
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "0.15",
                        "shrinkB": 1.0})

ax.set_xticks(xs)
# Horizontal, wrapped: rotated, "task+plan+subtask (n=95)" ran under its left
# neighbour's label.
_WRAP = {"task+plan+subtask": "task+plan\n+subtask"}
ax.set_xticklabels([f"{_WRAP.get(lab, lab)}\n(n={n})"
                    for (lab, _, _), n in zip(VARIANTS, ns)],
                   fontsize=TICK, linespacing=1.3)
ax.set_ylabel(r"Faithful rate  ($\Delta_\infty > " f"{TAU}" r"$)",
              fontsize=LAB, labelpad=1.5)
ax.set_ylim(0, 1.30)
ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.4, alpha=0.5)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

save(fig, "fig9_prompt_ablation")

print(f"[audit] source     : {os.path.relpath(REPORT, ROOT)}")
print(f"[audit] tau={TAU}  requested n={N_REQ}  floor(paraphrase_null)={FLOOR:.3f}")
for (lab, k, _), v, n in zip(VARIANTS, fr, ns):
    # The rate if every sample that failed to decode is counted as NO change --
    # the worst case for the claim, and the one the caption has to survive.
    lo = round(v * n) / N_REQ
    print(f"[audit]   {k:18} n={n:3d}  rate={v:.4f}"
          + (f"  (>= {lo:.2f} on all {N_REQ})" if n != N_REQ else "")
          + ("  <-- at the floor" if abs(v - FLOOR) < 0.01 and k != "full" else ""))
