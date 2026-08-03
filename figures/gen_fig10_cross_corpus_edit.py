"""Fig 10 -- cross-corpus causal-edit response (F5), 100 samples requested per
non-LIBERO corpus.

Loaded from results_v2/derived_metrics.json.  No hardcoded literals.
NOTE: these are magnitude-F values; the directional caveat of Fig. 12 applies
to the direction_flip column here too (self-decoded CoT logs on the lerobot
corpora do not store a_orig/a_edit, so directional-F cannot yet be computed
cross-corpus -- stated as a limitation).

Two things this figure used to leave to the reader, both now on the plot:

  * the LIBERO bar is NOT the same kind of measurement as the other three. It is
    the 3-seed main sweep on dataset CoT annotations (N=299 and 300 records); the
    other three are one run each of self-decoded CoT, visibility-filtered down to
    the N printed under them. The caption said "on LIBERO (N=100)", which is the
    per-seed sample count, not what the bar is computed over.

  * a THIRD family was run on all three non-LIBERO corpora. subject_swap landed
    on 0 samples in every one of them, so there is nothing to draw -- but "we ran
    it and it never applied" and "we did not run it" are different statements, and
    the figure previously made neither. The n=0 is read from the release and
    printed as a note, the way the rollout section prints its excluded arms.

AUTHORED AT THE INCLUDE WIDTH (see gen_fig4_dissociation.py): the previous canvas
was 6.8in and the figure was included at 0.92\\textwidth, so its 10pt labels
printed at 7.6pt. The canvas below IS the include width.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *
from _data import MODELS, CROSS, fam
import numpy as np, matplotlib.pyplot as plt

rows = [("LIBERO",
         fam("ecot-bridge", "direction_flip", "F_mag"),
         MODELS["ecot-bridge"]["families"]["direction_flip"]["n_total"],
         fam("ecot-bridge", "gripper_flip", "F_mag"),
         MODELS["ecot-bridge"]["families"]["gripper_flip"]["n_total"])]
for tag, name in (("bridge_v2", "Bridge V2"), ("fractal", "Fractal"), ("bcz", "BC-Z")):
    e = CROSS[tag]["edit"]
    rows.append((name, e["direction_flip"]["faithful_rate"], e["direction_flip"]["n"],
                 e["gripper_flip"]["faithful_rate"], e["gripper_flip"]["n"]))

labels = [f"{r[0]}\n(dir N={r[2]}; grip N={r[4]})" for r in rows]

# subject_swap: run on all three non-LIBERO corpora, landed on 0 samples in each.
_SUBJ = {tag: (CROSS[tag]["edit"].get("subject_swap") or {}).get("n")
         for tag in ("bridge_v2", "fractal", "bcz")}
assert set(_SUBJ.values()) == {0}, \
    f"subject_swap now has samples cross-corpus ({_SUBJ}); it has to be drawn, " \
    "not annotated as absent"
direction = [r[1] for r in rows]
gripper = [r[3] for r in rows]
x = np.arange(len(labels)); w = 0.35

# Sizes on the page. 6.5pt is the floor for run text; LAB is 8.6 rather than
# 6.6 because the y label contains $\Delta_\infty$ and mathtext sets a subscript
# at 0.7x the base, so 6.6pt printed the "inf" of the threshold this whole axis
# is measured at at 4.6pt.
TICK, LAB, VAL, LEG, TITLE, NOTE = 6.5, 8.6, 6.5, 6.5, 7.0, 6.5

fig, ax = plt.subplots(1, 1, figsize=(5.18, 2.62))
# Explicit margins: savefig crops to the artists, so default margins emit a page
# narrower than the canvas and LaTeX scales it back up.
fig.subplots_adjust(left=0.108, right=0.995, top=0.855, bottom=0.150)
ax.tick_params(axis="both", labelsize=TICK, length=2.2, pad=1.5)
b1 = ax.bar(x - w / 2, direction, w, label="direction_flip", color=C_COT_TRAINED,
            edgecolor="black", linewidth=0.4)
b2 = ax.bar(x + w / 2, gripper, w, label="gripper_flip", color=C_NO_COT,
            edgecolor="black", linewidth=0.4)
for bars, vals in ((b1, direction), (b2, gripper)):
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=VAL)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=TICK)
ax.set_ylabel(r"magnitude-$\mathcal{F}$  ($\Delta_\infty > 0.05$)",
              fontsize=LAB, labelpad=1.5)
ax.set_ylim(0, 1.34)
# The two facts a reader cannot get from the bars. The first is a protocol
# difference INSIDE the figure; the second is a family that was run and never
# landed, which is an absent measurement rather than a zero.
# 1.245 and 1.09, not 1.255/1.145: at 6.5pt each of these lines is 0.09 of the
# data range tall, and at the old spacing the second one set straight across the
# y=1.2 gridline. Each note now sits in the clear band between two gridlines.
ax.text(-0.42, 1.245,
        "LIBERO bar: 3-seed main sweep on dataset CoT. Other three: one run "
        "each, self-decoded CoT, visibility-filtered.",
        fontsize=NOTE, style="italic", color="0.3", ha="left", va="bottom")
ax.text(-0.42, 1.09,
        # Plain underscore: this is a matplotlib string, so "\\_" prints the
        # backslash.
        "subject_swap was run on all three non-LIBERO corpora and landed on "
        r"$0$" " samples in each: absent, not " r"$0.0$" ".",
        fontsize=NOTE, style="italic", color="0.3", ha="left", va="bottom")
# The legend goes ABOVE the axes, next to the title. Inside at upper right it
# printed through the 0.95 value label of the BC-Z direction_flip bar.
_hl, _ll = ax.get_legend_handles_labels()
fig.legend(_hl, _ll, ncol=2, frameon=False, fontsize=LEG, loc="upper right",
           bbox_to_anchor=(0.995, 1.000), handlelength=1.2, handletextpad=0.5,
           columnspacing=1.6, borderpad=0.1)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
ax.set_title("ECoT-bridge magnitude-$\\mathcal{F}$ across 4 corpora (F5)",
             fontsize=TITLE, loc="left", style="italic", pad=2.5)
ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.4, alpha=0.5)
save(fig, "fig10_cross_corpus_edit")

print("[audit] " + "  ".join(f"{r[0]}: dir={r[1]:.2f} (N={r[2]}), "
                             f"grip={r[3]:.2f} (N={r[4]})" for r in rows))
print(f"[audit] LIBERO bar is the 3-seed sweep: "
      f"per-run dir F_mag = "
      f"{MODELS['ecot-bridge']['families']['direction_flip']['F_mag_per_run']}")
print(f"[audit] subject_swap n cross-corpus: {_SUBJ} (absent, not null)")
