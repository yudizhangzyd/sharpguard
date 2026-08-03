"""Fig 5 — Bridge- vs LIBERO-trained CoT-VLA on ALL 11 shared families.

Grouped bar chart: 11 edit families x 2 models (Ours r=32 LIBERO fine-tune vs
public ECoT-bridge Bridge-V2), same architecture, same prompt, near-identical
attention distributions. Supports the O4 observation.

The previous version drew THREE families and its docstring and the caption both
called them "the 3 shared families". Eleven families are shared: ours-r32 carries
all 13 and ECoT-bridge carries 11, missing only bbox_jitter_null and
instr_random_sub. So "shared" was not the selection rule -- nothing was -- and the
three that survived were all semantic, which is the one subset that makes the
figure read as "Bridge-trained CoT is more faithful."

Drawing all 11 says the opposite, and it is the paper's own argument: ECoT-bridge
sits at 0.947 on paraphrase_null and 0.856 on syntactic_scramble, i.e. it is
near-uniformly high on the controls TOO, which is why its gap over ours cannot be
read as a faithfulness difference. The columns are grouped the way Figure 3
groups them, with the semantic set imported from _data.NON_CONTROL rather than
retyped, so the two figures cannot disagree about which families are semantic.

Per-bar value labels are gone: 11 groups in 6.28in give each bar 12pt of width
and "0.93" at 6pt is 14pt wide, so the labels collided. The per-group RATIO is
annotated instead, which is the quantity the figure exists to show, and the exact
values are Table 1 and Figure 3.

AUTHORED AT THE INCLUDE WIDTH (see gen_fig4_dissociation.py).

Both series carry their 3-sampling-seed std: drawing a 3-seed mean as a bare bar
states a precision the run does not have.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *
from _data import fam, NON_CONTROL
import matplotlib.pyplot as plt
import numpy as np

# Every family both models were run on, grouped as Figure 3 groups them. The
# semantic set is asserted against the derivation's own NON_CONTROL list.
_SEMANTIC = [("dir\nflip", "direction_flip"), ("negation", "negation"),
             ("verb\nswap", "verb_swap"), ("subj\nswap", "subject_swap"),
             ("loc\nswap", "location_swap"), ("grip\nflip", "gripper_flip"),
             ("adv\nplaus", "adversarial_plausible")]
assert {f for _, f in _SEMANTIC} == set(NON_CONTROL), \
    "this figure's 'semantic' group has drifted from derive_metrics.NON_CONTROL"
_TIER0 = [("selfspl\n(null)", "selfsplice_control"),
          ("syntactic\nscramble", "syntactic_scramble"),
          ("cross-task\nswap", "cross_task_swap")]
_CALIB = [("paraphrase\n(null)", "paraphrase_null")]
FAMS = _SEMANTIC + _TIER0 + _CALIB
GROUPS = [("semantic (7)", 0, len(_SEMANTIC)),
          ("Tier-0 controls", len(_SEMANTIC), len(_SEMANTIC) + len(_TIER0)),
          ("null", len(FAMS) - 1, len(FAMS))]

ours = [fam("ours-r32", k, "F_mag") for _, k in FAMS]
bridge = [fam("ecot-bridge", k, "F_mag") for _, k in FAMS]
ours_std = [fam("ours-r32", k, "F_mag_std") or 0.0 for _, k in FAMS]
bridge_std = [fam("ecot-bridge", k, "F_mag_std") or 0.0 for _, k in FAMS]
assert not any(v is None for v in ours + bridge), \
    "a shared family is missing from the release; the figure would draw a gap " \
    "as a value"
# The figure claims these are the families BOTH models were run on, so it has to
# be the case that no other family is shared.
_o = {f for f in fam.__self__.keys()} if hasattr(fam, "__self__") else None

xs = np.arange(len(FAMS))
w = 0.38

# Sizes on the page. Eleven two-line tick labels share 6.0in, so each gets 39pt
# and "syntactic" at 5.6pt is 26pt.
TICK, LAB, VAL, LEG, GRP = 5.6, 6.6, 5.4, 6.2, 5.6

fig, ax = plt.subplots(figsize=(6.28, 2.34))
# Explicit margins: savefig crops to the artists, so default margins emit a page
# narrower than the canvas and LaTeX scales it back up.
fig.subplots_adjust(left=0.075, right=0.996, top=0.845, bottom=0.185)
ax.tick_params(axis="both", labelsize=TICK, length=2.2, pad=1.5)

ax.bar(xs - w / 2, ours, w, yerr=ours_std, capsize=1.2,
       error_kw={"lw": 0.5}, label="Ours r=32 (LIBERO fine-tune)",
       color=C_COT_TRAINED, edgecolor="black", linewidth=0.4)
ax.bar(xs + w / 2, bridge, w, yerr=bridge_std, capsize=1.2,
       error_kw={"lw": 0.5}, label="ECoT-bridge (Bridge-V2)",
       color=C_ECOT_BRIDGE, edgecolor="black", linewidth=0.4)

# The ratio per family, which is the figure's content. Undefined on the identity
# null, where both models are 0.00 -- printed as "0/0" rather than skipped, so
# the reader sees the pair was measured and agreed.
for x, o, b, e in zip(xs, ours, bridge, bridge_std):
    txt = f"{b / o:.1f}$\\times$" if o else "0/0"
    ax.text(x, max(o, b) + e + 0.035, txt, ha="center", va="bottom",
            fontsize=VAL, color="0.25")

# The group boundaries, drawn rather than only named.
for _, _, end in GROUPS[:-1]:
    ax.axvline(end - 0.5, color="0.55", lw=0.6, ls=(0, (3, 2)), zorder=0)
for name, lo, hi in GROUPS:
    ax.text((lo + hi - 1) / 2, 1.145, name, ha="center", va="bottom",
            fontsize=GRP, style="italic", color="0.3")

ax.set_xticks(xs)
ax.set_xticklabels([f[0] for f in FAMS], fontsize=TICK)
ax.set_ylabel("Faithful rate", fontsize=LAB, labelpad=1.5)
ax.set_ylim(0, 1.235)
ax.set_yticks(np.arange(0, 1.01, 0.2))
# The legend goes ABOVE the axes: inside, it printed straight through the value
# labels of the tallest bars, which are in the left third of the panel.
_hl, _ll = ax.get_legend_handles_labels()
fig.legend(_hl, _ll, ncol=2, frameon=False, fontsize=LEG, loc="upper center",
           bbox_to_anchor=(0.55, 1.000), handlelength=1.3, handletextpad=0.5,
           columnspacing=1.8, borderpad=0.1)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

save(fig, "fig5_bridge_vs_libero")

nc = [k for _, k in _SEMANTIC]
mo = np.mean([fam("ours-r32", k, "F_mag") for k in nc])
mb = np.mean([fam("ecot-bridge", k, "F_mag") for k in nc])
print(f"[audit] {len(FAMS)} shared families drawn")
for (lab, k), o, b in zip(FAMS, ours, bridge):
    print(f"[audit]   {k:22} ours {o:.3f}  bridge {b:.3f}"
          + (f"  {b / o:.2f}x" if o else "  ratio undefined (both 0)"))
print(f"[audit] 7-non-control means: ours {mo:.3f} bridge {mb:.3f} "
      f"-> {mb / mo:.2f}x")
print(f"[audit] bridge on the two non-semantic high columns: "
      f"paraphrase_null {fam('ecot-bridge', 'paraphrase_null', 'F_mag'):.3f}, "
      f"syntactic_scramble "
      f"{fam('ecot-bridge', 'syntactic_scramble', 'F_mag'):.3f}")
