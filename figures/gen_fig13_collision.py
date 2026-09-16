"""Fig 13 -- what the magnitude score actually counts (S5).

Section 5 makes the paper's sharpest claim about the instrument -- that
F_mag is close to a decode-collision counter rather than a measure of how far
the action moved. Four panels now carry it:

(a) the one real camera frame Fig. 2 already leans on (all three arms are
    byte-identical at t=0, per Fig. 2's own caption), with the two real MOVE
    phrases that were decoded from it: "own CoT" and the direction-flipped
    edit of it.
(b) the two real decoded 7-DoF actions for that one frame, one dumbbell per
    dimension: 5 of 7 tie exactly (translation-x, all three rotation axes,
    the gripper bit) and 2 (translation-y, translation-z) do not, which is
    the mechanism in (b) is close to a collision counter, not a measure of
    how much meaning changed) made concrete at n=1.
(c) the unchanged aggregate this always rested on: F at tau against
    1 - P(Delta = 0), one point per (run, family) cell across all 324, so
    panel (b)'s anecdote is shown to generalize rather than stand alone.
(d) the resolution sweep: re-deriving the collision rate and the mean
    continuous Delta_inf at coarser groupings of the same 256-bin softmax
    (8/16/32/64/128/256 bins). Collision rate swings 4-5x across this range;
    the continuous displacement it is derived from barely moves, which is
    the panel that turns (c)'s correlation into a mechanism.

Panel (a)/(b)'s numbers are read from
results_v2/canonical_runs/rollout_filmstrip/rollout_edit_probe.json and
rollout_edit_report.json -- the same released files Figure 2 is drawn from,
not a new capture -- panel (c) is unchanged from the prior three-panel
version, read from collision_decomposition.json, and panel (d) is read from
results_v2/canonical_runs/action_bin_resolution_sweep/
action_bin_resolution_sweep_report.json. No number here is a literal:
Delta_inf, the tied/split dimensions, R^2 and the per-resolution rates are
all recomputed from the artifacts, so the figure cannot silently drift from
the audit.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *          # noqa: F401,F403  (rcParams + save)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results_v2", "canonical_runs",
                   "collision_decomposition", "collision_decomposition.json")
with open(SRC) as fh:
    D = json.load(fh)

TAU = D["tau"]
CELLS = D["cells"]

FILM_DIR = os.path.join(ROOT, "results_v2", "canonical_runs",
                        "rollout_filmstrip")
with open(os.path.join(FILM_DIR, "rollout_edit_probe.json")) as fh:
    PROBE = json.load(fh)
with open(os.path.join(FILM_DIR, "rollout_edit_report.json")) as fh:
    REPORT = json.load(fh)

SWEEP_SRC = os.path.join(ROOT, "results_v2", "canonical_runs",
                         "action_bin_resolution_sweep",
                         "action_bin_resolution_sweep_report.json")
with open(SWEEP_SRC) as fh:
    SWEEP = json.load(fh)
BINS = [8, 16, 32, 64, 128, 256]
SWEEP_AGG = SWEEP["aggregate"]
COLL_RATE = [SWEEP_AGG[str(b)]["collision_rate"] * 100 for b in BINS]
DLINF = [SWEEP_AGG[str(b)]["delta_linf_mean"] for b in BINS]
if SWEEP["n_mismatch_256_regression_guard"] != 0:
    raise SystemExit("[fig13] the 256-bin regression guard is non-zero -- "
                      "the resolution sweep disagrees with this paper's own "
                      "collision numbers, fix before plotting.")

ONE_FRAME = PROBE["one_frame_actions"]
CLEAN = np.array(ONE_FRAME["cot_clean"]["sent"])
FLIP = np.array(ONE_FRAME["cot_direction_flip"]["sent"])
DELTA_INF = ONE_FRAME["cot_direction_flip"]["delta_linf_vs_cot_clean"]
DIMS = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]
if len(CLEAN) != len(DIMS):
    raise SystemExit(f"[fig13] one_frame_actions has {len(CLEAN)} dims, "
                      f"not the {len(DIMS)} this figure labels -- fix DIMS "
                      f"before trusting the panel.")
diffs = np.abs(FLIP - CLEAN)
recovered_delta_inf = float(diffs.max())
if abs(recovered_delta_inf - DELTA_INF) > 1e-3:
    raise SystemExit(f"[fig13] recomputed Delta_inf {recovered_delta_inf} "
                      f"disagrees with the artifact's own "
                      f"{DELTA_INF} -- the two released files have drifted.")
TIED = [i for i in range(len(DIMS)) if diffs[i] < 1e-6]
SPLIT = [i for i in range(len(DIMS)) if diffs[i] >= 1e-6]

by_ep = {}
for e in REPORT.get("episodes", []):
    if e.get("trajectory"):
        by_ep.setdefault((e.get("task_idx"), e.get("episode")), {})[e["arm"]] = e
EP = by_ep[min(by_ep)]
MOVE = {}
for arm in ("cot_clean", "cot_direction_flip"):
    rec0 = next((t for t in EP[arm]["trajectory"] if t["step"] == 0), None)
    MOVE[arm] = (rec0 or {}).get("move") or "?"
FRAME_PATH = os.path.join(FILM_DIR, "frames",
                          f"t{min(by_ep)[0]}_ep{min(by_ep)[1]}",
                          "cot_clean_t0000.png")
FRAME = mpimg.imread(FRAME_PATH)

# 6.40x1.79 grows the canvas modestly from the prior three-panel version's
# 6.30x1.79 to make room for panel (d) without touching the height, so this
# adds no vertical page cost -- panel titles were shortened (detail moved to
# the LaTeX caption, which already carried it) to keep native width inside
# the legible include-scale band at ~100% \textwidth.
fig = plt.figure(figsize=(6.40, 1.79))
gs = fig.add_gridspec(1, 4, width_ratios=[0.80, 1.05, 1.45, 0.85], wspace=0.45)
ax0, ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(4))

# ---- (a) the one real frame, and the two real MOVE phrases ---------------
ax0.imshow(FRAME)
ax0.set_xticks([]); ax0.set_yticks([])
for s in ax0.spines.values():
    s.set_visible(False)
ax0.set_title("(a) one frame", loc="left",
              fontsize=FONT_SIZE, style="italic")
ax0.text(0.5, -0.08, MOVE["cot_clean"], transform=ax0.transAxes,
         ha="center", va="top", fontsize=FONT_SIZE - 3,
         color=C_COT_TRAINED)
ax0.text(0.5, -0.22, MOVE["cot_direction_flip"], transform=ax0.transAxes,
         ha="center", va="top", fontsize=FONT_SIZE - 3,
         color=C_ECOT_BRIDGE)

# ---- (b) the two real decoded actions, dimension by dimension ------------
ys = np.arange(len(DIMS))[::-1]
for y, i in zip(ys, range(len(DIMS))):
    tied = i in TIED
    ax1.plot([CLEAN[i], FLIP[i]], [y, y],
             color="0.75" if tied else "0.35", lw=1.1, zorder=1)
ax1.scatter(CLEAN, ys, s=14, color=C_COT_TRAINED, zorder=3, label="own CoT")
ax1.scatter(FLIP, ys, s=14, color=C_ECOT_BRIDGE, zorder=3,
            label="direction flipped")
ax1.set_yticks(ys)
ax1.set_yticklabels(DIMS, fontsize=FONT_SIZE - 2)
ax1.set_xlim(-1.15, 1.15)
ax1.axvline(0, color="0.85", lw=0.6, zorder=0)
ax1.set_xlabel("decoded action (normalized)", fontsize=FONT_SIZE - 2)
ax1.set_title(rf"(b) $\Delta_\infty={DELTA_INF:.2f}$",
              loc="left", fontsize=FONT_SIZE, style="italic")
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
           frameon=False, fontsize=FONT_SIZE - 3.5, handletextpad=0.3,
           columnspacing=0.8)
ax1.set_axisbelow(True)
ax1.xaxis.grid(True, ls=":", lw=0.4, alpha=0.5)

# ---- (c) F at tau vs the collision rate, unchanged from the prior version -
f = np.array([c["F_at_tau"] for c in CELLS])
g = np.array([c["one_minus_collision"] for c in CELLS])

ax2.plot([0, 1], [0, 1], color="0.35", lw=0.8, ls="--", zorder=1)
ax2.scatter(g, f, s=9, alpha=0.55, color=C_COT_TRAINED,
            edgecolor="none", zorder=2)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.set_xlabel(r"$1 - P(\Delta_\infty = 0)$", fontsize=FONT_SIZE - 1)
ax2.set_ylabel(rf"$\mathcal{{F}}$ at $\tau={TAU:g}$", fontsize=FONT_SIZE - 1)
ax2.set_title(rf"(c) {D['n_cells']} cells, $R^2={D['r_squared']:.2f}$",
              loc="left", fontsize=FONT_SIZE, style="italic")
ax2.set_axisbelow(True)
ax2.grid(True, ls=":", lw=0.4, alpha=0.5)

# The identical-to-the-last-digit cells are the strongest single number in the
# section, so it is stated on the panel rather than left to the caption.
ax2.text(0.97, 0.06, f"{D['n_cells_exactly_equal']} cells identical\n"
                     f"to the last digit",
         transform=ax2.transAxes, ha="right", va="bottom",
         fontsize=FONT_SIZE - 3.5, color="0.25")

# ---- (d) resolution sweep: collision rate vs. the continuous displacement -
xpos = np.arange(len(BINS))
l1, = ax3.plot(xpos, COLL_RATE, marker="o", ms=3, lw=1.1,
               color=C_ECOT_BRIDGE, zorder=3)
ax3.set_ylabel("collision rate (%)", fontsize=FONT_SIZE - 2.5,
               color=C_ECOT_BRIDGE)
ax3.tick_params(axis="y", labelcolor=C_ECOT_BRIDGE, labelsize=FONT_SIZE - 3)
ax3.set_ylim(0, max(COLL_RATE) * 1.25)
ax3.set_xticks(xpos)
ax3.set_xticklabels([str(b) for b in BINS], fontsize=FONT_SIZE - 3.5,
                    rotation=40, ha="right")
ax3.set_xlabel("action-token bins", fontsize=FONT_SIZE - 2.5)

ax3b = ax3.twinx()
l2, = ax3b.plot(xpos, DLINF, marker="s", ms=2.5, lw=1.0, ls="--",
                color=C_NO_COT, zorder=2)
ax3b.set_ylabel(r"$\Delta_\infty$", fontsize=FONT_SIZE - 2.5,
                color=C_NO_COT, labelpad=1)
ax3b.tick_params(axis="y", labelcolor=C_NO_COT, labelsize=FONT_SIZE - 3.5,
                 pad=1)
ax3b.set_ylim(0, 1.0)
for s in ("top",):
    ax3.spines[s].set_visible(False)
    ax3b.spines[s].set_visible(False)
ax3.set_title("(d) resolution sweep", loc="left",
              fontsize=FONT_SIZE, style="italic")
ax3.set_axisbelow(True)
ax3.grid(True, ls=":", lw=0.4, alpha=0.4)

save(fig, "fig13_collision")

print(f"[audit] source     : {os.path.relpath(SRC, ROOT)}")
print(f"[audit] cells      : {D['n_cells']}  R^2 = {D['r_squared']:.4f}")
print(f"[audit] one-frame  : Delta_inf={DELTA_INF}  "
      f"tied={[DIMS[i] for i in TIED]}  split={[DIMS[i] for i in SPLIT]}")
print(f"[audit] resolution sweep: bins={BINS}")
print(f"[audit]   collision_rate%={[round(v, 1) for v in COLL_RATE]}")
print(f"[audit]   delta_linf_mean={[round(v, 3) for v in DLINF]}")
