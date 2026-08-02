"""Fig 15 -- what the three rollout arms actually DO, frame by frame.

Every rollout number in this paper is a scalar: SR, step counts, edits skipped.
Section 6 reports 0/40 in both arms and says plainly that the clean-CoT policy
cannot finish the task, so no DSR is defined. That is an honest null, but it is
also invisible: two arms that both score 0/40 are indistinguishable in the
record even if one is reaching for the drawer and the other is driving into the
table, and which of those is happening is what an edit-sensitivity claim is
about. No released artifact of ours contained a single rendered frame.

So the harness now records them (`--capture-episodes`), and this draws them:

  (a) a filmstrip -- one row per arm, one column per timestep, the frame as the
      POLICY saw it. Under each CoT row is the MOVE phrase the policy was
      acting under at that step; on the edited row that is the EDITED phrase,
      so a reader sees the instruction that was read next to the motion it
      produced instead of taking on faith that the edit landed.
  (b) the end-effector path, top-down, all arms from the one shared init state.
  (c) the quantity the filmstrip is qualitative about: how far each arm's
      gripper is from the clean-CoT arm's at the same step. This is the
      rollout-level analogue of the first-step Delta_infinity the leaderboard
      is built on, and it is defined whether or not any arm ever succeeds --
      which is exactly what DSR is not.

Nothing here is drawn by hand and no number is a literal. Every pixel is a
captured frame, every line is a logged end-effector pose, and every caption
string is the MOVE phrase from the same step's record. If the capture artifact
is missing or carries no poses, this exits non-zero and says which -- a
half-drawn version of this figure would be a claim about motion that was not
measured.

Usage:  python figures/gen_fig15_rollout_filmstrip.py [<capture_dir>] [--no-eef]
where <capture_dir> holds rollout_edit_report.json and frames/. `--no-eef`
draws panel (a) alone, for a capture whose env did not expose end-effector
poses; it must be passed explicitly, because silently dropping two panels is
how a figure comes to show less than its caption claims.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import *          # noqa: F401,F403  (rcParams + save)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "results_v2", "canonical_runs",
                       "rollout_filmstrip")

# One column per timestep. Six is what fits at \textwidth with the row labels:
# 6 cells of ~65pt each read at print size, 8 do not.
N_COLS = 6

# Row order is the causal order the arms are defined in, not alphabetical:
# no reasoning, then the model's own reasoning, then that reasoning edited.
# Reading down a column is then reading the effect of adding, and then
# corrupting, the CoT.
ARMS = [
    ("nocot",              "no CoT",            C_NO_COT),
    ("cot_clean",          "own CoT",           C_COT_TRAINED),
    ("cot_direction_flip", "CoT, direction\nflipped", C_ECOT_BRIDGE),
]


def die(msg: str) -> None:
    print(f"[fig15] {msg}", file=sys.stderr)
    raise SystemExit(2)


def load(cap_dir: str):
    rep_p = os.path.join(cap_dir, "rollout_edit_report.json")
    if not os.path.exists(rep_p):
        die(f"no rollout_edit_report.json under {cap_dir}. This figure is "
            f"drawn from captured frames; it has no offline fallback.")
    with open(rep_p) as fh:
        rep = json.load(fh)

    # Episodes that were actually filmed, grouped by their (task, episode) --
    # the pairing the arms share an init state over. An episode from a
    # different init state is a different scene and cannot go in a column.
    by_ep = {}
    for e in rep.get("episodes", []):
        if not e.get("trajectory"):
            continue
        by_ep.setdefault((e.get("task_idx"), e.get("episode")), {})[e["arm"]] = e
    if not by_ep:
        die("the report carries no trajectory records. Either capture was off "
            "for this run or every frame write failed; run_cotfaith_rollout_"
            "edit_s3.sh exits 8 on the latter, so check which.")

    want = {a for a, _, _ in ARMS}
    full = {k: v for k, v in by_ep.items() if want <= set(v)}
    if not full:
        have = {k: sorted(v) for k, v in by_ep.items()}
        die(f"no filmed episode has all three arms {sorted(want)}; got {have}. "
            f"A filmstrip missing a row would invite the reader to compare two "
            f"arms as though the third had not been run.")
    # The lowest-numbered complete episode, so the choice is not a choice: a
    # "best-looking" one would be cherry-picked, and this figure's whole value
    # is that it is not.
    key = sorted(full)[0]
    return rep, key, full[key], sorted(full)


def pick_steps(eps: dict) -> list:
    """N_COLS evenly-spaced steps that EVERY arm has a frame for.

    The intersection, not each arm's own grid: a column has to be one instant
    across the three rows or it is not a column. Arms that terminate at
    different steps therefore shorten the strip rather than misalign it.
    """
    common = None
    for e in eps.values():
        s = {t["step"] for t in e["trajectory"] if t.get("frame")}
        common = s if common is None else (common & s)
    common = sorted(common or [])
    if len(common) < 2:
        die(f"only {len(common)} step(s) are common to all arms; there is no "
            f"strip to draw. Lower --capture-every and re-run.")
    idx = np.linspace(0, len(common) - 1, min(N_COLS, len(common)))
    return [common[int(round(i))] for i in idx]


def frame(cap_dir, key, arm, step, eps):
    rec = next((t for t in eps[arm]["trajectory"] if t["step"] == step), None)
    if rec is None or not rec.get("frame"):
        return None, rec
    p = os.path.join(cap_dir, "frames", f"t{key[0]}_ep{key[1]}", rec["frame"])
    if not os.path.exists(p):
        die(f"{rec['frame']} is named in the report but missing on disk "
            f"({p}). The report and the frames come from one run; a mismatch "
            f"means one of them is from another.")
    return plt.imread(p), rec


def short_move(s, width=22):
    """The MOVE phrase, trimmed to fit under a ~65pt cell.

    'move' prefixes every phrase the generator emits and says nothing that the
    figure does not already say, so it goes; the rest is truncated on a word
    boundary with an ellipsis, since a mid-word cut reads like a typo.
    """
    if not s:
        return "—"
    s = " ".join(str(s).split())
    for pre in ("move the gripper ", "move ", "the robot should "):
        if s.lower().startswith(pre):
            s = s[len(pre):]
            break
    if len(s) <= width:
        return s
    cut = s[:width].rsplit(" ", 1)[0]
    return (cut or s[:width]) + "…"


def start_mismatch(imgs: dict, step: int) -> "str | None":
    """None if every arm's first frame is the same pixels, else what differs.

    The rows of this figure are three prompts on ONE scene, so the first column
    must be one image repeated. That is worth checking rather than trusting: the
    first capture (bolt h8xzmqnhgg) had a 3-pixel translation of the cabinet
    between arms -- bit-identical outside the differing box, 8.8411 inside it --
    because `set_init_state` restores qpos and a welded fixture's pose is not in
    qpos, so robosuite's placement sampler drew it separately for each arm's env.
    Every scalar in the report looked right; only the pixels showed it. Fixing
    that surfaced a second one (bolt xyiztdu4n6): bit-identical outside the box
    again, but the box had moved onto the ROBOT, 31.7356 inside, and best rigid
    shift (0, 0) -- a pose no translation aligns, because each arm ran its own
    settling loop and the second inherited the first arm's controller goal and
    warm-start accelerations. Hence the exact-equality test below rather than a
    shift-tolerant one: the second defect has no shift to find. Both
    measurements are released under
    results_v2/canonical_runs/rollout_arm_pairing_defect/.

    There is deliberately no override. A strip whose rows are different scenes
    is not a smaller figure than intended, it is a wrong one: the reader would
    attribute to the CoT edit a difference that came from the furniture.
    """
    ref_name, ref = next(iter(imgs.items()))
    for name, im in imgs.items():
        if im.shape != ref.shape:
            return f"{name} is {im.shape} and {ref_name} is {ref.shape}"
        d = np.abs(im.astype(float) - ref.astype(float))
        if d.max() == 0:
            continue
        m = d.max(axis=2) if d.ndim == 3 else d
        ys, xs = np.where(m > m.max() * 0.05)
        return (f"{name} and {ref_name} differ at step {step} before "
                f"either arm has acted: mean |dpix| {d.mean():.3f}, max "
                f"{d.max():.3f}, over rows {ys.min()}-{ys.max()} cols "
                f"{xs.min()}-{xs.max()}. Re-run with --env-seed set (the "
                f"harness seeds np.random per episode so every arm draws the "
                f"same fixture placement); a shared init state is not enough, "
                f"because set_init_state does not restore a welded fixture.")
    return None


def eef_track(e):
    """(steps, xyz) for one episode, or (None, None) if poses were not logged."""
    rows = [(t["step"], t["eef"]) for t in e["trajectory"] if t.get("eef")]
    if not rows:
        return None, None
    return (np.array([r[0] for r in rows], float),
            np.array([r[1] for r in rows], float))


def column_motion(cap_dir, key, arm, steps, eps) -> dict:
    """How much the scene changes between the columns actually displayed.

    A reader who sees two near-identical cells has two readings available --
    "the policy stalled here" and "this figure is broken" -- and nothing in the
    strip distinguishes them. This measures which it is, so the caption can say
    it rather than leave the reader to guess.

    It is worth measuring because the answer is not uniform. On the capture that
    motivated this figure the policy moved hard for ~70 steps, then went nearly
    static: consecutive captured frames changed by a mean of 5-9 absolute pixel
    levels early and by 0.06-0.08 later, a two-order-of-magnitude drop. Evenly
    spaced columns therefore land several cells inside a stall. That is a
    property of a policy which scores 0/40, not an artifact of the sampling, and
    the figure should report it instead of hiding it behind a livelier choice of
    columns.
    """
    imgs = []
    for st in steps:
        im, _ = frame(cap_dir, key, arm, st, eps)
        imgs.append(None if im is None else im.astype(float))
    d = []
    for i in range(1, len(imgs)):
        if imgs[i] is None or imgs[i - 1] is None:
            continue
        if imgs[i].shape != imgs[i - 1].shape:
            continue
        d.append(float(np.abs(imgs[i] - imgs[i - 1]).mean()))
    if not d:
        return {}
    # Scaled to 0-255 levels whether matplotlib handed back floats or ints, so
    # the number means the same thing regardless of how the PNG was read.
    k = 255.0 if max(np.nanmax(np.abs(i)) for i in imgs if i is not None) <= 1.0 \
        else 1.0
    d = [x * k for x in d]
    return {"min": round(min(d), 3), "max": round(max(d), 3),
            "mean": round(float(np.mean(d)), 3)}


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    no_eef = "--no-eef" in sys.argv
    cap_dir = argv[0] if argv else DEFAULT
    rep, key, eps, all_keys = load(cap_dir)
    steps = pick_steps(eps)
    print(f"[fig15] {cap_dir}")
    print(f"[fig15] episode task{key[0]}/ep{key[1]} of {len(all_keys)} filmed; "
          f"columns at steps {steps}")

    tracks = {a: eef_track(eps[a]) for a, _, _ in ARMS}
    have_eef = all(t[0] is not None for t in tracks.values())

    # The rows must be one scene before they can be three conditions. Checked on
    # the earliest shared step, which is captured before that step's action is
    # applied, so the arms cannot legitimately differ there yet.
    first = {a: frame(cap_dir, key, a, steps[0], eps)[0] for a, _, _ in ARMS}
    if any(v is None for v in first.values()):
        die(f"an arm has no frame at the first shared step {steps[0]}, which "
            f"pick_steps() selected as common to all arms -- the report and the "
            f"frames on disk disagree.")
    bad = start_mismatch(first, steps[0])
    if bad:
        die("the three arms do not start from the same scene. " + bad)
    # Recorded, not just checked. The refusal above protects this run of the
    # generator; the number protects the released figure, which a reader meets
    # long after the check ran and otherwise has to take on trust. Two pairing
    # defects reached a rendered strip already (limitation (v)), so "the rows
    # are one scene" is the figure's most load-bearing and least visible claim.
    ref = next(iter(first.values())).astype(float)
    step0_pairing = max(
        round(float(np.abs(im.astype(float) - ref).mean()), 6)
        for im in first.values())
    print(f"[fig15] shared start verified: all arms identical at t={steps[0]}")

    if not have_eef and not no_eef:
        missing = [a for a, t in tracks.items() if t[0] is None]
        die(f"arms {missing} logged no end-effector pose, so panels (b) and "
            f"(c) would cover fewer arms than panel (a). The harness records "
            f"eef_available per episode for exactly this check. Pass --no-eef "
            f"to draw the filmstrip alone, which is a smaller figure and not a "
            f"weaker one -- but say so, rather than losing the panels quietly.")
    if not have_eef:
        print("[fig15] --no-eef: drawing panel (a) only; no pose was logged")

    # ---- geometry -------------------------------------------------------
    # 6.14in, not 6.20: save() trims to the tight bbox and adds 2*pad_inches,
    # and panel (b)'s y-label overhangs the leftmost axes by ~0.06in. Authoring
    # at 6.20 landed the PDF at 456.5pt against a 453.6pt \textwidth, so LaTeX
    # rescaled every captured frame by 0.995 for nothing.
    W = 6.14
    cell = (W - 0.62) / len(steps)         # 0.62in of row labels on the left
    strip_h = 3 * (cell + 0.20) + 0.16     # +0.20 per row for the MOVE caption
    bot_h = 1.30 if have_eef else 0.0
    H = strip_h + bot_h + 0.30
    fig = plt.figure(figsize=(W, H))

    # ---- (a) the filmstrip ----------------------------------------------
    y_top = 1.0 - 0.16 / H
    for r, (arm, label, colour) in enumerate(ARMS):
        row_top = y_top - r * (cell + 0.20) / H
        for c, st in enumerate(steps):
            img, rec = frame(cap_dir, key, arm, st, eps)
            ax = fig.add_axes([(0.62 + c * cell) / W, row_top - cell / H,
                               cell / W, cell / H])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(colour); s.set_linewidth(0.9)
            if img is None:
                ax.text(0.5, 0.5, "no frame", ha="center", va="center",
                        fontsize=FONT_SIZE - 3, color="0.5")
            else:
                ax.imshow(img)
            if r == 0:
                ax.set_title(f"$t={st}$", fontsize=FONT_SIZE - 2, pad=2.5)
            # The MOVE phrase under the cell it produced. Only the CoT arms get
            # one: the no-CoT arm reads no reasoning, so there is no phrase to
            # print, and an empty strip under that row says so.
            if arm != "nocot":
                ax.text(0.5, -0.055, short_move(rec and rec.get("move")),
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=FONT_SIZE - 4.2, color=colour, clip_on=False)
        fig.text(0.60 / W, row_top - 0.5 * cell / H, label, ha="right",
                 va="center", fontsize=FONT_SIZE - 2, color=colour,
                 linespacing=1.15)

    fig.text(0.02 / W, y_top + 0.10 / H, "(a)", fontsize=FONT_SIZE - 1,
             fontweight="bold", va="bottom")

    # Panels (b) and (c) exist only if poses were logged. `peak` stays empty in
    # that case rather than being filled with a stand-in, so the facts file says
    # "no measurement" instead of reporting one that was never taken.
    peak = {}
    if have_eef:
        # ---- (b) top-down end-effector path -----------------------------
        axb = fig.add_axes([0.62 / W, 0.42 / H, 1.85 / W, 0.80 / H])
        for arm, label, colour in ARMS:
            _, xyz = tracks[arm]
            axb.plot(xyz[:, 0], xyz[:, 1], color=colour, lw=1.1,
                     label=label.replace("\n", " "))
            axb.plot(xyz[0, 0], xyz[0, 1], "o", color=colour, ms=3.0, mew=0)
        axb.set_xlabel("gripper $x$ (m)", fontsize=FONT_SIZE - 2, labelpad=1.5)
        axb.set_ylabel("$y$ (m)", fontsize=FONT_SIZE - 2, labelpad=1.5)
        axb.tick_params(labelsize=FONT_SIZE - 4, pad=1.5)
        # matplotlib here is mathtext, not LaTeX (text.usetex is False in
        # paper_plot_style), so a "\," thin space prints its own backslash.
        axb.set_title("top-down path ($\\bullet$ = shared start)",
                      fontsize=FONT_SIZE - 2, pad=2.5)
        fig.text(0.02 / W, (0.42 + 0.80) / H, "(b)", fontsize=FONT_SIZE - 1,
                 fontweight="bold", va="bottom")

        # ---- (c) distance from the clean-CoT arm -------------------------
        # Defined at every step whether or not any arm succeeds, which is the
        # point: DSR is undefined here (SR(cot_clean) = 0), and this is not.
        axc = fig.add_axes([(0.62 + 2.35) / W, 0.42 / H, 2.60 / W, 0.80 / H])
        s_cl, xyz_cl = tracks["cot_clean"]
        for arm, label, colour in ARMS:
            if arm == "cot_clean":
                continue
            s_a, xyz_a = tracks[arm]
            # Paired on step, not on index: the arms terminate at different
            # steps, and comparing the k-th record of each would silently
            # compare two different instants.
            common = np.intersect1d(s_a, s_cl)
            d = np.linalg.norm(xyz_a[np.isin(s_a, common)]
                               - xyz_cl[np.isin(s_cl, common)], axis=1)
            axc.plot(common, d * 100.0, color=colour, lw=1.1,
                     label=label.replace("\n", " "))
            peak[arm] = float(d.max()) * 100.0
        axc.axhline(0.0, color="0.7", lw=0.6, zorder=0)
        # Where the strip's columns fall on this curve. Without these the two
        # halves of the figure are two unrelated pictures, and a reader cannot
        # tell whether a pair of near-identical cells sits in a stall or in a
        # stretch the sampling skipped over.
        for st in steps:
            axc.axvline(st, color="0.85", lw=0.5, zorder=0)
        axc.set_xlabel("rollout step (grey rules = filmstrip columns)",
                       fontsize=FONT_SIZE - 2, labelpad=1.5)
        axc.set_ylabel("distance from\nown-CoT arm (cm)",
                       fontsize=FONT_SIZE - 2, labelpad=1.5, linespacing=1.1)
        axc.tick_params(labelsize=FONT_SIZE - 4, pad=1.5)
        # `best` rather than a fixed corner: the curve's shape is whatever the
        # rollout did, and a legend pinned to a corner it happens to fill would
        # hide the divergence this panel exists to show.
        axc.legend(loc="best", fontsize=FONT_SIZE - 4, frameon=False,
                   handlelength=1.3, borderaxespad=0.2, labelspacing=0.25)
        fig.text((0.62 + 2.30) / W, (0.42 + 0.80) / H, "(c)",
                 fontsize=FONT_SIZE - 1, fontweight="bold", va="bottom")

    save(fig, "fig15_rollout_filmstrip")

    # Everything the caption needs to quote, printed so it is copied from a
    # measurement rather than recalled. The audit reads these back out of the
    # same artifact.
    facts = {
        "capture_dir": cap_dir,
        "suite": rep.get("config", {}).get("suite"),
        "task": eps["cot_clean"].get("task"),
        "task_idx": key[0], "episode": key[1],
        "n_filmed_episodes": len(all_keys),
        "cot_refresh_steps": eps["cot_clean"].get("cot_refresh_steps"),
        "columns_at_steps": steps,
        "steps_per_arm": {a: int(eps[a]["steps"]) for a, _, _ in ARMS},
        "success_per_arm": {a: bool(eps[a]["success"]) for a, _, _ in ARMS},
        "n_edit_skipped_flip": eps["cot_direction_flip"].get("n_edit_skipped"),
        # Which version of the figure this is. Without it, an empty
        # peak_cm_from_clean is ambiguous between "the arms never separated"
        # and "no pose was ever logged" -- opposite readings of the same file.
        "eef_logged": bool(have_eef),
        # The pairing measurement, so the audit can assert the released strip's
        # rows were one scene without loading the PNGs itself. Must be 0.0.
        "step0_pairing_max_mean_abs_pixel": step0_pairing,
        # Between-column change WITHIN each arm, so the caption can state the
        # stall rather than let a reader read two near-identical cells as a
        # broken figure. See column_motion().
        "column_change_mean_abs_pixel": {
            a: column_motion(cap_dir, key, a, steps, eps) for a, _, _ in ARMS},
        "peak_cm_from_clean": {k: round(v, 2) for k, v in peak.items()},
    }
    p = os.path.join(cap_dir, "fig15_facts.json")
    with open(p, "w") as fh:
        json.dump(facts, fh, indent=2)
    print(json.dumps(facts, indent=2))
    print(f"[fig15] caption facts -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
