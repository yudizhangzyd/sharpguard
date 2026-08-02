#!/usr/bin/env python3
"""Offline checks on the rollout harness's arm/refresh logic and the norm-stats
round trip (no GPU, no simulator, no network).

Why these four things and not others. Each guards a defect that would be
invisible in the report it produces:

  * `select_arms` -- `--arms` was added so the in-suite precondition (#10c) can
    be measured without paying for the edit arms, and a clean-CoT episode costs
    ~63 min at the per-step protocol. A dropped-instead-of-fatal typo would spend
    a 7 h budget on a set nobody asked for, and the report would look exactly
    like a run configured that way.
  * `should_refresh_cot` -- k=1 must reduce EXACTLY to the per-step protocol bolt
    nskmsunnpb ran, or #10c and #10b are not the two protocols they are described
    as. The ceil(n/k) count is what the report publishes as n_cot_generated,
    which is a reader's only handle on which protocol produced an SR.
  * `_summarize`'s precondition_note -- three states, not two. "cot_clean never
    succeeded" and "cot_clean was never attempted" both leave DSR undefined, and
    conflating them would let a two-arm SR run be read as a null result about CoT
    causality.
  * `normalize_action` / `unnormalize_action` -- exact inverses on the unmasked
    dimensions. The attention probe teacher-forces action tokens through the
    forward direction, and the first version quantized the raw action directly,
    i.e. off the token grid. A round trip that is not the identity means the
    tokens being attributed are not the tokens the model would emit.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHECKS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(cond), detail))


def load(name: str):
    """Import an experiments/ module by path.

    Both modules under test import numpy only at module scope (the torch and
    LIBERO imports live inside run()), which is what makes this suite runnable in
    the CI job that installs numpy alone.
    """
    path = ROOT / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_{name}_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# select_arms
# --------------------------------------------------------------------------

def test_select_arms(m) -> None:
    fams = ["direction_flip", "paraphrase_null"]

    default = m.select_arms(fams, "")
    check("arms: empty spec keeps every arm, in order",
          [n for n, _ in default] ==
          ["nocot", "cot_clean", "cot_direction_flip", "cot_paraphrase_null"])
    check("arms: the family travels with the arm, not just its name",
          dict(default)["cot_direction_flip"] == "direction_flip" and
          dict(default)["nocot"] is None)

    pre = m.select_arms(fams, "nocot,cot_clean")
    check("arms: the #10c precondition subset resolves to exactly two arms",
          [n for n, _ in pre] == ["nocot", "cot_clean"])
    check("arms: selection preserves canonical order, not spec order",
          [n for n, _ in m.select_arms(fams, "cot_clean,nocot")] ==
          ["nocot", "cot_clean"],
          "the report's arms_run must not depend on how the flag was typed")
    check("arms: whitespace in the spec is tolerated",
          [n for n, _ in m.select_arms(fams, " nocot , cot_clean ")] ==
          ["nocot", "cot_clean"])

    # The defect this raise exists to prevent: a plausible-looking name that is
    # not an arm of THIS run, because it is not in --families.
    for bad in ("cot_verb_swap", "cotclean", "cot_direction-flip", "clean"):
        raised = False
        try:
            m.select_arms(fams, f"nocot,{bad}")
        except ValueError as e:
            raised = bad in str(e) and "Available" in str(e)
        check(f"arms: unknown name {bad!r} is fatal and names itself", raised,
              "dropping it silently would run a subset nobody asked for")

    # An edit arm IS selectable when its family is present -- the check above
    # must not have made the whole flag unusable for edit runs.
    ok = m.select_arms(fams, "cot_clean,cot_direction_flip")
    check("arms: an edit arm is selectable when --families has it",
          [n for n, _ in ok] == ["cot_clean", "cot_direction_flip"])
    check("arms: no families still yields the two CoT-independent arms",
          [n for n, _ in m.select_arms([], "")] == ["nocot", "cot_clean"])


# --------------------------------------------------------------------------
# should_refresh_cot
# --------------------------------------------------------------------------

def n_generations(m, n_steps: int, k: int) -> int:
    """Replay the loop's own gating over n steps, as run_arm does."""
    cached = False
    n = 0
    for step in range(n_steps):
        if m.should_refresh_cot(step, k, cached):
            n += 1
            cached = True
    return n


def test_refresh_gate(m) -> None:
    check("refresh: k=1 regenerates at every step",
          all(m.should_refresh_cot(s, 1, True) for s in range(50)),
          "k=1 must reproduce bolt nskmsunnpb's per-step protocol exactly")
    check("refresh: k=1 over 400 steps is 400 generations",
          n_generations(m, 400, 1) == 400)

    # ceil(n/k), which is what #10c's economics were computed from: 400/25 = 16
    # generations per episode instead of 400.
    for n_steps, k in ((400, 25), (400, 1), (400, 400), (17, 5), (1, 25), (0, 25)):
        expected = -(-n_steps // k)
        check(f"refresh: {n_steps} steps at k={k} generates ceil = {expected}",
              n_generations(m, n_steps, k) == expected)

    check("refresh: the first step always generates, cache or not",
          m.should_refresh_cot(0, 25, False) and m.should_refresh_cot(0, 25, True))
    check("refresh: no cache forces a generation mid-episode",
          m.should_refresh_cot(7, 25, False),
          "an unstructured CoT leaves no cached prefix; the next step must "
          "generate rather than reuse a prefix that does not exist")
    check("refresh: mid-interval steps reuse the cached prefix",
          not any(m.should_refresh_cot(s, 25, True) for s in range(1, 25)))
    check("refresh: k<=0 degrades to per-step rather than dividing by zero",
          all(m.should_refresh_cot(s, 0, True) for s in range(5)) and
          all(m.should_refresh_cot(s, -3, True) for s in range(5)))


# --------------------------------------------------------------------------
# precondition_note: three states
# --------------------------------------------------------------------------

def _args(**kw):
    ns = types.SimpleNamespace(suite="libero_90", families="direction_flip",
                              cot_refresh_steps=25, arms="")
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _ep(arm, ti, ok, **kw):
    d = {"arm": arm, "task_idx": ti, "episode": 0, "success": ok, "steps": 400,
         "n_cot_generated": 16}
    d.update(kw)
    return d


def test_precondition_three_states(m) -> None:
    two_arms = [("nocot", None), ("cot_clean", None)]
    edit_arms = two_arms + [("cot_direction_flip", "direction_flip")]

    # State 1: cot_clean succeeded -> DSR is defined.
    r = m._summarize(_args(), edit_arms,
                     [_ep("cot_clean", 0, True), _ep("cot_clean", 1, False),
                      _ep("cot_direction_flip", 0, False),
                      _ep("cot_direction_flip", 1, False),
                      _ep("nocot", 0, True)], "complete")
    check("note: a success makes the precondition met", r["precondition_met"])
    check("note: met -> the note says the drop is attributable",
          "attributable" in r["precondition_note"])
    check("note: met -> DSR is computed for the edit arm",
          r["delta_sr_vs_cot_clean"]["cot_direction_flip"]["delta_sr"] == 0.5)
    check("summary: arms_run records what ran, not what was configured",
          r["arms_run"] == ["nocot", "cot_clean", "cot_direction_flip"])

    # State 2: cot_clean ran and never succeeded -> undefined, NOT a null.
    r2 = m._summarize(_args(), edit_arms,
                      [_ep("cot_clean", 0, False), _ep("cot_clean", 1, False),
                       _ep("cot_direction_flip", 0, False)], "complete")
    check("note: cot_clean at 0 successes leaves the precondition unmet",
          not r2["precondition_met"])
    check("note: unmet-but-attempted says 'NEVER succeeds' and names norm_stats",
          "NEVER succeeds" in r2["precondition_note"] and
          "norm_stats" in r2["precondition_note"])
    check("note: unmet -> no DSR table is emitted at all",
          r2["delta_sr_vs_cot_clean"] == {},
          "a table of 0.00 deltas reads like a null result")

    # State 3: cot_clean was never an arm -> makes no claim either way. This is
    # the state #10c's own report will be in if it finds no successes, and the
    # one that must not be readable as a null.
    r3 = m._summarize(_args(arms="nocot"), [("nocot", None)],
                      [_ep("nocot", 0, True), _ep("nocot", 1, False)], "complete")
    check("note: cot_clean absent -> not met (there is nothing to meet)",
          not r3["precondition_met"])
    check("note: absent -> the note says SR only, and refuses a null reading",
          "makes no claim about the precondition" in r3["precondition_note"] and
          "cannot be" in r3["precondition_note"] and
          "no clean-CoT episode was attempted" in r3["precondition_note"])
    check("note: the three states produce three distinct notes",
          len({r["precondition_note"], r2["precondition_note"],
               r3["precondition_note"]}) == 3)

    # The protocol record. Without these two numbers side by side, #10b and #10c
    # are indistinguishable in the artifact.
    a = r["by_arm"]["cot_clean"]
    check("summary: per-arm n_cot_generated and n_steps are both reported",
          a["n_cot_generated"] == 32 and a["n_steps"] == 800)
    check("summary: cot_refresh_steps reaches the report's config",
          r["config"]["cot_refresh_steps"] == 25)
    check("summary: SR and its Wilson interval are per-arm",
          a["sr"] == 0.5 and len(a["wilson95"]) == 2 and
          a["wilson95"][0] < 0.5 < a["wilson95"][1])

    # A 0/N arm must publish an interval, since the whole point of #10c is the
    # UPPER bound on a zero.
    z = m._summarize(_args(), two_arms, [_ep("cot_clean", i, False)
                                         for i in range(40)], "complete")
    zc = z["by_arm"]["cot_clean"]
    check("summary: 0/40 reports sr=0 with a nonzero Wilson upper bound",
          zc["sr"] == 0.0 and 0.0 < zc["wilson95"][1] < 0.15,
          "0/40 bounds the clean SR near 8.8%; that bound IS the deliverable")
    check("summary: an arm with no episodes reports sr=None, not 0.0",
          z["by_arm"]["nocot"]["sr"] is None,
          "0.0 would claim a measurement that was never made")

    # Errored episodes must not be counted as failures: that would manufacture
    # the zero this run exists to test for.
    e = m._summarize(_args(), two_arms,
                     [_ep("cot_clean", 0, True),
                      {"arm": "cot_clean", "task_idx": 1, "episode": 0,
                       "error": "boom"}], "complete")
    check("summary: errored episodes are excluded from n, not scored as failures",
          e["by_arm"]["cot_clean"]["n"] == 1 and
          e["by_arm"]["cot_clean"]["sr"] == 1.0)


# --------------------------------------------------------------------------
# norm-stats round trip
# --------------------------------------------------------------------------

def test_norm_roundtrip(a) -> None:
    rng = np.random.default_rng(0)
    q01 = np.array([-1.0, -0.5, -2.0, -0.1, -0.1, -0.1, 0.0])
    q99 = np.array([1.0, 0.5, 2.0, 0.1, 0.1, 0.1, 1.0])
    mask = np.array([True] * 6 + [False])

    # normalize(unnormalize(x)) == x on the token grid, which is where the
    # teacher-forced tokens come from.
    for _ in range(200):
        x = rng.uniform(-1.0, 1.0, size=7)
        back = a.normalize_action(a.unnormalize_action(x, q01, q99, mask),
                                  q01, q99, mask)
        if not np.allclose(back, x, atol=1e-12):
            check("norm: normalize inverts unnormalize on [-1,1]", False)
            break
    else:
        check("norm: normalize inverts unnormalize on [-1,1]", True)

    # And the other direction, for raw actions inside the quantile box.
    for _ in range(200):
        raw = rng.uniform(q01, q99)
        back = a.unnormalize_action(a.normalize_action(raw, q01, q99, mask),
                                    q01, q99, mask)
        if not np.allclose(back[:6], raw[:6], atol=1e-10):
            check("norm: unnormalize inverts normalize inside the quantile box",
                  False)
            break
    else:
        check("norm: unnormalize inverts normalize inside the quantile box", True)

    check("norm: the endpoints map to the grid's endpoints",
          np.allclose(a.normalize_action(q01, q01, q99, mask)[:6], -1.0) and
          np.allclose(a.normalize_action(q99, q01, q99, mask)[:6], 1.0))
    check("norm: the midpoint maps to 0",
          np.allclose(a.normalize_action(0.5 * (q01 + q99), q01, q99, mask)[:6],
                      0.0))
    # The gripper. OpenVLA passes masked dims through UNCHANGED in both
    # directions; rescaling it is a physical-scale error with no visible symptom.
    check("norm: masked dims pass through unnormalize untouched",
          a.unnormalize_action(np.full(7, 0.37), q01, q99, mask)[6] == 0.37)
    check("norm: masked dims pass through normalize untouched",
          a.normalize_action(np.full(7, 0.37), q01, q99, mask)[6] == 0.37)
    check("norm: normalize clips to the grid, unnormalize does not",
          a.normalize_action(q99 + 5.0, q01, q99, mask)[0] == 1.0 and
          a.unnormalize_action(np.full(7, 3.0), q01, q99, mask)[0] > 1.0,
          "out-of-box raw actions must land ON the grid; a token index off the "
          "grid is not a token the model could emit")
    # A degenerate dimension (q01 == q99) would divide by zero and produce nan,
    # which propagates silently through an L1 error.
    d01 = np.array([0.0, -1.0]); d99 = np.array([0.0, 1.0])
    dm = np.array([True, True])
    out = a.normalize_action(np.array([0.0, 0.5]), d01, d99, dm)
    check("norm: a zero-span dimension yields a finite value, not nan",
          np.all(np.isfinite(out)))


# --------------------------------------------------------------------------
# _capture_for: which episodes get filmed
# --------------------------------------------------------------------------

def test_capture_selection(m) -> None:
    """The capture flag decides where GPU-hours go, so its off-state matters.

    Frames are written inside the step loop. A spec that accidentally matched
    every episode would put a PNG write into a run whose budget was costed
    without one; a spec that silently matched nothing would produce a job that
    completes, reports SR, and yields no figure -- discovered only after the
    hours are spent. Both are checked here rather than in the simulator.
    """
    from pathlib import Path as P
    out = P("/tmp/does_not_need_to_exist")

    off = types.SimpleNamespace(capture_episodes="", capture_every=10)
    check("capture: the default captures nothing",
          all(m._capture_for(off, out, t, e) is None
              for t in range(3) for e in range(3)),
          "an SR run must not pay for I/O the figure runs need")
    check("capture: whitespace-only spec is also off",
          m._capture_for(types.SimpleNamespace(capture_episodes="  ",
                                               capture_every=10), out, 0, 0) is None)

    on = types.SimpleNamespace(capture_episodes="0:0,3:1", capture_every=8)
    check("capture: a named episode is selected",
          m._capture_for(on, out, 0, 0) is not None and
          m._capture_for(on, out, 3, 1) is not None)
    check("capture: an unnamed episode is not",
          m._capture_for(on, out, 0, 1) is None and
          m._capture_for(on, out, 3, 0) is None and
          m._capture_for(on, out, 1, 0) is None,
          "task and episode must both match; matching either alone would film "
          "a whole task at ~1 PNG per 8 steps per arm")

    c = m._capture_for(on, out, 3, 1)
    check("capture: the directory is per (task, episode), not per arm",
          c["dir"] == out / "frames" / "t3_ep1",
          "arms share it on purpose -- the filmstrip compares arms on one "
          "init state, and the filenames are already arm-prefixed")
    check("capture: --capture-every reaches the spec", c["every"] == 8)
    check("capture: every<=0 degrades to 1 rather than dividing by zero",
          m._capture_for(types.SimpleNamespace(capture_episodes="0:0",
                                               capture_every=0),
                         out, 0, 0)["every"] == 1)


def test_eef_reader(m) -> None:
    """A missing pose must read as None, never as zeros."""
    check("eef: a real pose round-trips as floats",
          m._eef({"robot0_eef_pos": np.array([0.1, -0.2, 1.0])}) ==
          [0.1, -0.2, 1.0])
    check("eef: an env without the low-dim observable yields None",
          m._eef({"agentview_image": np.zeros((2, 2, 3))}) is None,
          "zero-filling would draw a flat line at the origin, which reads as "
          "a policy that never moved -- a claim, and the wrong one")
    check("eef: a non-dict observation yields None", m._eef(None) is None)


def test_scene_seed(m) -> None:
    """Every arm of one episode must draw the same fixture placement."""
    def placements(seed_expr):
        """What robosuite's sampler would draw, standing in np.random for it."""
        out = []
        for _ in range(3):                       # three arms, three fresh envs
            m._seed_scene(seed_expr)
            out.append(float(np.random.uniform()))
        return out

    p = placements(7)
    check("scene: three arms seeded with one episode's seed draw the same "
          "fixture placement", p[0] == p[1] == p[2],
          "set_init_state restores qpos and a welded fixture's pose is not in "
          "qpos, so without this the arms are paired on the robot and "
          "mispaired on the furniture -- measured as a 3-pixel cabinet shift "
          "in bolt h8xzmqnhgg")
    check("scene: a different episode is still a different scene",
          placements(8)[0] != p[0],
          "seeding must pair the arms, not collapse the episodes into one "
          "scene filmed three times")


def test_start_mismatch_lives_elsewhere() -> None:
    """The pixel check is in tests/test_fig15_filmstrip.py, deliberately.

    This suite is the gate that runs on the pod before a rollout budget is
    spent, so it must import numpy and nothing else: the figure generator pulls
    in matplotlib, which setup-openvla.sh installs only on the branch that
    builds LIBERO from source. An ImportError here would exit 7 and cancel a
    20 h job to protect a figure that is drawn afterwards, on a laptop.
    """
    check("start-mismatch check exists, in the figure's own test file",
          (ROOT / "tests" / "test_fig15_filmstrip.py").exists())


def main() -> int:
    r = load("cotfaith_rollout_edit")
    test_select_arms(r)
    test_refresh_gate(r)
    test_precondition_three_states(r)
    test_capture_selection(r)
    test_eef_reader(r)
    test_scene_seed(r)
    test_start_mismatch_lives_elsewhere()
    test_norm_roundtrip(load("cotfaith_auroc"))

    bad = [c for c in CHECKS if not c[1]]
    for name, ok, detail in CHECKS:
        if not ok:
            print(f"FAIL {name}" + (f"  [{detail}]" if detail else ""))
    print(f"{len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
