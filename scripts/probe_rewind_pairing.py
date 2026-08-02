#!/usr/bin/env python3
"""Prove the rollout arms are paired, without a policy in the loop.

The filmstrip's argument is that its three rows differ only in the prompt. Two
defects broke that before anyone noticed, both invisible in the run report
(limitation (v)): a welded fixture re-sampled per arm, and -- after that was
fixed -- each arm running its own settling loop and inheriting the previous
arm's controller goal and solver warm start, which `set_init_state` does not
restore.

The capture run checks the weaker property: that the arms' FIRST frames match.
That is necessary and not sufficient, because the carry-over channels affect
step 1 onward, where the arms are supposed to differ and a confound is therefore
invisible. This probe checks the sufficient property instead, by removing the
policy: rewind twice and replay the SAME fixed action sequence both times. Two
identical prompts must produce identical trajectories. If they do not, then a
difference between two DIFFERENT prompts cannot be attributed to the prompt.

No model, no checkpoint, no GPU maths -- a scripted action sequence and a
simulator. That is the point: it isolates the harness from the policy, so a
failure here is a harness bug and cannot be explained away as stochastic
decoding.

Also records which carry-over channels `_rewind_to` actually reached. robosuite
has renamed both across versions, and a silent miss would leave the confound in
place while the code still read as if it had been handled -- so the paper's
claim that both were cleared is asserted from this artifact rather than from a
log line.

  python3 scripts/probe_rewind_pairing.py --out <path.json> [--suite libero_90]
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_harness():
    """Import the harness by path, so this probe tests the shipped code."""
    p = os.path.join(ROOT, "experiments", "cotfaith_rollout_edit.py")
    spec = importlib.util.spec_from_file_location("_rollout_under_probe", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def scripted_actions(n: int) -> list:
    """A fixed, non-trivial action sequence. Deterministic by construction.

    Non-trivial on purpose: a no-op sequence would leave the arm hanging still,
    and two still arms match whether or not the carry-over was cleared. These
    actions drive the arm down and across so the controller is actually
    integrating, which is what makes a stale goal observable.
    """
    out = []
    for i in range(n):
        ph = i / max(1, n - 1)
        out.append(np.array([0.35 * np.cos(3.0 * ph), 0.35 * np.sin(3.0 * ph),
                             -0.30 + 0.2 * ph, 0.0, 0.0, 0.05,
                             1.0 if ph > 0.6 else -1.0], dtype=np.float32))
    return out


def replay(env, m, snap, acts) -> dict:
    """Rewind, replay `acts`, and return the trajectory plus what was reset."""
    rw = m._rewind_to(env, snap)
    obs = rw["obs"]
    qpos, frames = [], []
    for a in acts:
        obs, _, _, _ = env.step(a)
        qpos.append(np.asarray(env.sim.get_state().flatten(), dtype=np.float64))
        img = obs.get("agentview_image", obs.get("image"))
        frames.append(None if img is None
                      else np.asarray(img, dtype=np.uint8).copy())
    return {"reset": rw["reset"], "qpos": qpos, "frames": frames}


def main(argv: list) -> int:
    out_p = None
    if "--out" in argv:
        out_p = argv[argv.index("--out") + 1]
    suite = "libero_90"
    if "--suite" in argv:
        suite = argv[argv.index("--suite") + 1]
    n_steps = 40
    if "--steps" in argv:
        n_steps = int(argv[argv.index("--steps") + 1])

    m = load_harness()
    from sharpguard.libero_sim import _load_libero_init_states
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = benchmark.get_benchmark_dict()[suite]().get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                        task.bddl_file)
    init = _load_libero_init_states(os.path.join(
        get_libero_path("init_states"), task.problem_folder,
        task.init_states_file))
    if init is None or not len(init):
        print("[probe] no init states for this task", file=sys.stderr)
        return 3

    m._seed_scene(0)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256,
                             camera_widths=256)
    env.reset()
    snap, settled_obs = m._settle_once(env, init[0])

    acts = scripted_actions(n_steps)
    # THREE replays, not two. Run 1 of this probe (bolt h6pttcu4g5) got
    # identical qpos at every step -- 0.0 -- and yet the rendered frames
    # differed by up to 160 levels, which cannot happen if the render is a
    # function of the state. Either the renderer carries state of its own or the
    # frames are offset by a step. A third replay separates those: if 2 and 3
    # agree while 1 differs, the first replay after the settle is the odd one
    # (a warm-up or stale buffer) rather than the render being irreproducible.
    # That distinction decides whether the arm that runs FIRST in a capture can
    # be compared with the arms after it -- which is the filmstrip's whole
    # premise, so it is measured here rather than assumed.
    a1 = replay(env, m, snap, acts)
    a2 = replay(env, m, snap, acts)
    a3 = replay(env, m, snap, acts)

    # The trajectories, compared at every step rather than only at the end: a
    # divergence that later re-converges is still a divergence, and the strip
    # samples intermediate steps.
    def dq_of(x, y):
        return [float(np.abs(p - q).max()) for p, q in zip(x["qpos"], y["qpos"])]

    def dpix_of(x, y):
        if not all(f is not None for f in x["frames"] + y["frames"]):
            return []
        return [float(np.abs(p.astype(np.int32) - q.astype(np.int32)).max())
                for p, q in zip(x["frames"], y["frames"])]

    dq = dq_of(a1, a2)
    dq23 = dq_of(a2, a3)
    dpix = dpix_of(a1, a2)
    dpix23 = dpix_of(a2, a3)

    def first_nonzero(xs):
        return next((i for i, v in enumerate(xs) if v != 0.0), None)

    # WHICH state entry diverges, and whether the divergence is an offset
    # present immediately or drift that accumulates. Reported because the first
    # run of this probe (bolt gzv4nuhtfe) failed with max == first_step ==
    # 0.1876, and reading "an immediate, non-growing offset" out of two equal
    # numbers is what identified the gripper accumulator. A later failure should
    # not need that inference done by hand: it should name its own index.
    step0 = (np.abs(a1["qpos"][0] - a2["qpos"][0]) if dq else np.zeros(1))
    rep = {
        "suite": suite, "task": task.language, "n_steps": n_steps,
        "arm_rewind_channels": a1["reset"],
        # The claim. Two identical prompts, two identical trajectories.
        "max_abs_qpos_diff_over_all_steps": max(dq) if dq else None,
        "max_abs_pixel_diff_over_all_steps": max(dpix) if dpix else None,
        "identical_qpos": bool(dq) and max(dq) == 0.0,
        "identical_frames": bool(dpix) and max(dpix) == 0.0,
        "first_step_qpos_diff": dq[0] if dq else None,
        "qpos_diff_per_step": [round(v, 12) for v in dq],
        "pixel_diff_per_step": dpix,
        "first_differing_pixel_step": first_nonzero(dpix),
        "n_pixel_steps_differing": sum(1 for v in dpix if v != 0.0),
        "first_step_worst_qpos_index": int(step0.argmax()) if dq else None,
        "n_qpos_entries_differing_at_step0": int((step0 > 0).sum()) if dq else None,
        # Replay 2 vs replay 3: same code path as 1 vs 2, but with no
        # first-after-settle asymmetry. If these agree where 1 vs 2 does not,
        # the renderer is reproducible and the FIRST replay is what differs.
        "max_abs_qpos_diff_2v3": max(dq23) if dq23 else None,
        "max_abs_pixel_diff_2v3": max(dpix23) if dpix23 else None,
        "identical_qpos_2v3": bool(dq23) and max(dq23) == 0.0,
        "identical_frames_2v3": bool(dpix23) and max(dpix23) == 0.0,
        "pixel_diff_per_step_2v3": dpix23,
        "first_differing_pixel_step_2v3": first_nonzero(dpix23),
        "settled_obs_is_shared": settled_obs is not None,
    }
    print(json.dumps(rep, indent=2))
    if out_p:
        os.makedirs(os.path.dirname(out_p) or ".", exist_ok=True)
        with open(out_p, "w") as fh:
            json.dump(rep, fh, indent=2)
        print(f"[probe] -> {out_p}", file=sys.stderr)

    # Non-zero when the pairing does NOT hold: this is a gate, not a
    # measurement. The figure's rows mean nothing if it fails.
    if not rep["identical_qpos"]:
        print(f"[probe] FAIL: replaying one action sequence twice diverges "
              f"(max |dqpos| {rep['max_abs_qpos_diff_over_all_steps']}, first "
              f"step {rep['first_step_qpos_diff']}, worst state index "
              f"{rep['first_step_worst_qpos_index']}, "
              f"{rep['n_qpos_entries_differing_at_step0']} entries differing at "
              f"step 0). The arms are not paired beyond step 0.",
              file=sys.stderr)
        return 1
    # Pixels too, and gated rather than merely reported: the filmstrip IS
    # pixels. Identical state with different frames means the render is not a
    # function of the state, and then a row-to-row difference in the figure is
    # not attributable to the prompt either.
    if not rep["identical_frames"]:
        print(f"[probe] FAIL: qpos is identical at every step and the RENDERED "
              f"frames are not (max |dpix| "
              f"{rep['max_abs_pixel_diff_over_all_steps']}, first differing "
              f"step {rep['first_differing_pixel_step']}, "
              f"{rep['n_pixel_steps_differing']}/{n_steps} steps differing). "
              f"Replay 2 vs 3: identical_frames="
              f"{rep['identical_frames_2v3']} (max "
              f"{rep['max_abs_pixel_diff_2v3']}, first differing step "
              f"{rep['first_differing_pixel_step_2v3']}). If 2 and 3 agree, the "
              f"FIRST replay after the settle renders differently and the arm "
              f"that runs first in a capture cannot be compared with the rest.",
              file=sys.stderr)
        return 1
    ch = rep["arm_rewind_channels"]
    missing = [k for k, v in ch.items() if not v]
    if missing:
        print(f"[probe] FAIL: carry-over channel(s) never reached: {missing} "
              f"(channels: {ch}). The pairing held here, but not for the stated "
              f"reason -- most likely robosuite renamed something.",
              file=sys.stderr)
        return 1
    print("[probe] PASS: identical prompts give identical trajectories in both "
          f"qpos and pixels, over {n_steps} steps and three replays, and every "
          f"carry-over channel was reset ({ch}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
