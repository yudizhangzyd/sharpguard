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
    a1 = replay(env, m, snap, acts)
    a2 = replay(env, m, snap, acts)

    # The trajectories, compared at every step rather than only at the end: a
    # divergence that later re-converges is still a divergence, and the strip
    # samples intermediate steps.
    dq = [float(np.abs(x - y).max()) for x, y in zip(a1["qpos"], a2["qpos"])]
    have_frames = all(f is not None for f in a1["frames"] + a2["frames"])
    dpix = ([float(np.abs(x.astype(np.int32) - y.astype(np.int32)).max())
             for x, y in zip(a1["frames"], a2["frames"])] if have_frames else [])

    rep = {
        "suite": suite, "task": task.language, "n_steps": n_steps,
        "arm_rewind_channels": a1["reset"],
        # The claim. Two identical prompts, two identical trajectories.
        "max_abs_qpos_diff_over_all_steps": max(dq) if dq else None,
        "max_abs_pixel_diff_over_all_steps": max(dpix) if dpix else None,
        "identical_qpos": bool(dq) and max(dq) == 0.0,
        "identical_frames": bool(dpix) and max(dpix) == 0.0,
        "first_step_qpos_diff": dq[0] if dq else None,
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
              f"(max |dqpos| {rep['max_abs_qpos_diff_over_all_steps']}). The "
              f"arms are not paired beyond step 0.", file=sys.stderr)
        return 1
    ch = rep["arm_rewind_channels"]
    if not ch.get("controller") or not ch.get("warmstart"):
        print(f"[probe] FAIL: a carry-over channel was never reached: {ch}. "
              f"The pairing held here, but not for the stated reason.",
              file=sys.stderr)
        return 1
    print("[probe] PASS: identical prompts give identical trajectories, and "
          "both carry-over channels were reset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
