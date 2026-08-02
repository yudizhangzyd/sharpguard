#!/usr/bin/env python3
"""Prove the rollout arms are paired, without a policy in the loop.

The filmstrip's argument is that its three rows differ only in the prompt. Four
defects broke that before anyone noticed, none of them visible in the run report
(limitation (v)): a welded fixture re-sampled per arm; each arm running its own
settling loop and inheriting the previous arm's controller goal and solver warm
start; the gripper's rate-limited action accumulator; and the observation
sampling phase, which decides at which physics substep of a control step the
camera is read. `set_init_state` restores qpos and qvel, and none of the four is
qpos or qvel.

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


# What the renderer actually reads. `sim.get_state().flatten()` is only
# time+qpos+qvel in robosuite's newer binding wrapper, and run 5 (bolt
# e268dqs2t8) showed that is not enough: qpos bit-identical at all 40 steps, the
# renderer bit-exact on one state rendered four times, no frame/state offset in
# +-3, and yet the frames differ inside a box whose complement is bit-identical.
# So something visible is outside the compared state. These are the arrays a
# MuJoCo frame is a function of -- derived pose (which must match if qpos does),
# then the model fields that are mutable at run time and change appearance
# rather than pose. Naming which one differs is the difference between a
# diagnosis and another round of guessing.
DATA_FIELDS = ("time", "qpos", "qvel", "act", "ctrl", "qacc",
               "qacc_warmstart", "xpos", "xquat", "geom_xpos", "geom_xmat",
               "site_xpos", "site_xmat", "cam_xpos", "cam_xmat", "light_xpos",
               "light_xdir", "mocap_pos", "mocap_quat")
MODEL_FIELDS = ("geom_rgba", "geom_size", "geom_pos", "geom_quat",
                "geom_matid", "geom_group", "site_rgba", "site_size",
                "site_pos", "mat_rgba", "mat_emission", "light_pos",
                "light_dir", "light_active", "cam_pos", "cam_quat",
                "cam_fovy", "body_pos", "body_quat")


def render_inputs(env) -> dict:
    """Every render-relevant array, copied. Missing fields are simply absent."""
    out = {}
    for tag, obj, names in (("data", getattr(env.sim, "data", None), DATA_FIELDS),
                            ("model", getattr(env.sim, "model", None),
                             MODEL_FIELDS)):
        for n in names:
            v = getattr(obj, n, None)
            if v is None:
                continue
            try:
                out[f"{tag}.{n}"] = np.array(v, dtype=np.float64, copy=True)
            except (TypeError, ValueError):
                continue
    return out


def sampling_phase(m, env) -> dict:
    """Flat, comparable view of the observation-sampling phase.

    Read before and after each rewind, so the artifact shows the channel BOTH
    carrying over and being restored, rather than asserting the second from the
    absence of a symptom. Scalars only: the observable timers are floats, and a
    cached image array being equal is what the pixel comparison already tests.
    """
    snap = m._snap_sampling(env)
    flat = {}
    for k, v in (snap.get("env") or {}).items():
        if isinstance(v, (bool, int, float)):
            flat[f"env.{k}"] = float(v)
    for name, d in (snap.get("observables") or {}).items():
        for k, v in d.items():
            if isinstance(v, (bool, int, float)):
                flat[f"{name}.{k}"] = float(v)
    return flat


def phase_diff(x: dict, y: dict) -> dict:
    keys = sorted(set(x) | set(y))
    diff = {k: [x.get(k), y.get(k)] for k in keys if x.get(k) != y.get(k)}
    return {"n_fields": len(keys), "n_fields_differing": len(diff),
            "differing": dict(list(diff.items())[:12])}


def replay(env, m, snap, acts) -> dict:
    """Rewind, replay `acts`, and return the trajectory plus what was reset."""
    phase_pre = sampling_phase(m, env)
    rw = m._rewind_to(env, snap)
    phase_post = sampling_phase(m, env)
    obs = rw["obs"]
    qpos, frames, inputs = [], [], []
    for a in acts:
        obs, _, _, _ = env.step(a)
        qpos.append(np.asarray(env.sim.get_state().flatten(), dtype=np.float64))
        inputs.append(render_inputs(env))
        img = obs.get("agentview_image", obs.get("image"))
        frames.append(None if img is None
                      else np.asarray(img, dtype=np.uint8).copy())
    return {"reset": rw["reset"], "qpos": qpos, "frames": frames,
            "inputs": inputs, "phase_pre": phase_pre, "phase_post": phase_post}


def field_diffs(xs: list, ys: list) -> dict:
    """Which render inputs differ between two replays, and where first.

    Reported per field rather than as one number: "the frames differ" was
    already known, and a single scalar over all of them would again be a
    statistic that cannot say WHICH channel carried the difference -- the same
    reason the pixel comparisons here carry a bbox and an inside/outside mean
    instead of a maximum.
    """
    names = sorted(set(xs[0]) | set(ys[0])) if xs and ys else []
    differ, same = {}, []
    for n in names:
        worst, first = 0.0, None
        for i, (a, b) in enumerate(zip(xs, ys)):
            if n not in a or n not in b or a[n].shape != b[n].shape:
                worst, first = float("inf"), i
                break
            d = float(np.abs(a[n] - b[n]).max()) if a[n].size else 0.0
            if d > 0.0 and first is None:
                first = i
            worst = max(worst, d)
        if worst > 0.0:
            differ[n] = {"max_abs": worst, "first_step": first}
        else:
            same.append(n)
    return {"fields_that_differ": differ, "n_fields_identical": len(same),
            "fields_identical": same}


def _renderer(env):
    """The env's re-render entry point, or None. robosuite renamed it."""
    base = getattr(env, "env", env)
    return getattr(base, "_get_observations", None) or \
        getattr(base, "_get_observation", None)


def _render_once(getter):
    """One frame from the CURRENT sim data, without stepping it."""
    try:                                  # robosuite caches obs unless forced
        o = getter(force_update=True)
    except TypeError:
        o = getter()
    return o.get("agentview_image", o.get("image")) if o else None


def render_noise_floor(env, m, snap, k: int = 4) -> dict:
    """Render ONE state k times without stepping: the renderer's own floor.

    Runs 2 and 3 of this probe (bolt h6pttcu4g5, qyh54st578) found qpos
    bit-identical at all 40 steps and the frames differing at all 40, by 60-160
    levels, in both 1-vs-2 and 2-vs-3. That rules out a warm-up (which would hit
    the first replay only) and a one-frame render lag (step 0 only), and leaves
    two possibilities that the max-over-the-image statistic cannot tell apart: a
    real difference in what is drawn, or a rasterizer that is not bit-exact
    between calls, where a handful of anti-aliased edge pixels saturate a `max`
    while the mean stays near zero.

    This measures which. If the same state rendered twice in a row is not
    bit-equal, then exact pixel equality is the wrong gate for pairing and the
    honest test is the replay difference against THIS floor. If it IS bit-equal,
    the replay difference is real and something outside qpos is being drawn.
    """
    m._rewind_to(env, snap)
    getter = _renderer(env)
    if getter is None:
        return {"available": False,
                "why": "no _get_observations on the env; cannot re-render one "
                       "state without stepping it"}
    imgs = []
    for _ in range(k):
        img = _render_once(getter)
        if img is None:
            return {"available": False, "why": "the re-render carried no image"}
        imgs.append(np.asarray(img, dtype=np.int32).copy())
    ref = imgs[0]
    d = [np.abs(im - ref) for im in imgs[1:]]
    return {"available": True, "n_renders": k,
            "max_abs": max(float(x.max()) for x in d),
            "mean_abs": max(round(float(x.mean()), 6) for x in d),
            "frac_pixels_differing": max(round(float((x > 0).mean()), 6)
                                         for x in d),
            "bit_identical": all(float(x.max()) == 0.0 for x in d)}


def state_determines_frame(env, m, state, frame) -> dict:
    """Restore ONE recorded state and re-render it: is the frame a function of it?

    This is the fork the array diff cannot close on its own. Set the state that
    replay 1 held at its worst step, re-render, and compare with the frame
    replay 1 actually produced there.

    Bit-equal means the flattened state DOES determine the frame, so the frames
    were taken at the state that was read and the difference is elsewhere.

    Not bit-equal means the frame is not a render of the end-of-step state at
    all -- robosuite samples camera observables inside the physics substep loop,
    on each observable's own timer -- so pairing the rows needs that timer
    rewound too, not just qpos. Either answer is a decision; the difference
    alone was not.
    """
    getter = _renderer(env)
    if getter is None or frame is None:
        return {"available": False, "why": "no re-render entry point or frame"}
    m._rewind_to(env, {"state": state, "grippers": []})
    img = _render_once(getter)
    if img is None:
        return {"available": False, "why": "the re-render carried no image"}
    got = np.asarray(img, dtype=np.uint8)
    st = pix_stats(got, frame)
    return {"available": True, "bit_identical": st["max"] == 0.0,
            "pixels": st, "region": region_stats(got, frame)}


def pix_stats(x, y) -> dict:
    """max, mean and the differing fraction -- the three together, deliberately.

    The pairing diagnostics for defects 1 and 2 argue from mean-inside versus
    mean-outside a bounding box, not from a max, because a max cannot separate
    "one object is somewhere else" from "one pixel is a shade off". The same
    applies here.
    """
    a, b = x.astype(np.int32), y.astype(np.int32)
    d = np.abs(a - b)
    return {"max": float(d.max()), "mean": round(float(d.mean()), 6),
            "frac": round(float((d > 0).mean()), 6)}


def alignment_offset(x, y, window: int = 3) -> dict:
    """Is frame i of one replay bit-equal to frame i+k of the other?

    Run 4 (bolt x8xbskv8tt) left exactly one explanation standing. The renderer
    is bit-exact -- one state rendered four times gave 0.0 max, 0.0 mean, 0.0
    differing pixels -- and the two replays' qpos matched bit-for-bit at all 40
    steps, and yet their frames differed at all 40, by a mean of 0.11-0.20 over
    1.4-2.8 pct of pixels with a max of 60-160. A tiny mean with a large max over
    a small fraction is a silhouette in a slightly different place, not a
    different scene: the shape of one moving arm sampled a step apart.

    So this tests alignment directly rather than by eye. If frame i of replay 1
    is bit-equal to frame i+k of replay 2 for a fixed non-zero k, the frames are
    offset by k renders relative to the states, the pixel difference is
    bookkeeping rather than physics, and it is fixable by capturing the frame at
    the same point in the step. If no k works, the difference is real and
    something visible is not in the flattened state.
    """
    out, offs = [], list(range(-window, window + 1))
    for i in range(len(x)):
        eq = [k for k in offs
              if 0 <= i + k < len(y) and
              np.array_equal(x[i], y[i + k])]
        out.append(eq[0] if eq else None)
    found = [k for k in out if k is not None]
    return {"per_step_offset": out,
            "n_steps_with_an_exact_match": len(found),
            "offsets_seen": sorted(set(found)),
            "constant_offset": (found[0] if found and
                                len(set(found)) == 1 and
                                len(found) == len(x) else None)}


def region_stats(x, y) -> dict:
    """Where the two frames differ: the bbox, and the mean inside vs outside.

    The same statistic the two earlier pairing defects are argued from, for the
    same reason -- a max says how loud the loudest pixel is and nothing about
    whether the difference is one object or the whole scene.
    """
    d = np.abs(x.astype(np.int32) - y.astype(np.int32))
    m = d.max(axis=2) if d.ndim == 3 else d
    ys, xs = np.where(m > 0)
    if not len(ys):
        return {"identical": True}
    r0, r1, c0, c1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    inside = d[r0:r1 + 1, c0:c1 + 1]
    out = d.copy()
    out[r0:r1 + 1, c0:c1 + 1] = 0
    n_out = d.size - inside.size
    return {"identical": False,
            "bbox": {"row0": r0, "row1": r1, "col0": c0, "col1": c1},
            "inside_mean_abs": round(float(inside.mean()), 6),
            "outside_mean_abs": round(float(out.sum() / max(1, n_out)), 6),
            "frac_of_frame_in_bbox": round(inside.size / d.size, 6)}


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
        return [pix_stats(p, q) for p, q in zip(x["frames"], y["frames"])]

    dq = dq_of(a1, a2)
    dq23 = dq_of(a2, a3)
    st12 = dpix_of(a1, a2)
    st23 = dpix_of(a2, a3)
    dpix = [s["max"] for s in st12]
    dpix23 = [s["max"] for s in st23]
    fd12 = field_diffs(a1["inputs"], a2["inputs"])
    fd23 = field_diffs(a2["inputs"], a3["inputs"])
    floor = render_noise_floor(env, m, snap)
    # Both of these are read AFTER the replays and the floor, because each one
    # sets the state and would perturb anything measured afterwards.
    worst_i = int(np.argmax(dpix)) if dpix else None
    sdf = ({"available": False, "why": "no differing step to check"}
           if worst_i is None else
           state_determines_frame(env, m, a1["qpos"][worst_i],
                                  a1["frames"][worst_i]))

    def first_nonzero(xs):
        return next((i for i, v in enumerate(xs) if v != 0.0), None)

    def worst(stats, key):
        return max((s[key] for s in stats), default=None)

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
        # max alone cannot separate "something else is drawn" from "an edge
        # pixel is a shade off", so the mean and the differing fraction travel
        # with it -- the same reason defects 1 and 2 are argued from
        # inside-versus-outside a box rather than from a maximum.
        "pixel_max_worst_step": worst(st12, "max"),
        "pixel_mean_worst_step": worst(st12, "mean"),
        "pixel_frac_worst_step": worst(st12, "frac"),
        "pixel_stats_per_step": st12,
        "render_noise_floor": floor,
        # Which render input differs, if any. If `data.geom_xpos` differs while
        # `data.qpos` does not, the rendered POSE differs and the frame is not
        # taken at the state that was read; if only a `model.*` appearance field
        # differs, nothing moved and something was recoloured; if every field is
        # identical, the difference is below the model and lives in the
        # rasterizer, which is what the noise floor is compared against.
        "render_input_diff_1v2": fd12,
        "render_input_diff_2v3": fd23,
        # The fourth carry-over channel, shown carrying over AND being
        # restored: replay 2 enters the rewind with replay 1's sampling phase
        # (these must differ), and leaves it with the snapshot's (these must
        # not). Without the second half, "restored" would be an inference from
        # the absence of a symptom.
        "sampling_phase_before_rewind_1v2": phase_diff(a1["phase_pre"],
                                                       a2["phase_pre"]),
        "sampling_phase_after_rewind_1v2": phase_diff(a1["phase_post"],
                                                      a2["phase_post"]),
        "sampling_phase_after_rewind_2v3": phase_diff(a2["phase_post"],
                                                      a3["phase_post"]),
        # And the fork that closes it: restore replay 1's state at its worst
        # step and re-render. Bit-equal means the state DOES determine the
        # frame, so the pairing is a capture-order bug and fixable here.
        "state_determines_frame": sdf,
        # Alignment and region, because "the frames differ" is not yet a
        # diagnosis: these say whether the difference is one silhouette a step
        # out of place or the whole scene.
        "frame_alignment_1v2": alignment_offset(a1["frames"], a2["frames"]),
        "frame_alignment_2v3": alignment_offset(a2["frames"], a3["frames"]),
        "region_worst_step_1v2": (
            region_stats(a1["frames"][worst_i], a2["frames"][worst_i])
            if dpix else None),
        "worst_step_index": worst_i,
        "first_differing_pixel_step": first_nonzero(dpix),
        "n_pixel_steps_differing": sum(1 for v in dpix if v != 0.0),
        "first_step_worst_qpos_index": int(step0.argmax()) if dq else None,
        "n_qpos_entries_differing_at_step0": int((step0 > 0).sum()) if dq else None,
        # Replay 2 vs replay 3: same code path as 1 vs 2, but with no
        # first-after-settle asymmetry. If these agree where 1 vs 2 does not,
        # the renderer is reproducible and the FIRST replay is what differs.
        "max_abs_qpos_diff_2v3": max(dq23) if dq23 else None,
        "max_abs_pixel_diff_2v3": max(dpix23) if dpix23 else None,
        "pixel_mean_worst_step_2v3": worst(st23, "mean"),
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
        # The frames themselves, at the worst step. Every earlier defect here
        # was diagnosed by LOOKING at pixels after the scalars had said nothing
        # was wrong, and a summary of a difference is not the difference.
        if dpix and rep["worst_step_index"] is not None:
            try:
                from PIL import Image
                i = rep["worst_step_index"]
                where = os.path.join(os.path.dirname(out_p) or ".", "worst_step")
                os.makedirs(where, exist_ok=True)
                d = np.abs(a1["frames"][i].astype(np.int32) -
                           a2["frames"][i].astype(np.int32)).astype(np.uint8)
                for nm, im in (("replay1", a1["frames"][i]),
                               ("replay2", a2["frames"][i]),
                               ("replay3", a3["frames"][i]),
                               ("absdiff", d),
                               # Amplified, since a mean of 0.2 is invisible on
                               # a 0-255 scale and the point is to be looked at.
                               ("absdiff_x8", np.clip(d.astype(np.int32) * 8,
                                                      0, 255).astype(np.uint8))):
                    Image.fromarray(im).save(
                        os.path.join(where, f"t{i:04d}_{nm}.png"))
                print(f"[probe] worst-step frames -> {where}", file=sys.stderr)
            except Exception as e:                      # noqa: BLE001
                print(f"[probe] could not write worst-step frames: {e}",
                      file=sys.stderr)

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
    # pixels. But gated against the renderer's own floor, not against exact
    # equality -- if one state rendered twice in a row is not bit-equal, then
    # bit-equality between two replays is a test of the rasterizer and not of
    # the pairing, and it would fail forever while telling us nothing.
    fl = rep["render_noise_floor"]
    if not rep["identical_frames"]:
        if not fl.get("available"):
            print(f"[probe] FAIL: qpos is identical at every step and the "
                  f"frames are not (max |dpix| "
                  f"{rep['max_abs_pixel_diff_over_all_steps']}, mean at the "
                  f"worst step {rep['pixel_mean_worst_step']}), and the "
                  f"renderer floor could not be measured "
                  f"({fl.get('why')}), so the two cannot be told apart.",
                  file=sys.stderr)
            return 1
        if fl.get("bit_identical"):
            fields = sorted(rep["render_input_diff_1v2"]["fields_that_differ"])
            sd = rep["state_determines_frame"]
            print(f"[probe] FAIL: one state rendered twice IS bit-identical, so "
                  f"the renderer is deterministic -- and two replays with "
                  f"identical qpos still differ (max "
                  f"{rep['max_abs_pixel_diff_over_all_steps']}, mean "
                  f"{rep['pixel_mean_worst_step']}, "
                  f"{rep['pixel_frac_worst_step']} of pixels, first differing "
                  f"step {rep['first_differing_pixel_step']}). Something "
                  f"outside qpos is being drawn. Render inputs that differ: "
                  f"{fields or 'NONE -- the difference is below the model'} "
                  f"({rep['render_input_diff_1v2']['n_fields_identical']} "
                  f"identical). Restoring the state and re-rendering "
                  f"{'REPRODUCES' if sd.get('bit_identical') else 'does NOT reproduce'}"
                  f" the frame, so the frame "
                  f"{'is' if sd.get('bit_identical') else 'is not'} a function "
                  f"of the state that was read.", file=sys.stderr)
            return 1
        # The renderer itself is not bit-exact. Then the only meaningful
        # question is whether two replays differ by MORE than one state
        # rendered twice does.
        over = (rep["pixel_mean_worst_step"] or 0.0) > fl["mean_abs"] or \
               (rep["pixel_frac_worst_step"] or 0.0) > fl["frac_pixels_differing"]
        if over:
            print(f"[probe] FAIL: two replays differ by more than the "
                  f"renderer's own floor. Replay: mean "
                  f"{rep['pixel_mean_worst_step']}, frac "
                  f"{rep['pixel_frac_worst_step']}. Floor (one state rendered "
                  f"{fl['n_renders']}x): mean {fl['mean_abs']}, frac "
                  f"{fl['frac_pixels_differing']}.", file=sys.stderr)
            return 1
        print(f"[probe] NOTE: the renderer is not bit-exact between calls "
              f"(floor: mean {fl['mean_abs']}, frac "
              f"{fl['frac_pixels_differing']}, max {fl['max_abs']} for ONE "
              f"state rendered {fl['n_renders']}x). Two replays stay within "
              f"that floor (mean {rep['pixel_mean_worst_step']}, frac "
              f"{rep['pixel_frac_worst_step']}), so the pairing holds to the "
              f"precision the renderer offers -- which is what the figure's "
              f"caption must say rather than claiming bit-identical rows.")
    ch = rep["arm_rewind_channels"]
    missing = [k for k, v in ch.items() if not v]
    if missing:
        print(f"[probe] FAIL: carry-over channel(s) never reached: {missing} "
              f"(channels: {ch}). The pairing held here, but not for the stated "
              f"reason -- most likely robosuite renamed something.",
              file=sys.stderr)
        return 1
    # And the sampling phase specifically, because it is the one channel a qpos
    # comparison cannot see: it changes WHEN the frame is taken, not where the
    # physics ends up. A count of restored observables says the code ran; this
    # says it worked.
    ph = rep["sampling_phase_after_rewind_1v2"]
    if ph["n_fields"] == 0:
        print("[probe] FAIL: no observation-sampling phase was found to compare, "
              "so the fourth channel is unverified on this robosuite.",
              file=sys.stderr)
        return 1
    if ph["n_fields_differing"]:
        print(f"[probe] FAIL: two replays leave the rewind with different "
              f"observation-sampling phases ({ph['n_fields_differing']} of "
              f"{ph['n_fields']} fields: {ph['differing']}), so their frames are "
              f"rendered at different substeps of a control step.",
              file=sys.stderr)
        return 1
    px = ("bit-identical pixels" if rep["identical_frames"]
          else "pixels within the renderer's own noise floor")
    print(f"[probe] PASS: identical prompts give identical trajectories -- "
          f"qpos bit-identical and {px} -- over {n_steps} steps and three "
          f"replays, every carry-over channel was reset ({ch}), and the "
          f"observation-sampling phase after the rewind is identical across "
          f"replays ({ph['n_fields']} fields).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
