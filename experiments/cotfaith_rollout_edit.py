"""Rollout-level CoT edit: does editing the CoT change TASK SUCCESS, not just
the first action?

This is the experiment limitation (v) says the paper does not have. Everything
in the leaderboard is a first-step, single-observation quantity: we perturb the
CoT, decode one action, and measure how far it moved. That establishes that the
CoT is *read*. It does not establish that reading it matters for doing the task,
because a 0.3-radian first-step deviation that the controller recovers from over
the next 200 steps is not the same finding as a failed episode.

So: run the same episode twice from the SAME canonical init state, once with the
model's own CoT and once with that CoT edited, and compare success rates.

Arms (all paired per (task, episode) on one init state):

  nocot        upstream's published prompt, no CoT. Reproduces the passing
               gate arm, and is the sanity check that this harness's env
               setup, gripper transform and step budget still give the SR the
               gate measured. If this arm regresses, nothing else here means
               anything.
  cot_clean    the model generates its own CoT, we parse it into the 9-tag
               dict and re-render it, then decode the action from that prefix.
               This is the CONTROL, and it is not optional: the parse/re-render
               round-trip is itself a perturbation (whitespace, tag order,
               dropped tags), so an edited arm must be compared against a
               round-tripped clean arm rather than against `nocot`. It is the
               rollout-level analogue of selfsplice_control.
  cot_<fam>    same, with an edit family applied to the parsed dict before
               re-rendering.

The headline is DSR = SR(cot_clean) - SR(cot_<fam>), paired per episode, with
an exact McNemar test over the discordant pairs.

Two preconditions this script REFUSES to paper over:

  1. If SR(cot_clean) == 0 the model cannot do the task with its own CoT in the
     prompt, so DSR is 0 for every family by construction and means nothing.
     The report sets precondition_met=false and says so, rather than emitting a
     table of 0.00 deltas that reads like a null result.
  2. If the checkpoint has no norm_stats entry for the suite's dataset, actions
     go to env.step() at the wrong physical scale, which drives SR to 0 for a
     reason that has nothing to do with CoT. The probe reports every available
     key so this is decided by looking rather than by hoping.

Decoder. Every arm in a run shares ONE decoder and differs in the PROMPT only,
so "the CoT was edited" is never confounded with "the de-quantizer changed".
Which decoder is right depends on the checkpoint, and --action-decoder makes
that explicit rather than assumed:

  upstream  the checkpoint's own predict_action with upstream's
            de-quantization, driven by norm_stats[unnorm_key]. Correct for
            weights that emit dataset-normalized values -- upstream's LIBERO
            checkpoints, which is what the passing gate used.
  ours      the identity [-1,1] de-quantization, no norm_stats. Correct for OUR
            fine-tunes, whose training target quantizes raw LIBERO actions
            clipped to [-1,1] with no dataset normalization
            (cotfaith_train.py:_quantize_action). For these weights an
            unnorm_key would rescale an already-correct action.

This matters because probe phenc9ygb4 established that ECoT-bridge carries
norm_stats for 'bridge_orig' only: the CoT checkpoint the leaderboard is built
on cannot be rolled out on LIBERO through the upstream decoder at all. Our
LIBERO fine-tunes can, through 'ours'.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

ECOT_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# Same tag order and key aliases as experiments/cotfaith_bridge.py, which is the
# self-decoded-CoT path the cross-corpus (F5) runs already use. Kept identical
# on purpose: if the two disagreed, a rollout-level result could not be compared
# against the first-step result it is supposed to extend.
ECOT_TAGS_ORDER = [
    (("task",), "TASK"), (("plan",), "PLAN"), (("bboxes",), "VISIBLE OBJECTS"),
    (("subtask_reasoning", "subtask_reason"), "SUBTASK REASONING"),
    (("subtask",), "SUBTASK"),
    (("movement_reasoning", "move_reasoning", "move_reason"), "MOVE REASONING"),
    (("movement", "move"), "MOVE"),
    (("gripper", "gripper_position"), "GRIPPER POSITION"),
]

_TAGS = ["TASK:", "PLAN:", "VISIBLE OBJECTS:", "SUBTASK REASONING:", "SUBTASK:",
         "MOVE REASONING:", "MOVE:", "GRIPPER POSITION:", "ACTION:"]
_KEY_MAP = {
    "TASK:": "task", "PLAN:": "plan", "VISIBLE OBJECTS:": "bboxes",
    "SUBTASK REASONING:": "subtask_reasoning", "SUBTASK:": "subtask",
    "MOVE REASONING:": "movement_reasoning", "MOVE:": "movement",
    "GRIPPER POSITION:": "gripper", "ACTION:": None,
}


def parse_generated_cot(text: str) -> dict:
    """Parse a decoded generation into a reasoning dict with the 9 tag keys."""
    positions = sorted((text.find(t), t) for t in _TAGS if text.find(t) >= 0)
    reasoning = {}
    for i, (pos, t) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        k = _KEY_MAP[t]
        if k:
            reasoning[k] = text[pos + len(t):end].strip()
    return reasoning


def build_cot_body(reasoning: dict) -> str:
    parts = []
    for keys, tag in ECOT_TAGS_ORDER:
        v = next((reasoning[k] for k in keys if k in reasoning), None)
        parts.append(f"{tag}: {str(v) if v else ''}")
    return " ".join(parts)


def cot_prompt(instruction: str, body: str) -> str:
    """The ECoT prompt with a reasoning prefix, ending at ACTION: so the next
    tokens generated are the action tokens."""
    return (f"{ECOT_SYSTEM_PROMPT} USER: What action should the robot take to "
            f"{str(instruction).lower()}? ASSISTANT: {body} ACTION:")


def gen_cot(model, processor, image, instruction, *, device, pixel_dtype,
            max_new_tokens: int) -> tuple:
    """Greedily generate the model's own CoT for this frame.

    Returns (reasoning_dict, n_generated_tokens, raw_text). Greedy so that the
    clean and edited arms see the same CoT for the same frame -- with sampling,
    an SR difference could just be two different CoTs.
    """
    import torch
    from PIL import Image
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    prompt = (f"{ECOT_SYSTEM_PROMPT} USER: What action should the robot take to "
              f"{str(instruction).lower()}? ASSISTANT: TASK:")
    proc = processor(images=pil, text=prompt, return_tensors="pt")
    ids = proc["input_ids"].to(device)
    px = proc["pixel_values"].to(device).to(pixel_dtype)
    with torch.no_grad():
        out = model.generate(input_ids=ids, pixel_values=px,
                             max_new_tokens=max_new_tokens, do_sample=False)
    n_new = int(out.shape[1] - ids.shape[1])
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return parse_generated_cot(text), n_new, text


def has_structured_cot(r: dict) -> bool:
    """A generation with no PLAN and no SUBTASK is not a CoT; editing it is
    editing nothing, and scoring that as a faithful-or-not outcome would be the
    same defect as the bbox_jitter_null artifact on DeepThinkVLA."""
    return bool(r.get("plan")) or bool(r.get("subtask"))


def mcnemar_exact(b: int, c: int) -> Optional[float]:
    """Two-sided exact McNemar p over discordant pairs (b = clean-success and
    edited-failure, c = the reverse). Binomial(b+c, 0.5). No scipy dependency:
    the bolt image's scipy has moved under us before, and this is 10 lines.
    """
    n = b + c
    if n == 0:
        return None
    from math import comb
    tail = [comb(n, k) for k in range(n + 1)]
    tot = float(sum(tail))
    k_obs = min(b, c)
    p = 2.0 * sum(tail[k] for k in range(k_obs + 1)) / tot
    return float(min(1.0, p))


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ----------------------------------------------------------------------
# probe: answer the two preconditions in minutes, not GPU-hours
# ----------------------------------------------------------------------

def _decoder(args):
    """Return the (fn, kwargs) pair every arm in this run will use.

    Resolved once, and the choice is recorded in the report: an arm that
    silently used a different de-quantizer than its pair would make the paired
    delta meaningless.
    """
    from sharpguard.libero_sim import predict_action, predict_action_upstream
    if args.action_decoder == "ours":
        # No unnorm_key on purpose: for our fine-tunes the identity map IS the
        # native scale, so passing one would rescale a correct action.
        return predict_action, {"unnorm_key": ""}
    return predict_action_upstream, {"unnorm_key": args.unnorm_key}


def probe(args, model, processor, device, pixel_dtype) -> dict:
    """Report what a full run depends on, without running one.

    Answers, in order of how badly each would waste a 24h job:
      * which norm_stats keys the checkpoint carries (a missing suite key means
        every action goes to the sim at the wrong scale -> SR=0 for reasons
        unrelated to CoT);
      * whether the checkpoint generates parseable 9-tag CoT at all;
      * whether an edit family actually changes the rendered prefix;
      * whether all three arms decode a 7-vector on one real frame.
    """
    from sharpguard.libero_sim import (_get_norm_stats, _preprocess_image,
                                       _apply_gripper_transform)
    from sharpguard.attacks import EDIT_FAMILIES

    decode, dec_kw = _decoder(args)
    out = {"probe": True, "ckpt": args.ckpt_path, "suite": args.suite,
           "action_decoder": args.action_decoder}

    stats = getattr(model, "norm_stats", None) or getattr(
        getattr(model, "config", object()), "norm_stats", None) or {}
    keys = sorted(stats.keys()) if isinstance(stats, dict) else []
    out["norm_stats_keys"] = keys
    out["unnorm_key_requested"] = args.unnorm_key
    out["unnorm_key_present"] = args.unnorm_key in keys if args.unnorm_key else False
    q01, q99, mask = _get_norm_stats(model, args.unnorm_key) if args.unnorm_key \
        else (None, None, None)
    out["norm_stats_usable"] = q01 is not None
    if q01 is not None:
        out["q01"] = [round(float(v), 4) for v in q01]
        out["q99"] = [round(float(v), 4) for v in q99]
        out["mask"] = [bool(v) for v in mask]
    # The blocker this exists to catch, stated as a fact rather than a guess.
    if args.action_decoder == "ours":
        # The identity path needs no norm_stats, so their absence is not a
        # blocker here -- it would be a blocker only for --action-decoder
        # upstream. Saying "ok" without saying why would hide that distinction.
        out["scale_precondition"] = (
            "ok (identity [-1,1] de-quantization; --action-decoder=ours needs "
            "no norm_stats because our fine-tunes quantize raw LIBERO actions "
            f"clipped to [-1,1]. norm_stats present: {keys})")
    else:
        out["scale_precondition"] = (
            "ok" if q01 is not None else
            f"MISSING: no usable norm_stats for unnorm_key={args.unnorm_key!r}; "
            f"available={keys}. Actions would reach env.step() at raw [-1,1] "
            f"scale, which pins SR at 0 independently of any CoT edit.")

    frame, instruction = _probe_frame(args)
    out["instruction"] = instruction
    if frame is None:
        out["frame"] = "unavailable (libero not importable); CoT probe only"
    img = None if frame is None else _preprocess_image(frame, args.image_preproc)

    t0 = time.time()
    reasoning, n_new, raw = gen_cot(
        model, processor, img if img is not None else np.zeros((256, 256, 3), np.uint8),
        instruction, device=device, pixel_dtype=pixel_dtype,
        max_new_tokens=args.max_new_tokens)
    out["cot_gen_seconds"] = round(time.time() - t0, 2)
    out["cot_tokens_generated"] = n_new
    out["cot_tags_parsed"] = sorted(reasoning.keys())
    out["cot_structured"] = has_structured_cot(reasoning)
    out["cot_raw_head"] = raw[:600]
    body = build_cot_body(reasoning)
    out["cot_body_head"] = body[:400]

    # Does each requested family actually move the rendered prefix? A family
    # that renders byte-identically is inapplicable here for the same reason
    # bbox_jitter_null is inapplicable on DeepThinkVLA, and must be recorded as
    # such rather than scored as a passing null.
    fam_status = {}
    for fname in args.families.split(","):
        fname = fname.strip()
        if not fname or fname not in EDIT_FAMILIES:
            fam_status[fname] = "unknown family"
            continue
        try:
            edited = EDIT_FAMILIES[fname](reasoning)
        except Exception as e:
            fam_status[fname] = f"raised {type(e).__name__}: {e}"
            continue
        if edited is None:
            fam_status[fname] = "not applicable to this CoT (returned None)"
            continue
        edited.pop("__edit_meta__", None)
        eb = build_cot_body(edited)
        fam_status[fname] = ("IDENTICAL RENDER - inapplicable"
                             if eb == body else "changes the rendered CoT")
    out["families"] = fam_status

    # One decode per arm, on the real frame, so a shape/dtype failure surfaces
    # here rather than 6 hours in.
    if img is not None:
        arms = {"nocot": None, "cot_clean": cot_prompt(instruction, body)}
        for fname, st in fam_status.items():
            if st == "changes the rendered CoT":
                ed = EDIT_FAMILIES[fname](reasoning)
                ed.pop("__edit_meta__", None)
                arms[f"cot_{fname}"] = cot_prompt(instruction, build_cot_body(ed))
        acts = {}
        for name, pr in arms.items():
            try:
                t1 = time.time()
                a = decode(model, processor, img, instruction, device=device,
                           pixel_dtype=pixel_dtype, prompt=pr, **dec_kw)
                a2 = _apply_gripper_transform(a, args.gripper_transform)
                acts[name] = {"raw": [round(float(v), 5) for v in a],
                              "sent": [round(float(v), 5) for v in a2],
                              "seconds": round(time.time() - t1, 3)}
            except Exception as e:
                acts[name] = {"error": f"{type(e).__name__}: {e}"}
        out["one_frame_actions"] = acts
        base = acts.get("cot_clean", {}).get("raw")
        for name, v in acts.items():
            if name.startswith("cot_") and name != "cot_clean" and base and "raw" in v:
                d = np.abs(np.asarray(v["raw"]) - np.asarray(base))
                v["delta_linf_vs_cot_clean"] = round(float(d.max()), 5)

    # Feasibility, computed rather than guessed: this is what decides whether a
    # full run fits in the 24h wall clock. The CoT term is divided by the
    # refresh interval, because that is the only reason the interval exists --
    # bolt nskmsunnpb spent 63 of every 66 minutes generating reasoning.
    refresh = max(1, int(getattr(args, "cot_refresh_steps", 1) or 1))
    per_step = out.get("cot_gen_seconds", 0.0) / refresh + sum(
        v.get("seconds", 0.0) for v in out.get("one_frame_actions", {}).values()
        if isinstance(v, dict)) / max(1, len(out.get("one_frame_actions", {})))
    n_arms = 1 + 1 + sum(1 for s in fam_status.values()
                         if s == "changes the rendered CoT")
    if args.arms:
        n_arms = len([a for a in args.arms.split(",") if a.strip()])
    out["feasibility"] = {
        "seconds_per_step_per_arm": round(per_step, 2),
        "n_arms": n_arms,
        "cot_refresh_steps": refresh,
        "est_hours_full_run": round(
            per_step * n_arms * args.max_steps * args.n_eps_per_task
            * args.n_tasks_est / 3600.0, 1),
        "note": "est assumes every episode runs to max_steps; successful "
                "episodes terminate early, so this is an upper bound. The "
                "no-CoT arm is cheaper than this per-arm average.",
    }
    return out


def _seed_scene(seed: int) -> None:
    """Make the next env construction place its fixtures identically.

    `set_init_state` restores the flattened MuJoCo state, which is qpos/qvel --
    the robot and every free object. A fixture welded to the world body has no
    joint, so its pose lives in the MODEL, and robosuite's placement sampler
    draws it from `np.random` at construction time. `set_init_state` therefore
    cannot restore it, and two arms built from separate envs get separately
    sampled fixtures.

    This is measured, not inferred. In the first filmstrip capture (bolt
    h8xzmqnhgg) the two arms' step-0 frames differed by exactly 0.000 mean
    absolute pixel over the robot and over the free objects on the table, and by
    7.97 over the cabinet -- a 3-pixel translation of the one articulated
    fixture in the scene, constant from step 0 and never moving thereafter. So
    the arms were paired on the robot and mispaired on the drawer they were
    being asked to close.

    Seeding np.random rather than calling env.seed(): the sampler reads the
    global numpy RNG directly, and this holds whether or not LIBERO's wrapper
    exposes a seed method.
    """
    np.random.seed(int(seed))


def _probe_frame(args):
    """One real (frame, instruction) from the suite, or (None, task text)."""
    try:
        from sharpguard.libero_sim import is_available, _load_libero_init_states
        if not is_available():
            return None, "pick up the black bowl and place it on the plate"
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        suite = benchmark.get_benchmark_dict()[args.suite]()
        task = suite.get_task(0)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        _seed_scene(args.env_seed)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256,
                                 camera_widths=256)
        env.reset()
        init = _load_libero_init_states(os.path.join(
            get_libero_path("init_states"), task.problem_folder,
            task.init_states_file))
        obs = env.set_init_state(init[0]) if init is not None else env.reset()
        no_op = np.array([0., 0., 0., 0., 0., 0., -1.], dtype=np.float32)
        for _ in range(10):
            obs, _, _, _ = env.step(no_op)
        img = np.asarray(obs["agentview_image"], dtype=np.uint8)[::-1, ::-1]
        env.close()
        return img, task.language
    except Exception as e:
        print(f"[probe] frame unavailable: {type(e).__name__}: {e}")
        return None, "pick up the black bowl and place it on the plate"


# ----------------------------------------------------------------------
# the rollout
# ----------------------------------------------------------------------

def _eef(obs) -> Optional[list]:
    """End-effector position from a LIBERO/robosuite observation, or None.

    Read by name rather than by index: the key is robosuite's, and an env
    configured without the low-dim observables simply does not have it. A
    missing pose must yield None so the trajectory record is short rather than
    silently zero-filled -- a flat line at the origin looks like a policy that
    never moved, which is a claim, and the wrong one.
    """
    v = obs.get("robot0_eef_pos") if isinstance(obs, dict) else None
    return None if v is None else [round(float(x), 5) for x in np.asarray(v)]


def run_arm(model, processor, env, task_lang, *, arm, family, device,
            pixel_dtype, args, edit_fn=None, capture: Optional[dict] = None) -> dict:
    """One episode under one arm. Env must already be at its init state.

    `capture`, when given, is {"dir": Path, "every": k}: every k-th step's
    rendered frame is written as a PNG and the step's end-effector pose, action
    and current MOVE phrase are appended to a trajectory record. This exists
    because the report is otherwise entirely scalar -- success, steps, counts --
    and a scalar cannot show what the arms DID. Two arms that both fail at 0/40
    are indistinguishable in the numbers and may be doing visibly different
    things, which is the whole question an edit-sensitivity paper is asking.
    Capture is off unless asked for: it is I/O inside the step loop, and the
    runs that measure SR should not pay for the runs that illustrate it.
    """
    from sharpguard.libero_sim import (_preprocess_image,
                                       _apply_gripper_transform)
    decode, dec_kw = _decoder(args)
    no_op = np.array([0., 0., 0., 0., 0., 0., -1.], dtype=np.float32)
    obs = None
    for _ in range(10):                       # Kim's NUM_STEPS_WAIT settling
        obs, _, _, _ = env.step(no_op)
    success, steps = False, 0
    n_cot_ok = n_cot_bad = n_edit_skipped = n_cot_gen = 0
    deltas = []
    # Regenerating the CoT every step costs ~8.9 s of the 9.5 s step on a 7B
    # (bolt nskmsunnpb: 3.6 min/episode with no CoT, 63 min with one), so a
    # per-step protocol buys ~9 episodes out of a 20 h budget and a 0/9 bound.
    # `--cot-refresh-steps k` regenerates every k steps and reuses the prefix in
    # between, which buys ~10x the episodes at the cost of feeding the policy
    # reasoning that is up to k-1 steps stale. That is a DIFFERENT protocol, not
    # a cheaper approximation of the same one -- GRIPPER POSITION and MOVE are
    # frame-dependent -- so k is recorded per arm in the report and k=1
    # reproduces the per-step run exactly.
    refresh = max(1, int(getattr(args, "cot_refresh_steps", 1) or 1))
    cached_prompt = None
    # The filmstrip record. `move` is the MOVE phrase the policy was acting
    # under at that step -- for an edit arm that is the EDITED phrase, which is
    # the point: it lets a reader see the instruction the policy read next to
    # the motion it produced, rather than taking on faith that the edit landed.
    cap_dir = None if capture is None else Path(capture["dir"])
    cap_every = 1 if capture is None else max(1, int(capture.get("every", 1)))
    traj, cur_move = [], None
    if cap_dir is not None:
        cap_dir.mkdir(parents=True, exist_ok=True)
    while steps < args.max_steps:
        img = obs.get("agentview_image", obs.get("image"))
        if img is None:
            break
        img = np.asarray(img, dtype=np.uint8)[::-1, ::-1]
        img = _preprocess_image(img, args.image_preproc)
        prompt = None
        if arm != "nocot":
            if should_refresh_cot(steps, refresh, cached_prompt is not None):
                reasoning, _, _ = gen_cot(model, processor, img, task_lang,
                                          device=device,
                                          pixel_dtype=pixel_dtype,
                                          max_new_tokens=args.max_new_tokens)
                n_cot_gen += 1
                if not has_structured_cot(reasoning):
                    # No CoT this frame. Falling back to the no-CoT prompt would
                    # silently mix arms, so the step is recorded and the clean
                    # prefix is used as-is (empty tags render as empty strings).
                    n_cot_bad += 1
                else:
                    n_cot_ok += 1
                body = build_cot_body(reasoning)
                cur_move = reasoning.get("movement") or reasoning.get("move")
                if edit_fn is not None:
                    try:
                        ed = edit_fn(reasoning)
                    except Exception:
                        ed = None
                    if ed is None:
                        n_edit_skipped += 1
                    else:
                        ed.pop("__edit_meta__", None)
                        eb = build_cot_body(ed)
                        if eb == body:
                            n_edit_skipped += 1
                        else:
                            body = eb
                            cur_move = ed.get("movement") or ed.get("move")
                cached_prompt = cot_prompt(task_lang, body)
            prompt = cached_prompt
        a = decode(model, processor, img, task_lang, device=device,
                   pixel_dtype=pixel_dtype, prompt=prompt, **dec_kw)
        a = _apply_gripper_transform(a, args.gripper_transform)
        if cap_dir is not None and steps % cap_every == 0:
            # The frame as the POLICY saw it, after the same flip and
            # preprocessing, so the figure shows the policy's input rather than
            # a differently-oriented render of the same instant.
            name = f"{arm}_t{steps:04d}.png"
            try:
                from PIL import Image
                Image.fromarray(np.asarray(img, dtype=np.uint8)).save(cap_dir / name)
            except Exception as e:
                print(f"[capture] frame {name} not written: {type(e).__name__}: {e}")
                name = None
            traj.append({"step": steps, "frame": name, "eef": _eef(obs),
                         "action": [round(float(v), 5) for v in a],
                         "move": cur_move})
        obs, reward, done, info = env.step(a)
        if ((isinstance(info, dict) and info.get("success", False))
                or (reward is not None and float(reward) > 0) or done):
            success = True
            break
        steps += 1
    r = {"arm": arm, "family": family, "success": bool(success),
         "steps": steps, "n_cot_structured": n_cot_ok,
         "n_cot_unstructured": n_cot_bad, "n_edit_skipped": n_edit_skipped,
         "n_cot_generated": n_cot_gen, "cot_refresh_steps": refresh,
         "n_delta_recorded": len(deltas)}
    if cap_dir is not None:
        r["trajectory"] = traj
        # Whether the poses are real is a property of the env, not of this run,
        # so it is recorded rather than assumed by whatever reads the file.
        r["eef_available"] = bool(traj) and traj[0]["eef"] is not None
    return r


def run(args):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from sharpguard.attacks import EDIT_FAMILIES
    from sharpguard.libero_sim import is_available, UPSTREAM_MAX_STEPS

    pixel_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.max_steps <= 0:
        args.max_steps = UPSTREAM_MAX_STEPS.get(args.suite, 400)
        print(f"[rollout-edit] max_steps=0 -> upstream's {args.max_steps} "
              f"for {args.suite}")

    print(f"[rollout-edit] loading {args.ckpt_path}")
    processor = AutoProcessor.from_pretrained(args.ckpt_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.ckpt_path, trust_remote_code=True, torch_dtype=pixel_dtype,
        low_cpu_mem_usage=True).to(device).eval()

    if args.probe_only:
        rep = probe(args, model, processor, device, pixel_dtype)
        p = out_dir / "rollout_edit_probe.json"
        p.write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=2)[:4000])
        print(f"[rollout-edit] probe written to {p}")
        return

    if not is_available():
        raise RuntimeError("libero/robosuite/mujoco not importable; a "
                           "rollout-level result cannot be faked from offline "
                           "records, so this exits rather than degrading.")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from sharpguard.libero_sim import _load_libero_init_states

    suite = benchmark.get_benchmark_dict()[args.suite]()
    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    arms = select_arms(fams, args.arms)
    print(f"[rollout-edit] arms={[n for n, _ in arms]} "
          f"cot_refresh_steps={args.cot_refresh_steps}")

    episodes = []
    rep_path = out_dir / "rollout_edit_report.json"

    def flush(status: str):
        rep_path.write_text(json.dumps(
            _summarize(args, arms, episodes, status), indent=2))

    t_start = time.time()
    for ti in range(min(args.n_tasks or suite.n_tasks, suite.n_tasks)):
        task = suite.get_task(ti)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        init = _load_libero_init_states(os.path.join(
            get_libero_path("init_states"), task.problem_folder,
            task.init_states_file))
        for ep in range(args.n_eps_per_task):
            if init is None or ep >= len(init):
                # A configuration error, not something to paper over with a
                # random reset -- that would mix two initial-state
                # distributions inside one reported SR. Recorded per arm rather
                # than raised, so the report says which episodes were skipped
                # instead of the run dying with its status still "running".
                msg = (f"episode {ep} has no canonical init state in "
                       f"{task.init_states_file}; lower --n-eps-per-task "
                       f"rather than mixing in a random reset")
                print(f"[rollout-edit] t{ti} ep{ep} SKIPPED: {msg}")
                for arm, fam in arms:
                    episodes.append({"arm": arm, "family": fam, "task_idx": ti,
                                     "episode": ep, "error": msg})
                flush("running")
                continue

            # ONE env for every arm of this episode. A fresh env per arm re-runs
            # robosuite's placement sampler, which places the welded fixtures,
            # and set_init_state cannot undo that: it restores qpos, and a body
            # welded to the world has no joint, so its pose lives in the MODEL.
            #
            # Measured, in the first filmstrip capture (bolt h8xzmqnhgg): the
            # arms' step-0 frames agreed to 0.000 mean absolute pixel over the
            # robot and over the free objects on the table, and differed by 7.97
            # over the cabinet -- a 3-pixel translation of the one fixture in
            # the scene, constant from step 0. The arms were paired on the robot
            # and mispaired on the drawer they were being asked to close, and
            # nothing in the report showed it, because the report is scalar.
            #
            # Sharing the env is what makes the pairing structural rather than a
            # property of an RNG we do not control: between arms only qpos is
            # restored, and no reset() intervenes to re-sample anything.
            env = None
            try:
                _seed_scene(args.env_seed + 1000 * ti + ep)
                env = OffScreenRenderEnv(bddl_file_name=bddl,
                                         camera_heights=256,
                                         camera_widths=256)
                env.reset()
                for arm, fam in arms:
                    if args.time_budget_h and \
                            (time.time() - t_start) / 3600.0 > args.time_budget_h:
                        print("[rollout-edit] time budget reached; stopping "
                              "with what is complete rather than dying "
                              "mid-episode")
                        flush("stopped_time_budget")
                        return
                    try:
                        # No reset() here. reset() re-samples the placement,
                        # which is the one thing that must not differ between
                        # arms; set_init_state restores the robot and every free
                        # object, which is the rest of the init state.
                        env.set_init_state(init[ep])
                        r = run_arm(model, processor, env, task.language,
                                    arm=arm, family=fam, device=device,
                                    pixel_dtype=pixel_dtype, args=args,
                                    edit_fn=EDIT_FAMILIES[fam] if fam else None,
                                    capture=_capture_for(args, out_dir, ti, ep))
                        r.update({"task_idx": ti, "episode": ep,
                                  "task": task.language})
                        episodes.append(r)
                        print(f"[rollout-edit] t{ti} ep{ep} {arm}: "
                              f"success={r['success']} steps={r['steps']} "
                              f"({(time.time()-t_start)/60:.1f} min elapsed)")
                    except Exception as e:
                        print(f"[rollout-edit] t{ti} ep{ep} {arm} FAILED: "
                              f"{type(e).__name__}: {e}\n"
                              f"{traceback.format_exc()[-500:]}")
                        episodes.append({"arm": arm, "family": fam,
                                         "task_idx": ti, "episode": ep,
                                         "error": str(e)})
                    flush("running")
            except Exception as e:
                # The env itself could not be built, so no arm of this episode
                # ran. Recorded for every arm, because a silently absent episode
                # would read as one that was never requested.
                print(f"[rollout-edit] t{ti} ep{ep} env FAILED: "
                      f"{type(e).__name__}: {e}")
                for arm, fam in arms:
                    episodes.append({"arm": arm, "family": fam, "task_idx": ti,
                                     "episode": ep, "error": f"env: {e}"})
                flush("running")
            finally:
                if env is not None:
                    env.close()
    flush("complete")
    print(f"[rollout-edit] done -> {rep_path}")


def _capture_for(args, out_dir: Path, ti: int, ep: int) -> Optional[dict]:
    """Capture spec for this (task, episode), or None if it is not on the list.

    `--capture-episodes` names episodes as `task:ep`, not a count, because the
    figure this feeds compares arms ON ONE INIT STATE -- the same pairing the
    DSR is computed over. A "first N episodes" flag would capture whichever
    episodes happened to run first, which across a time-budgeted run is not a
    stable set and would not necessarily be paired at all.
    """
    spec = (getattr(args, "capture_episodes", "") or "").strip()
    if not spec:
        return None
    want = {tuple(int(x) for x in s.split(":")) for s in spec.split(",") if s.strip()}
    if (ti, ep) not in want:
        return None
    every = getattr(args, "capture_every", 10)
    return {"dir": out_dir / "frames" / f"t{ti}_ep{ep}",
            # max() rather than `or 10`: 0 is falsy, so `or` would turn an
            # explicit --capture-every 0 into 10 -- a tenth of the frames the
            # caller asked for, with no error.
            "every": max(1, int(10 if every is None else every))}


def should_refresh_cot(step: int, refresh: int, have_cached: bool) -> bool:
    """Is this the step that regenerates the CoT?

    Pulled out of run_arm's loop so it is checkable without a simulator: with
    k=1 this must be True at every step (that is what makes the periodic
    protocol reduce exactly to the per-step one bolt nskmsunnpb ran), and with
    k>1 it must fire ceil(n/k) times over n steps -- the count the report
    publishes as n_cot_generated, which is the only thing distinguishing the two
    protocols for a reader.
    """
    return (not have_cached) or step % max(1, refresh) == 0


def select_arms(families: list, spec: str) -> list:
    """Resolve --arms against the arms this run actually has.

    `--arms` exists so the precondition can be measured without paying for the
    edit arms. SR(cot_clean) > 0 is what makes DSR defined at all (see
    _summarize), and it is the one arm no artifact of ours has ever reported for
    our own fine-tune -- only upstream's four LIBERO checkpoints were gated. A
    typo here would silently roll out the wrong set for hours, so an unknown name
    raises instead of being dropped: a run of two arms when three were asked for
    is indistinguishable in the report from a run that was configured that way.
    """
    all_arms = [("nocot", None), ("cot_clean", None)] + [
        (f"cot_{f}", f) for f in families]
    if not spec:
        return all_arms
    want = [a.strip() for a in spec.split(",") if a.strip()]
    known = {n for n, _ in all_arms}
    unknown = [a for a in want if a not in known]
    if unknown:
        raise ValueError(
            f"--arms names {unknown}, which are not arms of this run. "
            f"Available: {sorted(known)} (edit arms come from --families="
            f"{','.join(families)}). Refusing to run a subset nobody asked for.")
    return [(n, f) for n, f in all_arms if n in want]


def _summarize(args, arms, episodes, status: str) -> dict:
    by_arm = {}
    for arm, fam in arms:
        rows = [e for e in episodes if e.get("arm") == arm and "error" not in e]
        k = sum(1 for e in rows if e["success"])
        lo, hi = wilson(k, len(rows))
        by_arm[arm] = {"family": fam, "n": len(rows), "successes": k,
                       "sr": (k / len(rows)) if rows else None,
                       "wilson95": [lo, hi],
                       "n_cot_structured": sum(e.get("n_cot_structured", 0) for e in rows),
                       "n_cot_unstructured": sum(e.get("n_cot_unstructured", 0) for e in rows),
                       # How many times the CoT was actually generated, versus
                       # how many steps ran under it. With --cot-refresh-steps>1
                       # these differ, and a reader cannot otherwise tell a
                       # per-step run from a periodically-refreshed one.
                       "n_cot_generated": sum(e.get("n_cot_generated", 0) for e in rows),
                       "n_steps": sum(e.get("steps", 0) for e in rows),
                       "n_edit_skipped": sum(e.get("n_edit_skipped", 0) for e in rows)}

    clean = by_arm.get("cot_clean", {})
    precondition = bool(clean.get("successes", 0) > 0)
    deltas = {}
    if precondition:
        idx = {(e["task_idx"], e["episode"]): e for e in episodes
               if e.get("arm") == "cot_clean" and "error" not in e}
        for arm, fam in arms:
            if not arm.startswith("cot_") or arm == "cot_clean":
                continue
            b = c = n = 0
            for e in episodes:
                if e.get("arm") != arm or "error" in e:
                    continue
                base = idx.get((e["task_idx"], e["episode"]))
                if base is None:
                    continue
                n += 1
                if base["success"] and not e["success"]:
                    b += 1
                elif e["success"] and not base["success"]:
                    c += 1
            deltas[arm] = {
                "n_paired": n,
                "clean_success_edit_fail": b,
                "edit_success_clean_fail": c,
                "delta_sr": ((b - c) / n) if n else None,
                "mcnemar_exact_p": mcnemar_exact(b, c),
            }

    return {
        "experiment": "rollout_level_cot_edit",
        "status": status,
        "config": {k: v for k, v in vars(args).items()},
        "arms_run": [n for n, _ in arms],
        "precondition_met": precondition,
        "precondition_note": (
            "cot_clean solves the task at least once, so a drop under editing "
            "is attributable to the edit."
            if precondition else
            "cot_clean was not among --arms, so this run measures SR only and "
            "makes no claim about the precondition either way. It cannot be "
            "read as a null: no clean-CoT episode was attempted."
            if "cot_clean" not in {n for n, _ in arms} else
            "cot_clean NEVER succeeds, so DSR is 0 for every family by "
            "construction and carries no information about CoT causality. "
            "This is reported as an undefined measurement rather than as a "
            "null result. Check rollout_edit_probe.json: the usual cause is a "
            "missing norm_stats entry for this suite, i.e. a physical-scale "
            "error unrelated to any CoT."),
        "by_arm": by_arm,
        "delta_sr_vs_cot_clean": deltas,
        "episodes": episodes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--families", default="direction_flip,paraphrase_null",
                    help="comma-separated EDIT_FAMILIES keys. The default "
                         "pairs one semantic edit with the paraphrase null, "
                         "so a drop has its own control in the same run.")
    ap.add_argument("--n-tasks", type=int, default=0, help="0 = all in suite")
    ap.add_argument("--n-eps-per-task", type=int, default=1)
    ap.add_argument("--arms", default="",
                    help="comma-separated subset of nocot,cot_clean,cot_<fam>. "
                         "Empty = all. Use 'nocot,cot_clean' to measure the "
                         "SR precondition without paying for the edit arms.")
    ap.add_argument("--cot-refresh-steps", type=int, default=1,
                    help="regenerate the CoT every k steps and reuse it in "
                         "between. 1 = upstream-style per-step reasoning (what "
                         "bolt nskmsunnpb runs). k>1 is a different protocol, "
                         "recorded as such: it trades reasoning freshness for "
                         "~k x the episodes per GPU-hour.")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="0 = upstream's per-suite budget")
    ap.add_argument("--max-new-tokens", type=int, default=320,
                    help="cap on generated CoT length per step")
    ap.add_argument("--unnorm-key", default="")
    ap.add_argument("--action-decoder", default="upstream",
                    choices=["upstream", "ours"],
                    help="'upstream' for weights that emit dataset-normalized "
                         "values (upstream's LIBERO checkpoints, the passing "
                         "gate); 'ours' for the identity [-1,1] map our "
                         "fine-tunes were trained against.")
    ap.add_argument("--gripper-transform", default="openvla")
    ap.add_argument("--image-preproc", default="none")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--time-budget-h", type=float, default=0.0,
                    help=">0 stops cleanly before the wall clock kills the job")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--capture-episodes", default="",
                    help="comma-separated task:ep pairs whose frames and "
                         "end-effector trajectory are recorded, e.g. '0:0,3:0'. "
                         "Named rather than counted so the captured set is the "
                         "same paired init states across arms. Empty = capture "
                         "nothing, which is what every SR run should use.")
    ap.add_argument("--capture-every", type=int, default=10,
                    help="record every k-th step of a captured episode")
    ap.add_argument("--n-tasks-est", type=int, default=10,
                    help="only used by the probe's runtime estimate")
    ap.add_argument("--env-seed", type=int, default=0,
                    help="seeds np.random before each env is built, so every "
                         "arm of one episode gets the same fixture placement. "
                         "set_init_state restores qpos and a welded fixture's "
                         "pose is not in qpos, so without this the arms are "
                         "paired on the robot and mispaired on the furniture.")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
