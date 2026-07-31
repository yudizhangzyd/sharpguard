#!/usr/bin/env python3
"""Is our action decode the same function as the checkpoint's own?

Context. The decoder gate has failed four times, and three of the four candidate
causes are now measured rather than argued:

  * the gripper convention   -- bolt viyhc4kpft, four conventions, all 0/10
  * frame preprocessing      -- bolt i55ww23d5n and mmmnxeehda, 2x2, all 0/10,
                                with the approximate AND the exact resize
  * the per-suite step budget -- excluded by construction: those runs used
                                upstream's own 280 for libero_object

None is sufficient. What is left is the one quantity in this harness that has
only ever been validated against our own offline audit: the action
de-quantization. `sharpguard.libero_sim.predict_action` reimplements OpenVLA's
decode -- greedy argmax over a masked 256-token window, then bin centres of
linspace(-1, 1, 256), then a masked un-normalization -- and every one of those
choices is a place to be wrong in a way that degrades rather than breaks, which
is exactly the signature the gate measured (libero_goal 5/50, not a clean zero).

What this script does. It does not diff the reimplementation against upstream's
source and reason about it; four failures were produced by reasoning. It calls
BOTH decoders on the SAME frames and compares the numbers:

  stage 1 (equivalence)  N observations decoded by BOTH decoders and compared
                         per-dimension. The observations are taken along real
                         trajectories, not just at t=0: our decoder drives the
                         arm, and at every step both decoders are asked what to
                         do next. Comparing only the first frame of each episode
                         would miss a decoder that agrees on the canonical
                         starting pose -- which every episode shares -- and
                         diverges once the arm is somewhere the training
                         distribution covers less densely, and "degrades as the
                         episode proceeds" is precisely the failure shape the
                         gate measured.
  stage 2 (consequence)  a 2-arm rollout, action_decoder in {ours, upstream},
                         everything else held at the gate configuration. This is
                         what decides whether a decode difference is THE cause
                         rather than merely a difference -- the lesson of the
                         gripper and preprocessing nulls, both of which were real
                         differences that changed nothing.

Stage 2 runs even when stage 1 finds the decoders identical, because "identical
decode, still 0/10" is a stronger statement than "identical decode" and costs one
extra arm. Skip it with --no-rollout.

Output: action_decode_check.json. Exit 0 if the two decoders agree AND the
rollout arms agree; 1 if they differ anywhere, since a difference is a finding
that needs acting on rather than a job to retry.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DIMS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="openvla/openvla-7b-finetuned-libero-object")
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--unnorm-key", default="libero_object")
    p.add_argument("--n-obs", type=int, default=24,
                   help="frames compared in stage 1, spread over tasks and "
                        "timesteps; each costs one forward pass per decoder")
    p.add_argument("--traj-steps", type=int, default=8,
                   help="steps of real trajectory to walk per task in stage 1, "
                        "so the comparison covers mid-episode states and not "
                        "only the canonical starting pose every episode shares")
    p.add_argument("--image-preproc", default="none",
                   help="held identical between stage 1 and stage 2 so the "
                        "frames compared are the frames the rollout decodes")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="L-inf tolerance for calling two decodes equal; bf16 "
                        "logits make bit-equality the wrong test")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=0,
                   help="0 = upstream's own per-suite budget")
    p.add_argument("--no-rollout", action="store_true")
    p.add_argument("--out", default="action_decode_check.json")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    from sharpguard.libero_sim import (ACTION_DECODERS, RolloutConfig,
                                       _apply_gripper_transform,
                                       _load_libero_init_states,
                                       _preprocess_image, is_available,
                                       predict_action, predict_action_upstream,
                                       rollout_libero)

    if not is_available():
        sys.exit("[decode] libero/robosuite/mujoco not importable; cannot run")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[decode] model : {args.model}")
    print(f"[decode] device: {device}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to(device).eval()

    # The checkpoint must actually expose the method, or stage 1 is vacuous. Say
    # so as a finding rather than crashing with an AttributeError: "the
    # comparison could not be made" and "the comparison passed" must not look
    # alike in the report.
    has_upstream = hasattr(model, "predict_action")
    print(f"[decode] checkpoint exposes predict_action: {has_upstream}")

    report: dict = {
        "model": args.model, "suite": args.suite,
        "unnorm_key": args.unnorm_key,
        "checkpoint_exposes_predict_action": has_upstream,
        "decoders": list(ACTION_DECODERS),
        "tol": args.tol,
    }

    # ---------------- stage 1: same frames, both decoders ----------------
    if has_upstream:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        task_suite = benchmark.get_benchmark_dict()[args.suite]()
        pairs, per_obs = [], []
        for tid in range(task_suite.n_tasks):
            if len(pairs) >= args.n_obs:
                break
            task = task_suite.get_task(tid)
            init_states = _load_libero_init_states(
                os.path.join(get_libero_path("init_states"),
                             task.problem_folder, task.init_states_file))
            env = OffScreenRenderEnv(bddl_file_name=os.path.join(
                get_libero_path("bddl_files"), task.problem_folder,
                task.bddl_file), camera_heights=256, camera_widths=256)
            env.reset()
            obs = (env.set_init_state(init_states[0]) if init_states is not None
                   else env.reset())
            # Same settling period the rollout uses, so the frames compared here
            # are frames the rollout would actually have decoded.
            no_op = np.array([0., 0., 0., 0., 0., 0., -1.], dtype=np.float32)
            for _ in range(10):
                obs, _, _, _ = env.step(no_op)

            for t in range(args.traj_steps):
                if len(pairs) >= args.n_obs:
                    break
                img = np.asarray(obs["agentview_image"], dtype=np.uint8)[::-1, ::-1]
                img = _preprocess_image(img, args.image_preproc)
                a_ours = np.asarray(
                    predict_action(model, processor, img, task.language,
                                   device=device, unnorm_key=args.unnorm_key),
                    dtype=float)
                a_up = np.asarray(
                    predict_action_upstream(model, processor, img, task.language,
                                            device=device,
                                            unnorm_key=args.unnorm_key),
                    dtype=float)
                d = np.abs(a_ours - a_up)
                pairs.append((a_ours, a_up))
                per_obs.append({
                    "task": task.language, "t": t,
                    "ours": [round(float(v), 6) for v in a_ours],
                    "upstream": [round(float(v), 6) for v in a_up],
                    "linf": round(float(d.max()), 6),
                    "agrees": bool(d.max() <= args.tol),
                })
                print(f"[decode] task {tid:2d} t={t:2d} Linf={d.max():.6f} "
                      f"{'agree' if d.max() <= args.tol else 'DIFFER'}  "
                      f"{task.language[:40]}")
                # Ours drives, so the states visited are states the measured
                # (0/10) configuration actually visits. Driving with upstream
                # would compare the decoders on a trajectory that no run in this
                # paper was measured on.
                obs, _, done, _ = env.step(
                    _apply_gripper_transform(a_ours.astype(np.float32), "none"))
                if done:
                    break
            env.close()

        O = np.array([o for o, _ in pairs])
        U = np.array([u for _, u in pairs])
        dif = np.abs(O - U)
        n_agree = int((dif.max(axis=1) <= args.tol).sum())
        report["equivalence"] = {
            "n_obs": len(pairs),
            "n_tasks_covered": len({o["task"] for o in per_obs}),
            "traj_steps_per_task": args.traj_steps,
            "image_preproc": args.image_preproc,
            "n_agree": n_agree,
            "n_differ": len(pairs) - n_agree,
            "frac_agree": round(n_agree / max(len(pairs), 1), 4),
            "max_linf": round(float(dif.max()), 6),
            "mean_linf": round(float(dif.max(axis=1).mean()), 6),
            "per_dim_max_abs_diff": {
                DIMS[i]: round(float(dif[:, i].max()), 6) for i in range(7)},
            # The gripper is called out because a sign disagreement there is the
            # difference between grasping and not, and decoder_audit.json already
            # records 2% sign agreement against the demos.
            "gripper_sign_agreement": round(float(
                (np.sign(O[:, 6]) == np.sign(U[:, 6])).mean()), 4),
            "identical": bool(dif.max() <= args.tol),
            # Whether disagreement grows with time-in-episode. A decoder that
            # matches at t=0 and drifts later is a different diagnosis from one
            # that is wrong everywhere, and only the first is consistent with the
            # gate's partial successes (libero_goal 5/50 rather than 0/50).
            "linf_by_timestep": {
                str(t): round(float(np.mean(
                    [o["linf"] for o in per_obs if o["t"] == t])), 6)
                for t in sorted({o["t"] for o in per_obs})},
            "per_obs": per_obs,
        }
        print(f"\n[decode] stage 1: {n_agree}/{len(pairs)} frames agree within "
              f"{args.tol}; max Linf {dif.max():.6f}")
        for i, name in enumerate(DIMS):
            print(f"[decode]   {name:8s} max|diff| = {dif[:, i].max():.6f}")
    else:
        report["equivalence"] = {
            "identical": None,
            "note": "the checkpoint exposes no predict_action, so the two "
                    "decoders could not be compared. This is 'not measured', "
                    "not 'measured equal'.",
        }

    # ---------------- stage 2: does it change the rollout? ----------------
    if not args.no_rollout:
        arms = {}
        for dec in ACTION_DECODERS:
            if dec == "upstream" and not has_upstream:
                print(f"\n[decode] skipping rollout arm '{dec}': unavailable")
                continue
            print(f"\n{'=' * 62}\n[decode] rollout arm: action_decoder={dec}\n"
                  f"{'=' * 62}")
            cfg = RolloutConfig(
                suite=args.suite, n_episodes_per_suite=args.n_episodes,
                max_steps=args.max_steps, unnorm_key=args.unnorm_key,
                image_preproc=args.image_preproc, action_decoder=dec)
            try:
                arms[dec] = rollout_libero(model, processor, cfg, device=device)
            except Exception as e:  # noqa: BLE001
                print(f"[decode] arm {dec} RAISED: {type(e).__name__}: {e}")
                arms[dec] = {"error": f"{type(e).__name__}: {e}"}
                continue
            r = arms[dec]
            print(f"[decode] {dec}: SR={r['SR']:.3f} "
                  f"({r['n_success']}/{r['n_total']})  "
                  f"canonical_init={r['all_episodes_used_canonical_init']}  "
                  f"steps={r.get('max_steps')}")
        report["rollout"] = {
            "arms": arms,
            "sr_by_decoder": {k: v.get("SR") for k, v in arms.items()
                              if "error" not in v},
        }

    # ---------------- verdict ----------------
    print(f"\n{'=' * 62}\n[decode] VERDICT\n{'=' * 62}")
    ident = report["equivalence"].get("identical")
    srs = (report.get("rollout") or {}).get("sr_by_decoder") or {}
    same_sr = len(set(srs.values())) <= 1
    if ident is None:
        print("[decode] INCONCLUSIVE: the checkpoint exposes no predict_action, "
              "so this hypothesis is untested rather than refuted.")
    elif ident and same_sr:
        print("[decode] CONCLUSION: our decode is numerically the same function "
              "as the checkpoint's own on every frame compared, and swapping "
              "them changes no rollout outcome. The action de-quantization is "
              f"NOT the remaining cause (SR by decoder: {srs}). That exhausts "
              "the four candidates a source diff produced, so the next step is "
              "not another candidate from the same list -- it is instrumenting "
              "an episode against a known-good reference trajectory.")
    elif ident and not same_sr:
        print(f"[decode] CONTRADICTION: the decoders agree on every compared "
              f"frame yet the rollouts differ ({srs}). Something outside the "
              f"decode is nondeterministic; do not attribute the difference to "
              f"the decoder until that is explained.")
    else:
        print(f"[decode] CONCLUSION: the two decoders DISAGREE "
              f"(max Linf {report['equivalence']['max_linf']}, "
              f"{report['equivalence']['n_differ']}/"
              f"{report['equivalence']['n_obs']} frames). Upstream's is right by "
              f"definition -- it is what the published SR was measured with. "
              f"Per-dimension differences: "
              f"{report['equivalence']['per_dim_max_abs_diff']}. "
              f"SR by decoder: {srs}.")

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[decode] wrote {args.out}")

    if ident is None:
        return 1
    return 0 if (ident and same_sr) else 1


if __name__ == "__main__":
    raise SystemExit(main())
