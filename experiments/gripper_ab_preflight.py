#!/usr/bin/env python3
"""A/B/C/D pre-flight: which gripper convention does LIBERO actually want?

Context. The decoder gate has now returned Task SR = 0 three times against a
published ~85%, and the first two causes are fixed and verified (success read
from a key LIBERO never sets; `.pruned_init` read with np.load so every episode
silently fell back to a random env.reset()). The third run had correct init
states on 50/50 episodes and still scored 0/50 on both libero_object and
libero_spatial -- so it is a third, independent harness bug.

The evidence points at the gripper channel. It has mask=False in every OpenVLA
norm_stats, so it bypasses un-normalization and arrives as a raw bin centre in
[-1, 1]; `unnorm_key` therefore cannot be the cause. And
results_v2/decoder_audit.json records gripper_sign_agreement = 0.02: our
decoded gripper matches the ground-truth demo gripper on 2% of samples, where
chance is ~50%. A near-systematically inverted gripper cannot grasp, and
"cannot grasp" produces exactly 0/50 on pick-and-place rather than a degraded
SR.

What this script does NOT do is assume which correction is right. Four arms are
run under one model load on a deliberately small budget, and the arm that lifts
SR off the floor wins. Reasoning about upstream's conventions from memory is
what produced the first three failures; this measures instead.

Suite choice: libero_object. Its published SR is among the highest, and every
task is a pick-and-place whose success is impossible without a working gripper,
which makes it the sharpest discriminator available.

Output: gripper_ab.json, plus a verdict on stdout. Exit 0 if exactly one arm
clears --win-threshold, 1 otherwise (including the case where none does, which
means the gripper is not the remaining bug and the next hypothesis is needed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="openvla/openvla-7b-finetuned-libero-object")
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--unnorm-key", default="libero_object")
    p.add_argument("--n-episodes", type=int, default=4,
                   help="episodes per arm; 4 is enough to separate 0 from ~0.85")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--arms", default="none,invert,binvert,openvla")
    p.add_argument("--image-preprocs", default="none",
                   help="comma-separated image_preproc modes; the script runs "
                        "the full gripper x image cross product, so "
                        "'--arms none,openvla --image-preprocs none,tf_upstream' "
                        "is a 2x2 factorial whose (none,none) cell reproduces "
                        "the four failed gates as an anchor")
    p.add_argument("--win-threshold", type=float, default=0.5,
                   help="an arm wins if its SR is at least this")
    p.add_argument("--action-decoders", default="ours",
                   help="comma-separated action_decoder modes, crossed with the "
                        "other two axes. Added because bolt 7vpp28qfsk measured "
                        "that our decode and the checkpoint's own predict_action "
                        "disagree on 24/24 frames -- so every earlier cell in "
                        "this script ran a decode that is not upstream's, and in "
                        "particular the gripper channel that the 'openvla' arm "
                        "transforms is not the channel upstream's transform was "
                        "written for. Crossing the two axes is the only way to "
                        "see whether the corrections interact.")
    p.add_argument("--out", default="gripper_ab.json")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    from sharpguard.libero_sim import (ACTION_DECODERS, GRIPPER_TRANSFORMS,
                                       IMAGE_PREPROCS, RolloutConfig,
                                       is_available, rollout_libero)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    imgs = [i.strip() for i in args.image_preprocs.split(",") if i.strip()]
    decs = [d.strip() for d in args.action_decoders.split(",") if d.strip()]
    unknown = [a for a in arms if a not in GRIPPER_TRANSFORMS]
    if unknown:
        sys.exit(f"[ab] unknown arm(s) {unknown}; expected {GRIPPER_TRANSFORMS}")
    unknown = [i for i in imgs if i not in IMAGE_PREPROCS]
    if unknown:
        sys.exit(f"[ab] unknown image_preproc(s) {unknown}; "
                 f"expected {IMAGE_PREPROCS}")
    unknown = [d for d in decs if d not in ACTION_DECODERS]
    if unknown:
        sys.exit(f"[ab] unknown action_decoder(s) {unknown}; "
                 f"expected {ACTION_DECODERS}")
    if not is_available():
        sys.exit("[ab] libero/robosuite/mujoco not importable; cannot run")

    # Fail before the model load, not 20 minutes in, if an arm's backend is
    # missing. tf_upstream is the only mode with an extra dependency, and a
    # missing tensorflow must not silently degrade to a different resize
    # kernel -- that would mislabel the decisive cell.
    if "tf_upstream" in imgs:
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            sys.exit("[ab] image_preproc 'tf_upstream' was requested but "
                     "tensorflow is not installed. Set INSTALL_TF=1 so "
                     "bolt/setup-openvla.sh adds tensorflow-cpu, or ask for "
                     "'np_lanczos' and read the result as 8/255 LSB from "
                     "upstream's kernel rather than a bit-exact match.")

    cells = [(d, g, i) for d in decs for i in imgs for g in arms]

    # The cell key keeps its two-factor form when the decoder axis is degenerate,
    # so a re-run of any published configuration produces the same key it did
    # before and the anchor lookup below keeps working against it. A silently
    # renamed anchor would make the new run non-comparable to the ones it exists
    # to be compared against.
    def cell_key(d, g, i):
        return f"{g}+{i}" if len(decs) == 1 else f"{d}|{g}+{i}"

    anchor_key = cell_key(decs[0], "none", "none")

    print(f"[ab] model      : {args.model}")
    print(f"[ab] suite      : {args.suite}  unnorm_key={args.unnorm_key}")
    print(f"[ab] gripper    : {arms}")
    print(f"[ab] image      : {imgs}")
    print(f"[ab] decoder    : {decs}")
    print(f"[ab] cells      : {len(cells)} = {len(decs)} x {len(arms)} x "
          f"{len(imgs)}")
    print(f"[ab] budget     : {args.n_episodes} episodes (requested) x "
          f"{args.max_steps} steps per cell; episodes-per-task floors at 1, so "
          f"a request below the suite's task count runs more\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to(device).eval()

    results = {}
    for dec, arm, img in cells:
        key = cell_key(dec, arm, img)
        print(f"\n{'=' * 62}\n[ab] cell: decoder={dec} gripper={arm} "
              f"image={img}\n{'=' * 62}")
        cfg = RolloutConfig(
            suite=args.suite,
            n_episodes_per_suite=args.n_episodes,
            max_steps=args.max_steps,
            unnorm_key=args.unnorm_key,
            gripper_transform=arm,
            image_preproc=img,
            action_decoder=dec,
        )
        try:
            r = rollout_libero(model, processor, cfg, device=device)
        except Exception as e:
            print(f"[ab] cell {key} RAISED: {type(e).__name__}: {e}")
            results[key] = {"error": f"{type(e).__name__}: {e}"}
            continue
        results[key] = r
        print(f"[ab] {key}: SR={r['SR']:.3f} ({r['n_success']}/{r['n_total']})"
              f"  canonical_init={r['all_episodes_used_canonical_init']}"
              f"  steps={r.get('max_steps')}"
              f"  below_upstream={r.get('max_steps_below_upstream')}"
              f"  gripper_raw_mean={r['gripper_raw_mean']}"
              f"  frac_close_raw={r['gripper_frac_close_raw']}"
              f"  frac_close_sent={r['gripper_frac_close_sent']}")

    # -------- verdict --------
    print(f"\n{'=' * 62}\n[ab] VERDICT\n{'=' * 62}")
    print(f"{'cell':<24} {'SR':>7} {'succ/tot':>10} {'close_sent':>11}")
    scored = {}
    for dec, arm, img in cells:
        key = cell_key(dec, arm, img)
        r = results.get(key) or {}
        if "error" in r:
            print(f"{key:<24} {'ERROR':>7} {r['error'][:40]:>10}")
            continue
        sr = float(r.get("SR") or 0.0)
        scored[key] = sr
        tally = f"{r.get('n_success')}/{r.get('n_total')}"
        print(f"{key:<24} {sr:>7.3f} {tally:>10} "
              f"{r.get('gripper_frac_close_sent'):>11}")

    winners = [a for a, sr in scored.items() if sr >= args.win_threshold]
    # What was asked for and what actually ran, separately. rollout_libero
    # rounds eps_per_task up to 1, so a request below the suite's task count
    # runs more episodes than requested -- viyhc4kpft asked for 4 on
    # libero_object and ran 10 per arm. Reporting only the request made the
    # top-level "n_episodes_per_arm: 4" contradict every arm's "n_total": 10.
    n_ran = sorted({r.get("n_total") for r in results.values()
                    if r.get("n_total") is not None})
    steps_ran = sorted({r.get("max_steps") for r in results.values()
                        if r.get("max_steps") is not None})
    if n_ran and n_ran != [args.n_episodes]:
        print(f"\n[ab] NOTE: asked for {args.n_episodes} episodes per cell, "
              f"actually ran {n_ran}. rollout_libero floors episodes-per-task "
              f"at 1, so a request below the suite's task count runs more than "
              f"requested. Read the SR denominators above, not the request.")
    payload = {
        "model": args.model, "suite": args.suite,
        "unnorm_key": args.unnorm_key,
        "gripper_arms": arms, "image_preprocs": imgs,
        "action_decoders": decs,
        # Named explicitly because the key format depends on it: a reader who
        # assumes "gripper+image" on a three-factor report would silently
        # misattribute every cell.
        "cell_key_format": ("gripper+image" if len(decs) == 1
                            else "decoder|gripper+image"),
        "anchor_cell": anchor_key,
        "n_episodes_requested_per_arm": args.n_episodes,
        "n_episodes_actually_run_per_arm": n_ran[0] if len(n_ran) == 1 else n_ran,
        # Every cell must get the same budget or the comparison is not a
        # comparison; a list here rather than a scalar means it did not.
        "all_arms_same_episode_count": len(n_ran) == 1,
        # Requested and resolved, separately, for the same reason as the episode
        # count: --max-steps 0 means "use upstream's per-suite budget", so a bare
        # "max_steps": 0 reads as a zero-step run. f52r5nvnhb's report says 0 at
        # the top level while every arm inside it says 280.
        "max_steps_requested": args.max_steps,
        "max_steps": (steps_ran[0] if len(steps_ran) == 1
                      else (steps_ran or args.max_steps)),
        "win_threshold": args.win_threshold,
        "arms": results,
        "sr_by_arm": scored,
        "winners": winners,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[ab] wrote {args.out}")

    anchor = scored.get(anchor_key)
    if len(winners) == 1:
        print(f"\n[ab] CONCLUSION: cell '{winners[0]}' is the fix (SR "
              f"{scored[winners[0]]:.3f} against the {anchor_key} anchor "
              f"{anchor}). Cell format is "
              f"{payload['cell_key_format']}. Run the four-suite gate with that "
              f"cell and with max_steps left at 0 so each suite gets upstream's "
              f"own budget.")
        return 0
    if not winners:
        print(f"\n[ab] CONCLUSION: NO cell clears {args.win_threshold}. Do not "
              f"ship any of these as 'the fix'. Note what this does and does "
              f"not rule out: the gripper pipeline, the image resize and the "
              f"action decode are all known to differ from upstream -- the first "
              f"two from its source (bolt htrg4uchwi), the third by direct "
              f"measurement (bolt 7vpp28qfsk: 24/24 frames disagree, max "
              f"L-inf 1.12, and upstream holds the gripper at a constant open "
              f"while ours emits 11 different values over the same 24 early "
              f"frames). So a null here means these are real differences that "
              f"are not sufficient, not that they are non-differences. "
              f"Remaining candidates: (a) the per-suite step budget -- this run "
              f"used {args.max_steps}; (b) the checkpoint's norm_stats key "
              f"resolving to a different action space than this suite; (c) no "
              f"single-factor candidate at all, in which case the next step is "
              f"instrumenting one episode against a known-good reference "
              f"trajectory rather than adding a sixth factor to this grid.")
        return 1
    print(f"\n[ab] CONCLUSION: {len(winners)} cells tie ({winners}). Raise "
          f"--n-episodes to separate them before running the full gate.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
