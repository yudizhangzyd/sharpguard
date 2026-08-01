"""CoT-Faith rollout AUROC: attention pattern as failure predictor.

For each sample, greedy-decode an action conditioned on the CoT, compare it to
the corpus's ground-truth action, and ask whether the attention distribution
separates high-error from low-error samples.

Metric of interest:
  1. Per sample: (attn_action_to_cot, ->visual, ->instr) + L1 action error vs GT
  2. Binarize error at the median -> 'high-error' vs 'low-error'
  3. AUROC of each attention feature as failure predictor.

If action->cot attention distinguishes high-error from low-error samples
(AUROC > 0.6), attention IS predictive of failure -- SAFE/FIPER-style
downstream utility claim. If AUROC ~= 0.5, attention doesn't help predict
per-step action error. This mimics SAFE (2506.09937) / FIPER (2510.09459),
restricted to CoT-VLAs.

PROTOCOL NOTE (this is why P3 was withdrawn and re-run). The first version made
two errors that compound. It compared a NORMALIZED prediction against a
ground-truth action in the robot's own units, so the residual it thresholded was
dominated by a per-dimension constant offset rather than by model error. And it
scored a Bridge-trained checkpoint on LIBERO, where no correct normalization
exists on the checkpoint at all. Both are fixed here: the prediction is
un-normalized with the checkpoint's own q01/q99 for `--unnorm-key`, the run
aborts if that key is absent, and `--corpus bridge_v2` puts the checkpoint on
the corpus its statistics describe. The withdrawn quantity is recomputed on the
same forward passes and reported under `legacy_mixed_space` so the size of the
defect is readable from the artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np


ECOT_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

ECOT_TAGS_ORDER = [
    (("task",),                                    "TASK"),
    (("plan",),                                    "PLAN"),
    (("bboxes",),                                  "VISIBLE OBJECTS"),
    (("subtask_reasoning", "subtask_reason"),      "SUBTASK REASONING"),
    (("subtask",),                                 "SUBTASK"),
    (("movement_reasoning", "move_reasoning", "move_reason"), "MOVE REASONING"),
    (("movement", "move"),                         "MOVE"),
    (("gripper", "gripper_position"),              "GRIPPER POSITION"),
]


def build_target_text(reasoning):
    parts = []
    for keys, tag in ECOT_TAGS_ORDER:
        v = None
        for k in keys:
            if k in reasoning:
                v = reasoning[k]; break
        primary = keys[0]
        if primary == "bboxes":
            b = v
            if isinstance(b, dict):
                v_str = ", ".join(f"{n} {vv}" for n, vv in b.items())
            else:
                v_str = str(b) if b else ""
        elif primary in ("gripper", "gripper_position"):
            v_str = str(list(v)) if isinstance(v, (list, tuple)) else str(v or "")
        elif isinstance(v, dict):
            ks = sorted(v.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
            v_str = ". ".join(str(v[k]) for k in ks)
        else:
            v_str = str(v) if v else ""
        parts.append(f"{tag}: {v_str}")
    return " ".join(parts)


def dequantize(bin_ids, low=-1.0, high=1.0, bins=256):
    return low + (bin_ids + 0.5) * (high - low) / bins


def load_libero_samples(dataset_repo, tfds_subdir, reasoning_json, n_samples, seed=0):
    from sharpguard.hf_retry import snapshot_with_retry
    ds_dir = Path(snapshot_with_retry(repo_id=dataset_repo,
                                        repo_type="dataset"))
    tfds_dir = ds_dir / tfds_subdir
    with open(ds_dir / reasoning_json) as f:
        rdata = json.load(f)
    import tensorflow_datasets as tfds
    from PIL import Image as PILImage
    builder = tfds.builder_from_directory(str(tfds_dir))
    ds = builder.as_dataset(split="train",
                              shuffle_files=(seed != 0),
                              read_config=tfds.ReadConfig(shuffle_seed=seed))
    n = 0
    for ep in ds:
        if n >= n_samples: break
        meta = ep.get("episode_metadata", {})
        file_path = meta.get("file_path").numpy().decode()
        demo_id = int(meta.get("demo_id").numpy())
        file_base = os.path.basename(file_path)
        rep = rdata.get(file_path) or rdata.get(file_base) or {}
        rdemo = rep.get(str(demo_id), {})
        steps = list(ep["steps"].as_numpy_iterator())
        first = steps[0]
        gt = rdemo.get("0", {})
        if not gt: continue
        gt_action = first["action"].astype(np.float32)
        yield (PILImage.fromarray(first["observation"]["image"]).convert("RGB"),
                first["language_instruction"].decode(),
                gt, gt_action, file_base, demo_id)
        n += 1


def identity_norm_stats(model, corpus):
    """The other in-domain case, and it needs no statistics at all.

    Our LIBERO fine-tunes were trained by quantizing the RAW LIBERO action
    clipped to [-1,1] with no dataset normalization at all
    (`cotfaith_train.py:_quantize_action`), so the token grid and the corpus's
    own action units are the SAME frame and the correct map between them is the
    identity.

    Returning mask=all-False makes `unnormalize_action` and `normalize_action`
    pass values through unchanged (the latter still clipping to the grid), so the
    identity is expressed in the same code path as the affine rather than as a
    special case around it.

    The guard is NOT "the checkpoint carries no norm_stats". Our merged models
    inherit `bridge_orig` from the ECoT-bridge base they were LoRA'd from, even
    though training never consulted it -- so an existence test would refuse a
    legitimate run and, worse, teach the next reader that inherited keys mean the
    affine applies. What actually disqualifies the identity is the checkpoint
    carrying statistics that claim to describe the corpus being scored: then
    there are two competing maps and picking the identity is an unchecked
    assertion. Inherited keys for OTHER corpora are recorded in the report as
    unused rather than silently dropped.
    """
    stats = getattr(model, "norm_stats", None) or {}
    claims = sorted(k for k in stats if corpus.split("_")[0] in str(k).lower())
    if claims:
        raise RuntimeError(
            f"--action-scale identity refused: this checkpoint ships action "
            f"statistics that claim to describe corpus={corpus!r} ({claims}), "
            f"so the identity is a competing map rather than the native one. Use "
            f"--unnorm-key {claims[0]!r}.")
    if corpus != "libero":
        raise RuntimeError(
            f"--action-scale identity is only established for corpus=libero "
            f"(that is the corpus our fine-tunes clipped to [-1,1] and "
            f"quantized directly); got corpus={corpus!r}.")
    z = np.zeros(7, dtype=np.float64)
    return z, z, np.zeros(7, dtype=bool), sorted(stats)


def compute_auroc(scores, labels):
    """Non-parametric AUROC via rank-based Mann-Whitney U statistic."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg: return 0.5
    # Count pairs where pos > neg (+ 0.5 for ties).
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n: wins += 1
            elif p == n: wins += 0.5
    return wins / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# Action normalization. This is the protocol fix, and it is the whole reason
# the first P3 run was withdrawn rather than reported.
#
# `dequantize` returns a NORMALIZED action in [-1, 1]: that is the space the
# action tokens live in. A dataset's GT action is in the robot's own units. The
# first version of this script subtracted one from the other. Because the
# missing map is a per-dimension affine, the residual is dominated by a
# CONSTANT offset, and the AUROC labels here are a median split on that
# residual -- so the labels were mostly reporting which dimension had the
# largest offset, not which sample the model got wrong.
#
# The map is only defined if the checkpoint carries statistics for the corpus
# being evaluated. `ecot-openvla-7b-bridge` carries Bridge statistics and the
# first run scored it on LIBERO, so no correct map existed at all. Hence
# --corpus bridge_v2: evaluate the checkpoint on the corpus its own statistics
# describe. Scoring on LIBERO now requires --allow-cross-domain and says so in
# the report.
# ---------------------------------------------------------------------------

def action_norm_stats(model, unnorm_key):
    """(q01, q99, mask) for `unnorm_key`, or a hard failure naming the keys the
    checkpoint actually has. A silent fallback here is what produced a
    published number in the wrong units."""
    stats = getattr(model, "norm_stats", None) or {}
    if unnorm_key not in stats:
        raise RuntimeError(
            f"checkpoint has no action statistics for unnorm_key="
            f"{unnorm_key!r}. Without them the predicted action cannot be put "
            f"in the same units as the ground-truth action, and the error this "
            f"script thresholds is not an error. Available keys: "
            f"{sorted(stats)[:24]}")
    s = stats[unnorm_key]
    s = s["action"] if isinstance(s, dict) and "action" in s else s
    q01 = np.asarray(s["q01"], dtype=np.float64).reshape(-1)
    q99 = np.asarray(s["q99"], dtype=np.float64).reshape(-1)
    m = s.get("mask")
    mask = (np.ones_like(q01, dtype=bool) if m is None
            else np.asarray(m, dtype=bool).reshape(-1))
    return q01, q99, mask


def unnormalize_action(a_norm, q01, q99, mask):
    """OpenVLA's own convention, verbatim: masked dimensions (the gripper) are
    passed through rather than rescaled."""
    return np.where(mask, 0.5 * (np.asarray(a_norm, dtype=np.float64) + 1.0)
                    * (q99 - q01) + q01, a_norm)


def normalize_action(a_raw, q01, q99, mask):
    """Inverse of the above, clipped to the token grid's range. Used to build
    the teacher-forced action tokens for the attention probe -- which the first
    version quantized directly from the raw action, i.e. off the grid."""
    span = np.where((q99 - q01) == 0.0, 1.0, q99 - q01)
    a = np.asarray(a_raw, dtype=np.float64)
    return np.clip(np.where(mask, 2.0 * (a - q01) / span - 1.0, a), -1.0, 1.0)


def error_baselines(pred_mat, gt_mat):
    """Policy L1 error next to the two constants any policy must beat.

    `predict_mean` is the load-bearing one. The withdrawn P3 run was caught not
    by reading a config but by noticing its policy was ~13x WORSE than the
    dataset mean: when prediction and target sit in different units, a constant
    wins. `predict_zero` is reported alongside because on LIBERO the per-step
    deltas are small and near-zero-centred, so it is the number that shows how
    much of `predict_mean`'s strength is just "actions are tiny".
    """
    gt_mean = gt_mat.mean(axis=0)
    b = {
        "policy":       float(np.mean(np.abs(pred_mat - gt_mat))),
        "predict_mean": float(np.mean(np.abs(gt_mean[None, :] - gt_mat))),
        "predict_zero": float(np.mean(np.abs(gt_mat))),
    }
    b["policy_over_predict_mean"] = (
        b["policy"] / b["predict_mean"] if b["predict_mean"] > 0
        else float("inf"))
    return b


def policy_signal_check(frames, scored_as):
    """Did the policy beat predicting the dataset mean, and if not, why not?

    `frames` maps a frame name to `error_baselines(...)` for that frame, all
    computed from the SAME predictions. Splitting the failure in two is the
    point. A single frame's ratio cannot distinguish

      * WRONG FRAME -- we de-quantized in the wrong units, so some competing
        frame the checkpoint ships does beat the mean. This is the defect that
        withdrew P3, and it is fixable by using that frame.
      * NOT A FRAME ERROR -- no frame beats the mean, so the scale is not what
        is wrong and the policy's open-loop action prediction simply carries
        too little signal to threshold. Nothing to fix; P3 stays withdrawn on
        this checkpoint, for a reason about competence rather than units.

    Both block a P3 row. Reporting them as one verdict would have let the second
    be written up as the first.
    """
    here = frames[scored_as]
    best = min(frames, key=lambda k: frames[k]["policy_over_predict_mean"])
    best_ratio = frames[best]["policy_over_predict_mean"]
    passed = here["policy"] < here["predict_mean"]
    if passed:
        diagnosis = (f"policy beats predict-the-mean by "
                     f"{1.0 / here['policy_over_predict_mean']:.2f}x in the "
                     f"{scored_as} frame")
    elif best_ratio < 1.0:
        diagnosis = (
            f"WRONG FRAME: scored in {scored_as} the policy loses to the "
            f"dataset mean (ratio {here['policy_over_predict_mean']:.3f}), but "
            f"the competing frame {best} beats it (ratio {best_ratio:.3f}). The "
            f"predictions were de-quantized in the wrong units -- the defect "
            f"that withdrew P3 -- and {best} is the map to use.")
    else:
        diagnosis = (
            f"NOT A FRAME ERROR: no frame this checkpoint offers beats the "
            f"dataset mean (best is {best} at ratio {best_ratio:.3f}). The "
            f"scale is not what is wrong; this checkpoint's open-loop "
            f"single-step action prediction is no better than a constant, so a "
            f"per-sample action error computed from it carries too little "
            f"policy signal to threshold. P3 stays withdrawn on this "
            f"checkpoint for a reason about competence, not units.")
    return {
        "check": "policy_beats_predict_mean",
        "passed": bool(passed),
        "measured": {"scored_frame": scored_as,
                     "ratio": here["policy_over_predict_mean"],
                     "best_frame": best,
                     "best_frame_ratio": best_ratio},
        "threshold": "policy_over_predict_mean < 1.0",
        "diagnosis": diagnosis,
    }


def load_bridge_v2(dataset_repo, n_samples, seed=0):
    """In-domain corpus for the Bridge-trained ECoT checkpoint. Bridge carries
    no ground-truth CoT, so the CoT is self-generated -- which is also the only
    setting a deployed failure predictor ever sees."""
    sys.path.insert(0, str(_ROOT / "experiments"))
    from cotfaith_bridge import load_bridge_v2_samples
    for s in load_bridge_v2_samples(n_samples, seed=seed,
                                    dataset_repo=dataset_repo):
        yield (s["image"], s["instruction"], None,
               np.asarray(s["action"], dtype=np.float32),
               f"bridge_ep{s.get('episode')}", s.get("episode"))


def run(args):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from sharpguard.proguard import RVisHook, RVisConfig, CotAttentionAnalyzer

    dtype = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[auroc] loading model from {args.ckpt_path}")
    processor = AutoProcessor.from_pretrained(args.ckpt_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.ckpt_path, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device).eval()

    hook = RVisHook(model, RVisConfig(
        layers=tuple(int(x) for x in args.rvis_layers.split(",")),
        n_visual_tokens=256,
    ))
    analyzer = CotAttentionAnalyzer(hook, processor.tokenizer, n_visual=256)

    # ---- protocol preflight, before a single sample is scored ----
    inherited_keys = []
    if args.action_scale == "identity":
        q01, q99, mask, inherited_keys = identity_norm_stats(model, args.corpus)
        if inherited_keys:
            # Recorded, not hidden: the merged checkpoint carries the ECoT base's
            # keys because LoRA merging preserves the config, and the reader is
            # entitled to see that the identity was chosen over a key that was
            # physically present.
            print(f"[auroc] checkpoint carries UNUSED norm_stats for "
                  f"{inherited_keys} (inherited from the base it was LoRA'd "
                  f"from; none describes corpus={args.corpus}). Identity map is "
                  f"justified below by a measured frame match, not by absence.")
    else:
        q01, q99, mask = action_norm_stats(model, args.unnorm_key)
    print(f"[auroc] corpus={args.corpus}  unnorm_key={args.unnorm_key}  "
          f"action_scale={args.action_scale}")
    print(f"[auroc]   q01  = {np.round(q01, 4).tolist()}")
    print(f"[auroc]   q99  = {np.round(q99, 4).tolist()}")
    print(f"[auroc]   mask = {mask.tolist()}")
    # The identity path is in-domain BY CONSTRUCTION, so it must not trip the
    # cross-domain guard that exists to catch a borrowed percentile set.
    cross_domain = (args.corpus == "libero" and args.action_scale != "identity"
                    and "libero" not in args.unnorm_key)
    if cross_domain and not args.allow_cross_domain:
        raise RuntimeError(
            f"refusing to score corpus={args.corpus} with unnorm_key="
            f"{args.unnorm_key!r}: the normalization is from a different "
            f"corpus, so the un-normalized prediction and the ground-truth "
            f"action are in different units and the AUROC labels would be a "
            f"median split on that offset. This is the defect that withdrew "
            f"P3. Use --corpus bridge_v2 for a Bridge checkpoint, or pass "
            f"--allow-cross-domain to reproduce the withdrawn run on purpose.")

    per_sample = []
    if args.corpus == "bridge_v2":
        it = load_bridge_v2(args.bridge_repo, args.n_samples, seed=args.seed)
    else:
        it = load_libero_samples(args.dataset_repo, args.tfds_subdir,
                                 args.reasoning_json, args.n_samples,
                                 seed=args.seed)
    n_cot_selfgen = 0
    # Frame-match evidence for the identity path, counted rather than assumed.
    # If the corpus's raw actions were NOT already inside the token grid's
    # range, the identity would be as wrong as a borrowed percentile set and the
    # residual would again be dominated by scale. This is checked against the
    # data instead of inferred from the training script.
    n_gt_outside_grid = 0
    gt_abs_max = 0.0
    for si, (img, instr, gt, gt_action, fbase, dem) in enumerate(it):
        try:
            am = float(np.max(np.abs(gt_action)))
            gt_abs_max = max(gt_abs_max, am)
            if am > 1.0:
                n_gt_outside_grid += 1
            prompt = (f"{ECOT_SYSTEM_PROMPT} USER: What action should the "
                       f"robot take to {instr.lower()}? ASSISTANT: ")
            if gt:
                cot_body = build_target_text(gt)
            else:
                # No ground-truth CoT (Bridge). Let the model author its own,
                # then score the action it emits conditioned on it.
                from cotfaith_bridge import (build_prompt as _bp,
                                             parse_generated_cot as _pc)
                with torch.no_grad():
                    g = model.generate(
                        **processor(_bp(instr), img).to(device, dtype=dtype),
                        max_new_tokens=args.cot_max_new_tokens, do_sample=False)
                txt = processor.tokenizer.decode(g[0], skip_special_tokens=True)
                cot_body = build_target_text(_pc("TASK:" + txt.split("TASK:")[-1]))
                n_cot_selfgen += 1

            # Predict action via greedy generation.
            infer_txt = prompt + cot_body + " ACTION:"
            proc = processor(infer_txt, img).to(device, dtype=dtype)
            with torch.no_grad():
                out = model.generate(**proc, max_new_tokens=8, do_sample=False)
            gen_ids = out[0, -8:].cpu().tolist()
            vocab = processor.tokenizer.vocab_size
            action_lo = vocab - 256
            bins = [vocab - 1 - t for t in gen_ids if action_lo <= t < vocab][:7]
            if len(bins) < 7: continue
            pred_norm = dequantize(np.asarray(bins))
            pred_action = unnormalize_action(pred_norm, q01, q99, mask)

            # Action error vs GT, in the robot's own units.
            error_l1 = float(np.mean(np.abs(pred_action - gt_action)))
            error_linf = float(np.max(np.abs(pred_action - gt_action)))
            # The withdrawn run's quantity, kept so the withdrawal is checkable
            # from this artifact rather than only from the prose.
            legacy_l1 = float(np.mean(np.abs(pred_norm - gt_action)))

            # Capture attention via teacher-forced forward on cot+action. The
            # GT action has to be mapped ONTO the token grid first; quantizing
            # the raw action, as the first version did, put most dimensions in
            # the clipped end bins and made the probe's action block nearly
            # constant across samples.
            gt_norm = normalize_action(gt_action, q01, q99, mask)
            a_ids = np.clip(np.floor((gt_norm + 1) / 2 * 256).astype(np.int64), 0, 255)
            a_ids = vocab - 1 - a_ids
            input_ids = proc["input_ids"][0].to("cpu")
            eos = processor.tokenizer.eos_token_id
            full_ids = torch.cat([input_ids,
                                    torch.from_numpy(a_ids).to(input_ids.dtype),
                                    torch.tensor([eos], dtype=input_ids.dtype)])
            full_ids = full_ids.unsqueeze(0).to(device)
            attn_mask = torch.ones_like(full_ids)
            pixel = proc["pixel_values"].to(device, dtype=dtype)
            hook.clear()
            with torch.no_grad():
                _ = model(input_ids=full_ids, attention_mask=attn_mask,
                            pixel_values=pixel, output_attentions=True)
            seg = analyzer.compute_segments(full_ids[0], action_len=7)
            stats = analyzer.analyze(seg)

            per_sample.append({
                "sample": si,
                "instruction": instr[:200],
                "file_base": fbase,
                "cot_self_generated": not bool(gt),
                "action_error_l1":   error_l1,
                "action_error_linf": error_linf,
                "action_error_l1_mixed_space_LEGACY": legacy_l1,
                "action_pred_normalized": [float(x) for x in pred_norm],
                "action_pred":  [float(x) for x in pred_action],
                "action_gt":    [float(x) for x in gt_action],
                "action_gt_normalized": [float(x) for x in gt_norm],
                "action->cot":         stats.get("action->cot"),
                "action->visual":      stats.get("action->visual"),
                "action->instr":       stats.get("action->instr"),
                "action->action_prev": stats.get("action->action_prev"),
            })
            if (si + 1) % 20 == 0:
                print(f"[auroc] {si+1}/{args.n_samples} samples done")
        except Exception as e:
            print(f"[auroc] sample {si} failed: {e}\n{traceback.format_exc()[-400:]}")

    hook.close()

    # An empty run is a harness failure, not a null result. The withdrawn P3
    # row exists because a report was written from whatever survived.
    if len(per_sample) < 20:
        raise RuntimeError(
            f"only {len(per_sample)} of {args.n_samples} samples scored. An "
            f"AUROC over a median split needs both classes populated; below ~20 "
            f"samples the statistic is noise. No report written.")

    # ---- frame preconditions ----------------------------------------------
    # These used to `raise` before writing anything, which was backwards: the
    # first time one of them fired for real (bolt h3yb3s23qd) it destroyed the
    # only evidence for its own verdict, leaving a traceback in a log as the
    # sole record of a measurement worth releasing. They now record, write the
    # report, and exit non-zero. A failed frame check is a finding about the
    # checkpoint, not a crash.
    frame_checks = []

    # (1) Does the corpus's own action live inside the token grid? If not, the
    # identity map is as wrong as a borrowed percentile set and the residual is
    # again dominated by scale. 5% tolerance because LIBERO's gripper is +/-1
    # exactly and float round-trips can land a hair outside.
    frac_out = n_gt_outside_grid / max(1, len(per_sample))
    if args.action_scale == "identity":
        frame_checks.append({
            "check": "gt_actions_inside_token_grid",
            "passed": frac_out <= 0.05,
            "measured": {"frac_outside": frac_out,
                         "n_outside": n_gt_outside_grid,
                         "gt_abs_max": gt_abs_max},
            "threshold": "frac_outside <= 0.05",
        })

    # (2) The harder one, and the only one that separates "right frame" from
    # "plausible frame": the policy must beat predicting the dataset mean.
    gt_mat = np.asarray([s["action_gt"] for s in per_sample], dtype=np.float64)
    pred_mat = np.asarray([s["action_pred"] for s in per_sample], dtype=np.float64)
    norm_mat = np.asarray([s["action_pred_normalized"] for s in per_sample],
                          dtype=np.float64)
    baselines = error_baselines(pred_mat, gt_mat)

    # ...and, when it fails, whether ANY frame does better. A single frame's
    # ratio cannot tell "we de-quantized in the wrong units" (some competing
    # frame would win) from "this policy is simply weak open-loop" (none does).
    # Both are reasons not to publish a P3 row, but they are different findings,
    # so every frame the checkpoint physically ships is scored on these exact
    # same predictions rather than argued about.
    scored_as = ("identity" if args.action_scale == "identity"
                 else f"unnorm:{args.unnorm_key}")
    frames = {scored_as: baselines}
    for key in sorted(getattr(model, "norm_stats", None) or {}):
        name = f"unnorm:{key}"
        if name in frames:
            continue
        try:
            aq01, aq99, amask = action_norm_stats(model, key)
            alt = np.stack([unnormalize_action(r, aq01, aq99, amask)
                            for r in norm_mat])
        except Exception as e:                              # noqa: BLE001
            print(f"[auroc] competing frame {key!r} not scorable: {e}")
            continue
        frames[name] = error_baselines(alt, gt_mat)

    print(f"[auroc] action-error baselines (L1, corpus units): {baselines}")
    for name, b in sorted(frames.items(),
                          key=lambda kv: kv[1]["policy_over_predict_mean"]):
        print(f"[auroc]   frame {name:24s} policy={b['policy']:.4f}  "
              f"mean={b['predict_mean']:.4f}  "
              f"ratio={b['policy_over_predict_mean']:.3f}"
              f"{'   <-- scored here' if name == scored_as else ''}")

    frame_checks.append(policy_signal_check(frames, scored_as))
    diagnosis = frame_checks[-1]["diagnosis"]

    frame_check = {
        "why": "A P3 row is only meaningful if the action error it thresholds "
               "is in one frame and reflects the policy. Both preconditions are "
               "measured here and released whether or not they hold.",
        "passed": all(c["passed"] for c in frame_checks),
        "checks": frame_checks,
        "baselines_by_frame": frames,
        "diagnosis": diagnosis,
    }
    if not frame_check["passed"]:
        print(f"\n[auroc] *** FRAME CHECK FAILED *** {diagnosis}")
        print("[auroc] report still written; exiting 3. This artifact is a "
              "null result for P3, not a P3 row.")

    # Compute AUROC of each attention feature as high-error predictor.
    errors = [s["action_error_l1"] for s in per_sample]
    median_err = float(np.median(errors))
    labels = [1 if s["action_error_l1"] > median_err else 0 for s in per_sample]

    # The same split on the withdrawn run's mixed-space error, so the two
    # protocols can be compared inside one artifact instead of across runs.
    legacy = [s["action_error_l1_mixed_space_LEGACY"] for s in per_sample]
    legacy_median = float(np.median(legacy))
    legacy_labels = [1 if e > legacy_median else 0 for e in legacy]
    n_label_disagree = sum(1 for a, b in zip(labels, legacy_labels) if a != b)

    aurocs = {}
    legacy_aurocs = {}
    for feat in ["action->cot", "action->visual", "action->instr",
                  "action->action_prev"]:
        scores = [s[feat] for s in per_sample if s.get(feat) is not None]
        # AUROC works both directions; report max(auc, 1-auc) as absolute.
        if len(scores) != len(labels): continue
        raw = compute_auroc(scores, labels)
        aurocs[feat] = {
            "raw_auroc": float(raw),
            "abs_auroc": float(max(raw, 1.0 - raw)),
            "direction": "high-score → high-error" if raw > 0.5 else "low-score → high-error",
        }
        lraw = compute_auroc(scores, legacy_labels)
        legacy_aurocs[feat] = {"raw_auroc": float(lraw),
                               "abs_auroc": float(max(lraw, 1.0 - lraw))}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "cot_auroc_report.json").write_text(json.dumps({
        "n_samples": len(per_sample),
        "seed": args.seed,
        "ckpt_path": args.ckpt_path,
        "protocol": {
            "corpus": args.corpus,
            "unnorm_key": args.unnorm_key,
            "action_scale": args.action_scale,
            "in_domain": not cross_domain,
            "cross_domain_override": bool(args.allow_cross_domain),
            "error_space": (
                "robot units == token-grid units (this checkpoint was trained "
                "on raw actions clipped to [-1,1] with no dataset "
                "normalization, so the map between the two frames is the "
                "identity; verified by measurement, not by config inspection)"
                if args.action_scale == "identity" else
                "robot units (prediction un-normalized with the "
                "checkpoint's own q01/q99 for unnorm_key)"),
            "q01": q01.tolist(), "q99": q99.tolist(), "mask": mask.tolist(),
            "cot_source": ("self-generated" if n_cot_selfgen else
                           "ground-truth annotations"),
            "n_cot_self_generated": n_cot_selfgen,
            # Present but unused. A merged LoRA keeps the base checkpoint's
            # config, so our LIBERO fine-tunes still advertise the ECoT base's
            # bridge_orig statistics even though `_quantize_action` never reads
            # them. Released because "the identity was chosen while a norm_stats
            # key was sitting right there" is exactly the fact a skeptical
            # reader needs, and an existence-based guard would have wrongly
            # refused this run.
            "inherited_unused_norm_stats_keys": inherited_keys,
            # Measured frame-match evidence for the identity path. Released
            # whichever path ran, so a reader can check the precondition on the
            # affine path too rather than only where it is load-bearing.
            "n_gt_actions_outside_token_grid": n_gt_outside_grid,
            "gt_action_abs_max": gt_abs_max,
        },
        # The frame claim's real evidence: a policy scored in its own training
        # frame beats a constant. The withdrawn run was ~13x worse than
        # predict-the-mean, which is how the mixed-space defect was found.
        "action_error_baselines_l1": baselines,
        # Verdict on that evidence, plus the same baselines under every
        # competing frame the checkpoint ships. `passed: false` means this
        # artifact is a null result for P3 and must not be cited as a P3 row --
        # scripts/verify_paper_numbers.py enforces that.
        "frame_check": frame_check,
        "median_error_l1": median_err,
        "aurocs": aurocs,
        # The withdrawn protocol, recomputed on the same forward passes.
        "legacy_mixed_space": {
            "why": "prediction left in normalized [-1,1] space and subtracted "
                   "from a ground-truth action in robot units; this is the "
                   "defect that withdrew P3",
            "median_error_l1": legacy_median,
            "n_samples_whose_label_flips_vs_corrected": n_label_disagree,
            "aurocs": legacy_aurocs,
        },
        "per_sample": per_sample,
    }, indent=2, default=str))

    print(f"\n===== ROLLOUT-STYLE AUROC DONE  seed={args.seed} =====")
    print(f"  corpus={args.corpus}  unnorm_key={args.unnorm_key}  "
          f"in_domain={not cross_domain}")
    print(f"  n_samples used: {len(per_sample)}  "
          f"(CoT self-generated on {n_cot_selfgen})")
    print(f"  median action L1 error: {median_err:.4f}  (robot units)")
    for feat, a in aurocs.items():
        print(f"  {feat:22s}  AUROC={a['raw_auroc']:.3f}  abs={a['abs_auroc']:.3f}  ({a['direction']})")
    print(f"  --- withdrawn mixed-space protocol, same forward passes ---")
    print(f"  median mixed-space L1: {legacy_median:.4f}; "
          f"{n_label_disagree}/{len(per_sample)} labels flip vs corrected")
    for feat, a in legacy_aurocs.items():
        print(f"  {feat:22s}  AUROC={a['raw_auroc']:.3f}  abs={a['abs_auroc']:.3f}")
    print(f"  report -> {out / 'cot_auroc_report.json'}")
    if not frame_check["passed"]:
        print(f"  FRAME CHECK FAILED: {frame_check['diagnosis']}")
        print(f"  -> exiting 3; this is a released null result, not a P3 row.")
    sys.stdout.flush(); os._exit(0 if frame_check["passed"] else 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--out", default="./cotfaith-auroc")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rvis-layers", default="0,1,2,3")
    p.add_argument("--corpus", default="bridge_v2",
                     choices=["bridge_v2", "libero"],
                     help="Evaluation corpus. Default bridge_v2: the in-domain "
                          "corpus for a Bridge-trained ECoT checkpoint.")
    p.add_argument("--unnorm-key", default="bridge_orig",
                     help="Which of the checkpoint's action statistics to "
                          "un-normalize with. Must exist or the run aborts.")
    p.add_argument("--action-scale", default="unnorm",
                     choices=["unnorm", "identity"],
                     help="How to put the predicted action in the corpus's "
                          "units. 'unnorm' (default) uses the checkpoint's own "
                          "q01/q99 for --unnorm-key. 'identity' is for a "
                          "checkpoint trained on raw actions clipped to [-1,1] "
                          "with no dataset normalization -- ours -- where the "
                          "token grid IS the corpus frame; it is refused on any "
                          "checkpoint that ships norm_stats, and on any corpus "
                          "other than libero.")
    p.add_argument("--allow-cross-domain", action="store_true",
                     help="Score a corpus the unnorm_key does not describe. "
                          "Only for reproducing the withdrawn P3 run.")
    p.add_argument("--bridge-repo", default="IPEC-COMMUNITY/bridge_orig_lerobot")
    p.add_argument("--cot-max-new-tokens", type=int, default=1024)
    p.add_argument("--dataset-repo",
                     default="Embodied-CoT/embodied_features_and_demos_libero")
    p.add_argument("--tfds-subdir", default="libero_lm_90/1.0.0")
    p.add_argument("--reasoning-json", default="libero_reasonings.json")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
