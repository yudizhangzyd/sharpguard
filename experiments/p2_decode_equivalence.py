#!/usr/bin/env python3
"""Is P2's decode the same decode the checkpoint ships?

Context. bolt 7vpp28qfsk measured that `sharpguard/libero_sim.predict_action`
and the checkpoint's own `predict_action` disagree on 24 of 24 frames (max L-inf
1.12, near-zero per-dimension correlation, flat in t -- a wrong function, not a
drift). That defect lives on the rollout path, from which the manuscript
publishes no number. But it raises a question about the path that *does* carry
every published F: P2 decodes through `experiments/cotfaith_edit.infer_action`,
which is a *second*, independent reimplementation --

    model.generate(max_new_tokens=8, do_sample=False), then keep the first 7
    generated ids that happen to fall in [vocab-256, vocab-1], then
    bin = vocab - 1 - id

-- and it has never been compared to upstream. It has no logit mask over the
action window (a non-action token can be emitted and is silently dropped rather
than forbidden), and it does not do the input-id surgery upstream does (OpenVLA
appends the SentencePiece empty token 29871 when the prompt does not end in it,
which shifts what the first generated position even is).

The comparison has one trap. Upstream's `predict_action` normally builds its own
prompt from an instruction, while P2's prompt is the ECoT prompt with the whole
chain-of-thought in the context -- those are different prompts *by design*, so
comparing them would measure the CoT, not the decoder. This script therefore
hands upstream's `predict_action` P2's own `input_ids` and `pixel_values`. Then
the only difference left is decode mechanics.

Two things are reported, and only the second one can move the paper:

  1. token selection. Do the two paths select the same 7 action bins?
  2. does F change? For each (sample, family) pair the faithful flag is
     recomputed from upstream's bins under P2's own de-quantization, so token
     selection is isolated from the convention question that
     `p2_dequant_recompute.py` settles separately. A decoder that picks
     different tokens but flips no flag bounds the exposure; one that flips
     flags means P2 has to be re-run.

Upstream's returned action is un-normalized, so it is inverted back to bins
through a (256, 7) table built from the checkpoint's OWN `bin_centers` and OWN
action stats -- not from an assumed grid. The inversion residual is reported; if
it is not ~0 the inversion failed and no claim is made.

Three properties of the checkpoint's `predict_action` are load-bearing here and
were read out of the shipped remote code rather than assumed. The first run of
this script got all three wrong and compared 0 prompts, which is why they are
written down: (a) it forwards `**kwargs` into `generate()` without setting
`max_new_tokens`, so the caller must -- unset, generate stops at the default
`max_length=20` and raises on a 300-token ECoT prompt; (b) it returns
`(actions, generated_ids)`, not an array; (c) it slices
`generated_ids[0, -(action_dim+1):-1]`, so with `max_new_tokens=8` it reads
exactly the first 7 generated positions, which is the same span P2's filter
reads -- that is what makes the comparison apples to apples. It also indexes the
grid as `clip(self.vocab_size - id - 1, 0, len(bin_centers)-1)` where
`self.vocab_size = text_config.vocab_size - pad_to_multiple_of`, while P2 uses
`processor.tokenizer.vocab_size - 1 - id`; if those differ the two paths index
the grid with a constant offset, so the offset is measured and reported.

Output: p2_decode_equivalence.json plus a verdict. Exit 0 iff the two paths
select identical tokens on every prompt AND no faithful flag differs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.cotfaith_edit import (ECOT_SYSTEM_PROMPT,  # noqa: E402
                                       build_ecot_target_text,
                                       dequantize_action, infer_action,
                                       load_libero_samples)

N_ACT = 7


def _p2_bins_and_action(model, processor, text, image, device, dtype):
    """Mirror of cotfaith_edit.infer_action that also returns the bins.

    Kept a mirror rather than a refactor of the original so that the function
    the paper's numbers were produced by is not edited by this audit. The mirror
    is verified against the original once per run, on the first prompt.
    """
    import torch
    inputs = processor(text, image).to(device, dtype=dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    gen_ids = out[0, -8:].cpu().tolist()
    vocab = processor.tokenizer.vocab_size
    action_lo = vocab - 256
    bins = []
    for tid in gen_ids:
        if action_lo <= tid < vocab:
            bins.append(vocab - 1 - tid)
        if len(bins) == N_ACT:
            break
    if len(bins) < N_ACT:
        return None, None, gen_ids, inputs
    bins = np.asarray(bins, dtype=int)
    return bins, dequantize_action(bins, bins=256), gen_ids, inputs


def _normalized_lut(model, processor):
    """P2 bin index -> the normalized value upstream would produce for it.

    Indexed by *P2's* bin index rather than by token id, because that is what
    makes the two paths comparable: P2 computes `bin = vocab_p - 1 - id` from the
    processor's tokenizer, and the checkpoint's `predict_action` computes
    `clip(vocab_m - id - 1, 0, len(bin_centers) - 1)` from `self.vocab_size`,
    which is `text_config.vocab_size - pad_to_multiple_of`. Those two vocab sizes
    need not be equal, and if they differ the two paths index the grid with a
    constant offset -- a bigger difference than the 2/256-vs-2/255 spacing, so it
    is measured here and reported rather than assumed away.

    Returns (table, source, diagnostics). No fallback grid: if the checkpoint
    does not expose its own centres there is nothing to compare against and the
    caller reports INCONCLUSIVE instead of inventing an answer.
    """
    centers = getattr(model, "bin_centers", None)
    if centers is None:
        return None, "unavailable: model exposes no bin_centers", {}
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    vocab_m = int(getattr(model, "vocab_size"))
    vocab_p = int(processor.tokenizer.vocab_size)
    offset = vocab_m - vocab_p
    p2_bins = np.arange(256)
    idx = np.clip(p2_bins + offset, 0, centers.size - 1)
    diag = {
        "n_bin_centers": int(centers.size),
        "model_vocab_size": vocab_m,
        "processor_vocab_size": vocab_p,
        "bin_index_offset_upstream_minus_p2": offset,
        "bin_index_offset_is_zero": offset == 0,
        "n_p2_bins_clipped_by_upstream": int(np.sum(p2_bins + offset
                                                   > centers.size - 1)),
    }
    return centers[idx], "model.bin_centers (the checkpoint's own grid)", diag


def _action_stats(model, unnorm_key):
    getter = getattr(model, "get_action_stats", None)
    if callable(getter):
        try:
            st = getter(unnorm_key)
            q01 = np.asarray(st["q01"], dtype=np.float64)
            q99 = np.asarray(st["q99"], dtype=np.float64)
            mask = np.asarray(st.get("mask", [True] * q01.size), dtype=bool)
            return q01, q99, mask, "model.get_action_stats"
        except Exception as e:
            print(f"[eq] get_action_stats failed ({type(e).__name__}: {e})")
    from sharpguard.libero_sim import _get_norm_stats
    q01, q99, mask = _get_norm_stats(model, unnorm_key)
    if q01 is None:
        return None, None, None, "unavailable"
    return (np.asarray(q01, dtype=np.float64), np.asarray(q99, dtype=np.float64),
            np.asarray(mask, dtype=bool), "libero_sim._get_norm_stats")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-path", default="Embodied-CoT/ecot-openvla-7b-bridge")
    p.add_argument("--unnorm-key", default="bridge_orig")
    p.add_argument("--n-samples", type=int, default=12)
    p.add_argument("--families", default="subject_swap,location_swap,"
                                        "direction_flip,paraphrase_null",
                   help="a subset is enough: the question is whether ANY flag "
                        "differs, and each family costs one more generate per "
                        "sample per decoder")
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--dataset-repo",
                   default="Embodied-CoT/embodied_features_and_demos_libero")
    p.add_argument("--tfds-subdir", default="libero_lm_90/1.0.0")
    p.add_argument("--reasoning-json", default="libero_reasonings.json")
    p.add_argument("--out", default="p2_decode_equivalence.json")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from sharpguard.attacks import EDIT_FAMILIES

    wanted = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in wanted if f not in EDIT_FAMILIES]
    if unknown:
        sys.exit(f"[eq] unknown edit families {unknown}; "
                 f"expected a subset of {sorted(EDIT_FAMILIES)}")
    families = {k: EDIT_FAMILIES[k] for k in wanted}

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[eq] loading {args.ckpt_path}")
    processor = AutoProcessor.from_pretrained(args.ckpt_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.ckpt_path, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager", low_cpu_mem_usage=True).to(device).eval()

    if not hasattr(model, "predict_action"):
        print("[eq] INCONCLUSIVE: this checkpoint exposes no predict_action, so "
              "there is no upstream decode to compare against. Load it with "
              "trust_remote_code=True, or pick a checkpoint that ships one.")
        return 1

    norm_by_bin, lut_source, lut_diag = _normalized_lut(model, processor)
    q01, q99, mask, stats_source = _action_stats(model, args.unnorm_key)
    print(f"[eq] normalized LUT source : {lut_source}")
    print(f"[eq] action stats source   : {stats_source}")
    if norm_by_bin is None:
        print("[eq] INCONCLUSIVE: the checkpoint does not expose the bin centres "
              "its own predict_action uses, so upstream's un-normalized output "
              "cannot be inverted back to bins without assuming a grid -- and an "
              "assumed grid is exactly what this script exists to avoid.")
        return 1
    for k, v in lut_diag.items():
        print(f"[eq]   {k}: {v}")
    if not lut_diag.get("bin_index_offset_is_zero", True):
        print(f"[eq] NOTE P2 and upstream index the bin grid with a constant "
              f"offset of {lut_diag['bin_index_offset_upstream_minus_p2']} "
              f"(processor vocab {lut_diag['processor_vocab_size']} vs model "
              f"vocab {lut_diag['model_vocab_size']}). A constant offset cancels "
              f"in a paired difference, so it cannot move F_mag, but it does "
              f"move any sign- or norm-based predicate.")
    if q01 is None:
        print(f"[eq] INCONCLUSIVE: no action stats for unnorm_key="
              f"{args.unnorm_key!r}, so upstream's un-normalized output cannot "
              f"be inverted back to bins. Pass the key this checkpoint "
              f"registers.")
        return 1

    # Upstream's whole output space: row b = what predict_action returns if it
    # selects bin b in that dim. Built from the checkpoint's own tokenizer and
    # own stats, so inverting through it assumes nothing about the grid.
    unnorm_table = np.where(mask[None, :],
                            0.5 * (norm_by_bin[:, None] + 1.0) * (q99 - q01) + q01,
                            norm_by_bin[:, None])                  # (256, 7)
    # A dim is "degenerate" only if un-normalization loses distinctions the
    # checkpoint's own grid still had. The grid itself is legitimately coarser
    # than 256: linspace(-1,1,256) yields 255 midpoints, so the top bins collapse
    # and an earlier version of this script wrongly called all seven dims
    # degenerate for that reason and compared nothing at all.
    n_distinct_grid = int(len(np.unique(norm_by_bin)))
    degenerate_dims = [int(i) for i in range(N_ACT)
                       if len(np.unique(unnorm_table[:, i])) < n_distinct_grid]
    n_collapsed_bins = 256 - n_distinct_grid
    p2_table = dequantize_action(np.arange(256), bins=256)
    convention_diff = float(np.max(np.abs(p2_table - norm_by_bin)))
    print(f"[eq] max |P2 value - checkpoint value| over 256 bins: "
          f"{convention_diff:.6f} ({100 * convention_diff / args.tau:.1f}% of "
          f"tau={args.tau})")
    print(f"[eq] the checkpoint's grid has {n_distinct_grid} distinct values "
          f"over 256 bin indices ({n_collapsed_bins} collapse), which is a "
          f"property of linspace(-1,1,256) and not a defect")
    if degenerate_dims:
        print(f"[eq] NOTE dims {degenerate_dims} do not have {n_distinct_grid} distinct "
              f"un-normalized values, so a bin cannot always be recovered "
              f"there; those dims are excluded from bin-level agreement and "
              f"counted separately.")

    def invert(action):
        """upstream action -> (bins, per-dim residual)."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)[:N_ACT]
        d = np.abs(unnorm_table - a[None, :])          # (256, 7)
        b = np.argmin(d, axis=0)
        return b.astype(int), d[b, np.arange(N_ACT)]

    print(f"[eq] loading {args.n_samples} samples (seed={args.seed})")
    samples = list(load_libero_samples(args.dataset_repo, args.tfds_subdir,
                                       args.reasoning_json, args.n_samples,
                                       seed=args.seed))
    print(f"[eq] got {len(samples)} samples\n")

    records = []
    mirror_verified = None
    max_residual = 0.0
    n_prompts = 0
    n_token_identical = 0
    n_raw_ids_identical = 0
    # The naive slice below turned out to be misaligned on every prompt of
    # e2d58fvvn8, so the offset is now measured rather than assumed to be zero.
    n_slice_offset_by_one = 0
    n_upstream_dim0_at_grid_top = 0
    dim_agree = np.zeros(N_ACT, dtype=int)
    dim_absdiff_max = np.zeros(N_ACT, dtype=int)
    cot_changes_action = []

    import random
    rng = random.Random(args.seed + 7)

    for si, (img, instr, gt, fbase, dem) in enumerate(samples):
        try:
            prompt = (f"{ECOT_SYSTEM_PROMPT} USER: What action should the robot "
                      f"take to {instr.lower()}? ASSISTANT: ")
            texts = {"__orig__": prompt + build_ecot_target_text(gt) + " ACTION:"}
            for fname, fedit in families.items():
                if fname == "cross_task_swap":
                    alt = rng.randrange(len(samples))
                    if alt == si and len(samples) > 1:
                        alt = (alt + 1) % len(samples)
                    edited = fedit(gt, alt_reasoning=samples[alt][2], seed=args.seed)
                elif fname in ("syntactic_scramble", "bbox_jitter_null",
                               "instr_random_sub"):
                    edited = fedit(gt, seed=args.seed + si)
                else:
                    edited = fedit(gt)
                if edited is None:
                    continue
                edited.pop("__edit_meta__", None)
                texts[fname] = prompt + build_ecot_target_text(edited) + " ACTION:"

            decoded = {}
            for name, text in texts.items():
                b_p2, a_p2, gen_ids, inputs = _p2_bins_and_action(
                    model, processor, text, img, device, dtype)
                if b_p2 is None:
                    print(f"[eq] sample {si} {name}: P2 decode produced "
                          f"{len(gen_ids)} ids with <7 in the action window; "
                          f"skipping (cotfaith_edit records this as a skip too)")
                    continue

                # Verify the mirror against the function the paper used, once.
                if mirror_verified is None:
                    a_ref = infer_action(model, processor, text, img, device, dtype)
                    mirror_verified = bool(
                        a_ref is not None
                        and np.allclose(np.asarray(a_ref, dtype=float),
                                        np.asarray(a_p2, dtype=float), atol=0))
                    print(f"[eq] mirror reproduces cotfaith_edit.infer_action "
                          f"exactly: {mirror_verified}")

                # predict_action forwards **kwargs straight into generate()
                # WITHOUT setting max_new_tokens, so the caller must: left
                # unset, generate stops at the default max_length=20 and raises
                # on a 300-token ECoT prompt. 8 is not arbitrary -- upstream
                # slices `generated_ids[0, -(action_dim+1):-1]`, so with 8 new
                # tokens it reads exactly the first 7 generated positions, which
                # is the same span P2's filter reads. It also returns
                # (actions, generated_ids), not an array.
                with torch.no_grad():
                    a_up, up_gen_ids = model.predict_action(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        unnorm_key=args.unnorm_key, do_sample=False,
                        max_new_tokens=8)
                b_up, resid = invert(a_up)
                max_residual = max(max_residual, float(np.max(resid)))

                # Both paths ran their own greedy generate() on the same
                # input_ids, so the raw ids should coincide; if they do, any bin
                # difference is post-processing arithmetic rather than token
                # selection, which is a sharper answer than either alone.
                up_tail = up_gen_ids[0, -8:-1].cpu().tolist()
                raw_ids_same = bool(list(gen_ids[:N_ACT]) == list(up_tail))
                n_raw_ids_identical += int(raw_ids_same)

                n_prompts += 1
                good = [i for i in range(N_ACT) if i not in degenerate_dims]
                same = bool(np.array_equal(b_p2[good], b_up[good]))
                n_token_identical += int(same)
                # e2d58fvvn8 reported same=False on 66/66 prompts, and that was
                # this comparison being misaligned rather than the two decodes
                # disagreeing. The model emits one non-action separator token
                # before the seven action tokens, so P2's window filter reads
                # [a1..a7] while upstream's fixed `generated_ids[0, -8:-1]`
                # slice reads [sep, a1..a6]: on all 108 decodes
                # b_up[j+1] == b_p2[j] (up to upstream's 255->254 clip, since
                # its grid has only 255 centres), b_up[0] was the grid top on
                # 108/108 -- the clip of a non-action id, which no real action
                # token can produce -- and b_p2[6] was 255 on 108/108. P2's
                # selection is the correct one. Measured here so the report
                # states the offset instead of publishing a 0/66 that reads as
                # a disagreement.
                top = n_distinct_grid - 1
                shifted = bool(np.array_equal(
                    np.minimum(b_p2[:N_ACT - 1], top), b_up[1:N_ACT]))
                n_slice_offset_by_one += int(shifted)
                n_upstream_dim0_at_grid_top += int(int(b_up[0]) == top)
                for i in good:
                    dim_agree[i] += int(b_p2[i] == b_up[i])
                    dim_absdiff_max[i] = max(dim_absdiff_max[i],
                                             abs(int(b_p2[i]) - int(b_up[i])))
                decoded[name] = {
                    "bins_p2": b_p2.tolist(), "bins_upstream": b_up.tolist(),
                    "tokens_identical": same,
                    "raw_generated_ids_identical": raw_ids_same,
                    "gen_ids_p2": list(gen_ids),
                    "gen_ids_upstream_span": up_tail,
                    "max_bin_residual": float(np.max(resid)),
                    "action_p2": [float(x) for x in a_p2],
                    "action_upstream_raw": [float(x) for x in
                                            np.asarray(a_up, dtype=float).reshape(-1)[:N_ACT]],
                }

            if "__orig__" not in decoded:
                continue

            # Auxiliary: does the CoT context change the action at all, on
            # upstream's own instruction-only prompt? Not the question under
            # test, but a near-free read on whether the CoT is in the loop.
            try:
                from sharpguard.libero_sim import predict_action_upstream
                a_instr = predict_action_upstream(
                    model, processor, np.asarray(img, dtype=np.uint8), instr,
                    device=device, pixel_dtype=dtype, unnorm_key=args.unnorm_key)
                b_instr, _ = invert(a_instr)
                cot_changes_action.append(
                    not np.array_equal(b_instr,
                                       np.asarray(decoded["__orig__"]["bins_upstream"])))
            except Exception as e:
                print(f"[eq] sample {si}: instruction-only aux probe failed "
                      f"({type(e).__name__}: {e})")

            o = decoded["__orig__"]
            for fname in families:
                if fname not in decoded:
                    continue
                e = decoded[fname]
                # Both flags use P2's OWN de-quantization, so any difference is
                # token selection and not the convention.
                d_p2 = dequantize_action(np.asarray(e["bins_p2"]), bins=256) - \
                       dequantize_action(np.asarray(o["bins_p2"]), bins=256)
                d_up = dequantize_action(np.asarray(e["bins_upstream"]), bins=256) - \
                       dequantize_action(np.asarray(o["bins_upstream"]), bins=256)
                linf_p2 = float(np.max(np.abs(d_p2)))
                linf_up = float(np.max(np.abs(d_up)))
                records.append({
                    "sample": si, "family": fname, "file_base": fbase,
                    "instruction": instr[:160],
                    "tokens_identical_orig": o["tokens_identical"],
                    "tokens_identical_edit": e["tokens_identical"],
                    "linf_p2_decode": linf_p2,
                    "linf_upstream_decode": linf_up,
                    "faithful_p2_decode": linf_p2 > args.tau,
                    "faithful_upstream_decode": linf_up > args.tau,
                    "bins_orig_p2": o["bins_p2"],
                    "bins_orig_upstream": o["bins_upstream"],
                    "bins_edit_p2": e["bins_p2"],
                    "bins_edit_upstream": e["bins_upstream"],
                    # Released so a reader can check the span offset themselves:
                    # gen_ids_p2 is the full 8-token generation and
                    # gen_ids_upstream_span is the 7 that upstream's own slice
                    # read. Without these the offset is only inferable from the
                    # bins, which is how it went unnoticed in e2d58fvvn8.
                    "gen_ids_orig_p2": o["gen_ids_p2"],
                    "gen_ids_orig_upstream_span": o["gen_ids_upstream_span"],
                    "gen_ids_edit_p2": e["gen_ids_p2"],
                    "gen_ids_edit_upstream_span": e["gen_ids_upstream_span"],
                })
            if (si + 1) % 4 == 0:
                print(f"[eq] {si + 1}/{len(samples)} samples; "
                      f"{n_token_identical}/{n_prompts} prompts token-identical")
        except Exception as e:
            print(f"[eq] sample {si} failed: {type(e).__name__}: {e}\n"
                  f"{traceback.format_exc()[-400:]}")

    # -------- aggregate --------
    fam_stats = {}
    for fname in families:
        rows = [r for r in records if r["family"] == fname]
        if not rows:
            fam_stats[fname] = {"n": 0}
            continue
        k_p2 = sum(r["faithful_p2_decode"] for r in rows)
        k_up = sum(r["faithful_upstream_decode"] for r in rows)
        fam_stats[fname] = {
            "n": len(rows),
            "F_p2_decode": k_p2 / len(rows),
            "F_upstream_decode": k_up / len(rows),
            "delta_F": (k_up - k_p2) / len(rows),
            "n_flag_differs": sum(r["faithful_p2_decode"] !=
                                  r["faithful_upstream_decode"] for r in rows),
            "max_linf_gap": max(abs(r["linf_upstream_decode"] -
                                    r["linf_p2_decode"]) for r in rows),
        }

    n_flag_differs = sum(v.get("n_flag_differs", 0) for v in fam_stats.values())
    worst_dF, worst_where = 0.0, None
    for fname, v in fam_stats.items():
        if abs(v.get("delta_F") or 0.0) > abs(worst_dF):
            worst_dF, worst_where = v["delta_F"], fname

    payload = {
        "ckpt_path": args.ckpt_path, "unnorm_key": args.unnorm_key,
        "tau": args.tau, "seed": args.seed, "dtype": args.dtype,
        "families": list(families),
        "n_samples_loaded": len(samples),
        "n_prompts_compared": n_prompts,
        "mirror_reproduces_infer_action": mirror_verified,
        "normalized_lut_source": lut_source,
        "action_stats_source": stats_source,
        "degenerate_dims_excluded": degenerate_dims,
        "grid_n_distinct_values": n_distinct_grid,
        "grid_n_collapsed_bins": n_collapsed_bins,
        "lut_diagnostics": lut_diag,
        "max_bin_inversion_residual": max_residual,
        "convention_max_value_diff": convention_diff,
        "convention_max_value_diff_frac_of_tau": convention_diff / args.tau,
        "n_prompts_token_identical": n_token_identical,
        "n_prompts_raw_generated_ids_identical": n_raw_ids_identical,
        # The two keys that say whether `n_prompts_token_identical` is a
        # measurement of P2 or a measurement of this script's slice.
        "n_prompts_upstream_slice_offset_by_one": n_slice_offset_by_one,
        "n_prompts_upstream_dim0_at_grid_top": n_upstream_dim0_at_grid_top,
        "frac_prompts_raw_generated_ids_identical":
            (n_raw_ids_identical / n_prompts) if n_prompts else None,
        "frac_prompts_token_identical": (n_token_identical / n_prompts
                                        if n_prompts else None),
        "per_dim_bin_agreement": [
            (float(dim_agree[i] / n_prompts) if n_prompts and i not in degenerate_dims
             else None) for i in range(N_ACT)],
        "per_dim_max_bin_diff": [int(x) for x in dim_absdiff_max],
        "n_records": len(records),
        "n_faithful_flag_differs": n_flag_differs,
        "worst_delta_F": worst_dF, "worst_delta_F_family": worst_where,
        "per_family": fam_stats,
        "aux_cot_context_changes_upstream_action": (
            float(np.mean(cot_changes_action)) if cot_changes_action else None),
        "aux_n_probed": len(cot_changes_action),
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\n[eq] wrote {args.out}")

    print(f"\n{'=' * 66}\n[eq] VERDICT\n{'=' * 66}")
    print(f"prompts compared                : {n_prompts}")
    print(f"prompts with identical tokens   : {n_token_identical}"
          f"  ({payload['frac_prompts_token_identical']})")
    print(f"prompts with identical raw ids  : {n_raw_ids_identical}"
          f"  (if this is {n_prompts} and the bins still differ, the difference "
          f"is index arithmetic, not token selection)")
    print(f"per-dim bin agreement           : {payload['per_dim_bin_agreement']}")
    print(f"per-dim max bin difference      : {payload['per_dim_max_bin_diff']}")
    print(f"upstream slice off by one       : {n_slice_offset_by_one}/{n_prompts}"
          f"  (upstream dim0 at grid top on {n_upstream_dim0_at_grid_top})")
    print(f"max bin-inversion residual      : {max_residual:.3e}")
    print(f"(sample, family) records        : {len(records)}")
    print(f"faithful flags that differ      : {n_flag_differs}")
    for fname, v in fam_stats.items():
        if v.get("n"):
            print(f"   {fname:22s} n={v['n']:3d}  F_p2={v['F_p2_decode']:.3f}  "
                  f"F_up={v['F_upstream_decode']:.3f}  dF={v['delta_F']:+.3f}")

    if mirror_verified is False:
        print("\n[eq] INCONCLUSIVE: the mirror did not reproduce "
              "cotfaith_edit.infer_action, so the 'P2' arm here is not P2. Fix "
              "the mirror before reading anything else in this report.")
        return 1
    if max_residual > 1e-3:
        print(f"\n[eq] INCONCLUSIVE: upstream's action could not be inverted "
              f"back to bins (residual {max_residual:.3e}). The bin-level "
              f"comparison is unreliable; only the float-space fields are "
              f"meaningful.")
        return 1
    if n_prompts == 0:
        print("\n[eq] INCONCLUSIVE: no prompt was compared.")
        return 1

    # The question Stage B was asked is whether P2's *token selection* -- no
    # logit mask over the action window, no input-id surgery -- picks different
    # tokens than upstream. The sharpest available answer to that is the raw
    # generated ids, because both paths run their own greedy generate() on the
    # same input_ids: if those coincide, selection cannot differ. The bin
    # comparison below it is downstream of a fixed slice that e2d58fvvn8 showed
    # to be misaligned on every prompt, so it is reported but does not decide
    # the verdict.
    if (n_raw_ids_identical == n_prompts and n_flag_differs == 0
            and n_slice_offset_by_one == n_prompts):
        print(f"\n[eq] CONCLUSION: both paths generate byte-identical raw ids on "
              f"all {n_prompts} prompts, so P2's missing logit mask and missing "
              f"input-id surgery do not change which tokens are selected here. "
              f"No faithful flag differs on any of the {len(records)} records. "
              f"The bin-level slice disagrees on {n_prompts - n_token_identical}"
              f"/{n_prompts} prompts for a separate and now-measured reason: "
              f"upstream's `generated_ids[0, -8:-1]` reads one position earlier "
              f"than P2's action-window filter on {n_slice_offset_by_one}"
              f"/{n_prompts} prompts, putting a non-action separator token in "
              f"dim 0 (clipped to the grid top on "
              f"{n_upstream_dim0_at_grid_top}/{n_prompts}) and dropping the "
              f"gripper dim. P2's selection is the correct one. The 24/24 "
              f"mismatch measured by 7vpp28qfsk stays confined to the rollout "
              f"path, which publishes no number, and the de-quantization "
              f"convention remains a separate question "
              f"(p2_dequant_recompute.py).")
        return 0
    if n_token_identical == n_prompts and n_flag_differs == 0:
        print(f"\n[eq] CONCLUSION: P2's decode selects the SAME action tokens as "
              f"the checkpoint's own predict_action on all {n_prompts} prompts, "
              f"and no faithful flag differs. The 24/24 mismatch measured by "
              f"7vpp28qfsk is therefore confined to the rollout path, which "
              f"publishes no number. P2's missing logit mask and missing "
              f"input-id surgery are real differences in the code that do not "
              f"change its output on these prompts -- state that as measured, "
              f"and keep the de-quantization convention question separate "
              f"(p2_dequant_recompute.py).")
        return 0
    if n_flag_differs == 0:
        print(f"\n[eq] CONCLUSION: the two decodes pick different tokens on "
              f"{n_prompts - n_token_identical}/{n_prompts} prompts, but no "
              f"faithful flag differs on any of the {len(records)} records and "
              f"every family keeps its F (worst dF {worst_dF:+.3f}). So P2's "
              f"decode is not upstream's, and the exposure is bounded rather "
              f"than absent: this is one checkpoint and "
              f"{len(records)} records, not the whole leaderboard. Report the "
              f"bound with its N, and do not upgrade it to 'equivalent'.")
        return 1
    print(f"\n[eq] CONCLUSION: the decodes disagree AND it moves the metric -- "
          f"{n_flag_differs} of {len(records)} faithful flags differ, worst dF "
          f"{worst_dF:+.3f} on '{worst_where}'. Every published F is measured on "
          f"a decode that is not the checkpoint's. P2 has to be re-run under "
          f"upstream's decode, at least on a subset large enough to bound the "
          f"shift, before any F value is reported.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
