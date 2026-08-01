#!/usr/bin/env python3
"""Is our DeepThinkVLA decode path the same path the checkpoint ships?

`experiments/p2_decode_equivalence.py` answers this question for the OpenVLA
family and explicitly declines to answer it here: its README records that
records off P2's bin grid "pass through untouched and are counted in
GRID_PASSTHROUGH ... so a report from a checkpoint with a different action
tokenizer (DeepThinkVLA's FAST tokenizer, for instance) is not corrupted by the
conversion." That is a correct refusal, and it leaves the equivalence bound in
the manuscript scoped to one checkpoint family. This script closes the gap for
the other one.

What is NOT worth testing. `decode.decode_action_chunk` is arithmetic-identical
to the slice inside upstream's `predict_cot_action` -- same reversal
`(end - begin) - argmax`, same clip, same `bin_centers` lookup, same reshape.
Comparing the two would be comparing a transcription to its source and would
pass by construction.

What IS worth testing is the part that has no upstream counterpart in our code
path: PROMPT AND CoT ASSEMBLY. Upstream generates its own reasoning inside
`predict_cot_action` and then decodes actions from whatever sequence generation
happened to leave behind. Our harness never generates -- P2 *injects* a CoT and
calls `prompt_cot_predict_action` on a sequence we assembled ourselves in
`decode.build_input_cot_ids`:

    [prompt] + [<think>] + cot + [</think>, <action>]

Every element of that is a claim about upstream: that the think delimiters are
those two ids and not literal text, that `<action>` terminates the sequence, and
that the prompt is untouched. Getting exactly this class of thing wrong is what
made all three DeepThinkVLA edit runs report n=0 for all 11 families while
exiting 0, so it is measured rather than argued.

The comparison is possible because upstream's stopping criterion is a *sequence*
criterion -- `SeqEosTokenCriteria([think_end_token_index,
action_start_token_index])` fires only when the tail is exactly
`[</think>, <action>]`. So when generation stops on its own, the checkpoint's own
sequence has the shape our assembler produces, and the two can be compared
token for token. Per sample:

  1. run the checkpoint's end-to-end `predict_cot_action` and keep BOTH of its
     returns: the (10, 7) normalized chunk and the `input_cot_ids` it decoded it
     from;
  2. take the CoT it generated, hand it to `build_input_cot_ids`, and require the
     result to be byte-identical to upstream's own sequence;
  3. require `action_start_idx` to agree;
  4. re-run `prompt_cot_predict_action` on that sequence, decode it through
     `decode_action_chunk`, and require the (10, 7) chunk to match upstream's
     exactly;
  5. run `segment_boundaries` on it, which is what P1's four-bucket attention
     split depends on, and require it not to raise.

A control separates the two ways step 4 can fail: the same forward pass is run
twice and its two decodes compared. bf16 matmuls are not required to be bitwise
reproducible, so a mismatch with `determinism_ok == False` is a property of the
hardware and a mismatch with `determinism_ok == True` is a bug in our decode.
Without the control the second would be dismissable as the first.

Samples where generation never emitted the stop tail are counted, not silently
dropped: for those, upstream decoded from a sequence ending mid-CoT that no
assembler of ours would build, so the comparison is undefined rather than passed.

Exit 0 iff at least one sample was comparable and every comparable sample agreed
on all four counts.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.cotfaith_deepthink import load_libero_samples  # noqa: E402


def strip_pads(ids, pad_id):
    """Drop pad columns from a (1, L) id tensor, preserving order."""
    row = ids[0].tolist()
    return [t for t in row if t != pad_id]


def compare_one(model, processor, dtdec, torch, np_centers, img, instr, args,
                dtype) -> dict:
    """One sample: upstream end-to-end vs our assemble-and-decode path."""
    from sharpguard.vendor.deepthinkvla import (ACTION_DIM, ACTION_START,
                                                NUM_ACTIONS_CHUNK, THINK_END,
                                                THINK_START)
    from transformers import GenerationConfig

    device = next(model.parameters()).device
    pad_id = getattr(model, "pad_token_id", None)
    if pad_id is None:
        pad_id = processor.tokenizer.pad_token_id
    rec: dict = {"instruction": instr}

    prompt_text = dtdec.build_prompt_text(instr, n_images=1)
    proc = processor(text=[prompt_text], images=img, return_tensors="pt")
    prompt_ids = proc["input_ids"].to(device)
    pixel = proc["pixel_values"].to(device, dtype=dtype)
    rec["prompt_len"] = int(prompt_ids.shape[1])

    # ---- 1. the checkpoint's own end-to-end path --------------------------
    # Greedy, because the two paths must see the same CoT: with sampling, the
    # sequence upstream decoded from and the sequence we reassemble could differ
    # for a reason that has nothing to do with assembly.
    gcfg = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False,
                            num_beams=1, use_cache=True, pad_token_id=pad_id)
    with torch.no_grad():
        chunk_up, ids_up = model.predict_cot_action(
            input_ids=prompt_ids, pixel_values=pixel,
            attention_mask=torch.ones_like(prompt_ids),
            generation_config=gcfg)
    chunk_up = np.asarray(chunk_up, dtype=np.float64)
    up = strip_pads(ids_up, pad_id)
    rec["upstream_len"] = len(up)
    rec["n_generated"] = len(up) - int(prompt_ids.shape[1])

    # The prompt must survive generation unchanged, or the CoT we extract below
    # is offset and every later comparison is measuring the offset.
    p_list = prompt_ids[0].tolist()
    rec["prompt_preserved"] = bool(up[:len(p_list)] == p_list)

    # ---- 2. did upstream stop on its own stop tail? -----------------------
    # `[</think>, <action>]` is upstream's whole stopping criterion. When it
    # never fires, generation was cut by max_new_tokens and upstream decoded
    # actions from a sequence ending mid-CoT -- a sequence build_input_cot_ids
    # would never produce, so there is nothing here to agree or disagree with.
    tail_ok = len(up) >= 2 and up[-2:] == [THINK_END, ACTION_START]
    head_ok = rec["n_generated"] >= 1 and up[len(p_list)] == THINK_START
    rec["stop_tail_ok"] = bool(tail_ok)
    rec["think_start_first"] = bool(head_ok)
    if not tail_ok:
        rec["comparable"] = False
        rec["reason"] = (f"generation did not end on [</think>, <action>]; last "
                         f"ids {up[-3:]}. Raise --max-new-tokens or accept that "
                         f"this frame's upstream decode has no assembled "
                         f"counterpart.")
        return rec
    rec["comparable"] = True

    # ---- 3. our assembly, from the CoT upstream itself generated -----------
    # `up[len(prompt) + 1 : -2]` strips the leading <think> and the trailing
    # [</think>, <action>] -- exactly the three tokens build_input_cot_ids adds
    # back. If it adds them in a different order or omits one, the tensors below
    # differ in shape or content and this is where that shows up.
    cot_body = up[len(p_list) + 1:-2] if head_ok else up[len(p_list):-2]
    rec["n_cot_tokens"] = len(cot_body)
    ours = dtdec.build_input_cot_ids(prompt_ids, cot_body, torch)
    ours_list = ours[0].tolist()
    rec["ours_len"] = len(ours_list)
    rec["ids_equal"] = bool(ours_list == up)
    if not rec["ids_equal"]:
        first = next((i for i, (a, b) in enumerate(zip(ours_list, up)) if a != b),
                     min(len(ours_list), len(up)))
        rec["first_id_divergence"] = {
            "index": int(first),
            "ours": ours_list[first:first + 4],
            "upstream": up[first:first + 4],
        }

    # ---- 4. the forward pass and our decode of it --------------------------
    ids_t = ours if rec["ids_equal"] else torch.tensor(
        [up], dtype=prompt_ids.dtype, device=device)
    mask = torch.ones_like(ids_t)
    with torch.no_grad():
        logits, start = model.prompt_cot_predict_action(
            input_cot_ids=ids_t, pixel_values=pixel, attention_mask=mask)
    chunk_ours = np.asarray(
        dtdec.decode_action_chunk(logits, start, torch, np_centers),
        dtype=np.float64)
    rec["action_start_idx"] = int(start[0])
    rec["action_start_expected"] = int(ids_t.shape[1] - 1)
    rec["start_equal"] = rec["action_start_idx"] == rec["action_start_expected"]
    rec["chunk_shape"] = list(chunk_ours.shape)
    rec["chunk_equal"] = bool(np.array_equal(chunk_ours, chunk_up))
    rec["chunk_max_absdiff"] = float(np.max(np.abs(chunk_ours - chunk_up))) \
        if chunk_ours.shape == chunk_up.shape else None
    rec["n_bins_differing"] = (int(np.sum(chunk_ours != chunk_up))
                               if chunk_ours.shape == chunk_up.shape
                               else ACTION_DIM * NUM_ACTIONS_CHUNK)

    # Determinism control. Only meaningful when the chunks disagreed, but it is
    # run either way: a report that only measures the control on failures cannot
    # say the control was ever satisfied on the passing samples.
    with torch.no_grad():
        logits2, start2 = model.prompt_cot_predict_action(
            input_cot_ids=ids_t, pixel_values=pixel, attention_mask=mask)
    chunk_rep = np.asarray(
        dtdec.decode_action_chunk(logits2, start2, torch, np_centers),
        dtype=np.float64)
    rec["determinism_ok"] = bool(np.array_equal(chunk_ours, chunk_rep))

    # ---- 5. the segmentation P1 depends on --------------------------------
    try:
        rec["segments"] = dtdec.segment_boundaries(ids_t)
        rec["segments_ok"] = True
    except Exception as e:
        rec["segments_ok"] = False
        rec["segments_error"] = f"{type(e).__name__}: {e}"
    return rec


def run(args) -> int:
    import torch
    from transformers import AutoProcessor
    from sharpguard.vendor.deepthinkvla import (ACTION_DIM, NUM_ACTIONS_CHUNK,
                                                import_deepthinkvla)
    from sharpguard.vendor.deepthinkvla import decode as dtdec

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[dt-eq] loading {args.ckpt_path}")
    processor = AutoProcessor.from_pretrained(args.ckpt_path,
                                             trust_remote_code=True)
    DeepThinkVLA = import_deepthinkvla()
    model = DeepThinkVLA.from_pretrained(
        args.ckpt_path, torch_dtype=dtype, attn_implementation="eager",
        low_cpu_mem_usage=True).to(device).eval()
    dtdec.assert_config_matches(model.config)
    centers = dtdec.bin_centers()

    # The one place our constants could drift from the loaded object rather than
    # from config.json: upstream builds `self.bin_centers` in __init__, and every
    # action value on both sides of this comparison is a lookup into it.
    up_centers = np.asarray(model.bin_centers, dtype=np.float64)
    centers_equal = bool(up_centers.shape == centers.shape
                         and np.array_equal(up_centers, centers))
    print(f"[dt-eq] bin_centers: ours {centers.shape} upstream "
          f"{up_centers.shape} identical={centers_equal}")

    samples = list(load_libero_samples(args.dataset_repo, args.tfds_subdir,
                                       args.reasoning_json, args.n_samples,
                                       seed=args.seed))
    print(f"[dt-eq] {len(samples)} LIBERO frames")

    recs, errors = [], []
    for si, (img, instr, _gt, fbase, dem, _a) in enumerate(samples):
        try:
            r = compare_one(model, processor, dtdec, torch, centers, img, instr,
                            args, dtype)
            r.update({"sample": si, "file_base": fbase, "demo_id": dem})
            recs.append(r)
            print(f"[dt-eq] {si}: comparable={r.get('comparable')} "
                  f"ids={r.get('ids_equal')} start={r.get('start_equal')} "
                  f"chunk={r.get('chunk_equal')} "
                  f"maxdiff={r.get('chunk_max_absdiff')}")
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
            print(f"[dt-eq] sample {si} raised: {e}\n"
                  f"{traceback.format_exc()[-500:]}")

    comparable = [r for r in recs if r.get("comparable")]
    n_cmp = len(comparable)
    agg = {
        "n_samples_attempted": len(samples),
        "n_records": len(recs),
        "n_comparable": n_cmp,
        "n_no_stop_tail": sum(1 for r in recs if not r.get("comparable")),
        "n_prompt_preserved": sum(1 for r in recs if r.get("prompt_preserved")),
        "n_think_start_first": sum(1 for r in recs if r.get("think_start_first")),
        "n_ids_equal": sum(1 for r in comparable if r.get("ids_equal")),
        "n_start_equal": sum(1 for r in comparable if r.get("start_equal")),
        "n_chunk_equal": sum(1 for r in comparable if r.get("chunk_equal")),
        "n_determinism_ok": sum(1 for r in comparable
                                if r.get("determinism_ok")),
        "n_segments_ok": sum(1 for r in comparable if r.get("segments_ok")),
        "max_chunk_absdiff": max(
            [r["chunk_max_absdiff"] for r in comparable
             if r.get("chunk_max_absdiff") is not None] or [0.0]),
        "mean_cot_tokens": float(np.mean(
            [r["n_cot_tokens"] for r in comparable])) if n_cmp else None,
        "bin_centers_identical": centers_equal,
        "n_errors": len(errors),
        "error_examples": list(dict.fromkeys(errors))[:3],
    }

    ok = (n_cmp > 0
          and agg["n_ids_equal"] == n_cmp
          and agg["n_start_equal"] == n_cmp
          and agg["n_chunk_equal"] == n_cmp
          and agg["n_segments_ok"] == n_cmp
          and centers_equal)
    if n_cmp == 0:
        verdict = ("UNDEFINED: no sample reached upstream's own stop tail, so "
                   "the assembled sequence was never compared to one the "
                   "checkpoint produced. This measures nothing about our decode "
                   "-- it says generation never finished a CoT within "
                   f"--max-new-tokens={args.max_new_tokens}.")
    elif ok:
        verdict = (f"EQUIVALENT on {n_cmp}/{len(recs)} comparable frames: our "
                   f"build_input_cot_ids reproduces the checkpoint's own "
                   f"input_cot_ids byte-identically, action_start_idx agrees, "
                   f"and decode_action_chunk reproduces its (10,7) chunk "
                   f"exactly.")
    else:
        det = agg["n_determinism_ok"]
        verdict = (f"NOT EQUIVALENT: ids {agg['n_ids_equal']}/{n_cmp}, start "
                   f"{agg['n_start_equal']}/{n_cmp}, chunk "
                   f"{agg['n_chunk_equal']}/{n_cmp}, segments "
                   f"{agg['n_segments_ok']}/{n_cmp}, bin_centers "
                   f"identical={centers_equal}. Determinism control passed on "
                   f"{det}/{n_cmp}"
                   + (", so a chunk mismatch is OUR decode, not bf16 "
                      "irreproducibility." if det == n_cmp else
                      ", so some of the chunk mismatch may be forward-pass "
                      "nondeterminism rather than a decode defect."))

    report = {
        "experiment": "deepthink_decode_equivalence",
        "model": args.ckpt_path,
        "config": {k: v for k, v in vars(args).items()},
        "decode_under_test": {
            "assembly": "sharpguard/vendor/deepthinkvla/decode."
                        "build_input_cot_ids",
            "decode": "sharpguard/vendor/deepthinkvla/decode."
                      "decode_action_chunk",
            "reference": "the checkpoint's own predict_cot_action "
                         "(vendored upstream, MIT commit "
                         "4bbd0f4ea9010a421e4629e24177afc819f4b6d2)",
            "chunk_shape": [NUM_ACTIONS_CHUNK, ACTION_DIM],
            "n_bins": int(centers.shape[0]),
        },
        "aggregate": agg,
        "verdict": verdict,
        "per_sample": recs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print("\n===== DT DECODE EQUIVALENCE =====")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    print(f"  verdict: {verdict}")
    print(f"[dt-eq] report -> {out}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--out", default="dt_decode_equivalence.json")
    p.add_argument("--n-samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-new-tokens", type=int, default=320,
                   help="cap on the CoT upstream generates for itself. Too low "
                        "and the stop tail never fires and nothing is "
                        "comparable, which the report says rather than hides.")
    p.add_argument("--dataset-repo",
                   default="Embodied-CoT/embodied_features_and_demos_libero")
    p.add_argument("--tfds-subdir", default="libero_lm_90/1.0.0")
    p.add_argument("--reasoning-json", default="libero_reasonings.json")
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
