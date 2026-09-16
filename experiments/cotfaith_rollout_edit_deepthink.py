"""DeepThinkVLA rollout-level CoT edit -- probe first, matching
experiments/cotfaith_rollout_edit.py's own two-precondition probe pattern.

Why this file exists and is not an extension of cotfaith_rollout_edit.py.
That script decodes ONE 7-vector per forward pass via
AutoModelForVision2Seq.generate() + upstream/ours de-quantization, which is
the OpenVLA-family convention. DeepThinkVLA is a different architecture
end to end (see sharpguard/vendor/deepthinkvla/decode.py's own comparison
table): a (10, 7) action CHUNK from one forward pass under a hybrid
causal/bidirectional mask, QUANTILE un-normalization, and CoT delimiters
that are special token ids rather than literal text. Every model-facing
function here is new; every LIBERO-simulator function (settling, rewinding,
gripper-accumulator and observation-sampling residue, arm pairing) is
imported unchanged from cotfaith_rollout_edit.py, because those fixes are
properties of robosuite/MuJoCo, not of the policy, and re-deriving them
would risk silently reintroducing a bug that file's own docstrings show was
expensive to find.

Self-generated CoT was never attempted for DeepThinkVLA before this file:
every existing DeepThinkVLA report (deepthink_{base,sft,rl}_13family.json)
edits a teacher-forced ground-truth annotation via
experiments/cotfaith_deepthink.py's build_cot_text(gt). A rollout has no
ground truth at test time, so the CoT must come from the model itself. The
vendored class's own generate_action_verl() already does this -- upstream's
own method, presumably written for their own RL-rollout collection (hence
the name) -- so this file calls it rather than hand-rolling autoregressive
generation against a model class this codebase does not otherwise drive
that way.

Why DeepThinkVLA specifically: its own paper (Yin et al. 2025, arXiv
2511.15669) reports 97.0% success on LIBERO, and the RL-tuned checkpoint
used throughout this paper (yinchenghust/deepthinkvla_libero_cot_rl) is
LIBERO-COT-RL-native -- unlike ECoT-bridge, which carries only Bridge V2
norm-stats and cannot be rolled out on LIBERO at all (see
results_v2/canonical_runs/rollout_probe_ecot_bridge). If DT-RL is genuinely
competent, this is the first chance in this paper to test the floor-collapse
and F_dir behavior on a policy that can do the task, rather than on the six
LoRA configs whose action space is independently shown to be near-degenerate.

What this probe answers before any full run is attempted:
  1. Does generate_action_verl() run at all on a real LIBERO frame, with the
     prompt this checkpoint was evaluated on?
  2. What does the self-generated CoT actually look like as text? (Every
     other harness in this release renders a KNOWN dict via build_cot_text;
     this is the first time DeepThinkVLA's own generated text is read
     rather than assumed to match that rendering.)
  3. Does a naive parse of that text back into the {plan, subtask, movement,
     move} shape EDIT_FAMILIES expects round-trip through build_cot_text
     losslessly enough that an edit changes the rendered prefix at all?
  4. Does injecting the edited CoT back through prompt_cot_predict_action
     (the teacher-forced path cotfaith_deepthink.py already uses and
     trusts) produce a decodable, non-degenerate action chunk?

Usage:
    python experiments/cotfaith_rollout_edit_deepthink.py --probe-only \\
        --ckpt-path yinchenghust/deepthinkvla_libero_cot_rl --out ./probe-dt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

# Generic LIBERO-simulator plumbing, unchanged, from the OpenVLA-family
# harness -- see module docstring for why these are imported rather than
# re-derived.
from cotfaith_rollout_edit import (  # noqa: E402
    _seed_scene, _probe_frame, wilson, mcnemar_exact, NO_OP, SETTLE_STEPS,
    _settle_once, _rewind_to, _eef, _capture_for, _grippers,
)


def parse_deepthink_cot(text: str) -> dict:
    """Parse DeepThinkVLA's self-generated CoT text into the dict shape
    build_cot_text() renders and EDIT_FAMILIES operates on.

    build_cot_text() renders {plan, subtask, movement, move} as
    "key: value; key: value; ..." joined by "; ". If the model's own
    generated text follows that same shape (expected, since that rendering
    is presumably what its supervised targets looked like), this recovers
    the dict losslessly. If it does not, the probe reports the raw text and
    the parsed keys side by side so the mismatch is visible rather than
    silently producing an empty dict that no edit family can act on.
    """
    out = {}
    for part in text.split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip().lower()
        if k in ("plan", "subtask", "movement", "move"):
            out[k] = v.strip()
    return out


def _find_subsequence(ids_list: list, sub: list, start: int = 0) -> int:
    """First index of `sub` in `ids_list` at or after `start`, or -1.

    PROMPT_END_TOKEN_IDS is two ids, so a plain .index() (single-element
    lookup) cannot find it; this is the same sliding-window match
    decode.py's segment_boundaries() does inline for the same boundary,
    pulled out here because gen_selfcot_and_action needs it standalone.
    """
    n = len(sub)
    for i in range(start, len(ids_list) - n + 1):
        if ids_list[i:i + n] == sub:
            return i
    return -1


def build_cot_text(reasoning: dict) -> str:
    """Identical to experiments/cotfaith_deepthink.py's build_cot_text --
    copied rather than imported because that module's run() has import-time
    side effects (argparse at module scope would not, but this keeps the two
    renderers trivially diffable against each other rather than coupling
    this probe to that script's other contents)."""
    parts = []
    for k in ("plan", "subtask", "movement", "move"):
        v = reasoning.get(k)
        if isinstance(v, dict):
            ks = sorted(v.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
            parts.append(f"{k}: " + ". ".join(str(v[k2]) for k2 in ks))
        elif isinstance(v, str) and v:
            parts.append(f"{k}: {v}")
    return "; ".join(parts)


def load_model(ckpt_path, dtype):
    import torch
    from transformers import AutoProcessor
    from sharpguard.vendor.deepthinkvla import import_deepthinkvla
    from sharpguard.vendor.deepthinkvla import decode as dtdec

    processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
    DeepThinkVLA = import_deepthinkvla()
    model = DeepThinkVLA.from_pretrained(
        ckpt_path, torch_dtype=dtype, attn_implementation="eager",
        low_cpu_mem_usage=True)
    dtdec.assert_config_matches(model.config)
    norm = dtdec.load_quantile_norm_stats(ckpt_path)
    centers = dtdec.bin_centers()
    return model, processor, dtdec, norm, centers


def gen_selfcot_and_action(model, processor, dtdec, norm, img, instruction,
                            *, device, dtype, do_sample=False):
    """generate_action_verl() end to end: self-generate the CoT, decode the
    (10, 7) action chunk, unnormalize, and extract the CoT text for parsing.

    Returns (action_chunk[10,7] in physical units, full_decoded_text,
    text_from_first_non_image_token, cot_text, input_cot_ids,
    attention_mask, extraction_debug) so a caller can both act on the chunk
    and re-render an edited CoT through the teacher-forced path for the
    paired comparison; extraction_debug is every THINK_START/THINK_END
    position found, so an extraction failure is diagnosable from one run's
    output rather than needing another round-trip to add logging.
    """
    import torch
    from transformers import GenerationConfig
    prompt_text = dtdec.build_prompt_text(instruction, n_images=1)
    proc = processor(text=[prompt_text], images=img, return_tensors="pt")
    input_ids = proc["input_ids"].to(device)
    pixel_values = proc["pixel_values"].to(device, dtype=dtype)
    attention_mask = torch.ones_like(input_ids)

    # generate_action_verl() forwards ONLY `generation_config` to
    # super().generate() -- no max_new_tokens passthrough -- so leaving this
    # None means whatever the checkpoint's own default generation_config.json
    # sets, which the previous probe attempt's evidence (seq_len=309 total,
    # with the real <think> opening at position 289 and no closing
    # </think> anywhere in the sequence) shows is far too short to let the
    # model finish reasoning before being cut off mid-sentence. 400 is
    # generous rather than a tight guess at the true length: each extra
    # token costs one forward pass on a single probe sample, negligible next
    # to the ~20-90 minute checkpoint download that dominates this job's
    # wall clock, so overshooting here is far cheaper than a fourth
    # round-trip if 256 also turns out short.
    gen_cfg = GenerationConfig(max_new_tokens=400, do_sample=do_sample)
    with torch.no_grad():
        (normalized_chunk, action_token_ids, input_cot_ids,
         cot_attention_mask) = model.generate_action_verl(
            input_ids=input_ids, pixel_values=pixel_values,
            attention_mask=attention_mask, do_sample=do_sample,
            temperature=(1.0 if do_sample else None),
            generation_config=gen_cfg)
    chunk = dtdec.unnormalize(np.asarray(normalized_chunk), norm)

    # input_cot_ids is [prompt, <think>, CoT, <think_end>, ...]. THINK_START/
    # THINK_END decode to the literal "<think>"/"</think>", but so does
    # THINK_PREFIX's own instructional sentence ("...in <think></think>
    # tags..."), so BOTH the decorative occurrence and the real one carry
    # the same ids. Two fix attempts so far: a plain ids_list.index() (finds
    # the decorative pair, empty by construction) and searching only after
    # PROMPT_END_TOKEN_IDS (still returned empty -- meaning either that
    # 2-token id pair does not actually occur verbatim in this sequence's
    # tokenization, or something else is wrong; not re-guessed a third time).
    # This version finds EVERY occurrence of both ids and records them in
    # the probe output, then uses the LAST THINK_START and the first
    # THINK_END after it -- correct if there are exactly two <think> spans
    # (decorative, real), and self-diagnosing if that assumption is wrong
    # too, rather than silently returning empty a third time.
    ids_list = input_cot_ids[0].tolist()
    think_start_positions = [i for i, t in enumerate(ids_list)
                              if t == dtdec.THINK_START]
    think_end_positions = [i for i, t in enumerate(ids_list)
                            if t == dtdec.THINK_END]
    prompt_end_idx = _find_subsequence(ids_list, dtdec.PROMPT_END_TOKEN_IDS)
    extraction_debug = {
        "think_start_positions": think_start_positions,
        "think_end_positions": think_end_positions,
        "prompt_end_idx_found": prompt_end_idx,
        "seq_len": len(ids_list),
    }
    cot_text = ""
    if think_start_positions and think_end_positions:
        ts = think_start_positions[-1]
        te_candidates = [t for t in think_end_positions if t > ts]
        if te_candidates:
            te = te_candidates[0]
            cot_text = processor.tokenizer.decode(
                ids_list[ts + 1:te], skip_special_tokens=False).strip()
            extraction_debug["used_think_start"] = ts
            extraction_debug["used_think_end"] = te
    full_text = processor.tokenizer.decode(
        input_cot_ids[0], skip_special_tokens=False)
    # A preview starting at the first non-image token, not at index 0: 256
    # "<image>" tokens alone are >800 chars, so a naive [:N] preview of the
    # full decode never reaches the prompt/CoT/action region it exists to
    # show -- exactly what happened in the first probe attempt.
    post_image = [i for i, t in enumerate(ids_list) if t != dtdec.IMAGE_TOKEN]
    text_from_post_image = (
        processor.tokenizer.decode(ids_list[post_image[0]:],
                                    skip_special_tokens=False)
        if post_image else full_text)
    return (chunk, full_text, text_from_post_image, cot_text, input_cot_ids,
            cot_attention_mask, extraction_debug)


def extract_cot_span(full_text: str) -> str:
    """Text-level fallback, kept only as a diagnostic the probe reports
    alongside the id-based extraction in gen_selfcot_and_action -- if the
    two disagree, that is itself worth seeing rather than silently trusting
    one. Not depended on for parsing."""
    m = re.search(r"(?:<think>|\[think\])(.*?)(?:</think>|\[/think\])",
                  full_text, re.S)
    if m:
        return m.group(1).strip()
    idx = full_text.rfind(";")
    return full_text[idx + 1:].strip() if idx >= 0 else full_text.strip()


def probe(args):
    import torch
    from sharpguard.attacks import EDIT_FAMILIES
    from sharpguard.libero_sim import _preprocess_image, _apply_gripper_transform

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[dt-rollout-probe] loading {args.ckpt_path}")
    model, processor, dtdec, norm, centers = load_model(args.ckpt_path, dtype)
    model = model.to(device).eval()
    print(f"[dt-rollout-probe] action space verified: chunk "
          f"{dtdec.NUM_ACTIONS_CHUNK if hasattr(dtdec, 'NUM_ACTIONS_CHUNK') else '?'}, "
          f"q01={norm['q01'].round(3).tolist()} q99={norm['q99'].round(3).tolist()}")

    out = {"probe": True, "ckpt": args.ckpt_path, "suite": args.suite}

    frame, instruction = _probe_frame(args)
    out["instruction"] = instruction
    if frame is None:
        out["frame"] = "unavailable (libero not importable); nothing further to probe"
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "rollout_edit_probe.json").write_text(
            json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return
    img = _preprocess_image(frame, args.image_preproc)

    t0 = time.time()
    try:
        (chunk, full_text, text_from_post_image, cot_text_by_id, cot_ids,
         cot_mask, extraction_debug) = gen_selfcot_and_action(
            model, processor, dtdec, norm, img, instruction,
            device=device, dtype=dtype, do_sample=False)
    except Exception as e:
        out["generate_action_verl_error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-2000:]
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "rollout_edit_probe.json").write_text(
            json.dumps(out, indent=2))
        print(json.dumps(out, indent=2)[:4000])
        return
    out["gen_seconds"] = round(time.time() - t0, 2)
    out["extraction_debug"] = extraction_debug
    out["full_decoded_head"] = text_from_post_image[:800]
    cot_span = cot_text_by_id
    out["cot_span"] = cot_span
    out["cot_span_text_fallback"] = extract_cot_span(full_text)[:800]
    out["cot_span_id_vs_text_agree"] = (
        cot_span.strip() == out["cot_span_text_fallback"].strip())
    out["action_chunk"] = [[round(float(x), 5) for x in row] for row in chunk]
    out["action_chunk_shape"] = list(chunk.shape)
    # A cheap, cheaper-than-eyeballing-800-chars signal for whether the
    # generated CoT has ANY of the structured-field markers the parser
    # looks for, versus being free-form prose the {plan,subtask,movement,
    # move} parser was never going to match. If this is False and
    # parsed_keys ends up empty, that is why -- not a parser bug.
    out["cot_span_has_structured_markers"] = any(
        marker in cot_span.lower()
        for marker in ("plan:", "subtask:", "movement:", "move:"))

    reasoning = parse_deepthink_cot(cot_span)
    out["parsed_keys"] = sorted(reasoning.keys())
    out["parsed_reasoning"] = reasoning
    rerendered = build_cot_text(reasoning)
    out["rerendered_matches_span"] = (rerendered.strip() == cot_span.strip())
    out["rerendered"] = rerendered[:800]

    fam_status = {}
    edited_actions = {}
    for fname in args.families.split(","):
        fname = fname.strip()
        if not fname or fname not in EDIT_FAMILIES:
            fam_status[fname] = "unknown family"
            continue
        if not reasoning:
            fam_status[fname] = "no parsed reasoning to edit"
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
        edited_text = build_cot_text(edited)
        if edited_text == rerendered:
            fam_status[fname] = "IDENTICAL RENDER - inapplicable"
            continue
        fam_status[fname] = "changes the rendered CoT"
        try:
            cot_token_ids = processor.tokenizer(
                edited_text, add_special_tokens=False)["input_ids"]
            prompt_text = dtdec.build_prompt_text(instruction, n_images=1)
            p_proc = processor(text=[prompt_text], images=img, return_tensors="pt")
            p_ids = p_proc["input_ids"].to(device)
            pix = p_proc["pixel_values"].to(device, dtype=dtype)
            ids = dtdec.build_input_cot_ids(p_ids, cot_token_ids, torch)
            mask = torch.ones_like(ids)
            with torch.no_grad():
                logits, start = model.prompt_cot_predict_action(
                    input_cot_ids=ids, pixel_values=pix, attention_mask=mask,
                    output_attentions=False)
            ed_chunk = dtdec.unnormalize(
                dtdec.decode_action_chunk(logits, start, torch, centers), norm)
            edited_actions[fname] = {
                "chunk0": [round(float(x), 5) for x in ed_chunk[0]],
                "delta_linf_vs_clean": round(
                    float(np.max(np.abs(ed_chunk[0] - chunk[0]))), 5),
            }
        except Exception as e:
            edited_actions[fname] = {"error": f"{type(e).__name__}: {e}"}
    out["families"] = fam_status
    out["edited_actions"] = edited_actions

    Path(args.out).mkdir(parents=True, exist_ok=True)
    p = Path(args.out) / "rollout_edit_probe.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:6000])
    print(f"[dt-rollout-probe] -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="libero_90")
    ap.add_argument("--families", default="direction_flip,gripper_flip,paraphrase_null")
    ap.add_argument("--image-preproc", default="none")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--env-seed", type=int, default=0)
    ap.add_argument("--probe-only", action="store_true",
                     help="this file only implements the probe; kept as a "
                          "flag for symmetry with cotfaith_rollout_edit.py "
                          "and to fail loudly if omitted by habit")
    args = ap.parse_args()
    if not args.probe_only:
        raise SystemExit(
            "cotfaith_rollout_edit_deepthink.py currently only implements "
            "--probe-only. The full rollout loop (action-chunk execution, "
            "paired arms, DSR) is written once the probe confirms "
            "generate_action_verl() and the CoT parse round-trip work on a "
            "real checkpoint -- see module docstring.")
    probe(args)


if __name__ == "__main__":
    main()
