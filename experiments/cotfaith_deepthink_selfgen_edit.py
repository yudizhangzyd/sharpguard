"""DeepThinkVLA-RL self-generated-CoT edit-sensitivity measurement.

What this answers, and why it is scoped the way it is. The technical
reviewer's ask was "validated on at least one policy with a non-degenerate
action space and non-zero rollout success" -- a policy QUALIFIER, not a
requirement that this release itself run a closed-loop rollout. DT-RL
(yinchenghust/deepthinkvla_libero_cot_rl) already qualifies: its own paper
(Yin et al. 2025, arXiv 2511.15669) reports 97.0% success on LIBERO
independently of anything measured here, and
scripts/floor_convention_robustness.py's action_space_diagnostic already
shows its action space is not degenerate (ap1/noop3 near 0, unlike the six
LoRA configs). What is missing is the actual floor-collapse/F_dir-style
MEASUREMENT on it -- the same first-step edit-sensitivity statistic Table 1
reports for every other config -- which is a smaller, more direct target
than a full rollout: this release's whole leaderboard is ALREADY first-step,
so a first-step measurement here is directly comparable to it, not a lesser
standard.

Why self-generated CoT rather than teacher-forced (unlike
experiments/cotfaith_deepthink.py's existing DeepThinkVLA measurements): a
teacher-forced measurement on DT-RL would answer "does editing a
demonstration-derived CoT change the action", which this paper already
answers for DT-base/DT-SFT/DT-RL via deepthink_rl_13family.json. What it
does NOT answer is whether that holds for the CoT the policy would actually
be running on at test time -- exactly the self-generated-vs-teacher-forced
distinction Stage 2 (experiments/cotfaith_bridge.py,
stage1_stage2_analysis.py) already tests for the OpenVLA family. This
extends that same distinction to the one competent-policy candidate.

Why text-level edits, not the dict-shaped EDIT_FAMILIES. Probing
generate_action_verl() (see experiments/cotfaith_rollout_edit_deepthink.py's
probe, four iterations to get right) found DT-RL's self-generated CoT is
free-form prose ("This is the initial frame... Move gripper above the
handle of the middle drawer."), not the {plan, subtask, movement, move}
dict shape build_cot_text() renders for teacher-forced use. EDIT_FAMILIES
operates on that dict; it has nothing to act on here. Rather than inventing
a new vocabulary, this reuses sharpguard.attacks.cot_edit's own DIRECTION_PAIRS
and PARAPHRASE_SYNONYMS word/pattern lists -- the same substitutions
direction_flip and paraphrase_null apply to the dict's "movement" field --
applied as plain string edits to the CoT's final sentence instead. The
observed pattern across samples is that the CoT ends with one imperative
sentence ("Move gripper ..."); edits are scoped to THAT sentence only,
mirroring direction_flip's own scope (movement/move keys only, not
task/subtask/scene-description), rather than word-substituting the whole
paragraph and changing incidental scene-description words along with it.

What this does NOT claim. The word lists were validated (by the LLM judge,
Appendix sec:judge_edits) for meaning-preservation/meaning-change WHEN
APPLIED TO SHORT STRUCTURED PHRASES ("move left" -> "move right"), not to a
sentence embedded in a free-text paragraph. This measurement is disclosed
as exactly that -- an extension of the same substitution mechanism to a new
linguistic register, not a re-validated one -- so a reader can weigh it
accordingly rather than assume it carries the same judge-certified
guarantee Table 1's floors do.

Usage:
    python experiments/cotfaith_deepthink_selfgen_edit.py \\
        --ckpt-path yinchenghust/deepthinkvla_libero_cot_rl --out ./dt-selfgen-edit \\
        --n-samples 100
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

from cotfaith_rollout_edit_deepthink import (  # noqa: E402
    load_model, gen_selfcot_and_action,
)
from cotfaith_deepthink import load_libero_samples  # noqa: E402
from sharpguard.attacks.cot_edit import (  # noqa: E402
    DIRECTION_PAIRS, PARAPHRASE_SYNONYMS, _replace_word_pairs,
)

TAU_DEFAULT = 0.05
FAMILIES = ["direction_flip_text", "paraphrase_null_text"]


def split_sentences(text: str) -> list:
    """Naive sentence split on '.'/'!'/'?' boundaries. Adequate here: the
    text is short, machine-generated, and every sample observed so far uses
    plain periods -- not a general-purpose sentence tokenizer, and not
    trying to be one."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


_MOVEMENT_VERB_RE = re.compile(
    r"\bgripper\b|" + "|".join(p for p, _ in PARAPHRASE_SYNONYMS),
    re.IGNORECASE)
print(f"[version-check] find_movement_sentence regex = "
      f"{_MOVEMENT_VERB_RE.pattern!r} (attempt-13-fix, 2026-09-07)",
      flush=True)


def find_movement_sentence(text: str):
    """The sentence to edit: the LAST one carrying an actionable directive.

    Attempt 1 of this script matched only /move|gripper/, on the assumption
    (drawn from ECoT-bridge's structured "MOVE:" field) that every sample
    ends with a "move" sentence. Run against DT-RL, that missed 81 of 100
    samples: DT-RL's free-form CoT ends its pick-and-place tasks with
    "Pick up the X."/"Grasp the Y." as often as "Move...", and 0 of those 81
    contain "move" or "gripper" anywhere at all -- not noisily missing, this
    policy's actual vocabulary is broader than the pattern assumed. The fix
    matches on the same verb set paraphrase_null_text can already edit
    (PARAPHRASE_SYNONYMS' source patterns, which already cover
    grasp/pick/place/push/release/turn/open/close), so the sentence-finder
    stops being narrower than the editor it feeds. Verified against the
    completed run's own released per-sample CoT text before resubmitting:
    recovers all 81, changes nothing about the 19 that already matched
    (superset regex; backward search still returns the same, or a truly
    later, sentence -- never an earlier one). Attempt 13 shipped this fix
    but still showed n_no_movement_cot=81/100 in its own results: run()'s
    skip gate (just above the call site, "no movement-bearing self-
    generated CoT") had its OWN separate hardcoded /move|gripper/ check,
    upstream of this function, deciding skip status before this function
    was ever reached. Now both read from the same _MOVEMENT_VERB_RE.
    Returns (sentence, index) or (None, -1) if no sentence qualifies --
    recorded as inapplicable rather than editing an unrelated sentence."""
    sentences = split_sentences(text)
    for i in range(len(sentences) - 1, -1, -1):
        if _MOVEMENT_VERB_RE.search(sentences[i]):
            return sentences[i], i
    return None, -1


_PHRASAL_PICK_RE = re.compile(r"\bpick(ed|ing)?\s+up\b", re.IGNORECASE)


def _mask_phrasal_verbs(text: str):
    """Protect fixed phrasal verbs whose particle collides with a
    DIRECTION_PAIRS word but is not a spatial direction. "pick up" is the
    only one this dataset's vocabulary contains (verified by enumerating
    every distinct movement sentence in the released reports: "turn on"/
    "turn the knob" have no colliding particle, nothing else uses up/down/
    left/right/above/below/in/out as part of an idiom) -- masking that one
    phrase rather than excluding "up"/"down" globally, which would also
    block a genuine spatial "up"/"down" this dataset happens not to contain
    but a general fix should not assume away. Returns (masked_text, matches)
    where matches preserves each occurrence's original casing for restore."""
    originals = [m.group(0) for m in _PHRASAL_PICK_RE.finditer(text)]
    masked = _PHRASAL_PICK_RE.sub("\x00PICKUP\x00", text)
    return masked, originals


def _unmask_phrasal_verbs(text: str, originals):
    for orig in originals:
        text = text.replace("\x00PICKUP\x00", orig, 1)
    return text


def direction_flip_text(cot_text: str):
    """Text-level direction_flip: DIRECTION_PAIRS applied to the movement
    sentence only. Returns (edited_full_text, meta) or None if inapplicable
    (no movement sentence found, or it contains no direction word).

    "pick up" is masked before the substitution and restored after: its
    "up" is a fixed phrasal-verb particle (grasp), not a spatial direction,
    and DIRECTION_PAIRS' up<->down entry cannot tell the two apart on plain
    text. Found by an ICLR reviewer pulling the released edit_meta strings
    from the first N=100 run and verified against them before this fix: 72
    of 77 (93.5%) of that run's edits were exactly "Pick up the X." ->
    "Pick down the X." -- ungrammatical, not a direction reversal -- and
    only 5 (above/below, in/out, right/left) were genuine. That run's
    numbers are withdrawn; this fix is what the re-run uses."""
    sentence, idx = find_movement_sentence(cot_text)
    if sentence is None:
        return None
    masked, guard = _mask_phrasal_verbs(sentence)
    new_masked = _replace_word_pairs(masked, DIRECTION_PAIRS)
    new_sentence = _unmask_phrasal_verbs(new_masked, guard)
    if new_sentence == sentence:
        return None
    sentences = split_sentences(cot_text)
    sentences[idx] = new_sentence
    return " ".join(sentences), {"orig": sentence, "edited": new_sentence}


def paraphrase_null_text(cot_text: str):
    """Text-level paraphrase_null: PARAPHRASE_SYNONYMS applied to the
    movement sentence only, mirroring direction_flip_text's scope so the
    two are a floor/semantic PAIR over the same sentence, not different
    scopes that would confound the comparison.

    re.IGNORECASE matches "Pick" at the sentence start but every synonym in
    PARAPHRASE_SYNONYMS is lowercase, so the substitution silently
    decapitalized the sentence-initial word on every sample that reached
    this family in the first N=100 run (100 of 100, per the released
    edit_meta -- "Pick up the book." -> "lift up the book."), a systematic
    orthographic difference between the two arms of the comparison that had
    nothing to do with meaning. Restored below rather than left as a second
    confound alongside direction_flip_text's phrasal-verb bug."""
    sentence, idx = find_movement_sentence(cot_text)
    if sentence is None:
        return None
    new_sentence = sentence
    for pat, syn in PARAPHRASE_SYNONYMS:
        new_sentence = re.sub(pat, syn, new_sentence, flags=re.IGNORECASE)
    if new_sentence == sentence:
        return None
    if sentence[:1].isupper() and new_sentence[:1].islower():
        new_sentence = new_sentence[:1].upper() + new_sentence[1:]
    sentences = split_sentences(cot_text)
    sentences[idx] = new_sentence
    return " ".join(sentences), {"orig": sentence, "edited": new_sentence}


TEXT_EDIT_FAMILIES = {
    "direction_flip_text": direction_flip_text,
    "paraphrase_null_text": paraphrase_null_text,
}


def decode_edited(model, processor, dtdec, norm, centers, img, instruction,
                   edited_cot_text, *, device, dtype):
    """Re-render `edited_cot_text` through the SAME teacher-forced path
    experiments/cotfaith_deepthink.py already uses and trusts
    (prompt_cot_predict_action), so the edited-vs-clean comparison isolates
    the CoT text, not a different decode mechanism."""
    import torch
    prompt_text = dtdec.build_prompt_text(instruction, n_images=1)
    p_proc = processor(text=[prompt_text], images=img, return_tensors="pt")
    p_ids = p_proc["input_ids"].to(device)
    pix = p_proc["pixel_values"].to(device, dtype=dtype)
    cot_ids = processor.tokenizer(
        edited_cot_text, add_special_tokens=False)["input_ids"]
    ids = dtdec.build_input_cot_ids(p_ids, cot_ids, torch)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        logits, start = model.prompt_cot_predict_action(
            input_cot_ids=ids, pixel_values=pix, attention_mask=mask,
            output_attentions=False)
    chunk = dtdec.unnormalize(
        dtdec.decode_action_chunk(logits, start, torch, centers), norm)
    return chunk


def run(args):
    import torch

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[dt-selfgen-edit] loading {args.ckpt_path}")
    model, processor, dtdec, norm, centers = load_model(args.ckpt_path, dtype)
    model = model.to(device).eval()

    samples = list(load_libero_samples(
        args.dataset_repo, args.tfds_subdir, args.reasoning_json,
        args.n_samples, seed=args.seed))
    print(f"[dt-selfgen-edit] loaded {len(samples)} samples")

    per_sample = []
    n_no_structured_cot = 0
    n_decode_fail = 0
    for si, (img, instr, gt, fbase, dem, gt_action) in enumerate(samples):
        try:
            # img is already a PIL Image from load_libero_samples (a
            # recorded dataset frame, pre-oriented) -- unlike
            # cotfaith_rollout_edit_deepthink.py's probe, which reads raw
            # simulator frames and needs sharpguard.libero_sim's flip/resize.
            # experiments/cotfaith_deepthink.py's own run() passes this same
            # loader's img straight to the processor with no preprocessing;
            # matched here rather than reintroducing simulator-frame
            # handling this data source does not need.
            pil_img = img
            (chunk, full_text, text_from_post_image, cot_text, cot_ids,
             cot_mask, extraction_debug) = gen_selfcot_and_action(
                model, processor, dtdec, norm, pil_img, instr,
                device=device, dtype=dtype, do_sample=False)
        except Exception as e:
            n_decode_fail += 1
            print(f"[dt-selfgen-edit] sample {si} self-gen failed: "
                  f"{type(e).__name__}: {e}\n{traceback.format_exc()[-300:]}")
            continue
        if not cot_text or not _MOVEMENT_VERB_RE.search(cot_text):
            n_no_structured_cot += 1
            per_sample.append({
                "sample": si, "file_base": fbase, "skipped": True,
                "reason": "no movement-bearing self-generated CoT",
                "cot_text": cot_text,
            })
            continue

        for fname, fedit in TEXT_EDIT_FAMILIES.items():
            edited = fedit(cot_text)
            if edited is None:
                per_sample.append({
                    "sample": si, "family": fname, "file_base": fbase,
                    "skipped": True, "reason": "no plausible edit",
                })
                continue
            edited_text, meta = edited
            try:
                edited_chunk = decode_edited(
                    model, processor, dtdec, norm, centers, pil_img, instr,
                    edited_text, device=device, dtype=dtype)
            except Exception as e:
                per_sample.append({
                    "sample": si, "family": fname, "file_base": fbase,
                    "skipped": True,
                    "reason": f"edited decode raised {type(e).__name__}: {e}",
                })
                continue
            d0 = edited_chunk[0] - chunk[0]
            per_sample.append({
                "sample": si, "family": fname, "file_base": fbase,
                "edit_meta": meta,
                "a_orig": [float(x) for x in chunk[0]],
                "a_edit": [float(x) for x in edited_chunk[0]],
                "delta_l1_mean": float(np.mean(np.abs(d0))),
                "delta_linf": float(np.max(np.abs(d0))),
                "faithful": float(np.max(np.abs(d0))) > args.threshold,
            })
        if (si + 1) % 10 == 0:
            print(f"[dt-selfgen-edit] {si+1}/{len(samples)} done "
                  f"(no_movement_cot={n_no_structured_cot}, "
                  f"decode_fail={n_decode_fail})")

    edit_agg = {}
    for fname in TEXT_EDIT_FAMILIES:
        rows = [r for r in per_sample
                if r.get("family") == fname and not r.get("skipped")]
        n_skipped = sum(1 for r in per_sample
                        if r.get("family") == fname and r.get("skipped"))
        if not rows:
            edit_agg[fname] = {"n": 0, "n_skipped": n_skipped}
            continue
        li = [r["delta_linf"] for r in rows]
        fr = [r["faithful"] for r in rows]
        edit_agg[fname] = {
            "n": len(rows), "n_skipped": n_skipped,
            "delta_l1_mean": float(np.mean([r["delta_l1_mean"] for r in rows])),
            "delta_linf_mean": float(np.mean(li)),
            "delta_linf_median": float(np.median(li)),
            "faithful_rate": float(np.mean(fr)),
        }

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "dt_selfgen_edit_report.json").write_text(json.dumps({
        "model": args.ckpt_path,
        "n_samples_requested": args.n_samples,
        "n_samples_loaded": len(samples),
        "n_no_movement_cot": n_no_structured_cot,
        "n_selfgen_decode_fail": n_decode_fail,
        "threshold": args.threshold,
        "edit_aggregate": edit_agg,
        "per_sample": per_sample,
    }, indent=2, default=str))
    print("\n===== DT-RL SELF-GEN EDIT DONE =====")
    for fname, v in edit_agg.items():
        if v["n"] > 0:
            print(f"  {fname:22s} n={v['n']:3d} n_skipped={v['n_skipped']:3d} "
                  f"faithful_rate={v['faithful_rate']:.3f}")
        else:
            print(f"  {fname:22s} n=0 n_skipped={v['n_skipped']}")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--out", default="./dt-selfgen-edit")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=TAU_DEFAULT)
    ap.add_argument("--dataset-repo",
                     default="Embodied-CoT/embodied_features_and_demos_libero")
    ap.add_argument("--tfds-subdir", default="libero_lm_90/1.0.0")
    ap.add_argument("--reasoning-json", default="libero_reasonings.json")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
