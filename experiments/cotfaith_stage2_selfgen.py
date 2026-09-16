"""CoT-Faith Stage 2: score edits against the model's OWN self-generated CoT.

Motivation. Every LIBERO number this benchmark has published so far --
`experiments/cotfaith_edit.py`, `experiments/cotfaith_stage1_continuous.py`,
every `*_edit_13family*.json` under `results_v2/canonical_runs/` -- edits a
TEACHER-FORCED ground-truth annotation from Embodied-CoT's
`libero_reasonings.json`. The policy never produced that text; it is spliced
into the prompt and the policy conditions on it. That is the single deepest
construct-validity objection this paper has drawn: a policy conditioned on
reasoning it did not produce is trivially OOD-sensitive to ANY perturbation of
that reasoning, which could explain the paper's whole headline finding without
the policy being "faithful" to anything in the sense the paper claims. Stage 2
does not argue against that objection -- it removes its premise, by re-running
the exact same edit-and-measure protocol on a CoT the checkpoint decoded
itself, online, from the LIBERO frame, with no ground-truth text anywhere in
the prompt.

Self-generation call pattern -- reused, not reinvented. The proof of concept
for this already exists: `experiments/cotfaith_rollout_edit.py`'s `gen_cot()`
(greedy free-generation primed with "ASSISTANT: TASK:", capturing whatever the
model emits), `parse_generated_cot()` (slices the raw text into the same 8
content keys `cotfaith_edit.py`'s ground-truth dicts use), and `build_cot_body()`
/ `has_structured_cot()`. Confirmed against a real checkpoint by bolt
`phenc9ygb4` (`results_v2/canonical_runs/rollout_probe_ecot_bridge/
rollout_edit_probe.json`, read-only reference, not reproduced here): ECoT-bridge
free-generates a well-formed 8-tag CoT from a LIBERO frame unprompted, 255
tokens, 8.57s. This script imports those four functions verbatim and calls
`gen_cot()` exactly once per sample -- generation is the expensive step, and
its result is reused across all `--families` edits of that sample, the same
way `experiments/cotfaith_stage1_continuous.py` computes one ORIGINAL decode
per sample and reuses it across families.

CAVEAT surfaced by that same real artifact, load-bearing for how to read this
script's per-family skip counts: `rollout_edit_probe.json`'s
`cot_body_head` shows `VISIBLE OBJECTS: a black robot arm [80, 1, 168, 87], ...`
-- a free-text STRING, because `parse_generated_cot()` slices raw generated
text between tag markers. The ground-truth annotation's `bboxes` field, by
contrast, is a dict of `name -> [[x1,y1],[x2,y2]]` (confirmed against
`libero_reasonings.json` directly). `sharpguard.attacks.cot_edit`'s
`_iter_bbox_names()` only recognizes a dict or a list; on a plain string it
returns `[]`. Consequence, deterministic and by construction, not a dispatch
bug: `subject_swap`, `adversarial_plausible`, and `selfsplice_control` (all
three call `_iter_bbox_names` via `_find_primary_object`) and
`bbox_jitter_null` (type-checks `bboxes` itself) will report
`skipped=True, reason="no plausible edit"` on essentially EVERY
self-generated sample, on every checkpoint, including the two checkpoints that
free-generate perfectly well-formed CoT. Confirmed empirically, not just by
reading `cot_edit.py`'s source: a control-flow dry run of this script, with
the model-forward and self-generation calls stubbed, scored all four of
these families at n=0/skipped=100% while the other nine families scored
normally on the same synthetic self-generated dicts -- this is exactly why
that dry run matters rather than trusting a manual trace of the dispatch
logic alone (the manual trace this docstring started from missed
selfsplice_control's shared dependency on `_find_primary_object`). This is
not patched here -- the task this script was written for is explicit that the
13 edit generators are "the same deterministic CPU-only generators as
always", i.e. reused verbatim, not extended with a new bbox-text parser. A
reviewer should expect these four families' `n` (scored) to be near 0 and
`n_skipped` to be near total for ALL THREE checkpoints, and should not read
that as evidence this script's dispatch is broken -- the evidence FOR
"dispatch is broken" would be the OTHER nine families showing the same
pattern, which they should not.

Exact protocol.

  Checkpoints (KNOWN_CHECKPOINTS below): ecot-bridge (public HF, ECoT's own
  weights), lora-r32 (this project's LoRA r=32 fine-tune, S3 task bcihypv3gu),
  no-cot (this project's no-CoT-supervision ablation, S3 task a8eegzcg4r).
  no-cot is a CONTROL: it should not have a meaningful CoT for `gen_cot()` to
  produce in the first place, so seeing it skip nearly everything across ALL
  13 families (not just the four bbox-dependent ones above) is the expected,
  useful outcome, not a failure of this script.

  EXCLUDED: DeepThinkVLA (base/SFT/RL). It decodes a `<think>`-block format,
  not the ECoT 9-tag format `gen_cot`/`parse_generated_cot` know how to prime
  and parse; self-generation for it needs a different priming string and a
  different parser, neither of which exists yet. Forcing DeepThinkVLA through
  this script's ECoT-shaped priming would silently produce ill-formed or
  empty `reasoning` dicts for a reason that has nothing to do with whether
  DeepThinkVLA's CoT is faithful, and would burn this job's GPU budget to
  learn that. This is a SCOPE LIMIT, not an oversight: do not attempt it here.
  `--model-family` is deliberately not exposed as a flag (unlike Stage 1's),
  so there is no dead code path that could be pointed at "deepthink" by
  mistake.

  Families: all 13 keys of `sharpguard.attacks.EDIT_FAMILIES` (imported, not
  retyped, so this can never quietly diverge from the canonical set every
  other script in this project already dispatches against): subject_swap,
  direction_flip, gripper_flip, location_swap, verb_swap, negation,
  adversarial_plausible, selfsplice_control, syntactic_scramble,
  cross_task_swap, paraphrase_null, bbox_jitter_null, instr_random_sub. Each
  is dispatched with the exact same seed offsets and special-case kwargs
  `experiments/cotfaith_edit.py:run()` uses (`apply_edit()` below), so the
  only thing that differs between that script and this one is where
  `reasoning` came from -- never how a family is invoked.

  Sampling: ~100 self-generated samples per (checkpoint), scored against all
  13 families, at t=0 (first step of the episode) -- matching every existing
  table's protocol, for direct comparability. PLUS a cheap add-on: for the 3
  families that carry the paper's headline (paraphrase_null,
  syntactic_scramble, direction_flip) only, also self-generate and score at
  two more points within each episode's recorded length T
  (t ~ round(T/3), t ~ round(2T/3)), to test whether "first step only" is
  itself a confound.

  ASSUMPTION, flagged rather than silently picked: "T" here is the length of
  the PRE-RECORDED demonstration trajectory this TFDS dataset already stores
  (`len(list(ep["steps"].as_numpy_iterator()))`), not a live policy rollout.
  `load_libero_samples_multistep()` below is that same fetch/parse/iterate
  pipeline `cotfaith_edit.load_libero_samples` already runs (identical HF
  snapshot call, identical tfds builder/shuffle config, identical
  file_path/demo_id join against the reasoning JSON, used ONLY to reproduce
  the same episode-selection filter -- never as CoT content) with two more
  indices read out of the SAME already-materialized per-episode step list
  instead of discarded. It opens no MuJoCo/robosuite env. The alternative
  reading -- T as a live policy rollout's length -- is what
  `experiments/cotfaith_rollout_edit.py`'s `OffScreenRenderEnv` machinery is
  for, and building that INTO this script is exactly the "new simulator
  harness" the add-on is scoped to avoid; if that reading is the intended one,
  this add-on measures something related but not identical, and that
  substitution should be reviewed before trusting its numbers.

  Scoring, per sample per family: self-generate the CoT once (the "original"
  for this experiment -- there is no ground-truth text anywhere in this
  script's prompts), apply the edit, and record BOTH:
    (a) the existing thresholded metric -- `a_orig`/`a_edit`/`delta_per_dim`/
        `delta_l1_mean`/`delta_linf`/`faithful` (`delta_linf > --threshold`),
        field names verbatim from `cotfaith_edit.py`, for direct comparison
        against every existing `*_edit*.json` table;
    (b) Stage 1's continuous metric -- `tv_mean`/`kl_mean`/`expected_action_l2`
        /`delta_logp` plus the text covariates, computed by IMPORTING
        `experiments/cotfaith_stage1_continuous.py`'s
        `openvla_action_softmax`/`openvla_teacher_forced_logp`/
        `total_variation`/`kl_divergence`/`text_covariates`/
        `build_openvla_prefix`/`load_model_openvla` rather than
        reimplementing any of them, so the two jobs' numbers are guaranteed
        consistent by construction, not by two independent implementations
        happening to agree.
  (a)'s `a_orig`/`a_edit` are read off the SAME forward pass (b) already
  computes -- `values[argmax(orig_probs, axis=1)]` -- rather than calling
  `cotfaith_edit.infer_action` a second time. Two consequences, both
  deliberate: this uses the CANONICAL checkpoint de-quantization grid (see
  next section) rather than `cotfaith_edit.dequantize_action`'s grid, and it
  costs zero extra model calls instead of doubling every "lightweight op" in
  this already self-generation-dominated budget. `p2_dequant_recompute.py`
  already established the `faithful` boolean is invariant to which of the
  two grids computes it (a pure grid shift cannot cross the tau boundary);
  the raw `delta_linf` FLOATS will therefore differ from existing tables' by
  the two grids' ~0.4% spacing difference, while `faithful_rate` aggregates
  remain directly comparable -- which is the comparability this script's
  spec actually asked for.

CANONICAL DE-QUANTIZATION GRID: reused from
`experiments/cotfaith_stage1_continuous.py` (which imports it from
`experiments/p2_dequant_recompute.py:up_value`, checkpoint-verified to
4.7e-07 by bolt `7vpp28qfsk`) via that script's own `openvla_action_softmax`.
Not re-imported directly here and not retyped a third time -- see Stage 1's
module docstring for the full argument; the point of canonicalizing it once
is that nothing downstream should need to choose again.

Output. One JSON per checkpoint process, written by the caller's
`--out` (this project's `results_v2/canonical_runs/` is the eventual, curated
home for a run worth keeping, following that directory's existing
README+JSON convention -- exactly like
`experiments/cotfaith_stage1_continuous.py`, this script itself writes to a
plain `--out` directory, e.g. under `$BOLT_ARTIFACT_DIR`, and curating a
finished run into `results_v2/canonical_runs/stage2_selfgen/` is a separate,
later step for whoever reviews the run, not something this script does
itself). Every per-sample record carries `"cot_source": "self_generated"`
(also stamped at the payload's top level) precisely so a Stage 2 record can
never be silently pooled with a ground-truth-annotation record from every
other table in this project -- and carries `"timestep_kind"` (`"t0"` or
`"addon"`) plus `"t_index"`/`"episode_len"`/`"t_frac"` so the two sampling
protocols in one file are never accidentally averaged together either.

What a bug in this script would look like, beyond the failure modes Stage 1's
docstring already documents for the functions imported from it (inherited
for free, not re-argued here -- e.g. `_assert_argmax_consistency`'s raise on
an action-window slicing bug applies to every `openvla_action_softmax` call
this script makes, exactly as it does in Stage 1):
  (1) Self-generation collapsing into a degenerate repeat loop under greedy
      decoding -- `cot_structured=False` with `n_cot_tokens_generated` pinned
      at `--max-new-tokens` is the signature; the check is reading a handful
      of real `orig_cot_head` values before trusting an aggregate, the same
      way `rollout_edit_probe.json`'s README did once for ECoT-bridge.
  (2) The four bbox-dependent families (see CAVEAT above) skipping near-100%
      on ecot-bridge/lora-r32 is EXPECTED. The bug this could mask is the
      OTHER nine families also skipping near-100% on those two checkpoints --
      that pattern would mean self-generation itself is failing, not that
      the edits are structurally inapplicable, and is not the same finding.
  (3) `--max-new-tokens` too small for a checkpoint whose free CoT runs
      longer than 320 tokens (this project's one confirmed sample ran 255)
      would truncate mid-tag rather than error. Signature: many rows'
      `n_cot_tokens_generated` sitting exactly at the ceiling; the fix is
      raising `--max-new-tokens`, not trusting a truncated `reasoning` dict.
  (4) A `sample` index in a `"timestep_kind": "addon"` row that does not
      correspond to the SAME episode as the `"t0"` row with the same
      `checkpoint`+`sample` would break the one thing the add-on is for
      (comparing one episode's t=0 sensitivity against its own t~T/3, t~2T/3
      sensitivity). This is why `load_libero_samples_multistep` reproduces
      `load_libero_samples_openvla`'s exact shuffle/seed/filter -- see that
      function's docstring -- rather than sampling episodes independently.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

# Reused verbatim -- see module docstring's "Self-generation call pattern".
from experiments.cotfaith_rollout_edit import (
    gen_cot,
    build_cot_body,
    has_structured_cot,
)
# Reused verbatim -- episode loader (base t0 pass) and the canonical family
# dict. See module docstring's "Families" and "Sampling" sections.
from experiments.cotfaith_edit import load_libero_samples as load_libero_samples_openvla
# Reused verbatim -- Stage 1's model loader + BOTH metrics' scoring functions.
# See module docstring's "Scoring" section for exactly which of these are
# imported instead of reimplemented.
from experiments.cotfaith_stage1_continuous import (
    load_model_openvla,
    build_openvla_prefix,
    openvla_action_softmax,
    openvla_teacher_forced_logp,
    total_variation,
    kl_divergence,
    text_covariates,
)
from sharpguard.attacks import EDIT_FAMILIES, apply_instr_random_sub


# The 3 checkpoints this experiment is scoped to (DeepThinkVLA excluded -- see
# module docstring "EXCLUDED"), and where each identifier was confirmed from
# this repo, recorded here so bolt/run_stage2_selfgen.sh has one place to read
# them from and a reviewer can check each against its source file directly.
KNOWN_CHECKPOINTS = {
    "ecot-bridge": {
        "model_family": "openvla",
        "ckpt_path": "Embodied-CoT/ecot-openvla-7b-bridge",
        "source": "bolt/boltconfig-cotfaith-edit-ecot-bridge.yaml (CKPT_HF_ID); "
                   "same identifier experiments/cotfaith_stage1_continuous.py's "
                   "ecot-bridge row uses.",
    },
    "lora-r32": {
        "model_family": "openvla",
        "ckpt_path": None,  # resolved at run time from CKPT_TASK_ID's S3 artifacts
        "ckpt_task_id": "bcihypv3gu",
        "source": "bolt/boltconfig-cotfaith-edit13-r32.yaml (CKPT_TASK_ID); also "
                   "bolt/submit_edit13.sh's ROWS mapping 'r32:bcihypv3gu'; same "
                   "checkpoint experiments/cotfaith_stage1_continuous.py's lora-r32 "
                   "row uses, so its LORA_R32_* S3 credentials can be reused as-is.",
    },
    "no-cot": {
        "model_family": "openvla",
        "ckpt_path": None,
        "ckpt_task_id": "a8eegzcg4r",
        "source": "bolt/boltconfig-cotfaith-edit13-no-cot.yaml (CKPT_TASK_ID); also "
                   "bolt/submit_edit13.sh's ROWS mapping 'no-cot:a8eegzcg4r'. NOT one "
                   "of Stage 1's 6 checkpoints -- Stage 1 has no no-CoT row -- so this "
                   "checkpoint's S3 credentials are NEW to this job, not something "
                   "bolt/submit_stage1.sh already renders.",
    },
}

# Families that take a `seed=` kwarg (their edit is itself randomized, not
# just which sample it is applied to). Mirrors the dispatch in
# experiments/cotfaith_edit.py:run() exactly, restricted to the tuple that
# script special-cases (cross_task_swap ALSO takes a seed kwarg, but is
# dispatched in its own branch below because it additionally needs
# `alt_reasoning`).
_SEEDED_FAMILIES = {"syntactic_scramble", "bbox_jitter_null", "instr_random_sub"}


# --------------------------------------------------------------------------
# Episode loading: base t0 pass reuses cotfaith_edit.load_libero_samples
# directly (imported above as load_libero_samples_openvla). The timestep
# add-on needs two more indices per episode, which is the one genuinely new
# piece of data-loading code in this file -- see module docstring's
# "ASSUMPTION" paragraph for exactly what it does and does not reuse.
# --------------------------------------------------------------------------
def load_libero_samples_multistep(dataset_repo, tfds_subdir, reasoning_json,
                                    n_episodes, seed, fracs, stats):
    """Yield extra-timestep frames from the SAME pre-recorded episodes
    `load_libero_samples_openvla` iterates, at fractional offsets into each
    qualifying episode's recorded length.

    Byte-identical fetch/shuffle/filter to `cotfaith_edit.load_libero_samples`
    (same HF snapshot call, same tfds builder + `shuffle_files`/`shuffle_seed`,
    same file_path/demo_id join against `reasoning_json`, same "truthy GT
    annotation at step '0'" episode filter) so that, given the same `seed`,
    the k-th qualifying episode here is the SAME episode as the k-th sample
    `load_libero_samples_openvla` yields -- this is what lets a caller stamp
    both passes' rows with the same `sample` index and compare one episode's
    t=0 sensitivity against its own t~T/3 / t~2T/3 sensitivity. The GT
    annotation itself is looked up only to reproduce that filter and is never
    read as content -- this script has no ground-truth CoT anywhere in it.

    Yields (image, instruction, file_base, demo_id, episode_ordinal, t_index,
    episode_len, frac) once per (qualifying episode, distinct valid frac).
    Episodes shorter than 3 steps cannot place two distinct interior indices
    and are counted in `stats["n_short_skipped"]` (mutated in place) rather
    than silently omitted -- `stats` must be a dict the caller can read after
    the generator is drained.
    """
    from sharpguard.hf_retry import snapshot_with_retry
    ds_dir = Path(snapshot_with_retry(repo_id=dataset_repo, repo_type="dataset"))
    tfds_dir = ds_dir / tfds_subdir
    with open(ds_dir / reasoning_json) as f:
        rdata = json.load(f)

    import tensorflow_datasets as tfds
    from PIL import Image as PILImage
    builder = tfds.builder_from_directory(str(tfds_dir))
    ds = builder.as_dataset(split="train", shuffle_files=(seed != 0),
                              read_config=tfds.ReadConfig(shuffle_seed=seed))
    ep_ordinal = 0
    for ep in ds:
        if ep_ordinal >= n_episodes:
            break
        meta = ep.get("episode_metadata", {})
        file_path = meta.get("file_path").numpy().decode()
        demo_id = int(meta.get("demo_id").numpy())
        file_base = os.path.basename(file_path)
        rep = rdata.get(file_path) or rdata.get(file_base) or {}
        rdemo = rep.get(str(demo_id), {})
        if not rdemo.get("0", {}):
            continue
        steps = list(ep["steps"].as_numpy_iterator())
        T = len(steps)
        this_ep = ep_ordinal
        ep_ordinal += 1
        if T < 3:
            stats["n_short_skipped"] = stats.get("n_short_skipped", 0) + 1
            continue
        instr = steps[0]["language_instruction"].decode()
        seen_t = set()
        for frac in fracs:
            t = min(T - 1, max(1, int(round(T * frac))))
            if t in seen_t:
                continue
            seen_t.add(t)
            img = PILImage.fromarray(
                np.asarray(steps[t]["observation"]["image"])).convert("RGB")
            yield (img, instr, file_base, demo_id, this_ep, t, T, frac)


# --------------------------------------------------------------------------
# Edit dispatch: mirrors experiments/cotfaith_edit.py:run()'s per-family
# special-casing exactly (same seed offsets, same cross_task_swap alt
# selection, same instr_random_sub side effect), generalized only in WHERE
# `reasoning` comes from.
# --------------------------------------------------------------------------
def apply_edit(fname, reasoning, *, instr, args, si, buffered, rng_cross):
    """Dispatch one EDIT_FAMILIES[fname] call.

    Returns (edited, edit_meta, edit_prefix). `edited` is None if the family
    judged itself inapplicable (most commonly subject_swap /
    adversarial_plausible / bbox_jitter_null on self-generated `reasoning` --
    see module docstring CAVEAT). `edit_prefix` is a full prompt prefix
    (built by `build_openvla_prefix`, so it is directly comparable to the
    sample's own prefix) and is non-None ONLY for instr_random_sub, whose
    edit lives in the instruction rather than the CoT -- callers must decode
    against `edit_prefix` in place of the sample's own prefix whenever it is
    not None.

    `si` and `buffered` are used only by cross_task_swap, to draw an
    unrelated sample's ALREADY self-generated reasoning as the donor (the
    reason the base pass buffers every sample's self-generation before
    scoring any family -- see `_self_gen_pass`'s docstring). `rng_cross` is
    the caller's persistent `random.Random(args.seed + 7)`, matching
    `cotfaith_edit.py:run()`'s own offset, reused across every family/sample
    of one process so alt-selection is a single reproducible stream rather
    than re-seeded per call.
    """
    fedit = EDIT_FAMILIES[fname]
    if fname == "cross_task_swap":
        if len(buffered) < 2:
            return None, {}, None
        alt_idx = rng_cross.randrange(len(buffered))
        if alt_idx == si and len(buffered) > 1:
            alt_idx = (alt_idx + 1) % len(buffered)
        edited = fedit(reasoning, alt_reasoning=buffered[alt_idx]["reasoning"],
                       seed=args.seed)
    elif fname in _SEEDED_FAMILIES:
        edited = fedit(reasoning, seed=args.seed + si)
    else:
        edited = fedit(reasoning)
    if edited is None:
        return None, {}, None
    edit_meta = edited.pop("__edit_meta__", {})
    edit_prefix = None
    if "instr_random_sub" in edit_meta:
        instr_pert = apply_instr_random_sub(
            instr.lower(), seed=args.seed + si, n_tokens=int(edit_meta["instr_random_sub"]))
        # build_openvla_prefix() lower()s its argument itself; instr_pert is
        # already lowercase (apply_instr_random_sub only substitutes whole
        # words with entries from its own lowercase vocabulary), so this is
        # byte-identical to cotfaith_edit.py's inline f-string reconstruction
        # of the same prefix, without retyping the ECoT prompt template here.
        edit_prefix = build_openvla_prefix(instr_pert)
        edit_meta["instruction_perturbed"] = instr_pert[:200]
    return edited, edit_meta, edit_prefix


# --------------------------------------------------------------------------
# Shared scoring core: one ORIGINAL decode per sample (self-generated CoT),
# reused across every family; one EDITED decode per (sample, family).
# --------------------------------------------------------------------------
def _decode_original(model, processor, device, dtype, img, instr, reasoning):
    """Render + decode the ORIGINAL (self-generated, unedited) sample once.

    Returns None if the checkpoint's greedy decode did not yield 7 valid
    action-token positions (see openvla_action_softmax's own docstring for
    when that happens) -- callers must skip the whole sample in that case,
    exactly as experiments/cotfaith_stage1_continuous.py:run() does, rather
    than attempt any family against a half-decoded original.
    """
    prefix = build_openvla_prefix(instr)
    orig_cot = build_cot_body(reasoning)
    orig_probs, values = openvla_action_softmax(
        model, processor, img, prefix + orig_cot + " ACTION:", device, dtype)
    if orig_probs is None:
        return None
    orig_logp, orig_n_tok = openvla_teacher_forced_logp(
        model, processor, img, prefix, orig_cot, device, dtype)
    return {
        "prefix": prefix, "orig_cot": orig_cot, "orig_probs": orig_probs,
        "values": values, "orig_logp": orig_logp, "orig_n_tok": orig_n_tok,
        "orig_expected": orig_probs @ values,
        "orig_greedy_action": values[np.argmax(orig_probs, axis=1)],
    }


def _score_family(model, processor, device, dtype, args, *, fname, reasoning,
                   instr, img, dec, base_record, si, buffered, rng_cross):
    """One (sample, family) row: dispatch the edit, decode if applicable,
    and compute BOTH metric (a) and metric (b) fields -- see module docstring
    "Scoring". Shared by the base t0 pass and the timestep add-on pass so the
    two protocols can never silently diverge in how a family is scored.
    """
    edited, edit_meta, edit_prefix = apply_edit(
        fname, reasoning, instr=instr, args=args, si=si, buffered=buffered,
        rng_cross=rng_cross)
    if edited is None:
        return {**base_record, "skipped": True, "reason": "no plausible edit"}

    edited_cot = build_cot_body(edited)
    prefix = dec["prefix"]
    use_prefix = edit_prefix if edit_prefix is not None else prefix
    if edited_cot == dec["orig_cot"] and use_prefix == prefix:
        # Same admissibility guard as experiments/cotfaith_stage1_continuous.py,
        # generalized to "nothing the model is shown changed" now that a
        # prefix (instr_random_sub) can change instead of the CoT: skip only
        # if BOTH the rendered CoT and the prefix are unchanged.
        return {**base_record, "skipped": True,
                "reason": "identical render (inapplicable)", "edit_meta": edit_meta}

    edit_probs, _ = openvla_action_softmax(
        model, processor, img, use_prefix + edited_cot + " ACTION:", device, dtype)
    if edit_probs is None:
        return {**base_record, "skipped": True, "reason": "edit decode failed",
                "edit_meta": edit_meta}
    edit_logp, edit_n_tok = openvla_teacher_forced_logp(
        model, processor, img, use_prefix, edited_cot, device, dtype)

    values, orig_probs = dec["values"], dec["orig_probs"]
    edit_expected = edit_probs @ values
    edit_greedy_action = values[np.argmax(edit_probs, axis=1)]
    delta = edit_greedy_action - dec["orig_greedy_action"]
    delta_l1 = float(np.mean(np.abs(delta)))
    delta_linf = float(np.max(np.abs(delta)))
    tv_per_dim = [total_variation(orig_probs[k], edit_probs[k]) for k in range(7)]
    kl_per_dim = [kl_divergence(orig_probs[k], edit_probs[k], eps=args.kl_eps)
                 for k in range(7)]
    expected_l2 = float(np.linalg.norm(edit_expected - dec["orig_expected"]))
    cov = text_covariates(dec["orig_cot"], edited_cot)
    orig_logp = dec["orig_logp"]
    delta_logp = None if orig_logp is None or edit_logp is None else edit_logp - orig_logp

    return {
        **base_record, "skipped": False, "edit_meta": edit_meta,
        "a_orig": [float(x) for x in dec["orig_greedy_action"]],
        "a_edit": [float(x) for x in edit_greedy_action],
        "delta_per_dim": [float(x) for x in delta],
        "delta_l1_mean": delta_l1, "delta_linf": delta_linf,
        "faithful": delta_linf > args.threshold,
        "tv_per_dim": tv_per_dim, "tv_mean": float(np.mean(tv_per_dim)),
        "kl_per_dim": kl_per_dim, "kl_mean": float(np.mean(kl_per_dim)),
        "expected_action_orig": [float(x) for x in dec["orig_expected"]],
        "expected_action_edit": [float(x) for x in edit_expected],
        "expected_action_l2": expected_l2,
        "logp_orig_cot": orig_logp, "logp_edit_cot": edit_logp,
        "delta_logp": delta_logp,
        "n_tokens_orig_cot": dec["orig_n_tok"], "n_tokens_edit_cot": edit_n_tok,
        **cov,
    }


# --------------------------------------------------------------------------
# Base (t0) pass: two sub-passes, self-generation then scoring, because
# cross_task_swap needs OTHER samples' already-self-generated reasoning as
# its donor and self-generation (~8.6s/call) is far too expensive to redo
# per family. Mirrors cotfaith_edit.py:run() materializing `all_samples`
# up front for the identical reason ("build an alt reasoning reservoir").
# --------------------------------------------------------------------------
def _self_gen_pass(args, model, processor, device, dtype):
    buffered = []
    stats = {"n_requested": args.n_samples, "n_loaded": 0, "n_self_gen_failed": 0}
    for si, sample in enumerate(load_libero_samples_openvla(
            args.dataset_repo, args.tfds_subdir, args.reasoning_json,
            args.n_samples, seed=args.seed)):
        img, instr, _gt_unused, fbase, dem = sample  # GT annotation intentionally
                                                       # discarded -- see module docstring
        t0 = time.time()
        # Printed before, not just after, gen_cot(): the two failed self-gen
        # Bolt attempts (cbe2xdc9qt, 76twybafqw) both died with zero output
        # between dataset-load and the crash, and the only other progress
        # print in this loop fires every 20 *completed* samples -- so a death
        # on sample 0-19 was, and would still be, indistinguishable from a
        # death during dataset loading itself. This line alone separates
        # those two cases the next time it happens.
        print(f"[stage2] sample {si}: generating self CoT "
              f"(file_base={fbase})", flush=True)
        try:
            reasoning, n_new, raw_text = gen_cot(
                model, processor, img, instr, device=device, pixel_dtype=dtype,
                max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f"[stage2] sample {si}: self-generation raised "
                  f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}")
            stats["n_self_gen_failed"] += 1
            continue
        buffered.append({
            "si": si, "img": img, "instr": instr, "fbase": fbase,
            "reasoning": reasoning, "raw_text": raw_text, "n_new": n_new,
            "cot_gen_seconds": round(time.time() - t0, 3),
            "structured": has_structured_cot(reasoning),
        })
        stats["n_loaded"] += 1
        if stats["n_loaded"] % 20 == 0:
            print(f"[stage2] self-gen {stats['n_loaded']}/{args.n_samples} done")
    return buffered, stats


def _score_base_pass(args, model, processor, device, dtype, buffered):
    rng_cross = random.Random(args.seed + 7)  # same offset as cotfaith_edit.py:run
    rows = []
    for pos, entry in enumerate(buffered):
        reasoning, instr, img = entry["reasoning"], entry["instr"], entry["img"]
        try:
            dec = _decode_original(model, processor, device, dtype, img, instr, reasoning)
            if dec is None:
                print(f"[stage2] sample {entry['si']}: original decode failed "
                      f"(self-generated CoT)")
                continue
            common = {
                "sample": entry["si"], "checkpoint": args.checkpoint_label,
                "seed": args.seed, "instruction": instr[:200],
                "file_base": entry["fbase"], "cot_source": "self_generated",
                "timestep_kind": "t0", "t_index": 0, "episode_len": None, "t_frac": 0.0,
                "cot_gen_seconds": entry["cot_gen_seconds"],
                "n_cot_tokens_generated": entry["n_new"],
                "cot_structured": entry["structured"],
                "orig_cot_head": entry["raw_text"][:300],
            }
            for fname in args.families_list:
                rows.append(_score_family(
                    model, processor, device, dtype, args, fname=fname,
                    reasoning=reasoning, instr=instr, img=img, dec=dec,
                    base_record={**common, "family": fname}, si=pos,
                    buffered=buffered, rng_cross=rng_cross))
        except Exception as e:
            print(f"[stage2] sample {entry['si']} failed: {e}\n"
                  f"{traceback.format_exc()[-800:]}")
        if (pos + 1) % 20 == 0:
            print(f"[stage2] scored {pos+1}/{len(buffered)} buffered samples")
    return rows


# --------------------------------------------------------------------------
# Timestep add-on pass: single-pass (self-generate then immediately score),
# because none of args.timestep_families_list is cross_task_swap -- validated
# in main() -- so there is nothing that needs another sample's buffer.
# --------------------------------------------------------------------------
def _timestep_addon_pass(args, model, processor, device, dtype):
    if args.n_samples_addon <= 0 or not args.timestep_families_list:
        return [], {"n_episodes_requested": args.n_samples_addon, "skipped_entirely": True}
    stats = {"n_episodes_requested": args.n_samples_addon, "n_short_skipped": 0,
             "n_self_gen_failed": 0, "n_rows_attempted": 0}
    rows = []
    rng_cross = random.Random(args.seed + 7)  # cross_task_swap never appears in
                                               # timestep_families_list (see main());
                                               # kept only so apply_edit's signature
                                               # needs no special case for this pass.
    for img, instr, fbase, dem, ep_ord, t_index, episode_len, t_frac in \
            load_libero_samples_multistep(
                args.dataset_repo, args.tfds_subdir, args.reasoning_json,
                args.n_samples_addon, seed=args.seed, fracs=args.timestep_fracs_list,
                stats=stats):
        stats["n_rows_attempted"] += 1
        t0 = time.time()
        try:
            reasoning, n_new, raw_text = gen_cot(
                model, processor, img, instr, device=device, pixel_dtype=dtype,
                max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f"[stage2-addon] episode {ep_ord} t={t_index}: self-generation "
                  f"raised {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}")
            stats["n_self_gen_failed"] += 1
            continue
        cot_gen_seconds = round(time.time() - t0, 3)
        try:
            dec = _decode_original(model, processor, device, dtype, img, instr, reasoning)
            if dec is None:
                print(f"[stage2-addon] episode {ep_ord} t={t_index}: "
                      f"original decode failed")
                continue
            base_record = {
                "sample": ep_ord, "checkpoint": args.checkpoint_label,
                "seed": args.seed, "instruction": instr[:200], "file_base": fbase,
                "cot_source": "self_generated", "timestep_kind": "addon",
                "t_index": int(t_index), "episode_len": int(episode_len),
                "t_frac": float(t_frac), "cot_gen_seconds": cot_gen_seconds,
                "n_cot_tokens_generated": n_new,
                "cot_structured": has_structured_cot(reasoning),
                "orig_cot_head": raw_text[:300],
            }
            for fname in args.timestep_families_list:
                rows.append(_score_family(
                    model, processor, device, dtype, args, fname=fname,
                    reasoning=reasoning, instr=instr, img=img, dec=dec,
                    base_record={**base_record, "family": fname}, si=ep_ord,
                    buffered=[], rng_cross=rng_cross))
        except Exception as e:
            print(f"[stage2-addon] episode {ep_ord} t={t_index} failed: {e}\n"
                  f"{traceback.format_exc()[-800:]}")
    return rows, stats


def _aggregate(rows, families):
    metric_keys = ["tv_mean", "kl_mean", "expected_action_l2", "delta_logp",
                   "delta_linf", "delta_l1_mean", "edit_distance_tokens",
                   "n_tokens_changed", "len_delta_tokens"]
    agg = {}
    for fname in families:
        by_kind = {}
        for kind in ("t0", "addon"):
            kind_rows = [r for r in rows if r["family"] == fname and r["timestep_kind"] == kind]
            if not kind_rows:
                continue
            scored = [r for r in kind_rows if not r.get("skipped")]
            n_skip = sum(1 for r in kind_rows if r.get("skipped"))
            entry = {"n": len(scored), "n_skipped": n_skip}
            if scored:
                entry["faithful_rate"] = float(np.mean([r["faithful"] for r in scored]))
                for k in metric_keys:
                    vals = [r[k] for r in scored if r.get(k) is not None]
                    if vals:
                        entry[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                                    "median": float(np.median(vals)), "n": len(vals)}
            by_kind[kind] = entry
        agg[fname] = by_kind
    return agg


def run(args):
    import torch
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[stage2] loading OpenVLA-family model from {args.ckpt_path}")
    model, processor = load_model_openvla(args.ckpt_path, device, dtype)

    buffered, gen_stats = _self_gen_pass(args, model, processor, device, dtype)
    print(f"[stage2] self-gen pass done: {gen_stats}")
    base_rows = _score_base_pass(args, model, processor, device, dtype, buffered)

    addon_rows, addon_stats = _timestep_addon_pass(args, model, processor, device, dtype)
    print(f"[stage2] timestep add-on done: {addon_stats}")

    all_rows = base_rows + addon_rows
    agg = _aggregate(all_rows, args.families_list)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "cotfaith_stage2_selfgen",
        "cot_source": "self_generated",
        "checkpoint": args.checkpoint_label,
        "model_family": "openvla",
        "ckpt_path": args.ckpt_path,
        "n_samples_requested": args.n_samples,
        "n_samples_addon_requested": args.n_samples_addon,
        "seed": args.seed,
        "families": args.families_list,
        "timestep_addon_families": args.timestep_families_list,
        "timestep_addon_fracs": args.timestep_fracs_list,
        "max_new_tokens_cot": args.max_new_tokens,
        "threshold_linf": args.threshold,
        "kl_eps": args.kl_eps,
        "grid_convention": "checkpoint (see experiments/cotfaith_stage1_continuous.py "
                            "module docstring: CANONICAL DE-QUANTIZATION GRID); reused "
                            "for both metric (a) and metric (b) here -- see this "
                            "script's own module docstring, 'Scoring'.",
        "deepthink_scope_note": "DeepThinkVLA is out of scope for this script -- see "
                                 "module docstring 'EXCLUDED'.",
        "self_gen_pass_stats": gen_stats,
        "timestep_addon_stats": addon_stats,
        "aggregate": agg,
        "per_sample": all_rows,
    }
    report_path = out / "stage2_selfgen_report.json"
    report_path.write_text(json.dumps(payload, indent=2, default=str))

    print(f"\n===== STAGE2 SELFGEN DONE  checkpoint={args.checkpoint_label} "
          f"seed={args.seed} =====")
    for fname, by_kind in agg.items():
        for kind, a in by_kind.items():
            if a["n"] == 0:
                print(f"  {fname:22s} [{kind:5s}]  (0 scored; skipped={a.get('n_skipped', 0)})")
            else:
                print(f"  {fname:22s} [{kind:5s}]  n={a['n']:3d}  "
                      f"faithful={a.get('faithful_rate', float('nan')):.3f}  "
                      f"TVmean={(a.get('tv_mean') or {}).get('mean', float('nan')):.4f}")
    print(f"  report -> {report_path}")
    sys.stdout.flush()
    os._exit(0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-path", required=True,
                   help="HF repo id (ecot-bridge) or local dir (lora-r32 / no-cot -- "
                        "the S3 sync in bolt/run_stage2_selfgen.sh writes the "
                        "merged_model to a local dir, NOT an HF id).")
    p.add_argument("--checkpoint-label", required=True,
                   help="Free-form identifier stamped onto every record's "
                        "'checkpoint' field, e.g. one of KNOWN_CHECKPOINTS' keys.")
    p.add_argument("--out", default="./cotfaith-stage2-selfgen")
    p.add_argument("--n-samples", type=int, default=100,
                   help="episodes scored at t=0 against ALL --families.")
    p.add_argument("--n-samples-addon", type=int, default=40,
                   help="episodes ALSO scored at the two --timestep-fracs, against "
                        "ONLY --timestep-families. Defaults to 40, not 100: this "
                        "project's existing convention for a deliberately cheap "
                        "add-on pass is N=40 (see JUDGE_N_SAMPLES in "
                        "bolt/boltconfig-stage1-continuous.yaml); 0 disables the "
                        "add-on entirely.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--families", default=",".join(EDIT_FAMILIES.keys()),
                   help="comma-separated EDIT_FAMILIES keys. Default is all 13, "
                        "read from the dict itself rather than retyped.")
    p.add_argument("--timestep-families", default="paraphrase_null,syntactic_scramble,direction_flip",
                   help="the subset of --families that also gets the timestep add-on.")
    p.add_argument("--timestep-fracs", default="0.333333,0.666667",
                   help="fractions of each episode's recorded length T to add-on "
                        "sample, as t=round(T*frac) (clamped to [1, T-1]).")
    p.add_argument("--kl-eps", type=float, default=1e-6)
    p.add_argument("--threshold", type=float, default=0.05,
                   help="delta_linf > threshold decision for metric (a); default "
                        "matches experiments/cotfaith_edit.py's own default.")
    p.add_argument("--dataset-repo", default="Embodied-CoT/embodied_features_and_demos_libero")
    p.add_argument("--tfds-subdir", default="libero_lm_90/1.0.0")
    p.add_argument("--reasoning-json", default="libero_reasonings.json",
                   help="Used ONLY to reproduce the existing episode-selection "
                        "filter (a truthy GT annotation must exist at step 0), so "
                        "this script's sample pool matches every other script's. "
                        "Never read as CoT content -- see module docstring.")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--max-new-tokens", type=int, default=320,
                   help="cap on self-generated CoT length; default matches "
                        "experiments/cotfaith_rollout_edit.py's own default and "
                        "comfortably exceeds the one confirmed real sample (255 "
                        "tokens, bolt phenc9ygb4). See module docstring failure mode (3).")
    args = p.parse_args()

    args.families_list = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in args.families_list if f not in EDIT_FAMILIES]
    if unknown:
        raise ValueError(f"--families contains unknown families: {unknown}")

    args.timestep_families_list = [f.strip() for f in args.timestep_families.split(",") if f.strip()]
    unknown_ts = [f for f in args.timestep_families_list if f not in EDIT_FAMILIES]
    if unknown_ts:
        raise ValueError(f"--timestep-families contains unknown families: {unknown_ts}")
    if "cross_task_swap" in args.timestep_families_list:
        raise ValueError(
            "cross_task_swap cannot be a --timestep-families member: "
            "_timestep_addon_pass runs single-pass with no cross-sample buffer "
            "(see its docstring), so cross_task_swap would always report "
            "skipped=True there for a reason unrelated to the family itself.")

    args.timestep_fracs_list = [float(x) for x in args.timestep_fracs.split(",") if x.strip()]

    run(args)


if __name__ == "__main__":
    # cbe2xdc9qt and 76twybafqw (the two Bolt attempts before this fix) both
    # ended with no Python-level trace at all: run.log stops cleanly right
    # after dataset load, and the parent shell's `wait` on the child PID
    # returned an exit status that was not even a valid 0-255 POSIX code --
    # bash itself lost track of what happened, not just this script. If that
    # recurs, a SIGKILL (uncatchable in user space, by definition) is the
    # only explanation this wrapper cannot rule in or out; anything else --
    # a Python exception, an os._exit from a native library, a SystemExit --
    # now gets written to a dedicated crash file with an explicit flush
    # before the process can disappear silently a third time.
    crash_path = os.environ.get("COTFAITH_CRASH_LOG",
                                 "/tmp/cotfaith_stage2_crash.log")
    try:
        main()
    except BaseException:
        try:
            with open(crash_path, "a") as fh:
                fh.write(f"\n=== crash at pid {os.getpid()} ===\n")
                traceback.print_exc(file=fh)
                fh.flush()
                os.fsync(fh.fileno())
        finally:
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
        raise
