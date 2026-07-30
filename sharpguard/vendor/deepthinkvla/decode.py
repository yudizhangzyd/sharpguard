"""Correct action decode and prompt assembly for DeepThinkVLA.

This module exists because our first attempt at DeepThinkVLA reused OpenVLA's
conventions, and every one of them is wrong for a pi0-FAST-initialized
PaliGemma checkpoint:

  | quantity        | what we did (wrong)      | what the checkpoint declares    |
  |-----------------|--------------------------|---------------------------------|
  | action ids      | top 256 of vocab_size    | [254976, 257023], base vocab    |
  | bin count       | 256                      | 2047 midpoints of 2048 edges    |
  | id -> bin       | V - 1 - id               | 2047 - (id - 254976), reversed  |
  | extraction      | model.generate()         | one forward pass, argmax over   |
  |                 |                          | the action slice at 70 fixed    |
  |                 |                          | positions, bidirectional mask   |
  | output shape    | one 7-vector             | a (10, 7) action chunk          |
  | un-normalize    | min/max                  | QUANTILE (q01/q99)              |
  | prompt          | "Instruction: ...\\nAction:" | THINK_PREFIX + "Task: x;"   |
  | CoT delimiters  | literal "<think>" text   | token ids 257153 / 257154       |

The consequence of the id-range error alone was that zero generated tokens ever
landed in the assumed window, so `a_orig` was None on every sample and all three
DeepThinkVLA edit runs recorded n=0 for all 11 families while exiting 0.

Everything here is checked against the checkpoint at load time rather than
trusted: assert_config_matches() compares our pinned constants to config.json
and raises on any disagreement, so a checkpoint with a different action range
fails loudly instead of producing plausible-looking numbers.
"""
from __future__ import annotations

import json

import numpy as np

from . import (ACTION_DIM, ACTION_START, IMAGE_TOKEN, NUM_ACTIONS_CHUNK,
               PROMPT_END_TOKEN_IDS, THINK_END, THINK_PREFIX, THINK_START)

# config.json of every yinchenghust/deepthinkvla_* checkpoint.
ACTION_TOKEN_BEGIN = 254976
ACTION_TOKEN_END = 257023
# bins = linspace(-1, 1, 2048) -> 2047 midpoints. Upstream:
#   self.bins = np.linspace(-1, 1, 2048)
#   self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
N_BIN_EDGES = 2048


def bin_centers() -> np.ndarray:
    edges = np.linspace(-1.0, 1.0, N_BIN_EDGES)
    return (edges[:-1] + edges[1:]) / 2.0


def assert_config_matches(config) -> None:
    """Fail if the checkpoint's action space is not the one we decode for."""
    got = {
        "action_token_begin_idx": getattr(config, "action_token_begin_idx", None),
        "action_token_end_idx": getattr(config, "action_token_end_idx", None),
        "action_start_token_index": getattr(config, "action_start_token_index", None),
        "think_start_token_index": getattr(config, "think_start_token_index", None),
        "think_end_token_index": getattr(config, "think_end_token_index", None),
        "image_token_index": getattr(config, "image_token_index", None),
    }
    want = {
        "action_token_begin_idx": ACTION_TOKEN_BEGIN,
        "action_token_end_idx": ACTION_TOKEN_END,
        "action_start_token_index": ACTION_START,
        "think_start_token_index": THINK_START,
        "think_end_token_index": THINK_END,
        "image_token_index": IMAGE_TOKEN,
    }
    bad = {k: (want[k], got[k]) for k in want if want[k] != got[k]}
    if bad:
        raise RuntimeError(
            "DeepThinkVLA checkpoint does not match the action space this "
            "decoder was written for; refusing to decode rather than emit "
            "numbers under the wrong convention. "
            + ", ".join(f"{k}: expected {w}, config says {g}"
                        for k, (w, g) in sorted(bad.items())))


def load_quantile_norm_stats(ckpt_dir_or_repo: str) -> dict:
    """Return {'q01': (7,), 'q99': (7,)} for the LIBERO action space.

    Upstream normalizes LIBERO actions with the QUANTILE scheme, so
    un-normalizing needs q01/q99 and NOT min/max -- they differ, and using the
    wrong pair rescales every action by a constant factor per dimension, which
    would inflate or deflate every delta_linf we report.
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(ckpt_dir_or_repo, "norm_stats.json")
    except Exception:
        import os
        path = os.path.join(ckpt_dir_or_repo, "norm_stats.json")
    stats = json.loads(open(path).read())["action"]

    missing = [k for k in ("q01", "q99") if k not in stats]
    if missing:
        raise RuntimeError(
            f"norm_stats.json for {ckpt_dir_or_repo} has no {missing}; the "
            f"LIBERO action space is QUANTILE-normalized upstream, so falling "
            f"back to min/max would silently change the scale of every "
            f"reported action delta. Keys present: {sorted(stats)}")
    q01 = np.asarray(stats["q01"], dtype=np.float64)[:ACTION_DIM]
    q99 = np.asarray(stats["q99"], dtype=np.float64)[:ACTION_DIM]
    return {"q01": q01, "q99": q99}


def unnormalize(normalized: np.ndarray, norm: dict) -> np.ndarray:
    """Invert QUANTILE normalization: [-1, 1] -> [q01, q99]."""
    q01, q99 = norm["q01"], norm["q99"]
    return 0.5 * (normalized + 1.0) * (q99 - q01) + q01


def build_prompt_text(instruction: str, n_images: int = 1) -> str:
    """Upstream's evaluation prompt, exactly.

    `"<image>" * n_images + THINK_PREFIX + f"Task: {instruction.lower()};"`

    The lowercasing and the trailing semicolon are both load-bearing: the
    semicolon plus the processor's appended newline are the two ids
    (`[235289, 108]`) the model class uses to find where the prompt ends and the
    CoT begins.
    """
    return "<image>" * n_images + THINK_PREFIX + f"Task: {instruction.lower()};"


def build_input_cot_ids(prompt_ids, cot_ids, torch):
    """Assemble `[prompt, <think>, cot, </think>, <action>]`.

    This is the sequence `prompt_cot_predict_action` expects. Upstream's no-CoT
    path appends exactly `[257153, 257154, 257155]` -- an empty think block --
    which confirms the ordering: the action marker comes last and the CoT sits
    between the two think tokens.

    Injecting a CoT here is precisely the P2 intervention: the model is handed a
    reasoning trace it did not generate and we read the action it produces.
    """
    tail = [THINK_START] + list(cot_ids) + [THINK_END, ACTION_START]
    tail_t = torch.tensor([tail], dtype=prompt_ids.dtype, device=prompt_ids.device)
    return torch.cat([prompt_ids, tail_t], dim=-1)


def decode_action_chunk(logits, action_start_idx, torch, centers=None):
    """logits -> (NUM_ACTIONS_CHUNK, ACTION_DIM) normalized action chunk.

    Mirrors upstream `predict_cot_action`: restrict the argmax to the action
    slice so the model cannot emit a non-action token, reverse the index, clip,
    then look up bin centers.
    """
    if centers is None:
        centers = bin_centers()
    start = action_start_idx.unsqueeze(1)
    offsets = torch.arange(ACTION_DIM * NUM_ACTIONS_CHUNK,
                           device=logits.device).unsqueeze(0)
    seq_idx = start + offsets
    slice_argmax = (
        logits[torch.arange(logits.shape[0], device=logits.device).unsqueeze(-1),
               seq_idx,
               ACTION_TOKEN_BEGIN:ACTION_TOKEN_END + 1]
        .argmax(dim=-1).cpu().numpy())
    ids = (ACTION_TOKEN_END - ACTION_TOKEN_BEGIN) - slice_argmax
    ids = np.clip(ids, 0, centers.shape[0] - 1)
    return centers[ids].reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)


def segment_boundaries(input_cot_ids) -> dict:
    """4-bucket segment boundaries for the sequence the language model sees.

    Returns exclusive ends {visual_end, instr_end, cot_end, action_end} as
    absolute indices into that sequence.

    This replaces the text-marker search the shared analyzer uses for OpenVLA.
    Two things make markers unusable here. PaliGemma puts the 256 image tokens
    INLINE in input_ids (id 257152) rather than prepending them at embedding
    time, so there is no `+ n_visual` shift to apply; and the CoT delimiters are
    special token ids, not literal text. Searching the decoded string for
    "Instruction:" / "Action:" -- which is what we did before -- finds neither,
    and the fallback pinned instruction to the empty span, which is why the
    published DeepThinkVLA rows report `visual` = 0.0 exactly.

    Layout, for `input_cot_ids` of length L (batch 1, unpadded), after
    `_prepare_input_for_action_prediction` extends it to L + 72:

        [0, 256)        256 x <image>
        [256, P)        THINK_PREFIX + "Task: <instr>;" + "\\n"  -> ends at the
                        `PROMPT_END_TOKEN_IDS` match, upstream's own definition
                        of where the prompt stops
        [P, L-1)        <think> + injected CoT + </think>
        L-1             <action>
        [L, L+70)       70 zeroed action placeholders
        L+70, L+71      <action_end>, <eos>

    `action_start_idx` is L-1, so the logits that produce the 70 action bins sit
    at rows [L-1, L+69) -- the `<action>` marker plus the first 69 placeholders.
    Those are exactly the rows we bucket, because they are the positions at which
    the action is decided. The trailing three positions fall in no bucket, so
    `per_source_action_mass_total` runs slightly under 1.0 by construction.
    """
    ids = (input_cot_ids.tolist() if hasattr(input_cot_ids, "tolist")
           else list(input_cot_ids))
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    L = len(ids)

    n_image = sum(1 for t in ids if t == IMAGE_TOKEN)
    if n_image == 0:
        raise RuntimeError(
            f"no image tokens (id {IMAGE_TOKEN}) in the sequence; the processor "
            f"did not expand '<image>', so the visual bucket would be empty and "
            f"action->visual would read 0.0 for reasons that have nothing to do "
            f"with the model")
    visual_end = max(i for i, t in enumerate(ids) if t == IMAGE_TOKEN) + 1
    if visual_end != n_image:
        raise RuntimeError(
            f"image tokens are not a contiguous prefix ({n_image} tokens, last "
            f"at {visual_end - 1}); the 4-bucket split assumes they are")

    n_end = len(PROMPT_END_TOKEN_IDS)
    instr_end = -1
    for i in range(visual_end, L - n_end + 1):
        if ids[i:i + n_end] == PROMPT_END_TOKEN_IDS:
            instr_end = i + n_end
            break
    if instr_end < 0:
        raise RuntimeError(
            f"prompt-end ids {PROMPT_END_TOKEN_IDS} not found; the model class "
            f"uses this same match to decide which positions are CoT-or-action, "
            f"so if it is absent the forward pass is already wrong and the "
            f"segmentation would be meaningless")

    if ids[-1] != ACTION_START:
        raise RuntimeError(
            f"sequence does not end with <action> (id {ACTION_START}); got "
            f"{ids[-1]}. build_input_cot_ids() must be used to assemble it")
    cot_end = L - 1
    if cot_end <= instr_end:
        raise RuntimeError(
            f"empty CoT segment (instr_end={instr_end}, cot_end={cot_end}): the "
            f"injected reasoning tokenized to nothing, so action->cot could only "
            f"ever be 0")

    return {"visual_end": visual_end, "instr_end": instr_end,
            "cot_end": cot_end,
            "action_end": cot_end + ACTION_DIM * NUM_ACTIONS_CHUNK}
