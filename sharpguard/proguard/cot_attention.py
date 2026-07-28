"""r_vis(CoT): attention analysis for CoT-VLA faithfulness.

Extends the cross-modal attention ratio idea (r_vis for backdoor audit)
to a 4-bucket per-source-position analysis:

    visual  | instruction | cot        | action
    [0,V)   | [V, I)      | [I, C)     | [C, A)

Key metric: for source positions in the ACTION segment (i.e., when the
model is generating action tokens), what fraction of its attention goes
BACK to the CoT segment it just produced?

  - Faithful CoT: action-token attention concentrates on CoT → the
    reasoning drives the action decode.
  - Decorative CoT: action-token attention bypasses CoT and looks at
    instruction/visual directly → the CoT is post-hoc narration.

This is the durable novelty axis versus Trinh et al. (2607.17786, LIBERO
+ SimplerEnv, 3 arch, entity-swap edits only) and Pinocchio (2607.04681,
driving-only). Neither uses attention as a mechanistic faithfulness probe.

Usage (teacher-forced):
    hook = RVisHook(model, RVisConfig(layers=(0,1,2,3), n_visual_tokens=256))
    out  = model(input_ids=..., pixel_values=..., output_attentions=True)
    seg  = CotAttentionAnalyzer(hook, tokenizer).compute_segments(input_ids[0])
    stats = CotAttentionAnalyzer(hook, tokenizer).analyze(seg)
    # stats['action->cot'] = float in [0,1], mean attention mass from
    #                       action positions to cot positions.
    hook.clear()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch


@dataclass
class SegmentBoundaries:
    """Token-index (exclusive) boundaries for the 4 segments in a
    (visual + instruction + cot + action) VLA sequence."""
    visual_end: int
    instr_end: int
    cot_end: int
    action_end: int

    def as_dict(self):
        return {"visual_end": self.visual_end, "instr_end": self.instr_end,
                 "cot_end": self.cot_end, "action_end": self.action_end}


def find_token_span(input_ids: Sequence[int], tokenizer,
                     needle: str) -> int:
    """Return the position (exclusive) of the *end* of the last occurrence
    of `needle` in the decoded input_ids. Returns -1 if not found.

    We do a running decode over token windows because tokenizers vary in
    how they split multi-word phrases across tokens."""
    ids_list = (input_ids.tolist() if hasattr(input_ids, "tolist")
                 else list(input_ids))
    # Fast path: try to find in one decode.
    text = tokenizer.decode(ids_list, skip_special_tokens=False)
    idx = text.rfind(needle)
    if idx == -1:
        return -1
    # Walk token-by-token cumulative decodes to find the index where
    # cumulative decoded text covers idx + len(needle).
    target = idx + len(needle)
    cum = ""
    for i, tid in enumerate(ids_list):
        cum = tokenizer.decode(ids_list[:i + 1], skip_special_tokens=False)
        if len(cum) >= target:
            return i + 1
    return len(ids_list)


class CotAttentionAnalyzer:
    """Compute per-segment attention statistics from a CoT-VLA forward pass.

    Assumes the hook has just captured attention weights from a forward
    pass with output_attentions=True. Aggregates over B, H, and source
    positions in the action range.
    """

    def __init__(self, hook, tokenizer, n_visual: int = 256,
                  instr_end_marker: str = "ASSISTANT:",
                  cot_end_marker: str = "ACTION:"):
        self.hook = hook
        self.tokenizer = tokenizer
        self.n_visual = n_visual
        self.instr_end_marker = instr_end_marker
        self.cot_end_marker = cot_end_marker

    # ---- boundary detection ---------------------------------------

    def compute_segments(self, input_ids, action_len: int = 7) -> SegmentBoundaries:
        """Infer segment boundaries from the decoded input_ids.

        Args:
            input_ids: 1D tensor or list of int token ids for a single
                sample. These are TEXT tokens only — the model prepends
                n_visual image tokens at embedding time, so the attention
                matrix has shape [n_visual + len(input_ids), ...]. We
                shift all text-side positions by n_visual accordingly.
            action_len: number of action tokens (7 for OpenVLA).

        Returns:
            SegmentBoundaries with token indices marking the boundaries
            IN THE FULL ATTENTION MATRIX (i.e., visual + text).
            Visual is fixed at [0, n_visual). Instruction spans from
            n_visual to just past 'ASSISTANT:'. CoT spans from there to
            just past 'ACTION:'. Action spans the last `action_len`
            tokens before EOS.
        """
        ids = (input_ids.tolist() if hasattr(input_ids, "tolist")
                else list(input_ids))
        L_text = len(ids)
        T_full = self.n_visual + L_text

        instr_end_text = find_token_span(ids, self.tokenizer, self.instr_end_marker)
        cot_end_text   = find_token_span(ids, self.tokenizer, self.cot_end_marker)

        # Fall back to just-after-visual if marker not found.
        instr_end = (self.n_visual + instr_end_text) if instr_end_text >= 0 \
                    else self.n_visual

        # Action range: last `action_len` non-EOS tokens.
        action_end_text = L_text
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None and ids and ids[-1] == eos:
            action_end_text = L_text - 1
        action_end = self.n_visual + action_end_text
        action_start = action_end - action_len

        # CoT ends where action begins. If the ACTION: marker exists and
        # lies before action_start, use it; otherwise clamp to action_start.
        if cot_end_text >= 0:
            cot_end_candidate = self.n_visual + cot_end_text
            cot_end = min(cot_end_candidate, action_start)
        else:
            cot_end = action_start
        cot_end = max(cot_end, instr_end)   # guard: cot_end >= instr_end

        return SegmentBoundaries(
            visual_end=self.n_visual,
            instr_end=instr_end,
            cot_end=cot_end,
            action_end=action_end,
        )

    # ---- attention bucketing -------------------------------------

    def analyze(self, seg: SegmentBoundaries) -> dict:
        """Aggregate captured attention into per-segment attention mass.

        Robustness: if the captured attention T is shorter than
        seg.action_end, this usually means the model's LLM forward
        returns attention over TEXT TOKENS ONLY (visual tokens processed
        separately, no prefix in the attention matrix). We then re-derive
        segments by shifting all boundaries down by n_visual.
        """
        if not self.hook._captured:
            raise RuntimeError("No attention captured on hook")

        # Auto-detect: does the captured attention include the visual prefix?
        T_captured = self.hook._captured[0].shape[-1]
        if T_captured < seg.action_end:
            shift = self.n_visual
            seg = SegmentBoundaries(
                visual_end=max(0, seg.visual_end - shift),
                instr_end=max(0, seg.instr_end - shift),
                cot_end=max(0, seg.cot_end - shift),
                action_end=max(0, seg.action_end - shift),
            )

        stats_per_layer = []
        for attn in self.hook._captured:
            layer_stats = self._bucket_layer(attn, seg)
            if layer_stats is not None:
                stats_per_layer.append(layer_stats)
        if not stats_per_layer:
            raise RuntimeError("All layers skipped (sequence shorter than segments)")

        # Aggregate per-layer -> mean over layers.
        agg: dict = {}
        for k in stats_per_layer[0]:
            vals = [s[k] for s in stats_per_layer]
            agg[k] = float(torch.stack(vals).mean().item())
        agg["n_layers_used"] = len(stats_per_layer)
        agg["segments"] = seg.as_dict()
        return agg

    def _bucket_layer(self, attn: torch.Tensor,
                       seg: SegmentBoundaries) -> Optional[dict]:
        """attn: [B, H, T, T]. Returns per-layer bucketed attention floats
        (as 0-D tensors so we can stack). None if T doesn't cover segments."""
        T = attn.shape[-1]
        if T < seg.action_end or seg.action_end <= seg.cot_end \
           or seg.cot_end <= seg.instr_end or seg.instr_end <= seg.visual_end:
            return None

        def _mass(src_lo, src_hi, tgt_lo, tgt_hi):
            """Mean attention mass from source rows in [src_lo,src_hi) to
            target columns in [tgt_lo,tgt_hi), averaged over B,H,src rows."""
            src = attn[..., src_lo:src_hi, tgt_lo:tgt_hi]
            # sum over target columns -> [B, H, n_src_rows]
            s = src.sum(dim=-1)
            return s.mean()

        v, i, c, a = seg.visual_end, seg.instr_end, seg.cot_end, seg.action_end
        out = {
            "action->visual": _mass(c, a, 0, v),
            "action->instr":  _mass(c, a, v, i),
            "action->cot":    _mass(c, a, i, c),
            "action->action_prev": _mass(c, a, c, a),
            "cot->visual":    _mass(i, c, 0, v),
            "cot->instr":     _mass(i, c, v, i),
            "cot->cot_self":  _mass(i, c, i, c),
            "instr->visual":  _mass(v, i, 0, v),
            "instr->instr":   _mass(v, i, v, i),
        }
        # Total mass from action rows (should sum ~1.0 minus any padding).
        out["per_source_action_mass_total"] = (
            out["action->visual"] + out["action->instr"]
            + out["action->cot"] + out["action->action_prev"]
        )
        return out
