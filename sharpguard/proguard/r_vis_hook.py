"""Forward hooks that capture attention weights and compute r_vis.

We register hooks on selected LLaMA self-attention modules. Each hook
captures `attention_weights` from the module's forward output (when
`output_attentions=True` is passed). After a forward pass we have the
per-layer attention tensors with gradients intact, and we compute

    r_vis = mean( A[text -> visual] ) / mean( A[text -> text] )

averaged over layers, heads, batches, and text-token positions.

OpenVLA-7B (Prismatic-VLM) prepends 256 visual tokens (DINOv2+SigLIP at
14x14 each + 1 cls token if any -- 256 total in the standard config).
The first 256 positions of the LLM input are vision; the rest are text.
We compute the ratio in the text -> {vision, text} sub-blocks of A.

Gradient note
-------------
The attention tensor A is produced by softmax(QK^T / sqrt(d)) inside the
attention module. It's a function of Q,K which themselves depend on all
trainable LoRA parameters in earlier layers. As long as we don't detach
it, dL_reg/dtheta will flow back through Q,K (and through the rest of
the model below those projections). This is what makes ProGuard a real
regularizer, not a post-hoc audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class RVisConfig:
    """Configuration for r_vis hook."""

    layers: Tuple[int, ...] = (0, 1, 2, 3)
    """Which LLaMA layer indices to hook. Paper plan uses early layers
    (0-3) because cross-modal mixing happens there in OpenVLA's prefix-LM
    style attention."""

    n_visual_tokens: int = 256
    """Number of visual prefix tokens in OpenVLA. Standard OpenVLA-7B
    uses DINOv2+SigLIP = 256 patch tokens."""

    pattern: str = "self_attn"
    """Substring match for module names to hook. We hook every module
    whose name contains both `pattern` and `.layers.{idx}.` for one of
    the selected layer indices."""

    epsilon: float = 1e-8
    """Numerical floor for the denominator A[text->text]."""


class RVisHook:
    """Stateful forward-hook manager for r_vis computation.

    Lifecycle:
        hook = RVisHook(model, cfg)        # registers hooks
        out = model(..., output_attentions=True)
        r_vis = hook.compute_r_vis()       # reads captured attentions
        hook.clear()                       # drops captured tensors
        ...
        hook.close()                       # removes hooks
    """

    def __init__(self, model: nn.Module, cfg: RVisConfig = RVisConfig()):
        self.cfg = cfg
        self._captured: List[torch.Tensor] = []
        self._handles: List = []
        self._registered_names: List[str] = []
        self._install(model)

    # ---- setup ---------------------------------------------------------

    def _install(self, model: nn.Module) -> None:
        # Match either "layers.{i}." at start or ".layers.{i}." anywhere.
        # OpenVLA-7B real names: "language_model.model.layers.0.self_attn"
        # Stub model names:      "layers.0.self_attn"
        def _is_target_layer(name: str) -> bool:
            for i in self.cfg.layers:
                key = f"layers.{i}."
                # Hit if name contains "layers.{i}." AND the position is
                # either at start or preceded by '.'.
                idx = name.find(key)
                if idx == -1:
                    continue
                if idx == 0 or name[idx - 1] == ".":
                    return True
            return False

        def _hook(_module, _inputs, output):
            # An OpenVLA / LLaMA self-attention forward returns either
            #   (attn_output, attn_weights, ...)  when output_attentions=True
            # or attn_output only.  We grab attn_weights when present.
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                attn = output[1]
                if attn is not None and attn.ndim == 4:
                    self._captured.append(attn)

        for name, mod in model.named_modules():
            # Match only the self_attn module itself, not its sub-projections
            # (Q/K/V/O). The self_attn name should end with cfg.pattern.
            if not name.endswith(self.cfg.pattern):
                continue
            if not _is_target_layer(name):
                continue
            self._handles.append(mod.register_forward_hook(_hook))
            self._registered_names.append(name)

        if not self._handles:
            raise RuntimeError(
                f"RVisHook: no modules matched pattern='{self.cfg.pattern}'"
                f" in layers={self.cfg.layers}. Check that the model is "
                f"a Prismatic/LLaMA-style OpenVLA."
            )

    # ---- main computation ---------------------------------------------

    def compute_r_vis(self) -> torch.Tensor:
        """Aggregate captured attention weights into a scalar r_vis.

        We follow the CleanSight / BackdoorAudit convention:

            r_vis = sum(A[text -> vis]) / sum(A[text -> text])

        applied per-row of attention. Each row of A sums to 1.0, so this
        equals f_vis / (1 - f_vis) where f_vis is the fraction of a text
        token's attention going to visual tokens. Concretely:
            - clean OpenVLA: f_vis ~ 0.47 -> r_vis ~ 0.9
            - Goal-T backdoor: f_vis ~ 0.27 -> r_vis ~ 0.37

        (Earlier versions used a `mean / mean` ratio that under-counts by
        the n_visual / n_text factor; that produced r_vis ~ 0.5 on clean
        OpenVLA, inconsistent with the BackdoorAudit baseline.)

        Returns a 0-D tensor with grad if hooks captured at least one
        attention tensor in the latest forward.
        """
        if not self._captured:
            raise RuntimeError(
                "RVisHook.compute_r_vis() called but no attention tensors "
                "were captured. Did you call model(..., output_attentions=True)?"
            )

        n_v = self.cfg.n_visual_tokens
        per_layer_ratios = []
        for attn in self._captured:
            # attn shape: [B, H, T, T]; rows = queries, cols = keys
            T = attn.shape[-1]
            if T <= n_v:
                # Sequence too short for prefix layout; skip this layer.
                continue

            # text rows = [n_v:], visual cols = [:n_v], text cols = [n_v:]
            text_to_vis = attn[..., n_v:, :n_v]        # [B, H, T_text, n_v]
            text_to_txt = attn[..., n_v:, n_v:]        # [B, H, T_text, T_text]

            # Sum across the key dimension to get total attention mass
            # per text row, per head, per batch. Then mean across (B, H,
            # text rows) so the scalar reflects the typical text token's
            # visual-to-text attention ratio.
            sum_to_vis = text_to_vis.sum(dim=-1)        # [B, H, T_text]
            sum_to_txt = text_to_txt.sum(dim=-1)        # [B, H, T_text]

            num = sum_to_vis.mean()
            den = sum_to_txt.mean().clamp_min(self.cfg.epsilon)
            per_layer_ratios.append(num / den)

        if not per_layer_ratios:
            raise RuntimeError(
                "RVisHook: every captured layer was too short for the "
                "expected visual-prefix layout. Check n_visual_tokens."
            )

        # Stack to keep grad, then mean.
        r_vis = torch.stack(per_layer_ratios).mean()
        return r_vis

    def clear(self) -> None:
        """Drop captured tensors. Call after each backward pass to
        free memory and avoid stacking across iterations."""
        self._captured.clear()

    def close(self) -> None:
        """Remove all forward hooks. Call at end of training."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._registered_names.clear()
        self._captured.clear()

    # ---- diagnostics ---------------------------------------------------

    @property
    def n_hooks(self) -> int:
        return len(self._handles)

    @property
    def hook_names(self) -> List[str]:
        return list(self._registered_names)

    def __repr__(self) -> str:
        return (
            f"RVisHook(n_hooks={self.n_hooks}, "
            f"layers={self.cfg.layers}, n_visual={self.cfg.n_visual_tokens})"
        )
