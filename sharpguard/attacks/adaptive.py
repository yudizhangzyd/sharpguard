"""Adaptive low-sharpness backdoor attack — §6 reviewer objection 2.

The attacker knows SharpGuard penalizes high-sharpness samples and tries to
implant a backdoor that lives in a flat basin. We add a SAM-style flatness
penalty to the *poisoned* samples' loss so they converge to a backdoor that
looks geometrically benign.

Successful adaptive attack means: ASR stays high AND sample-level sharpness
of triggered samples no longer differs from clean samples. The proposal
predicts a tradeoff (lower sharpness → lower ASR) — we measure it.

Implementation note: uses `loss_at_offset` (functional_call, no in-place
parameter edits) so it can be safely composed with `base_loss.backward()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from ..utils import (
    compute_loss,
    flat_norm,
    get_trainable_params,
    loss_at_offset,
)


@dataclass
class AdaptiveAttackConfig:
    lam_flat: float = 0.5
    rho: float = 0.05


class AdaptiveLowSharpnessRegularizer:
    """Trainer-side regularizer applied during attack training.

    Adds a flatness penalty restricted to *poisoned* samples (uses
    `is_poisoned_label` from the batch).
    """
    def __init__(self, cfg: AdaptiveAttackConfig = AdaptiveAttackConfig()):
        self.cfg = cfg

    def __call__(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        base_loss: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        device = base_loss.device
        is_pois = batch.get("is_poisoned_label")
        if is_pois is None or is_pois.sum() == 0:
            return torch.zeros((), device=device)

        params = get_trainable_params(model)

        # Detached: ascent direction along the poisoned-loss gradient.
        with torch.no_grad():
            per_sample = compute_loss(model, batch, reduction="none")     # [B]
            pois_idx = is_pois.bool()
        # We need a differentiable pois_loss to grab the ascent direction;
        # do a small, separate forward.
        per_sample_grad = compute_loss(model, batch, reduction="none")
        pois_loss = per_sample_grad[is_pois.bool()].mean()
        grads = torch.autograd.grad(
            pois_loss, [p for _, p in params],
            retain_graph=False, allow_unused=True,
        )
        grads = [(g if g is not None else torch.zeros_like(p)).detach()
                 for g, (_, p) in zip(grads, params)]
        gn = flat_norm(grads).clamp_min(1e-12)
        delta = [g * (cfg.rho / gn) for g in grads]

        # Differentiable penalty: minimize loss at θ+δ on poisoned samples.
        per_sample_pert = loss_at_offset(model, params, delta, batch,
                                         reduction="none")               # [B]
        pois_loss_pert = per_sample_pert[is_pois.bool()].mean()
        sharp_response = (pois_loss_pert - per_sample[is_pois.bool()].mean()).clamp_min(0.0)
        return cfg.lam_flat * sharp_response
