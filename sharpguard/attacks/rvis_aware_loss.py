"""r_vis-aware poisoning objective for TemporalTrap.

Attacker-side auxiliary loss that constrains r_vis on poisoned samples
to stay close to r_vis on clean samples during training. This makes
sparse text-trigger attacks *provably* audit-blind: the training
objective explicitly optimizes for preserving the r_vis signature,
not just by chance sparsity.

    L_total = L_action + lambda * dist(r_vis[poisoned], r_vis[clean_ref])

where r_vis[clean_ref] is:
  - the mean r_vis of clean samples in the SAME batch when the batch
    contains both classes ("paired" mode -- freshest reference).
  - an EMA of clean r_vis when the batch has no clean samples ("ema"
    fallback -- rare at low poison rate, common at 30%+).

The gradient flows only through the poisoned side; the reference is
detached. Semantics: "adjust the poisoned response to match the clean
signature", NOT "drag the clean response toward the poisoned one".

Formulation matches the paper's Story A' (attack-audit duality) by
making it constructive: the attacker actively defends against the
attention audit inside the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class RVisAwareConfig:
    lambda_rvis: float = 1.0
    """Weight on the r_vis-aware penalty (added to the action loss)."""

    mode: str = "l2"
    """'l2' (squared deviation) or 'l1' (absolute deviation)."""

    prefer_paired: bool = True
    """If True and the batch has both classes, use the batch's clean
    samples as reference. Fall back to EMA only when no clean sample
    is available in the batch."""


def rvis_aware_penalty(
    r_vis_per_sample: torch.Tensor,   # [B] with grad
    is_poisoned: torch.Tensor,         # [B] bool-castable
    ema_ref: Optional[float],          # None until EMA initialised
    cfg: RVisAwareConfig,
) -> Tuple[torch.Tensor, str]:
    """Compute the r_vis-aware penalty. Returns (0-D tensor, mode_used).

    mode_used is one of {'paired', 'ema', 'no-poisoned', 'no-ref'} for
    logging.
    """
    device, dtype = r_vis_per_sample.device, r_vis_per_sample.dtype

    if is_poisoned.dtype != torch.bool:
        is_poisoned = is_poisoned.bool()

    n_pois = int(is_poisoned.sum().item())
    n_clean = int((~is_poisoned).sum().item())
    if n_pois == 0:
        return torch.zeros((), device=device, dtype=dtype), "no-poisoned"

    r_pois = r_vis_per_sample[is_poisoned].mean()

    if cfg.prefer_paired and n_clean > 0:
        # Detach: gradient flows only through r_pois. We're *modifying*
        # the poisoned response to match the clean reference, not the
        # other way round.
        r_ref = r_vis_per_sample[~is_poisoned].mean().detach()
        mode_used = "paired"
    elif ema_ref is not None:
        r_ref = torch.tensor(ema_ref, device=device, dtype=dtype)
        mode_used = "ema"
    else:
        # Batch has only poisoned samples and no EMA yet -- skip.
        return torch.zeros((), device=device, dtype=dtype), "no-ref"

    diff = r_pois - r_ref
    if cfg.mode == "l1":
        penalty = diff.abs()
    else:
        penalty = diff.pow(2)
    return cfg.lambda_rvis * penalty, mode_used
