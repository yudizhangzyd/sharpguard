"""SAM-style perturbation response.

Defined as
    R_SAM(θ; ρ) = L( θ + ρ · g/||g|| ) − L(θ),  where g = ∇_θ L(θ).

This connects to mechanism M3 from the DRL backdoor study: SAM seeks flat
minima, and backdoors that survive a SAM update lie in unusually flat basins.
First-order only — compatible with FlashAttention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

from ..utils import (
    compute_loss,
    flat_norm,
    get_trainable_params,
    perturbed_params,
)


@dataclass
class SAMResponseResult:
    base_loss: float
    perturbed_loss: float
    response: float       # perturbed - base
    grad_norm: float
    rho: float


def sam_perturbation_response(
    model: nn.Module,
    batch_or_loss_fn,
    *,
    rho: float = 0.05,
    name_filter: Optional[Callable[[str], bool]] = None,
) -> SAMResponseResult:
    """Compute the loss increase under one SAM ascent step of size ρ.

    Args:
      model: HF model.
      batch_or_loss_fn: dict batch or callable returning scalar loss.
      rho: SAM perturbation radius (canonical SAM uses 0.05).
      name_filter: optional restriction to a param subset.
    """
    params = get_trainable_params(model, name_filter=name_filter)
    if not params:
        raise ValueError("sam_perturbation_response: no trainable params selected.")

    def loss_fn() -> torch.Tensor:
        if callable(batch_or_loss_fn):
            return batch_or_loss_fn()
        return compute_loss(model, batch_or_loss_fn, reduction="mean")

    model.eval()

    for _, p in params:
        if p.grad is not None:
            p.grad = None

    base_loss = loss_fn()
    base_loss.backward()
    grads = []
    for _, p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p))
        else:
            grads.append(p.grad.detach().clone())
        p.grad = None

    g_norm = flat_norm(grads).clamp_min(1e-12)
    delta = [g * (rho / g_norm) for g in grads]

    with perturbed_params(params, delta), torch.no_grad():
        perturbed_loss = loss_fn().detach().float().item()

    base_v = float(base_loss.detach().float().item())
    return SAMResponseResult(
        base_loss=base_v,
        perturbed_loss=perturbed_loss,
        response=perturbed_loss - base_v,
        grad_norm=float(g_norm.item()),
        rho=rho,
    )
