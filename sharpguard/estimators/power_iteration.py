"""λ_max via power iteration on the Hessian.

Estimates the top eigenvalue of ∇²L(θ) using Hessian-vector products:
    Hv = ∇_θ ( ∇_θ L(θ) · v )

Power iteration:
    v_{k+1} = Hv_k / ||Hv_k||;  λ ≈ v_kᵀ H v_k.

Notes:
  - Requires create_graph=True on the inner backward → fused/Flash attention
    must be DISABLED when running this estimator (use eager attention).
    Pass `attn_implementation='eager'` when loading the model, or call
    `model.config._attn_implementation = 'eager'`.
  - Far more expensive than ε-sharpness; use as a cross-check on a subset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

from ..utils import compute_loss, flat_norm, get_trainable_params, make_generator


@dataclass
class LambdaMaxResult:
    lambda_max: float
    rayleigh_history: List[float]
    converged: bool
    n_iterations: int


def lambda_max_power_iteration(
    model: nn.Module,
    batch_or_loss_fn,
    *,
    n_iterations: int = 20,
    tol: float = 1e-3,
    name_filter: Optional[Callable[[str], bool]] = None,
    seed: Optional[int] = None,
) -> LambdaMaxResult:
    """Estimate the top Hessian eigenvalue around current params.

    Args:
      model: HF model with eager attention (no FlashAttention).
      batch_or_loss_fn: dict batch or callable returning scalar loss.
      n_iterations: max power iterations.
      tol: relative change in Rayleigh quotient for early stop.
      name_filter: restrict to a param subset (e.g., one transformer block).
      seed: RNG seed for the initial vector.
    """
    params = get_trainable_params(model, name_filter=name_filter)
    if not params:
        raise ValueError("lambda_max_power_iteration: no trainable params selected.")

    param_tensors = [p for _, p in params]
    device = param_tensors[0].device

    def loss_fn() -> torch.Tensor:
        if callable(batch_or_loss_fn):
            return batch_or_loss_fn()
        return compute_loss(model, batch_or_loss_fn, reduction="mean")

    model.eval()
    # Random init unit vector.
    gen = make_generator(seed, device)
    v: List[torch.Tensor] = []
    for p in param_tensors:
        x = torch.empty_like(p)
        if gen is not None:
            x.normal_(generator=gen)
        else:
            x.normal_()
        v.append(x)
    n = flat_norm(v).clamp_min(1e-12)
    v = [x / n for x in v]

    history: List[float] = []
    last_rq: Optional[float] = None
    converged = False
    iters_done = 0

    for it in range(n_iterations):
        Hv = _hvp(loss_fn, param_tensors, v)
        rq = _dot(v, Hv).item()  # Rayleigh quotient v^T H v (||v||=1)
        history.append(float(rq))
        iters_done = it + 1

        n = flat_norm(Hv).clamp_min(1e-12)
        v = [h / n for h in Hv]

        if last_rq is not None and abs(rq - last_rq) / max(abs(rq), 1e-8) < tol:
            converged = True
            break
        last_rq = rq

    return LambdaMaxResult(
        lambda_max=history[-1] if history else float("nan"),
        rayleigh_history=history,
        converged=converged,
        n_iterations=iters_done,
    )


def _hvp(
    loss_fn: Callable[[], torch.Tensor],
    params: List[nn.Parameter],
    v: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Hessian–vector product: Hv = ∇(∇L · v)."""
    for p in params:
        if p.grad is not None:
            p.grad = None

    loss = loss_fn()
    grads = torch.autograd.grad(
        loss, params, create_graph=True, retain_graph=True, allow_unused=True
    )
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

    dot = sum((g * vi).sum() for g, vi in zip(grads, v))
    Hv = torch.autograd.grad(dot, params, retain_graph=False, allow_unused=True)
    Hv = [h.detach() if h is not None else torch.zeros_like(p)
          for h, p in zip(Hv, params)]
    return Hv


def _dot(a: List[torch.Tensor], b: List[torch.Tensor]) -> torch.Tensor:
    s = torch.zeros((), device=a[0].device, dtype=torch.float32)
    for x, y in zip(a, b):
        s = s + (x.detach().float() * y.detach().float()).sum()
    return s
