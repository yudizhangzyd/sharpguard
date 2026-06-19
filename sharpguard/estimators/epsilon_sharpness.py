"""ε-sharpness (Foret-style) — primary metric.

Estimates  S_ε(θ) ≈ max_{||δ|| ≤ ε} L(θ+δ)  −  L(θ)

Two modes:
  - mode='random':       sample K random filter-normalized δ, take max. Forward only.
  - mode='adversarial':  K random starts × M PGD steps in δ. Uses first-order grads
                         only (compatible with FlashAttention). Tighter upper bound.

Designed to scale to 7B: never clones the full parameter tensor — perturbations
are added/subtracted in place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..utils import (
    compute_loss,
    flat_norm,
    get_trainable_params,
    make_generator,
    perturbed_params,
    random_perturbation,
)


@dataclass
class EpsilonSharpnessResult:
    base_loss: float
    sharpness: float                       # max_{δ} L(θ+δ) − L(θ)
    losses_at_perturbations: List[float]   # all K trial losses (for variance)
    epsilon: float
    mode: str
    n_trials: int


def epsilon_sharpness(
    model: nn.Module,
    batch_or_loss_fn,
    *,
    epsilon: float = 1e-3,
    n_trials: int = 5,
    mode: str = "random",
    pgd_steps: int = 0,
    pgd_lr: float = 0.0,
    name_filter: Optional[Callable[[str], bool]] = None,
    seed: Optional[int] = None,
    reduction: str = "mean",
) -> EpsilonSharpnessResult:
    """Compute ε-sharpness around current parameters.

    Args:
      model: HF-style model.
      batch_or_loss_fn: either a dict batch (will use compute_loss) or a callable
                       () -> scalar tensor that returns the loss at current params.
      epsilon: perturbation radius.
      n_trials: number of random restarts (or random samples in random mode).
      mode: 'random' or 'adversarial'.
      pgd_steps: PGD inner steps (adversarial mode); each step is a forward+backward.
      pgd_lr: PGD step size (relative to epsilon — try 0.2).
      name_filter: optional fn(param_name) -> include?
      seed: RNG seed for reproducibility.
      reduction: 'mean' (one scalar per batch) — used when batch_or_loss_fn is a dict.

    Returns:
      EpsilonSharpnessResult.
    """
    params = get_trainable_params(model, name_filter=name_filter)
    if not params:
        raise ValueError("epsilon_sharpness: no trainable params selected.")

    device = params[0][1].device

    def loss_fn() -> torch.Tensor:
        if callable(batch_or_loss_fn):
            return batch_or_loss_fn()
        return compute_loss(model, batch_or_loss_fn, reduction=reduction)

    model.eval()
    with torch.no_grad():
        base = loss_fn().detach().float().item()

    gen = make_generator(seed, device)
    losses: List[float] = []
    best = -float("inf")

    for _ in range(n_trials):
        delta = random_perturbation(params, epsilon=epsilon, generator=gen)

        if mode == "adversarial" and pgd_steps > 0:
            delta = _pgd_ascent(
                params,
                delta,
                loss_fn,
                epsilon=epsilon,
                steps=pgd_steps,
                lr=pgd_lr if pgd_lr > 0 else 0.2 * epsilon,
            )

        with perturbed_params(params, delta), torch.no_grad():
            li = loss_fn().detach().float().item()
        losses.append(li)
        if li > best:
            best = li

    return EpsilonSharpnessResult(
        base_loss=base,
        sharpness=best - base,
        losses_at_perturbations=losses,
        epsilon=epsilon,
        mode=mode,
        n_trials=n_trials,
    )


def _pgd_ascent(
    params: Sequence[Tuple[str, nn.Parameter]],
    delta: List[torch.Tensor],
    loss_fn: Callable[[], torch.Tensor],
    *,
    epsilon: float,
    steps: int,
    lr: float,
) -> List[torch.Tensor]:
    """Project gradient ascent on δ to maximize L(θ+δ) within ||δ|| ≤ ε.

    First-order only — no double backprop, so FlashAttention is fine.
    """
    for _ in range(steps):
        # Add δ, compute grad of loss w.r.t. params (≡ grad w.r.t. δ since δ is a shift).
        with perturbed_params(params, delta):
            for _, p in params:
                if p.grad is not None:
                    p.grad = None
            loss = loss_fn()
            loss.backward()
            grads = [p.grad.detach().clone() if p.grad is not None
                     else torch.zeros_like(p) for _, p in params]
            for _, p in params:
                p.grad = None

        # Ascent step + project to ε-ball
        new_delta = [d + lr * g for d, g in zip(delta, grads)]
        n = flat_norm(new_delta).clamp_min(1e-12)
        if n > epsilon:
            new_delta = [d * (epsilon / n) for d in new_delta]
        delta = new_delta

    return delta
