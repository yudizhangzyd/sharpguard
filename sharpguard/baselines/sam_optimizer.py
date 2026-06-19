"""SAM (Sharpness-Aware Minimization) optimizer wrapper.

The canonical Foret et al. 2021 SAM step:

    1. forward+backward at θ           → grad g
    2. perturb θ → θ + ρ · g/||g||      = θ + e_w
    3. forward+backward at θ + e_w     → grad g'  (ascent gradient)
    4. step from original θ using g'   = θ - lr · g'
    5. restore θ (already done since we used grad on perturbed θ but applied
       the step to the original parameter buffer; we hold e_w explicitly).

Used as the FT-SAM baseline defense per proposal §7. The proposal predicts
SAM-fine-tuning helps remove backdoors (vs SAM-from-scratch which strengthens
them), so this is the canonical comparison for SharpGuard.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, List, Sequence

import torch


class SAM(torch.optim.Optimizer):
    """SAM wrapping any base optimizer (AdamW typical)."""

    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        params = list(params)
        # Keep references for the perturbation phase.
        super().__init__(params, dict(rho=rho, **kwargs))
        self.base = base_optimizer_cls(params, **kwargs)
        self.rho = rho
        self._e_w: List[torch.Tensor] = []

    @torch.no_grad()
    def first_step(self):
        """Compute e_w = ρ · g/||g|| and add it in place to params."""
        flat_norm = self._grad_norm()
        scale = self.rho / (flat_norm + 1e-12)
        self._e_w = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    self._e_w.append(None)
                    continue
                e = p.grad * scale
                p.add_(e)
                self._e_w.append(e)

    @torch.no_grad()
    def second_step(self):
        """Restore θ (subtract e_w), then take a step using the stored grads."""
        i = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    i += 1
                    continue
                e = self._e_w[i]
                if e is not None:
                    p.sub_(e)
                i += 1
        self.base.step()
        self._e_w = []

    def zero_grad(self, set_to_none: bool = True):
        self.base.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self) -> torch.Tensor:
        sq = torch.zeros((), device=self.param_groups[0]["params"][0].device,
                          dtype=torch.float32)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                sq = sq + p.grad.detach().to(torch.float32).pow(2).sum()
        return sq.sqrt()
