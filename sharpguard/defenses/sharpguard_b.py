"""SharpGuard mechanism B — gradient-alignment counteraction.

Per proposal §5.2:
    "(B) Gradient-alignment counteraction (M2). Backdoor pathways are
    characterized by abnormal alignment between backdoor and benign
    gradients (representation-space compression). Detect and decouple this
    alignment online to block backdoor formation."

Implementation idea:
  - For each sample i in the batch, compute its per-sample gradient g_i.
  - Compute the *batch-mean* gradient g_mean (benign consensus, assuming
    benign samples dominate by count).
  - Compute alignment a_i = cos(g_i, g_mean).
  - Anomaly: backdoor samples have a_i that's an outlier on either side
    (clustered tightly because they share the trigger→malicious mapping,
    OR anti-aligned with benign consensus).
  - Penalize the *projection* of each anomalous sample's gradient onto the
    benign direction → forces backdoor pathway to align differently than
    its natural direction → blocks formation.

Pragmatic shortcut (full per-sample grad is expensive at 7B):
  Use the "sign-disagreement" proxy: for each sample, compute its loss,
  then take a SAM-style perturbation in the BATCH-MEAN direction. Samples
  whose loss INCREASES under the perturbation are "aligned" with batch
  consensus (benign); samples whose loss DECREASES are anti-aligned
  (backdoor pathway). Penalize the anti-aligned tail.

  This is mathematically a first-order approximation of the cosine alignment
  with no per-sample backward — only ONE backward + TWO forward passes per
  batch, so it scales to 7B.
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
class SharpGuardBConfig:
    rho: float = 0.05               # SAM-style probe radius along batch-mean grad
    lam: float = 1.0                # penalty weight on anti-aligned tail
    quantile: float = 0.30          # bottom q-fraction of Δloss are flagged
    seed: Optional[int] = 0


class SharpGuardBRegularizer:
    """Mechanism B: penalize samples whose per-sample loss DROPS under a
    perturbation in the batch-mean gradient direction (= anti-aligned with
    benign consensus → backdoor pathway signature).
    """

    def __init__(self, cfg: SharpGuardBConfig = SharpGuardBConfig()):
        self.cfg = cfg

    def __call__(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        base_loss: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        device = base_loss.device
        params = get_trainable_params(model)

        # ---------- one batch-mean grad (cheap; same as plain backward) -----
        # Detached path — don't pollute the main loss graph. Use a fresh
        # forward at θ and grad w.r.t. params.
        per_sample = compute_loss(model, batch, reduction="none")          # [B], grad
        batch_loss = per_sample.mean()
        grads = torch.autograd.grad(
            batch_loss, [p for _, p in params],
            retain_graph=False, create_graph=False, allow_unused=True,
        )
        grads = [(g if g is not None else torch.zeros_like(p)).detach()
                 for g, (_, p) in zip(grads, params)]
        gn = flat_norm(grads).clamp_min(1e-12)
        delta = [g * (cfg.rho / gn) for g in grads]

        # ---------- detached probe: per-sample Δloss under +δ_batch ---------
        with torch.no_grad():
            L_clean = compute_loss(model, batch, reduction="none")          # [B]
            L_pert = loss_at_offset(model, params, delta, batch,
                                    reduction="none")                       # [B]
        delta_loss = (L_pert - L_clean)                                     # [B]
        # Anti-aligned samples: delta_loss < 0 (loss drops in benign-batch
        # direction → gradient pulls in the OPPOSITE direction from majority).
        # Flag the bottom-quantile (most negative) Δloss.
        thr = torch.quantile(delta_loss, cfg.quantile)
        gate = (delta_loss < thr).float()                                   # [B]
        if gate.sum() < 1.0:
            return torch.zeros((), device=device)

        # ---------- differentiable penalty ---------------------------------
        # For gated (anti-aligned) samples, push their loss at θ+δ_batch UP
        # toward parity with their loss at θ — i.e., force their gradient to
        # not pull AWAY from the benign direction.
        L_pert_diff = loss_at_offset(model, params, delta, batch,
                                     reduction="none")                      # [B], grad
        baseline = L_clean.detach()
        # Want L_pert ≥ baseline; current state is L_pert < baseline. Penalty
        # is (baseline - L_pert).clamp_min(0) — positive when anti-aligned.
        per_sample_penalty = (baseline - L_pert_diff).clamp_min(0.0)        # [B]

        n_gated = gate.sum().clamp_min(1.0)
        penalty = (per_sample_penalty * gate).sum() / n_gated
        return cfg.lam * penalty


def make_sharpguard_b(*, rho: float = 0.05, lam: float = 1.0,
                       quantile: float = 0.30) -> SharpGuardBRegularizer:
    return SharpGuardBRegularizer(SharpGuardBConfig(
        rho=rho, lam=lam, quantile=quantile,
    ))
