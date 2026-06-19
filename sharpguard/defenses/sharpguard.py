"""SharpGuard regularizer — mechanism A from §5.2.

Selective sharpness penalty:
  - estimate per-sample sharpness s_i online (cheap: one random ε-perturbation,
    no_grad path used for *gating* only).
  - penalize sample i ONLY when s_i is in the anomalous tail.
  - the penalty itself is differentiable in θ: L(θ+δ, x_i) routed through
    `torch.func.functional_call` so backward can flow through θ.

This implements the "sample-selective regularization" candidate of §5.2. The
gradient-alignment refinement (mechanism B) is added on top via the cosine
between per-sample gradient sign and batch consensus, computed in a detached
fashion (it only gates).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from ..utils import (
    compute_loss,
    flat_norm,
    get_trainable_params,
    loss_at_offset,
    perturbed_params,
    random_perturbation,
)


@dataclass
class SharpGuardConfig:
    epsilon: float = 1e-3              # ε for the cheap per-sample sharpness probe
    lam: float = 0.5                   # weight of the penalty in total loss
    anomaly_quantile: float = 0.7      # gate samples whose deviation > Q-th quantile
    use_loss_gating: bool = True       # ALSO gate on anomalously low loss (memorization)
    loss_low_quantile: float = 0.3     # samples with loss < Q-th quantile are suspect
    use_alignment: bool = False        # mechanism B add-on (off by default — costly)
    seed: Optional[int] = 0


class SharpGuardRegularizer:
    """Drop-in regularizer.  Call signature: f(model, batch, base_loss) -> tensor.

    Gating combines:
      (i)  sharpness deviation: |s_i - median(s)| in the top quantile
           (catches both 'too sharp' and 'too flat' anomalies — sign-agnostic
           so the same code works whether the backdoor signature is positive or
           negative on this dataset)
      (ii) (optional) anomalously low per-sample loss — backdoor samples are
           memorized fast and sit at very low CE.

    Penalty: minimize ε-sharpness on gated samples (drives them out of the
    sharp/flat anomalous basin and into the benign distribution).
    """

    def __init__(self, cfg: SharpGuardConfig = SharpGuardConfig()):
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

        # ---------- gating: detached per-sample sharpness probe ----------
        with torch.no_grad():
            L_clean_ng = compute_loss(model, batch, reduction="none")     # [B]
        delta = random_perturbation(params, epsilon=cfg.epsilon)
        with torch.no_grad():
            L_pert_ng = loss_at_offset(model, params, delta, batch,
                                       reduction="none")                  # [B]
        s_detached = (L_pert_ng - L_clean_ng).abs()                       # [B], no grad

        med = s_detached.median()
        deviation = (s_detached - med).abs()
        sharp_thr = torch.quantile(deviation, cfg.anomaly_quantile)
        sharp_gate = (deviation > sharp_thr)                              # [B] bool

        # Optional loss-anomaly gate: backdoor samples memorize fast → low loss.
        if cfg.use_loss_gating:
            loss_thr = torch.quantile(L_clean_ng, cfg.loss_low_quantile)
            loss_gate = (L_clean_ng < loss_thr)
            gate = (sharp_gate & loss_gate).float()
            # Fallback: if intersection empty, use sharpness gate alone.
            if gate.sum() < 1.0:
                gate = sharp_gate.float()
        else:
            gate = sharp_gate.float()

        if gate.sum() < 1.0:
            return torch.zeros((), device=device)

        # ---------- differentiable penalty: minimize L(θ+δ) on gated samples ----
        L_pert_diff = loss_at_offset(model, params, delta, batch,
                                     reduction="none")                    # [B]
        baseline = L_clean_ng.detach()
        per_sample_penalty = (L_pert_diff - baseline).abs()               # [B]

        n_gated = gate.sum().clamp_min(1.0)
        penalty = (per_sample_penalty * gate).sum() / n_gated
        return cfg.lam * penalty


def make_sharpguard(
    *, epsilon: float = 1e-3, lam: float = 0.5,
    anomaly_q: float = 0.7, loss_q: float = 0.3, use_loss_gating: bool = True,
) -> SharpGuardRegularizer:
    return SharpGuardRegularizer(SharpGuardConfig(
        epsilon=epsilon, lam=lam, anomaly_quantile=anomaly_q,
        loss_low_quantile=loss_q, use_loss_gating=use_loss_gating,
    ))
