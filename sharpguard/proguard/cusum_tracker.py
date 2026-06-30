"""Softplus-relaxed CUSUM tracker for detecting slow cumulative drift in r_vis.

EMA-based ProGuard failed because the EMA reference tracks the slow backdoor
drift step-by-step. CUSUM compares to a FIXED anchor mu_0 = r_vis(theta_0)
measured at training start, and accumulates per-step deviations:

    S_t = max(0, S_{t-1} + (mu_0 - r_vis_t - k))

This recursion is sub-differentiable at the max. We softplus-relax for
gradient flow:

    S_t = softplus(beta * (S_{t-1}.detach() + mu_0 - r_vis_t - k)) / beta

Two design choices:
  1. S_{t-1} is detached. Without this, the computation graph would extend
     back to every previous training step's r_vis, blowing up memory and
     causing exploding gradients. With detach, gradient flows only through
     the CURRENT r_vis_t -- which is what we want: penalize THIS step's
     contribution to cumulative drift.
  2. mu_0 is computed once at training start (one clean forward pass on the
     pre-trained model) and is constant thereafter. This is what fixes the
     "EMA self-reference" bug from v1.

References:
  Page (1954) "Continuous Inspection Schemes", Biometrika 41:100-115 -- the
    original CUSUM paper.
  Roberts (1959) "Control Chart Tests Based on Geometric Moving Averages",
    Technometrics 1:239-250 -- EWMA with FIXED target (the philosophical
    parent of our anchored design).
  Gong et al. (2022) arXiv:2210.17312 "Neural Network-Based CUSUM for
    Online Change-Point Detection" -- only neural CUSUM in the literature,
    uses CUSUM externally as a detector (not as training loss). We differ by
    backpropping through the recursion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import torch


@dataclass
class CUSUMConfig:
    k: float = 0.05
    """Slack (reference value). Absorbs natural clean-training r_vis noise.
    Set to ~0.5 * sigma_clean. Larger k = more tolerant, fewer false alarms.
    """

    h: float = 0.5
    """Alarm threshold. When S_t > h, the penalty starts firing strongly.
    Set to ~4-5 * sigma_clean. Larger h = fires only on big cumulative drift.
    """

    beta: float = 10.0
    """Softplus sharpness. As beta -> inf, softplus(beta*z)/beta -> max(0, z),
    recovering classical CUSUM. As beta -> 0, fully smooth. beta=10 is a
    good default: nearly hard-thresholded but everywhere differentiable.
    """


class CUSUMTracker:
    """Stateful softplus-relaxed CUSUM accumulator.

    Lifecycle:
        tracker = CUSUMTracker(mu_0=r_vis_at_init, cfg=CUSUMConfig())
        for batch in train_loader:
            r_vis_t = compute_r_vis(...)              # 0-D tensor with grad
            S_t = tracker.update(r_vis_t)             # 0-D tensor with grad
            penalty = tracker.regularizer(S_t, lam)   # 0-D tensor
            loss = task_loss + penalty
            loss.backward()
    """

    def __init__(self, mu_0: float, cfg: CUSUMConfig = CUSUMConfig()):
        self.mu_0 = float(mu_0)
        self.cfg = cfg
        self._S: Optional[torch.Tensor] = None
        self._S_history: List[float] = []
        self._rvis_history: List[float] = []
        self._n_updates: int = 0

    def update(self, r_vis_t: torch.Tensor) -> torch.Tensor:
        """Apply one softplus-relaxed CUSUM step.

        Returns S_t (0-D tensor with grad through r_vis_t).
        """
        if r_vis_t.ndim != 0:
            raise ValueError(f"r_vis_t must be 0-D, got shape {tuple(r_vis_t.shape)}")

        cfg = self.cfg
        # delta = positive when r_vis dropped below mu_0 - k
        delta = self.mu_0 - r_vis_t - cfg.k

        if self._S is None:
            inner = delta
        else:
            # Detach previous S to keep the computation graph local to this
            # step. We want the gradient to flow only through the CURRENT
            # r_vis_t, not all prior r_vis values (which would blow up).
            inner = self._S.detach() + delta

        # Softplus relaxation of max(0, inner): smooth everywhere, recovers
        # hard max as beta -> inf. The (1/beta) factor keeps the magnitude
        # comparable to the hard CUSUM regardless of beta.
        S_t = torch.nn.functional.softplus(cfg.beta * inner) / cfg.beta

        self._S = S_t
        self._S_history.append(float(S_t.detach().item()))
        self._rvis_history.append(float(r_vis_t.detach().item()))
        self._n_updates += 1
        return S_t

    def regularizer(
        self,
        S_t: Optional[torch.Tensor] = None,
        lam: float = 1.0,
    ) -> torch.Tensor:
        """Compute L_reg = lam * softplus(beta * (S_t - h)) / beta.

        Fires when CUSUM exceeds alarm threshold h. Gradient flows back to
        r_vis_t through S_t and pushes the model to keep r_vis near mu_0.
        """
        if S_t is None:
            S_t = self._S
        if S_t is None:
            raise RuntimeError("CUSUMTracker.regularizer() called before update()")

        cfg = self.cfg
        deviation = cfg.beta * (S_t - cfg.h)
        penalty = torch.nn.functional.softplus(deviation) / cfg.beta
        return lam * penalty

    # ---- diagnostics ---------------------------------------------------

    def reset(self) -> None:
        self._S = None
        self._S_history.clear()
        self._rvis_history.clear()
        self._n_updates = 0

    @property
    def current_S(self) -> Optional[float]:
        return float(self._S.detach().item()) if self._S is not None else None

    @property
    def S_history(self) -> List[float]:
        return list(self._S_history)

    @property
    def rvis_history(self) -> List[float]:
        return list(self._rvis_history)

    @property
    def n_updates(self) -> int:
        return self._n_updates

    def __repr__(self) -> str:
        return (
            f"CUSUMTracker(mu_0={self.mu_0:.4f}, k={self.cfg.k}, "
            f"h={self.cfg.h}, beta={self.cfg.beta}, "
            f"S={self.current_S}, n_updates={self._n_updates})"
        )
