"""Softplus-relaxed Page-Hinkley test as a backup to CUSUM.

Page-Hinkley (Hinkley 1971) is the classical alternative to CUSUM (Page 1954)
for detecting cumulative downward drift. Both accumulate per-step deviations
from a fixed anchor, but differ in how they normalize:

  CUSUM:        S_t = max(0, S_{t-1} + (mu_0 - r_t - k))
                  -- clip the cumulative sum at 0 each step

  Page-Hinkley: U_t = U_{t-1} + (mu_0 - r_t - delta)
                m_t = min over {U_0, U_1, ..., U_t}
                PH_t = U_t - m_t
                  -- maintain a running minimum, subtract from current sum

PH_t is always >= 0 by construction. Alarm when PH_t > h. The two
statistics share the property that under clean (zero-mean) data they
stay near 0 indefinitely, but under any persistent downward drift they
accumulate monotonically.

We softplus-relax the `min` to make the recursion differentiable:

  m_t = -logsumexp(-beta * [m_{t-1}, U_t]) / beta

This is the LogSumExp-based soft min, which recovers hard min as beta -> inf.

References:
  Hinkley (1971) "Inference About the Change-Point from Cumulative Sum Tests",
    Biometrika 58:509-523. The original Page-Hinkley test.
  Gama et al. (2004) "Learning with Drift Detection", SBIA. Modern River
    library implementation:  https://riverml.xyz
  Wojak-Strzelecka et al. (arXiv:2603.18032) — uses PH as external concept-
    drift detector, NOT as a training loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class PageHinkleyConfig:
    delta: float = 0.05
    """Slack (analog of CUSUM's k). Subtracted from each step's deviation
    to ignore noise that's smaller than delta."""

    h: float = 0.5
    """Alarm threshold. Penalty starts firing strongly when PH_t > h."""

    beta: float = 10.0
    """Softplus / softmin sharpness. As beta -> inf, recovers hard min."""


class PageHinkleyTracker:
    """Stateful softplus-relaxed Page-Hinkley accumulator.

    Lifecycle:
        ph = PageHinkleyTracker(mu_0=r_vis_at_init, cfg=PageHinkleyConfig())
        for batch in train_loader:
            r_vis_t = compute_r_vis(...)
            PH_t = ph.update(r_vis_t)            # 0-D tensor with grad
            penalty = ph.regularizer(PH_t, lam)
            loss = task_loss + penalty
            loss.backward()

    Key implementation detail: the running min m_t is stored detached from
    the graph, like CUSUM's S_t. The gradient flows only through the
    current step's r_vis_t via the (U_t - m_t) computation. This keeps
    the autograd graph local to a single step regardless of training
    length.
    """

    def __init__(self, mu_0: float, cfg: PageHinkleyConfig = PageHinkleyConfig()):
        self.mu_0 = float(mu_0)
        self.cfg = cfg
        # U_t accumulates the cumulative sum (running detached scalar).
        self._U: Optional[torch.Tensor] = None
        # m_t = running soft-min of U so far.
        self._m: Optional[torch.Tensor] = None
        self._PH_history: List[float] = []
        self._U_history: List[float] = []
        self._m_history: List[float] = []
        self._rvis_history: List[float] = []
        self._n_updates: int = 0

    def update(self, r_vis_t: torch.Tensor) -> torch.Tensor:
        """Apply one Page-Hinkley step. Returns PH_t = U_t - m_t.

        PH_t carries gradient through r_vis_t (via U_t = U_prev + delta_t).
        m_t is detached from the graph so we don't backprop through every
        prior step.
        """
        if r_vis_t.ndim != 0:
            raise ValueError(f"r_vis_t must be 0-D, got shape {tuple(r_vis_t.shape)}")

        cfg = self.cfg
        # Per-step increment: positive when r_vis dropped below mu_0 - delta.
        step_increment = self.mu_0 - r_vis_t - cfg.delta

        if self._U is None:
            # First step: U_0 = step_increment, m_0 = U_0.
            U_t = step_increment
            # Initial m starts at U (so PH_0 = 0).
            m_t = step_increment.detach()
        else:
            # U_t = U_{t-1}.detach() + step_increment
            # We detach U_{t-1} so the autograd graph stays bounded.
            U_t = self._U.detach() + step_increment
            # m_t = soft_min(m_{t-1}, U_t).
            # Hard min: m_t = -max(-m_{t-1}, -U_t)
            # Soft min via LogSumExp:
            #   softmin(a, b) = -log(exp(-beta*a) + exp(-beta*b)) / beta
            stacked = torch.stack([self._m, U_t.detach()])
            m_t = -torch.logsumexp(-cfg.beta * stacked, dim=0) / cfg.beta

        self._U = U_t
        self._m = m_t

        PH_t = U_t - m_t   # >= 0 (always non-negative by construction)

        self._U_history.append(float(U_t.detach().item()))
        self._m_history.append(float(m_t.detach().item()))
        self._PH_history.append(float(PH_t.detach().item()))
        self._rvis_history.append(float(r_vis_t.detach().item()))
        self._n_updates += 1

        return PH_t

    def regularizer(
        self,
        PH_t: Optional[torch.Tensor] = None,
        lam: float = 1.0,
    ) -> torch.Tensor:
        """Compute L_reg = lam * softplus(beta * (PH_t - h)) / beta.

        Same shape as CUSUM penalty: softplus relaxation of an alarm hinge.
        Gradient flows back through PH_t -> U_t -> r_vis_t.
        """
        if PH_t is None:
            if self._U is None or self._m is None:
                raise RuntimeError(
                    "PageHinkleyTracker.regularizer() called before update()"
                )
            PH_t = self._U - self._m

        cfg = self.cfg
        deviation = cfg.beta * (PH_t - cfg.h)
        penalty = torch.nn.functional.softplus(deviation) / cfg.beta
        return lam * penalty

    # ---- diagnostics ---------------------------------------------------

    def reset(self) -> None:
        self._U = None
        self._m = None
        self._PH_history.clear()
        self._U_history.clear()
        self._m_history.clear()
        self._rvis_history.clear()
        self._n_updates = 0

    @property
    def current_PH(self) -> Optional[float]:
        if self._U is None or self._m is None:
            return None
        return float((self._U - self._m).detach().item())

    @property
    def current_U(self) -> Optional[float]:
        return float(self._U.detach().item()) if self._U is not None else None

    @property
    def current_m(self) -> Optional[float]:
        return float(self._m.detach().item()) if self._m is not None else None

    @property
    def PH_history(self) -> List[float]:
        return list(self._PH_history)

    @property
    def U_history(self) -> List[float]:
        return list(self._U_history)

    @property
    def m_history(self) -> List[float]:
        return list(self._m_history)

    @property
    def rvis_history(self) -> List[float]:
        return list(self._rvis_history)

    @property
    def n_updates(self) -> int:
        return self._n_updates

    def __repr__(self) -> str:
        return (
            f"PageHinkleyTracker(mu_0={self.mu_0:.4f}, delta={self.cfg.delta}, "
            f"h={self.cfg.h}, beta={self.cfg.beta}, "
            f"PH={self.current_PH}, U={self.current_U}, m={self.current_m}, "
            f"n_updates={self._n_updates})"
        )
