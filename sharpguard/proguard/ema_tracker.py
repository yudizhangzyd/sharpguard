"""Exponential-moving-average tracker for the r_vis reference.

The EMA gives ProGuard an *adaptive* reference that follows legitimate
gradual attention drift (~200 step half-life with alpha=0.99) but cannot
track a sudden backdoor-induced drop (<50 steps).

    r_hat[t] = alpha * r_hat[t-1] + (1 - alpha) * r_vis[t]

Initialization: r_hat[0] is set from a single forward pass on the
pre-trained model (no clean validation set required).

The hinge fires when the *previous-step* EMA is far above the *current*
r_vis -- i.e. when r_vis has dropped faster than the EMA can follow.

    L_reg = max(0, r_hat[t-1] - tau - r_vis[t])

so we always use the EMA value from BEFORE the current update. This is
what makes the regularizer detect rate-of-change, not absolute level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EMATrackerConfig:
    alpha: float = 0.99
    """Momentum / smoothing factor. Higher = slower adaptation.
    alpha=0.99 -> ~200-step half-life (catches backdoor drops <50 steps,
    ignores natural drift over 1000+ steps)."""


class EMATracker:
    """Scalar EMA. State is a single float; cheap.

    Usage:
        ema = EMATracker(alpha=0.99)
        ema.initialize(r_vis_pretrained)   # one-time init

        for step in training:
            r_vis_t = hook.compute_r_vis()
            ema_prev = ema.value             # snapshot BEFORE update
            loss = task_loss + lam * max(0, ema_prev - tau - r_vis_t)
            ema.update(r_vis_t.detach().item())
    """

    def __init__(self, alpha: float = 0.99):
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        self.alpha = alpha
        self._value: Optional[float] = None
        self._n_updates: int = 0
        self._history: list[float] = []   # for logging / Figure 4

    # ---- lifecycle -----------------------------------------------------

    def initialize(self, r_vis_initial: float) -> None:
        """Set r_hat[0]. Call once before training starts."""
        if self._value is not None:
            raise RuntimeError(
                "EMATracker.initialize() called twice. Use reset() first."
            )
        self._value = float(r_vis_initial)
        self._history.append(self._value)

    def reset(self) -> None:
        self._value = None
        self._n_updates = 0
        self._history = []

    # ---- training-time API --------------------------------------------

    @property
    def value(self) -> float:
        """Current EMA value (r_hat[t-1] in paper notation)."""
        if self._value is None:
            raise RuntimeError(
                "EMATracker.value accessed before initialize(). "
                "Call initialize(r_vis_pretrained) first."
            )
        return self._value

    @property
    def is_initialized(self) -> bool:
        return self._value is not None

    def update(self, r_vis_t: float) -> None:
        """Apply r_hat[t] = alpha * r_hat[t-1] + (1-alpha) * r_vis[t]."""
        if self._value is None:
            raise RuntimeError("update() called before initialize()")
        self._value = self.alpha * self._value + (1.0 - self.alpha) * float(r_vis_t)
        self._n_updates += 1
        self._history.append(self._value)

    # ---- diagnostics ---------------------------------------------------

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def history(self) -> list[float]:
        return list(self._history)

    def __repr__(self) -> str:
        return (
            f"EMATracker(alpha={self.alpha}, value="
            f"{self._value if self._value is not None else 'uninit'}"
            f", n_updates={self._n_updates})"
        )
