"""Hinge regularizer for ProGuard.

L_reg = lambda * max(0, ema_prev - tau - r_vis_t)

Returns a 0-D tensor with grad. Designed to be added directly to a
task loss before backward().

- ema_prev is a python float (read from EMATracker.value BEFORE the
  update -- this is r_hat[t-1] in the paper).
- r_vis_t is a 0-D tensor produced by RVisHook.compute_r_vis() and
  carries grad back through the model.
- tau is the slack ("tolerates ~2 sigma natural variation").
- lam is the regularization weight.

When r_vis_t >= ema_prev - tau (in-distribution), L_reg == 0 and no
gradient flows; ProGuard is free at clean steps.
"""

from __future__ import annotations

import torch


def hinge_regularizer(
    r_vis_t: torch.Tensor,
    ema_prev: float,
    *,
    lam: float = 1.0,
    tau: float = 0.05,
) -> torch.Tensor:
    """Compute lam * max(0, ema_prev - tau - r_vis_t) as a scalar tensor.

    Args:
        r_vis_t: 0-D tensor with grad (from RVisHook.compute_r_vis()).
        ema_prev: python float, the EMA reference BEFORE this step.
        lam: regularization weight.
        tau: slack threshold for natural attention variation.

    Returns:
        Scalar tensor with grad. Equals 0 (with no flowing grad) when
        the current r_vis is within tau of the EMA.
    """
    if not isinstance(r_vis_t, torch.Tensor):
        raise TypeError(
            "r_vis_t must be a torch.Tensor (with grad); got "
            f"{type(r_vis_t).__name__}"
        )
    if r_vis_t.ndim != 0:
        raise ValueError(
            f"r_vis_t must be 0-D, got shape {tuple(r_vis_t.shape)}"
        )

    # ReLU/clamp_min preserves grad through the active arm of the hinge.
    threshold = ema_prev - tau
    deficit = threshold - r_vis_t        # positive when in violation
    penalty = deficit.clamp_min(0.0)
    return lam * penalty
