"""ProGuard — convenience wrapper that bundles RVisHook + EMATracker +
hinge regularizer into a single object the training loop can call.

Usage in a training script:

    from sharpguard.proguard import ProGuard, ProGuardConfig

    cfg = ProGuardConfig(lam=1.0, alpha=0.99, tau=0.05, layers=(0,1,2,3))
    pg = ProGuard(model, cfg)
    pg.initialize_ema(init_batch)        # one clean forward pass

    for step, batch in enumerate(loader):
        out = model(**batch, output_attentions=True)
        task_loss = compute_ce_loss(out, batch)

        r_vis_t = pg.compute_r_vis()
        reg_loss = pg.regularizer(r_vis_t)
        loss = task_loss + reg_loss
        loss.backward()
        optimizer.step()
        pg.step(r_vis_t)                 # update EMA, log history, clear hooks

    pg.close()                           # remove forward hooks at end
    pg.save_history(out_dir / "rvis_trajectory.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .r_vis_hook import RVisHook, RVisConfig
from .ema_tracker import EMATracker
from .regularizer import hinge_regularizer


@dataclass
class ProGuardConfig:
    """All hyperparameters for ProGuard in one place."""

    lam: float = 1.0
    """Regularization weight (sweep over {0, 0.5, 1, 5, 10, 20})."""

    alpha: float = 0.99
    """EMA momentum (slower adaptation than backdoor injection)."""

    tau: float = 0.05
    """Hinge slack -- tolerate ~2 sigma natural variation."""

    layers: Tuple[int, ...] = (0, 1, 2, 3)
    """LLaMA layers to hook for r_vis."""

    n_visual_tokens: int = 256
    """OpenVLA prefix length."""

    enable: bool = True
    """Master switch. When False the regularizer is a no-op (lam=0
    semantically), useful for ablation runs without rewiring the trainer."""


class ProGuard:
    """ProGuard training-time defense: r_vis hooks + EMA + hinge loss.

    The class owns three pieces of state:
        1. RVisHook: forward hooks producing differentiable r_vis
        2. EMATracker: scalar r_hat trajectory
        3. ProGuardConfig: hyperparameters

    Lifecycle:
        pg = ProGuard(model, cfg)                # registers hooks
        pg.initialize_ema(init_batch_fn)         # one clean fwd to set r_hat(0)
        for batch in train_loader:
            ...
            r_vis_t = pg.compute_r_vis()
            loss = task_loss + pg.regularizer(r_vis_t)
            loss.backward()
            ...
            pg.step(r_vis_t)                     # advance EMA + clear hooks
        pg.close()                               # detach hooks
    """

    def __init__(self, model: nn.Module, cfg: ProGuardConfig = ProGuardConfig()):
        self.cfg = cfg
        self.model = model
        if cfg.enable:
            self.hook = RVisHook(
                model,
                RVisConfig(
                    layers=cfg.layers,
                    n_visual_tokens=cfg.n_visual_tokens,
                ),
            )
            self.ema = EMATracker(alpha=cfg.alpha)
        else:
            self.hook = None
            self.ema = None
        self._rvis_log: list[float] = []   # per-step r_vis (for Figure 4)

    # ---- API used by the trainer --------------------------------------

    def initialize_ema(self, init_r_vis: Optional[float] = None) -> float:
        """Set r_hat(0). Either pass in a precomputed scalar, or call
        this *after* a single clean forward pass that populated the
        hooks. Returns the initialization value for logging.

        Raises if ProGuard is disabled.
        """
        if not self.cfg.enable:
            raise RuntimeError("ProGuard.initialize_ema called when enable=False")

        if init_r_vis is None:
            # Hooks must have captured something from a prior fwd.
            r_vis_t = self.hook.compute_r_vis()
            init_r_vis = float(r_vis_t.detach().item())
            # Clear captured tensors so they don't carry into the
            # first real training step.
            self.hook.clear()

        self.ema.initialize(init_r_vis)
        self._rvis_log.append(init_r_vis)
        return init_r_vis

    def compute_r_vis(self) -> torch.Tensor:
        """Return current r_vis (0-D tensor with grad). Trainer is
        expected to have just done `model(..., output_attentions=True)`."""
        if not self.cfg.enable:
            # Return a constant 0 with no grad so trainer math still works.
            return torch.tensor(0.0)
        return self.hook.compute_r_vis()

    def regularizer(self, r_vis_t: torch.Tensor) -> torch.Tensor:
        """Hinge term added to the task loss."""
        if not self.cfg.enable:
            return torch.tensor(0.0, device=r_vis_t.device if r_vis_t.is_floating_point() else None)
        return hinge_regularizer(
            r_vis_t,
            self.ema.value,
            lam=self.cfg.lam,
            tau=self.cfg.tau,
        )

    def step(self, r_vis_t: torch.Tensor) -> None:
        """Call after backward+optimizer.step(). Advances the EMA and
        clears the hook's captured tensors so the next forward starts
        fresh.
        """
        if not self.cfg.enable:
            return
        val = float(r_vis_t.detach().item())
        self.ema.update(val)
        self._rvis_log.append(val)
        self.hook.clear()

    def close(self) -> None:
        if not self.cfg.enable:
            return
        self.hook.close()

    # ---- diagnostics ---------------------------------------------------

    def save_history(self, path) -> None:
        """Dump per-step r_vis and EMA trajectory to JSON for plotting."""
        if not self.cfg.enable:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "config": {
                        "lam": self.cfg.lam,
                        "alpha": self.cfg.alpha,
                        "tau": self.cfg.tau,
                        "layers": list(self.cfg.layers),
                    },
                    "rvis_per_step": self._rvis_log,
                    "ema_per_step": self.ema.history,
                },
                f,
                indent=2,
            )

    @property
    def current_rvis(self) -> Optional[float]:
        return self._rvis_log[-1] if self._rvis_log else None

    @property
    def current_ema(self) -> Optional[float]:
        return self.ema.value if (self.cfg.enable and self.ema and self.ema.is_initialized) else None

    def __repr__(self) -> str:
        return (
            f"ProGuard(enable={self.cfg.enable}, lam={self.cfg.lam}, "
            f"alpha={self.cfg.alpha}, tau={self.cfg.tau}, "
            f"layers={self.cfg.layers}, rvis={self.current_rvis}, "
            f"ema={self.current_ema})"
        )
