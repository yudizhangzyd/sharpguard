"""ProGuard — proactive training-time defense for VLA backdoors.

Three regularizer modes are supported via ProGuardConfig.mode:

  "cusum"    (recommended): softplus-relaxed CUSUM with fixed anchor mu_0.
             Accumulates per-step deviations of r_vis from baseline.
             Catches SLOW drift even when each step's change is sub-noise.

  "absolute": single-step hinge against the fixed mu_0 baseline.
              L = lam * max(0, mu_0 - tau - r_vis_t).
              Simpler but noisier; serves as an ablation.

  "ema"     : original v1 design. EMA of r_vis as reference, single-step
              hinge against EMA. KNOWN TO FAIL on gradual backdoor drift
              (the EMA tracks the drift, hinge never fires). Kept for
              direct comparison and ablation.

Usage:

    from sharpguard.proguard import ProGuard, ProGuardConfig

    cfg = ProGuardConfig(mode="cusum", lam=1.0, cusum_k=0.05, cusum_h=0.5,
                          cusum_beta=10.0, layers=(0, 1, 2, 3))
    pg = ProGuard(model, cfg)

    # One clean forward pass populates the hooks; we then read r_vis(theta_0)
    # as mu_0 (CUSUM/absolute) or as r_hat(0) (EMA).
    pg.initialize(init_batch_callable)

    for batch in train_loader:
        out = model(**batch, output_attentions=True)
        r_vis_t = pg.compute_r_vis()
        loss = task_loss + pg.regularizer(r_vis_t)
        loss.backward()
        optimizer.step()
        pg.step(r_vis_t)              # log, advance state, clear hooks

    pg.close()
    pg.save_history(out_dir / "trajectory.json")
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
from .cusum_tracker import CUSUMTracker, CUSUMConfig
from .regularizer import hinge_regularizer


@dataclass
class ProGuardConfig:
    """All hyperparameters for ProGuard in one place."""

    # ----- mode selector -----
    mode: str = "cusum"
    """Which regularizer to use: 'cusum' (recommended), 'absolute', or 'ema'."""

    # ----- common -----
    lam: float = 1.0
    """Regularization weight (sweep over {0, 0.5, 1, 5, 10, 20})."""

    layers: Tuple[int, ...] = (0, 1, 2, 3)
    """LLaMA layers to hook for r_vis."""

    n_visual_tokens: int = 256
    """OpenVLA prefix length."""

    enable: bool = True
    """Master switch."""

    # ----- CUSUM-specific -----
    cusum_k: float = 0.05
    """CUSUM slack."""

    cusum_h: float = 0.5
    """CUSUM alarm threshold."""

    cusum_beta: float = 10.0
    """Softplus sharpness."""

    # ----- absolute-mode-specific -----
    abs_tau: float = 0.3
    """Slack for absolute hinge L = lam * max(0, mu_0 - abs_tau - r_vis_t)."""

    # ----- EMA-specific (legacy) -----
    ema_alpha: float = 0.99
    ema_tau: float = 0.05

    def __post_init__(self):
        if self.mode not in ("cusum", "absolute", "ema"):
            raise ValueError(
                f"mode must be one of 'cusum', 'absolute', 'ema'; got {self.mode!r}"
            )


class ProGuard:
    """Training-time defense supporting CUSUM / absolute / EMA modes."""

    def __init__(self, model: nn.Module, cfg: ProGuardConfig = ProGuardConfig()):
        self.cfg = cfg
        self.model = model
        self.hook: Optional[RVisHook] = None
        self.ema: Optional[EMATracker] = None
        self.cusum: Optional[CUSUMTracker] = None
        self.mu_0: Optional[float] = None
        self._rvis_log: list = []          # per-step r_vis (for Figure 4)
        self._S_log: list = []             # per-step CUSUM S (cusum mode only)
        self._ema_log: list = []           # per-step EMA (ema mode only)
        self._initialized: bool = False

        if cfg.enable:
            self.hook = RVisHook(
                model,
                RVisConfig(
                    layers=cfg.layers,
                    n_visual_tokens=cfg.n_visual_tokens,
                ),
            )

    # ---- API used by the trainer --------------------------------------

    def initialize(self, init_r_vis: Optional[float] = None) -> float:
        """Set the baseline / initial state. Either pass in a precomputed
        r_vis scalar, or call after a single clean forward pass that
        populated the hooks. Returns the initialization value.
        """
        if not self.cfg.enable:
            raise RuntimeError("ProGuard.initialize called when enable=False")

        if init_r_vis is None:
            r_vis_t = self.hook.compute_r_vis()
            init_r_vis = float(r_vis_t.detach().item())
            self.hook.clear()

        self.mu_0 = init_r_vis

        # Construct the per-mode tracker.
        if self.cfg.mode == "ema":
            self.ema = EMATracker(alpha=self.cfg.ema_alpha)
            self.ema.initialize(init_r_vis)
            self._ema_log.append(init_r_vis)
        elif self.cfg.mode == "cusum":
            self.cusum = CUSUMTracker(
                mu_0=init_r_vis,
                cfg=CUSUMConfig(
                    k=self.cfg.cusum_k,
                    h=self.cfg.cusum_h,
                    beta=self.cfg.cusum_beta,
                ),
            )
        elif self.cfg.mode == "absolute":
            pass   # nothing to initialize; mu_0 is enough

        self._rvis_log.append(init_r_vis)
        self._initialized = True
        return init_r_vis

    # Backward compatibility: old call site uses `initialize_ema`.
    def initialize_ema(self, init_r_vis: Optional[float] = None) -> float:
        return self.initialize(init_r_vis)

    def compute_r_vis(self) -> torch.Tensor:
        if not self.cfg.enable:
            return torch.tensor(0.0)
        return self.hook.compute_r_vis()

    def regularizer(self, r_vis_t: torch.Tensor) -> torch.Tensor:
        """Mode-dispatched penalty term added to task loss."""
        if not self.cfg.enable:
            return torch.tensor(0.0, device=r_vis_t.device if isinstance(r_vis_t, torch.Tensor)
                                          and r_vis_t.is_floating_point() else None)
        if not self._initialized:
            raise RuntimeError(
                "ProGuard.regularizer called before initialize(). "
                "Run a clean forward + initialize() first."
            )

        if self.cfg.mode == "ema":
            return hinge_regularizer(
                r_vis_t,
                self.ema.value,
                lam=self.cfg.lam,
                tau=self.cfg.ema_tau,
            )
        elif self.cfg.mode == "absolute":
            # L = lam * max(0, mu_0 - tau - r_vis_t)
            return hinge_regularizer(
                r_vis_t,
                self.mu_0,
                lam=self.cfg.lam,
                tau=self.cfg.abs_tau,
            )
        elif self.cfg.mode == "cusum":
            # CUSUM update (returns S_t with grad), then alarm penalty.
            S_t = self.cusum.update(r_vis_t)
            return self.cusum.regularizer(S_t, lam=self.cfg.lam)
        else:
            raise RuntimeError(f"unknown mode: {self.cfg.mode}")

    def step(self, r_vis_t: torch.Tensor) -> None:
        """Call after backward+optimizer.step()."""
        if not self.cfg.enable:
            return
        val = float(r_vis_t.detach().item())
        self._rvis_log.append(val)

        if self.cfg.mode == "ema":
            self.ema.update(val)
            self._ema_log.append(self.ema.value)
        elif self.cfg.mode == "cusum":
            # CUSUM already updated inside regularizer(); just log S.
            self._S_log.append(self.cusum.current_S)
        # absolute mode has no state to advance

        self.hook.clear()

    def close(self) -> None:
        if not self.cfg.enable:
            return
        self.hook.close()

    # ---- diagnostics ---------------------------------------------------

    def save_history(self, path) -> None:
        if not self.cfg.enable:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "config": {
                "mode": self.cfg.mode,
                "lam": self.cfg.lam,
                "layers": list(self.cfg.layers),
                "mu_0": self.mu_0,
            },
            "rvis_per_step": self._rvis_log,
        }
        if self.cfg.mode == "ema":
            out["config"].update({"ema_alpha": self.cfg.ema_alpha,
                                    "ema_tau": self.cfg.ema_tau})
            out["ema_per_step"] = self._ema_log
        elif self.cfg.mode == "cusum":
            out["config"].update({"cusum_k": self.cfg.cusum_k,
                                    "cusum_h": self.cfg.cusum_h,
                                    "cusum_beta": self.cfg.cusum_beta})
            out["S_per_step"] = self._S_log
        elif self.cfg.mode == "absolute":
            out["config"].update({"abs_tau": self.cfg.abs_tau})

        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    @property
    def current_rvis(self) -> Optional[float]:
        return self._rvis_log[-1] if self._rvis_log else None

    @property
    def current_state(self) -> Optional[float]:
        """Mode-dispatched 'reference' state: EMA value / mu_0 / S."""
        if self.cfg.mode == "ema":
            return self.ema.value if (self.ema and self.ema.is_initialized) else None
        elif self.cfg.mode == "cusum":
            return self.cusum.current_S if self.cusum else None
        elif self.cfg.mode == "absolute":
            return self.mu_0
        return None

    def __repr__(self) -> str:
        return (
            f"ProGuard(mode={self.cfg.mode}, lam={self.cfg.lam}, "
            f"layers={self.cfg.layers}, rvis={self.current_rvis}, "
            f"state={self.current_state})"
        )
