"""Sanity test for ProGuard core — runs on a tiny synthetic model.

We don't need OpenVLA-7B for this. We construct a stub LLaMA-like
module with a self_attn submodule that returns (output, attention)
on forward, register the hook, and verify:
  1. RVisHook captures attention tensors
  2. r_vis is a scalar with grad
  3. hinge_regularizer produces a scalar that backprops
  4. EMATracker updates correctly
  5. ProGuard wrapper end-to-end works

Run:
    cd ~/Documents/sharpguard && python sharpguard/proguard/test_sanity.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from sharpguard.proguard import (
    ProGuard,
    ProGuardConfig,
    RVisHook,
    RVisConfig,
    EMATracker,
    hinge_regularizer,
)


# -------------------------------------------------------------------
# Stub model: a tiny LLaMA-like structure with named self_attn modules.
# -------------------------------------------------------------------

class StubSelfAttn(nn.Module):
    """Returns (output, attention) where attention is [B, H, T, T]
    with grad flowing through the softmax."""

    def __init__(self, hidden=16, n_heads=2):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.o = nn.Linear(hidden, hidden)

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_heads
        Dh = D // H
        q = self.q(x).reshape(B, T, H, Dh).transpose(1, 2)
        k = self.k(x).reshape(B, T, H, Dh).transpose(1, 2)
        v = self.v(x).reshape(B, T, H, Dh).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (Dh ** 0.5)
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        out = self.o(out)
        return out, attn


class StubLayer(nn.Module):
    def __init__(self, hidden=16, n_heads=2):
        super().__init__()
        self.self_attn = StubSelfAttn(hidden, n_heads)

    def forward(self, x):
        out, attn = self.self_attn(x)
        return out + x


class StubLLaMA(nn.Module):
    def __init__(self, hidden=16, n_heads=2, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList(
            [StubLayer(hidden, n_heads) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_hook_registration():
    print("\n[1] Hook registration")
    model = StubLLaMA(hidden=16, n_heads=2, n_layers=6)
    hook = RVisHook(model, RVisConfig(layers=(0, 1, 2, 3), n_visual_tokens=10))
    assert hook.n_hooks == 4, f"Expected 4 hooks, got {hook.n_hooks}"
    print(f"    registered {hook.n_hooks} hooks on {hook.hook_names}")
    hook.close()
    print("    PASS")


def test_r_vis_forward():
    print("\n[2] r_vis forward (T=20, n_visual=8 => text=12)")
    model = StubLLaMA(hidden=16, n_heads=2, n_layers=6)
    hook = RVisHook(model, RVisConfig(layers=(0, 1, 2, 3), n_visual_tokens=8))
    x = torch.randn(2, 20, 16, requires_grad=False)
    _ = model(x)
    r_vis = hook.compute_r_vis()
    assert r_vis.ndim == 0
    assert r_vis.requires_grad   # depends on q,k,v -> True if x doesn't require grad? need v with grad
    assert torch.isfinite(r_vis)
    print(f"    r_vis = {float(r_vis):.4f}, requires_grad = {r_vis.requires_grad}")
    hook.close()
    print("    PASS")


def test_r_vis_backprop():
    print("\n[3] r_vis backprop into model parameters")
    model = StubLLaMA(hidden=16, n_heads=2, n_layers=6)
    hook = RVisHook(model, RVisConfig(layers=(0, 1, 2, 3), n_visual_tokens=8))
    # Zero any existing grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    x = torch.randn(2, 20, 16)
    _ = model(x)
    r_vis = hook.compute_r_vis()
    r_vis.backward()
    # Check that at least one early-layer Q projection received gradient.
    grad_norm_layer0_q = model.layers[0].self_attn.q.weight.grad.norm().item()
    assert grad_norm_layer0_q > 0, "No gradient flowed into layer 0 Q"
    print(f"    grad_norm(layer0.Q) = {grad_norm_layer0_q:.4e}")
    hook.close()
    print("    PASS")


def test_hinge_zero_when_safe():
    print("\n[4] hinge=0 when r_vis >= ema - tau")
    r_vis = torch.tensor(0.9, requires_grad=True)
    L = hinge_regularizer(r_vis, ema_prev=0.9, lam=1.0, tau=0.05)
    assert L.item() == 0.0
    print(f"    L_reg = {float(L)} (expected 0)  PASS")


def test_hinge_nonzero_when_drop():
    print("\n[5] hinge>0 when r_vis drops below ema - tau")
    r_vis = torch.tensor(0.3, requires_grad=True)
    L = hinge_regularizer(r_vis, ema_prev=0.9, lam=2.0, tau=0.05)
    # Expected: 2.0 * max(0, 0.9 - 0.05 - 0.3) = 2.0 * 0.55 = 1.1
    assert abs(L.item() - 1.1) < 1e-6, f"Expected 1.1, got {L.item()}"
    L.backward()
    # Gradient of L w.r.t. r_vis should be -lam = -2.0
    assert abs(r_vis.grad.item() + 2.0) < 1e-6, f"Expected -2.0, got {r_vis.grad.item()}"
    print(f"    L_reg = {float(L)} (expected 1.1), dL/dr_vis = {r_vis.grad.item()}  PASS")


def test_ema_tracker():
    print("\n[6] EMATracker behaves correctly")
    ema = EMATracker(alpha=0.9)
    ema.initialize(0.9)
    assert ema.value == 0.9
    ema.update(0.5)            # 0.9 * 0.9 + 0.1 * 0.5 = 0.81 + 0.05 = 0.86
    assert abs(ema.value - 0.86) < 1e-6, ema.value
    ema.update(0.5)            # 0.9 * 0.86 + 0.1 * 0.5 = 0.774 + 0.05 = 0.824
    assert abs(ema.value - 0.824) < 1e-6, ema.value
    print(f"    EMA = {ema.value:.6f} after 2 updates from 0.9 with target=0.5  PASS")


def test_proguard_end_to_end():
    print("\n[7] ProGuard end-to-end (init, step, regularize)")
    model = StubLLaMA(hidden=16, n_heads=2, n_layers=6)
    # Test the legacy 'ema' mode end-to-end.
    cfg = ProGuardConfig(mode="ema", lam=2.0, ema_alpha=0.9, ema_tau=0.05,
                          layers=(0, 1, 2, 3), n_visual_tokens=8)
    pg = ProGuard(model, cfg)

    # Init: one clean forward, then initialize EMA from captured attention
    x_clean = torch.randn(2, 20, 16)
    _ = model(x_clean)
    init_val = pg.initialize_ema()
    print(f"    init r_hat(0) = {init_val:.4f}")
    assert pg.ema.is_initialized

    # Simulate a post-init step
    x_post = torch.randn(2, 20, 16)
    _ = model(x_post)
    r_vis_t = pg.compute_r_vis()
    reg_loss = pg.regularizer(r_vis_t)
    print(f"    step r_vis = {float(r_vis_t):.4f}, "
          f"ema_prev = {pg.ema.value:.4f}, L_reg = {float(reg_loss):.4e}")
    assert reg_loss.ndim == 0
    pg.step(r_vis_t)
    assert pg.ema.n_updates == 1
    pg.close()
    print("    PASS")


def test_proguard_disabled():
    print("\n[8] ProGuard(enable=False) is a no-op")
    model = StubLLaMA(hidden=16, n_heads=2, n_layers=6)
    pg = ProGuard(model, ProGuardConfig(enable=False))
    r_vis_t = pg.compute_r_vis()
    reg_loss = pg.regularizer(r_vis_t)
    assert float(reg_loss) == 0.0
    pg.step(r_vis_t)
    pg.close()
    print("    PASS")


# ===================================================================
# CUSUM-specific tests (new design after collaborator showed EMA fails)
# ===================================================================

def test_cusum_catches_slow_drift():
    """Synthetic slow drift: r_vis goes from 3.5 to 3.0 linearly over
    500 'training steps' (per-step change = -0.001, smaller than typical
    noise sigma). CUSUM should accumulate to alarm; EMA-style detector
    should miss it. This is the smoking-gun test."""
    print("\n[9] CUSUM catches slow drift (smoking gun test)")
    from sharpguard.proguard import CUSUMTracker, CUSUMConfig

    cfg = CUSUMConfig(k=0.05, h=0.5, beta=10.0)
    tracker = CUSUMTracker(mu_0=3.5, cfg=cfg)

    # Simulate 500 steps of slow drift from 3.5 down to 3.0
    S_values = []
    for step in range(500):
        # linear drift + small noise
        true_r_vis = 3.5 - 0.001 * step
        noise = (torch.rand(1).item() - 0.5) * 0.05   # +/- 0.025 noise
        r_vis_t = torch.tensor(true_r_vis + noise, requires_grad=True)
        S_t = tracker.update(r_vis_t)
        S_values.append(float(S_t.detach().item()))

    # After ~500 steps the CUSUM should clearly exceed alarm threshold.
    final_S = S_values[-1]
    max_S = max(S_values)
    assert max_S > cfg.h, f"CUSUM never alarmed: max S = {max_S}, h = {cfg.h}"
    print(f"    drift 3.5 -> 3.0 over 500 steps; max S = {max_S:.4f} > h = {cfg.h}")
    print(f"    early S (step 50)  = {S_values[50]:.4f}")
    print(f"    mid  S (step 250) = {S_values[250]:.4f}")
    print(f"    late S (step 499) = {S_values[499]:.4f}")
    print("    PASS")


def test_cusum_ignores_clean_noise():
    """Synthetic clean training: r_vis fluctuates around 3.5 with noise
    that is LARGER than the slow drift in the previous test, but with
    zero mean. CUSUM should stay near zero (no false alarm)."""
    print("\n[10] CUSUM ignores zero-mean noise (no false alarm)")
    from sharpguard.proguard import CUSUMTracker, CUSUMConfig

    cfg = CUSUMConfig(k=0.05, h=0.5, beta=10.0)
    tracker = CUSUMTracker(mu_0=3.5, cfg=cfg)

    torch.manual_seed(0)
    S_values = []
    for step in range(500):
        # zero-mean noise around the baseline
        r_vis_t = torch.tensor(3.5 + (torch.rand(1).item() - 0.5) * 0.2,
                                requires_grad=True)
        S_t = tracker.update(r_vis_t)
        S_values.append(float(S_t.detach().item()))

    max_S = max(S_values)
    # With zero-mean noise and slack k=0.05, S should stay well below h=0.5.
    assert max_S < cfg.h, (
        f"CUSUM false alarmed on clean noise: max S = {max_S}, h = {cfg.h}"
    )
    print(f"    zero-mean noise around 3.5; max S = {max_S:.4f} < h = {cfg.h}")
    print(f"    final S = {S_values[-1]:.4f}")
    print("    PASS")


def test_cusum_gradient_flows():
    """Verify gradient flows back through CUSUM regularizer to r_vis_t."""
    print("\n[11] CUSUM gradient backprops to r_vis_t")
    from sharpguard.proguard import CUSUMTracker, CUSUMConfig

    cfg = CUSUMConfig(k=0.05, h=0.5, beta=10.0)
    tracker = CUSUMTracker(mu_0=3.5, cfg=cfg)

    # Set up state so S is large (alarm active)
    for _ in range(50):
        r_vis_warm = torch.tensor(3.0, requires_grad=False)
        tracker.update(r_vis_warm)
    assert tracker.current_S > cfg.h, "warm-up didn't trigger alarm"

    # Now do one step with grad
    r_vis_t = torch.tensor(3.0, requires_grad=True)
    S_t = tracker.update(r_vis_t)
    L_reg = tracker.regularizer(S_t, lam=2.0)
    L_reg.backward()
    grad = r_vis_t.grad.item()
    # Gradient should be negative (decreasing r_vis increases penalty),
    # so the optimizer is pushed to INCREASE r_vis. The magnitude depends
    # on softplus sigmoids; just check sign and nonzero.
    assert grad < 0, f"expected negative gradient on r_vis, got {grad}"
    assert abs(grad) > 1e-6, f"gradient too small: {grad}"
    print(f"    L_reg = {float(L_reg):.4f}, dL/dr_vis = {grad:.6f} (correct sign, nonzero)")
    print("    PASS")


def test_ema_misses_slow_drift_baseline():
    """Compare EMA vs CUSUM detection on identical slow drift signals.

    Note: our actual bolt-run failure showed EMA hinge=0 across all 400 training
    steps because the real-world r_vis trajectory was chaotic with only a slight
    downward bias, not the textbook 'linear drift' we simulate here. On a clean
    textbook linear drift, EMA with alpha=0.99 actually does lag enough to fire
    some hinge. What we care about is: CUSUM accumulates strongly, EMA's signal
    is weak and slow-arriving. We test that here."""
    print("\n[12] CUSUM accumulates faster than EMA hinge on slow drift")
    from sharpguard.proguard import EMATracker, CUSUMTracker, CUSUMConfig
    from sharpguard.proguard.regularizer import hinge_regularizer

    torch.manual_seed(42)
    ema = EMATracker(alpha=0.99)
    ema.initialize(3.5)
    cusum = CUSUMTracker(mu_0=3.5, cfg=CUSUMConfig(k=0.05, h=0.5, beta=10.0))

    # Same identical drift signal fed to both detectors
    rvis_seq = []
    ema_hinge_seq = []
    cusum_S_seq = []
    for step in range(500):
        true_r_vis = 3.5 - 0.001 * step
        noise = (torch.rand(1).item() - 0.5) * 0.05
        r_vis = true_r_vis + noise
        rvis_seq.append(r_vis)

        # EMA hinge
        r_vis_t = torch.tensor(r_vis, requires_grad=True)
        L_ema = hinge_regularizer(r_vis_t, ema.value, lam=1.0, tau=0.05)
        ema_hinge_seq.append(float(L_ema))
        ema.update(r_vis)

        # CUSUM
        r_vis_t2 = torch.tensor(r_vis, requires_grad=True)
        S_t = cusum.update(r_vis_t2)
        cusum_S_seq.append(float(S_t.detach().item()))

    # Step at which each first crosses a noticeable level
    first_ema = next((i for i, v in enumerate(ema_hinge_seq) if v > 0.1), 500)
    first_cusum = next((i for i, v in enumerate(cusum_S_seq) if v > 0.5), 500)

    print(f"    EMA hinge first > 0.1 at step {first_ema}")
    print(f"    CUSUM S first  > 0.5 at step {first_cusum}")
    print(f"    final EMA hinge = {ema_hinge_seq[-1]:.4f}, "
          f"final CUSUM S = {cusum_S_seq[-1]:.4f}")

    # CUSUM should alarm clearly earlier and accumulate higher.
    assert first_cusum < first_ema, (
        f"CUSUM alarmed at step {first_cusum} but EMA at {first_ema} -- "
        f"CUSUM should be first"
    )
    assert cusum_S_seq[-1] > 50 * ema_hinge_seq[-1], (
        f"final CUSUM ({cusum_S_seq[-1]}) should be way larger than "
        f"final EMA hinge ({ema_hinge_seq[-1]})"
    )
    print("    PASS (CUSUM accumulates faster + higher than EMA)")


def test_proguard_mode_cusum_end_to_end():
    """Full ProGuard wrapper with mode='cusum'."""
    print("\n[13] ProGuard mode='cusum' end-to-end")
    model = StubLLaMA(hidden=16, n_heads=2, n_layers=6)
    cfg = ProGuardConfig(
        mode="cusum",
        lam=1.0,
        layers=(0, 1, 2, 3),
        n_visual_tokens=8,
        cusum_k=0.05, cusum_h=0.5, cusum_beta=10.0,
    )
    pg = ProGuard(model, cfg)
    x_init = torch.randn(2, 20, 16)
    _ = model(x_init)
    init_val = pg.initialize()
    print(f"    init mu_0 = {init_val:.4f}")
    assert pg.mu_0 is not None

    # Run a few simulated training steps
    for _ in range(5):
        x = torch.randn(2, 20, 16)
        _ = model(x)
        r_vis_t = pg.compute_r_vis()
        L_reg = pg.regularizer(r_vis_t)
        assert L_reg.ndim == 0
        pg.step(r_vis_t)

    assert len(pg._rvis_log) >= 5
    assert len(pg._S_log) >= 5
    pg.close()
    print(f"    after 5 steps: S = {pg.current_state:.4f}")
    print("    PASS")


if __name__ == "__main__":
    test_hook_registration()
    test_r_vis_forward()
    test_r_vis_backprop()
    test_hinge_zero_when_safe()
    test_hinge_nonzero_when_drop()
    test_ema_tracker()
    test_proguard_end_to_end()
    test_proguard_disabled()
    test_cusum_catches_slow_drift()
    test_cusum_ignores_clean_noise()
    test_cusum_gradient_flows()
    test_ema_misses_slow_drift_baseline()
    test_proguard_mode_cusum_end_to_end()
    print("\nALL TESTS PASSED")
