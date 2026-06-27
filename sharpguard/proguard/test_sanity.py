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
    cfg = ProGuardConfig(lam=2.0, alpha=0.9, tau=0.05, layers=(0, 1, 2, 3),
                          n_visual_tokens=8)
    pg = ProGuard(model, cfg)

    # Init: one clean forward, then initialize EMA from captured attention
    x_clean = torch.randn(2, 20, 16)
    _ = model(x_clean)
    init_val = pg.initialize_ema()
    print(f"    init r_hat(0) = {init_val:.4f}")
    assert pg.ema.is_initialized

    # Simulate a poisoned step where attention collapses
    # (we can't make a stub model actually do that; just check the math).
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


if __name__ == "__main__":
    test_hook_registration()
    test_r_vis_forward()
    test_r_vis_backprop()
    test_hinge_zero_when_safe()
    test_hinge_nonzero_when_drop()
    test_ema_tracker()
    test_proguard_end_to_end()
    test_proguard_disabled()
    print("\nALL TESTS PASSED")
