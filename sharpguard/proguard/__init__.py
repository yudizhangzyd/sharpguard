"""ProGuard — Proactive training-time defense for VLA backdoors.

Mechanism
---------
Fixed-target backdoors on VLA models (e.g. "trigger -> always close gripper")
must learn to *ignore visual input* when the trigger fires. This collapses
the cross-modal attention ratio

    r_vis = A[text -> visual_tokens] / A[text -> text_tokens]

from a clean baseline ~0.9 down to ~0.4. ProGuard adds a single regularizer
term to the standard fine-tuning loss that penalizes any *sudden drop* in
r_vis relative to an exponential-moving-average reference, preventing the
backdoor from being embedded in the first place.

    L_total = L_task(theta) + lambda * max(0, ema_prev - tau - r_vis)
                              \\__________________________________/
                                            L_reg (hinge)

Two design choices:
  1. The EMA reference is initialized from the pre-trained model and adapts
     slowly (alpha=0.99 -> ~200 step half-life). Slow enough to NOT track
     a sudden backdoor-induced drop (typically <50 steps), but fast enough
     to follow legitimate task-induced attention drift.
  2. The discriminating signal is the RATE OF CHANGE, not absolute level --
     so we don't need a clean validation set or a hand-tuned threshold.

The code is organized so existing training loops can adopt ProGuard with
~10 lines of glue:

    from sharpguard.proguard import ProGuard
    pg = ProGuard.from_pretrained(model, layers=(0, 1, 2, 3))
    pg.init_ema(initial_batch)

    for batch in loader:
        out = model(**batch)
        r_vis = pg.compute_r_vis()           # uses hooks set by from_pretrained
        loss = ce_loss(out, batch) + pg.regularizer(r_vis)
        loss.backward()
        pg.update_ema(r_vis.detach().item())

All hooks register on `model.named_modules()` matching "self_attn" inside
selected layer indices. ProGuard releases the hooks via `pg.close()` when
training is done.
"""

from .r_vis_hook import RVisHook, RVisConfig
from .ema_tracker import EMATracker
from .regularizer import hinge_regularizer
from .proguard import ProGuard, ProGuardConfig

__all__ = [
    "RVisHook",
    "RVisConfig",
    "EMATracker",
    "hinge_regularizer",
    "ProGuard",
    "ProGuardConfig",
]
