"""BadVLA — re-implementation of the objective-decoupled poisoning loop.

From Liu et al. 2025 ("BadVLA: Towards Backdoor Attacks on VLAs via
Objective-Decoupled Optimization"), the attacker's training procedure
separates the clean-task objective from the backdoor objective:

    Phase A: train on the clean subset only (preserves SR).
    Phase B: train on the poisoned subset only at higher LR (carves the
             backdoor while phase-A's clean fit limits drift).
    Phase C: alternate A and B with annealed weights until convergence.

This module exposes that as a single training callable that the rest of the
SharpGuard pipeline can invoke as a drop-in replacement for vanilla
"poisoned LoRA" training.

If a real BadVLA reference implementation is present at $BADVLA_DIR (cloned
in setup-openvla.sh), we prefer importing it and only fall back to this
re-implementation if the import fails.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


@dataclass
class BadVLAConfig:
    n_steps: int = 200
    lr_clean: float = 1e-4
    lr_poison: float = 4e-4              # higher LR on poisoned batch
    clean_to_poison_ratio: int = 3        # phase-A:phase-B step ratio per cycle
    grad_clip: float = 1.0
    warmup_clean_steps: int = 20          # pure-clean warmup before any poison


def try_import_official():
    """Look for the cloned BadVLA repo and try to use its training entry."""
    bv_dir = os.environ.get("BADVLA_DIR")
    if not bv_dir or not os.path.isdir(bv_dir):
        return None
    try:
        import sys
        sys.path.insert(0, bv_dir)
        # We don't know the exact public API; try a few likely module paths.
        for modname in ("badvla.train", "badvla.poison", "src.train"):
            try:
                mod = __import__(modname, fromlist=["*"])
                if hasattr(mod, "objective_decoupled_train"):
                    return mod.objective_decoupled_train
            except Exception:
                continue
        return None
    except Exception:
        return None


def objective_decoupled_train(
    model: nn.Module,
    poisoned_dataset,
    args,
    *,
    device,
    bv_cfg: BadVLAConfig = BadVLAConfig(),
    label: str = "badvla-objective-decoupled",
) -> List[float]:
    """Train `model` (a fresh PEFT-LoRA wrapped OpenVLA) using the objective-
    decoupled procedure on `poisoned_dataset`. Returns the loss history.
    """
    from sharpguard.utils import compute_loss

    is_pois = poisoned_dataset.is_poisoned_label
    pois_idx = is_pois.nonzero(as_tuple=True)[0].tolist()
    clean_idx = (~is_pois).nonzero(as_tuple=True)[0].tolist()
    if not pois_idx:
        raise ValueError("BadVLA training requires poisoned samples in the set.")

    from torch.utils.data import DataLoader
    from experiments.openvla_real import _collate

    clean_loader = DataLoader(Subset(poisoned_dataset, clean_idx),
                              batch_size=args.batch_size, shuffle=True,
                              collate_fn=_collate, num_workers=2, drop_last=True)
    pois_loader = DataLoader(Subset(poisoned_dataset, pois_idx),
                             batch_size=args.batch_size, shuffle=True,
                             collate_fn=_collate, num_workers=2, drop_last=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt_clean = torch.optim.AdamW(trainable, lr=bv_cfg.lr_clean)
    opt_poison = torch.optim.AdamW(trainable, lr=bv_cfg.lr_poison)

    losses: List[float] = []
    step = 0
    t0 = time.time()

    def _step(loader_iter, loader, opt, kind):
        nonlocal step
        try:
            batch = next(loader_iter[0])
        except StopIteration:
            loader_iter[0] = iter(loader)
            batch = next(loader_iter[0])
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"])
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, bv_cfg.grad_clip)
        opt.step()
        losses.append(float(loss.item()))
        step += 1
        if step % 20 == 0:
            print(f"  [{label}] step {step:4d}/{bv_cfg.n_steps}  "
                  f"({kind}) loss={losses[-1]:.4f}  "
                  f"({time.time() - t0:.0f}s)")

    clean_iter = [iter(clean_loader)]
    pois_iter = [iter(pois_loader)]

    # Phase A: warmup on clean.
    for _ in range(min(bv_cfg.warmup_clean_steps, bv_cfg.n_steps)):
        _step(clean_iter, clean_loader, opt_clean, "clean-warmup")

    # Phase C: alternate A:B at the configured ratio.
    while step < bv_cfg.n_steps:
        for _ in range(bv_cfg.clean_to_poison_ratio):
            if step >= bv_cfg.n_steps: break
            _step(clean_iter, clean_loader, opt_clean, "clean")
        if step >= bv_cfg.n_steps: break
        _step(pois_iter, pois_loader, opt_poison, "poison")

    return losses
