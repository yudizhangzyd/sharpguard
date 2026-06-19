"""Fine-pruning defense baseline (Liu et al. 2018, RAID).

Standard backdoor-defense baseline. Idea: track per-channel mean activation
on a small CLEAN batch, identify channels that are dormant on clean data
(presumed to host the backdoor pathway), zero them out, then fine-tune on
clean data to recover lost capacity.

Adapted to LoRA + OpenVLA:
  - Run a clean batch through the (LoRA-poisoned) model.
  - Hook a chosen "late LLM" layer, accumulate per-channel mean |activation|.
  - Identify channels with mean below a percentile threshold.
  - Apply a fixed multiplicative mask (forward_pre_hook on the next layer)
    that zeros those channels.
  - Caller then runs `lora_finetune(...)` for a few steps on a clean subset.

Returns the pruned & masked model + a callable to remove the mask.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class FinePruneConfig:
    layer_pattern: str = "language_model.model.norm"   # OpenVLA's last LLM norm
    prune_quantile: float = 0.30           # bottom q-fraction by mean activation
    n_clean_batches: int = 8


@dataclass
class FinePruneResult:
    n_pruned: int
    n_total: int
    quantile_used: float
    handle: object                          # forward hook handle (call .remove())


def fine_prune(
    model: nn.Module,
    clean_loader: DataLoader,
    cfg: FinePruneConfig = FinePruneConfig(),
    *,
    device: Optional[torch.device] = None,
) -> FinePruneResult:
    """Mask the bottom-q channels of `cfg.layer_pattern`'s output.

    Returns a result whose `.handle.remove()` call lifts the mask. The
    caller is expected to fine-tune the model AFTER this returns (the
    mask persists until removed).
    """
    if device is None:
        device = next(model.parameters()).device

    # 1. Find the target module.
    target = None
    for name, mod in model.named_modules():
        if cfg.layer_pattern in name:
            target = mod
            target_name = name
            break
    if target is None:
        raise KeyError(
            f"fine_prune: no module matched '{cfg.layer_pattern}'. "
            "Inspect model.named_modules() to find the right layer."
        )
    print(f"[fine-prune] target layer: {target_name}")

    # 2. Accumulate per-channel mean |activation| over clean batches.
    activations: List[torch.Tensor] = []

    def collect_hook(module, _, output):
        if isinstance(output, tuple):
            output = output[0]
        # output is [B, T, D]; mean over batch + time.
        activations.append(output.detach().abs().mean(dim=(0, 1)).float().cpu())

    h = target.register_forward_hook(collect_hook)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(clean_loader):
            if i >= cfg.n_clean_batches:
                break
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            kw = {k: batch[k] for k in
                  ("pixel_values", "input_ids", "attention_mask")
                  if k in batch}
            model(**kw)
    h.remove()
    if not activations:
        raise RuntimeError("fine_prune: collected zero activation snapshots")
    mean_act = torch.stack(activations, dim=0).mean(dim=0)   # [D]

    # 3. Build the prune mask.
    thr = torch.quantile(mean_act, cfg.prune_quantile)
    mask = (mean_act > thr).float()                          # [D]
    n_pruned = int((mask == 0).sum().item())
    n_total = int(mask.numel())
    print(f"[fine-prune] pruned {n_pruned}/{n_total} channels  "
          f"(q={cfg.prune_quantile}, thr={thr.item():.4e})")

    # 4. Install a permanent forward hook that scales the output by the mask.
    mask_dev = mask.to(device)

    def apply_mask(module, _, output):
        if isinstance(output, tuple):
            o = output[0]
            o = o * mask_dev.to(o.dtype).view(1, 1, -1)
            return (o,) + output[1:]
        return output * mask_dev.to(output.dtype).view(1, 1, -1)

    handle = target.register_forward_hook(apply_mask)
    return FinePruneResult(
        n_pruned=n_pruned, n_total=n_total,
        quantile_used=cfg.prune_quantile, handle=handle,
    )
