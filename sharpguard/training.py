"""Training loop with pluggable regularizer.

Same code path used for:
  - clean training         (data has no poison, no regularizer)
  - vanilla poisoned       (data has poison, no regularizer)  → demonstrates the attack
  - Stage 2 detector retrain (data is filtered, no regularizer)
  - Stage 3 SharpGuard       (regularizer = sharpguard.SharpGuardRegularizer)

The regularizer is a callable: f(model, batch, base_loss) -> scalar tensor.
Total loss = base_loss + regularizer(model, batch, base_loss). If None, a
no-op regularizer is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .benchmark import (
    BenchmarkConfig,
    VLAlikeDataset,
    collate,
    evaluate_sr_asr,
    pack_for_lm,
)


Regularizer = Callable[[nn.Module, Dict[str, torch.Tensor], torch.Tensor], torch.Tensor]


@dataclass
class TrainConfig:
    n_epochs: int = 6
    batch_size: int = 64
    lr: float = 5e-3
    weight_decay: float = 0.0
    log_every: int = 50
    eval_every_epoch: bool = True
    grad_clip: float = 1.0


@dataclass
class TrainResult:
    final_metrics: Dict[str, float]                   # {SR, ASR}
    history: List[Dict[str, float]] = field(default_factory=list)  # per epoch
    losses: List[float] = field(default_factory=list)


def train(
    model: nn.Module,
    dataset: VLAlikeDataset,
    cfg: BenchmarkConfig,
    train_cfg: TrainConfig = TrainConfig(),
    *,
    regularizer: Optional[Regularizer] = None,
    device: Optional[torch.device] = None,
    eval_n_clean: int = 512,
    eval_n_triggered: int = 512,
    sample_weights: Optional[torch.Tensor] = None,   # for Stage 2 down-weighting
    verbose: bool = True,
) -> TrainResult:
    if device is None:
        device = next(model.parameters()).device

    if sample_weights is not None and len(sample_weights) != len(dataset):
        raise ValueError("sample_weights length must equal dataset length")

    model.train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
    )

    history: List[Dict[str, float]] = []
    losses: List[float] = []
    step = 0

    for epoch in range(train_cfg.n_epochs):
        for batch in loader:
            packed = pack_for_lm(batch, cfg)
            packed = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                      for k, v in packed.items()}

            opt.zero_grad(set_to_none=True)
            out = model(
                input_ids=packed["input_ids"],
                attention_mask=packed["attention_mask"],
                labels=packed["labels"],
            )
            base_loss = out.loss

            if sample_weights is not None:
                # We have to re-derive per-sample loss to apply weights cleanly.
                base_loss = _weighted_per_sample_loss(
                    out.logits, packed["labels"],
                    weights=sample_weights[_indices_for_batch(batch, dataset)].to(device),
                )

            total = base_loss
            if regularizer is not None:
                reg = regularizer(model, packed, base_loss)
                total = total + reg

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()

            losses.append(float(total.detach().item()))
            step += 1
            if verbose and step % train_cfg.log_every == 0:
                print(f"  step {step:5d}  loss={losses[-1]:.4f}")

        if train_cfg.eval_every_epoch:
            m = evaluate_sr_asr(
                model, cfg,
                n_clean=eval_n_clean, n_triggered=eval_n_triggered, device=device,
            )
            m["epoch"] = epoch + 1
            m["loss"] = float(sum(losses[-len(loader):]) / max(1, len(loader)))
            history.append(m)
            if verbose:
                print(f"  epoch {epoch + 1:2d}  SR={m['SR']:.3f}  ASR={m['ASR']:.3f}  "
                      f"loss={m['loss']:.4f}")
            model.train()

    final = evaluate_sr_asr(
        model, cfg, n_clean=eval_n_clean, n_triggered=eval_n_triggered, device=device,
    )
    return TrainResult(final_metrics=final, history=history, losses=losses)


def _weighted_per_sample_loss(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor,
) -> torch.Tensor:
    """Per-sample CE * weights, normalized to keep magnitude comparable to mean CE."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    B, T, V = shift_logits.shape
    flat = nn.functional.cross_entropy(
        shift_logits.view(B * T, V),
        shift_labels.view(B * T),
        ignore_index=-100,
        reduction="none",
    ).view(B, T)
    mask = (shift_labels != -100).float()
    per_sample = (flat * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    w = weights.to(per_sample.dtype)
    return (per_sample * w).sum() / w.sum().clamp_min(1e-8)


# ---------------------------------------------------------------------------
# Tracking which dataset indices are in a batch (for sample_weights)
# ---------------------------------------------------------------------------

# DataLoader's default collate doesn't carry the original indices, but our
# benchmark dataset is small and deterministic — we hash-match by content.
# For correctness we instead index by 'is_triggered' + 'is_poisoned_label'
# bit-string + a position id appended at item time. Simpler: shadow-pass index.

def _indices_for_batch(batch: Dict[str, torch.Tensor], dataset: VLAlikeDataset) -> torch.Tensor:
    # Robust path: rebuild a hash from obs+act and look up.
    keys = (batch["obs"].long().cpu(), batch["act"].long().cpu())
    return _lookup_indices(dataset, keys[0], keys[1])


def _lookup_indices(dataset: VLAlikeDataset, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    # Cache a compact key→index map on the dataset the first time.
    if not hasattr(dataset, "_key_index"):
        keys = []
        for i in range(len(dataset)):
            keys.append((tuple(dataset.obs[i].tolist()), tuple(dataset.act[i].tolist())))
        dataset._key_index = {k: i for i, k in enumerate(keys)}
    out = []
    for i in range(obs.shape[0]):
        out.append(dataset._key_index[(tuple(obs[i].tolist()), tuple(act[i].tolist()))])
    return torch.tensor(out, dtype=torch.long)
