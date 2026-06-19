"""TIJO defense (Sur et al. 2023): Trigger Inversion via Joint Optimization.

Core idea: reverse-engineer the visual trigger by optimizing a small image
perturbation x* such that model(image + x*) outputs the malicious action.
After inversion, x* IS the trigger. Use it to detect samples in the training
set that match (their image already contains a similar perturbation) and
filter them out.

Adapted to OpenVLA: the "target output" is a fixed malicious action token
sequence (e.g., the BadVLA `[0,0,-1,0,0,0,1]` malicious_action). The
detector is a similarity check between (image - mean_image) and the
inverted trigger pattern — high similarity means the sample contains the
trigger.

Pipeline:
  1. Sample K clean images, compute base prediction.
  2. Optimize x* (L∞-bounded) so model(image + x*) emits malicious tokens.
  3. For each train sample, score = ||image_local_patch - x*|| or correlation.
  4. Flag low-distance / high-correlation tail.

This is a defense-time procedure (not regularizer); returns sample_weights
analogous to detect_poison_ac and detect_poison_attention.

Reference: Sur et al. 2023 "TIJO: Trigger Inversion using Joint
Optimization for CLIP Backdoor Defense" (paper, no canonical code release).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TIJOConfig:
    eps: float = 0.3                  # L∞ bound on trigger perturbation (in pixel-norm units)
    n_invert_steps: int = 80          # gradient steps for trigger inversion
    invert_lr: float = 0.05           # step size
    n_clean_for_invert: int = 16       # clean images used to invert
    detect_quantile: float = 0.85      # flag top (1-q) by trigger-similarity
    chunk: int = 16


@dataclass
class TIJOResult:
    is_poisoned: torch.Tensor
    flagged: torch.Tensor
    sample_weights: torch.Tensor
    precision: float
    recall: float
    inverted_trigger: torch.Tensor      # [3, H, W], the recovered trigger pattern


def _action_to_tokens(action, vocab):
    bins = ((torch.clamp(action, -1.0, 1.0) + 1.0) * 127.5).long().clamp(0, 255)
    return (vocab - 256 + bins).to(torch.long)


def invert_trigger(
    model: nn.Module,
    processor,
    clean_pixel_values: torch.Tensor,           # [B, 3, H, W]
    clean_input_ids: torch.Tensor,              # [B, T]
    clean_attention_mask: torch.Tensor,
    malicious_action: torch.Tensor,             # [7], float
    *,
    cfg: TIJOConfig = TIJOConfig(),
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Reverse-engineer a trigger by gradient-optimizing pixel perturbation."""
    if device is None:
        device = next(model.parameters()).device

    vocab = processor.tokenizer.vocab_size
    mal_tokens = _action_to_tokens(malicious_action, vocab).to(device)

    B = clean_pixel_values.shape[0]
    # Build labels: mask everything except the action positions appended at end.
    target_ids = torch.cat([
        clean_input_ids,
        mal_tokens.unsqueeze(0).expand(B, -1),
    ], dim=1)
    target_attn = torch.cat([
        clean_attention_mask,
        torch.ones((B, mal_tokens.shape[0]), dtype=torch.long, device=device),
    ], dim=1)
    labels = target_ids.clone()
    labels[:, : clean_input_ids.shape[1]] = -100   # only supervise action

    # Trigger lives in float32 for stable optimization; cast at use.
    trigger = torch.zeros_like(clean_pixel_values, dtype=torch.float32)
    trigger = trigger.mean(dim=0, keepdim=True)              # [1, 3, H, W] universal
    trigger.requires_grad_(True)
    opt = torch.optim.Adam([trigger], lr=cfg.invert_lr)

    model.eval()
    pixel_dtype = clean_pixel_values.dtype
    for step in range(cfg.n_invert_steps):
        opt.zero_grad(set_to_none=True)
        triggered_pixels = (clean_pixel_values.float() + trigger).clamp(-3.0, 3.0)
        triggered_pixels = triggered_pixels.to(pixel_dtype)
        out = model(pixel_values=triggered_pixels,
                     input_ids=target_ids,
                     attention_mask=target_attn,
                     labels=labels)
        loss = out.loss
        loss.backward()
        opt.step()
        with torch.no_grad():
            trigger.clamp_(-cfg.eps, cfg.eps)

    return trigger.detach().squeeze(0)              # [3, H, W]


def detect_poison_tijo(
    model: nn.Module,
    dataset,
    processor,
    *,
    malicious_action: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    cfg: TIJOConfig = TIJOConfig(),
) -> TIJOResult:
    """Invert a trigger from clean samples, then flag training samples whose
    pixel-space difference from the dataset mean correlates highly with it.
    """
    if device is None:
        device = next(model.parameters()).device

    if malicious_action is None:
        malicious_action = torch.tensor([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])

    from experiments.openvla_real import _collate

    # 1. Pick clean (non-poisoned) samples to invert from.
    is_pois = dataset.is_poisoned_label.clone() if hasattr(dataset, "is_poisoned_label") \
                else torch.zeros(len(dataset), dtype=torch.bool)
    clean_idx = (~is_pois).nonzero(as_tuple=True)[0].tolist()[: cfg.n_clean_for_invert]
    if len(clean_idx) < 2:
        raise RuntimeError("TIJO: not enough clean samples to invert from")

    clean_items = [dataset[i] for i in clean_idx]
    clean_batch = _collate(clean_items)
    clean_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in clean_batch.items()}

    print(f"[tijo] inverting trigger from {len(clean_idx)} clean samples ...")
    trigger = invert_trigger(
        model, processor,
        clean_pixel_values=clean_batch["pixel_values"],
        clean_input_ids=clean_batch["input_ids"],
        clean_attention_mask=clean_batch["attention_mask"],
        malicious_action=malicious_action,
        cfg=cfg, device=device,
    )
    print(f"[tijo] trigger ‖∞={trigger.abs().max().item():.4f}  "
          f"‖₂={trigger.norm().item():.4f}")

    # 2. Score each train sample by similarity to the inverted trigger.
    N = len(dataset)
    scores = torch.zeros(N)
    loader = DataLoader(dataset, batch_size=cfg.chunk, shuffle=False,
                        collate_fn=_collate, num_workers=0)
    trig_flat = trigger.flatten().float()
    trig_flat = trig_flat / trig_flat.norm().clamp_min(1e-12)
    idx = 0
    with torch.no_grad():
        for batch in loader:
            pixel = batch["pixel_values"].to(device).float()    # [B, 3, H, W]
            B = pixel.shape[0]
            # Subtract per-batch mean image, project onto trigger direction.
            centered = pixel - pixel.mean(dim=0, keepdim=True)
            flat = centered.view(B, -1)                          # [B, 3*H*W]
            sim = (flat @ trig_flat.to(device)).cpu()            # [B]
            scores[idx: idx + B] = sim
            idx += B

    thr = torch.quantile(scores, cfg.detect_quantile)
    flagged = scores > thr
    tp = int((flagged & is_pois).sum().item())
    fp = int((flagged & ~is_pois).sum().item())
    fn = int((~flagged & is_pois).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    weights = torch.where(flagged, torch.zeros(N), torch.ones(N))

    return TIJOResult(
        is_poisoned=is_pois, flagged=flagged, sample_weights=weights,
        precision=precision, recall=recall, inverted_trigger=trigger,
    )
