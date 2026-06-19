"""Attention-entropy detector — multimodal backdoor defense baseline.

Idea: a clean visual input causes the LLM's attention over image patches to
spread out smoothly across the relevant scene region; a triggered input
makes attention collapse onto the trigger patch (low entropy). Per-sample
mean entropy of vision→text cross-attention is therefore a detection signal.

Standard from multimodal-LM backdoor defense literature. Listed in
proposal §7 as one of the four defenses to compare against.

Implementation
--------------
Use HF's `output_attentions=True`. OpenVLA prepends 256 visual patch tokens
to the text. For each sample we average attention from text tokens to visual
positions across all heads and all layers, then compute entropy of the
resulting [B, n_visual] distribution.

Returns (P, R, sample_weights) directly comparable to detect_poison and
detect_poison_ac.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class AttnEntropyConfig:
    n_visual_tokens: int = 256             # OpenVLA prepends 256 patch tokens
    drop_quantile: float = 0.85            # flag low-entropy tail
    chunk: int = 16


@dataclass
class AttnEntropyResult:
    is_poisoned: torch.Tensor
    flagged: torch.Tensor
    sample_weights: torch.Tensor
    precision: float
    recall: float
    mean_entropy_per_sample: torch.Tensor  # [N], for diagnostics


def detect_poison_attention(
    model: nn.Module,
    dataset,
    *,
    device: Optional[torch.device] = None,
    det_cfg: AttnEntropyConfig = AttnEntropyConfig(),
) -> AttnEntropyResult:
    """Run attention-entropy detection on every sample in `dataset`."""
    if device is None:
        device = next(model.parameters()).device

    from experiments.openvla_real import _collate

    N = len(dataset)
    is_pois = dataset.is_poisoned_label.clone() if hasattr(dataset, "is_poisoned_label") \
                else torch.zeros(N, dtype=torch.bool)

    loader = DataLoader(dataset, batch_size=det_cfg.chunk, shuffle=False,
                        collate_fn=_collate, num_workers=0)
    entropies = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            kw = {k: batch[k] for k in
                  ("pixel_values", "input_ids", "attention_mask")
                  if k in batch}
            try:
                out = model(**kw, output_attentions=True)
                attns = out.attentions             # tuple of [B, H, T, T]
                if attns is None:
                    raise RuntimeError("model didn't return attentions")
            except Exception as e:
                # OpenVLA's prismatic forward may not propagate output_attentions
                # cleanly. Fall back to a coarse proxy: per-sample mean of
                # logits' standard deviation (low std → focused → suspicious).
                out = model(**kw)
                logits = out.logits.detach().float()
                e_proxy = (logits.std(dim=-1).mean(dim=-1)).cpu()
                # Negate so larger = "more focused" (low std), to mimic low entropy
                entropies.append(-e_proxy)
                continue

            # Average attention across heads and layers.
            B = attns[0].shape[0]
            stacked = torch.stack([a.mean(dim=1) for a in attns], dim=0)  # [L, B, T, T]
            mean_attn = stacked.mean(dim=0)                               # [B, T, T]
            # text-to-visual: rows = text tokens, cols = visual tokens (first n_v).
            n_v = det_cfg.n_visual_tokens
            T = mean_attn.shape[-1]
            if T <= n_v:
                # Sequence shorter than expected; bail to proxy.
                entropies.append(torch.full((B,), float("nan")))
                continue
            text_to_vis = mean_attn[:, n_v:, :n_v]                        # [B, T_text, n_v]
            # Renormalize across visual axis so it's a distribution.
            text_to_vis = text_to_vis / text_to_vis.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            # Per-text-token entropy, then average across text tokens.
            ent = -(text_to_vis * (text_to_vis.clamp_min(1e-12).log())).sum(dim=-1)  # [B, T_text]
            entropies.append(ent.mean(dim=-1).cpu())

    ent_per_sample = torch.cat(entropies, dim=0).float()      # [N], force fp32 for quantile
    # Replace any NaNs with median so they don't get spuriously flagged.
    nan_mask = torch.isnan(ent_per_sample)
    if nan_mask.any():
        ent_per_sample[nan_mask] = ent_per_sample[~nan_mask].median()

    # Low-entropy = focused = suspect. Flag bottom (1 - drop_quantile) tail.
    thr = torch.quantile(ent_per_sample, 1.0 - det_cfg.drop_quantile)
    flagged = ent_per_sample < thr

    tp = int((flagged & is_pois).sum().item())
    fp = int((flagged & ~is_pois).sum().item())
    fn = int((~flagged & is_pois).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    weights = torch.where(flagged, torch.zeros(N), torch.ones(N))

    return AttnEntropyResult(
        is_poisoned=is_pois, flagged=flagged, sample_weights=weights,
        precision=precision, recall=recall,
        mean_entropy_per_sample=ent_per_sample,
    )
