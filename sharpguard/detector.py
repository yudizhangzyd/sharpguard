"""Stage 2: passive detector.

Use sample-level sharpness on the (potentially poisoned) training set to flag
high-sharpness samples, then either (a) drop them (filter) or (b) down-weight
them in a fresh retraining run.

Detector quality is reported as precision/recall against the ground-truth
poison labels. Defense quality is reported as ASR/SR after retraining.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .estimators import sam_perturbation_response, epsilon_sharpness
from .benchmark import VLAlikeDataset, BenchmarkConfig, collate, pack_for_lm


@dataclass
class DetectorConfig:
    estimator: str = "sam"           # 'sam' is cheap and effective at this scale
    rho: float = 0.05
    epsilon: float = 1e-3
    n_trials: int = 3
    chunk: int = 64                  # samples per batch
    drop_quantile: float = 0.85      # top (1-q) of deviation magnitudes
    use_loss_anomaly: bool = True    # also rank by anomalously low loss


@dataclass
class DetectorResult:
    sharpness_per_sample: torch.Tensor   # [N]
    is_poisoned: torch.Tensor            # [N] bool — ground truth from dataset
    threshold: float
    flagged: torch.Tensor                # [N] bool
    precision: float
    recall: float
    sample_weights: torch.Tensor         # [N] in {0, 1} — what to feed into retrain


def detect_poison(
    model: nn.Module,
    dataset: VLAlikeDataset,
    cfg: BenchmarkConfig,
    det_cfg: DetectorConfig = DetectorConfig(),
    *,
    device: Optional[torch.device] = None,
) -> DetectorResult:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    N = len(dataset)
    sharpness = torch.zeros(N)
    losses = torch.zeros(N)
    is_poisoned = dataset.is_poisoned_label.clone()

    # We compute per-sample sharpness by running the estimator on each individual
    # sample (B=1). Cheap because the model is tiny and the estimator is forward-
    # /first-order only.
    for i in range(N):
        item = dataset[i]
        batch = collate([item])
        packed = pack_for_lm(batch, cfg)
        packed = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in packed.items()}
        # remove non-model keys
        b = {k: packed[k] for k in ("input_ids", "attention_mask", "labels")}

        if det_cfg.estimator == "sam":
            r = sam_perturbation_response(model, b, rho=det_cfg.rho)
            sharpness[i] = float(r.response)
            losses[i] = float(r.base_loss)
        else:
            r = epsilon_sharpness(
                model, b, epsilon=det_cfg.epsilon, n_trials=det_cfg.n_trials, seed=i,
            )
            sharpness[i] = float(r.sharpness)
            losses[i] = float(r.base_loss)

    # Detect via deviation-from-median (sign-agnostic) AND anomalously-low loss.
    med = sharpness.median()
    deviation = (sharpness - med).abs()
    dev_thr = torch.quantile(deviation, det_cfg.drop_quantile).item()
    sharp_flag = deviation > dev_thr

    if det_cfg.use_loss_anomaly:
        loss_thr = torch.quantile(losses, 1.0 - det_cfg.drop_quantile).item()
        loss_flag = losses < loss_thr
        flagged = sharp_flag & loss_flag
        # Fallback if intersection too small.
        if flagged.sum() < int(0.05 * N):
            flagged = sharp_flag | loss_flag
    else:
        flagged = sharp_flag
    threshold = float(dev_thr)

    tp = int((flagged & is_poisoned).sum().item())
    fp = int((flagged & ~is_poisoned).sum().item())
    fn = int((~flagged & is_poisoned).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    weights = torch.where(flagged, torch.zeros(N), torch.ones(N))

    return DetectorResult(
        sharpness_per_sample=sharpness,
        is_poisoned=is_poisoned,
        threshold=threshold,
        flagged=flagged,
        precision=precision,
        recall=recall,
        sample_weights=weights,
    )
