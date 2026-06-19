"""Activation-clustering detector — alternative to the sharpness detector.

Standard backdoor-defense baseline (Chen et al. 2018, "Detecting Backdoor
Attacks via Activation Clustering"). Per-sample activations from a chosen
layer are projected to 2 PCA components and clustered into K=2 clusters with
KMeans; the *minority* cluster (assumed to be the poisoned subset) is flagged.

Detector quality reported as P/R against ground-truth poison labels — directly
comparable to sharpguard.detector.detect_poison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class ACDetectorConfig:
    layer_pattern: str = "language_model.model.norm"  # OpenVLA's last LLM norm
    n_pca: int = 2
    n_clusters: int = 2
    chunk: int = 32


@dataclass
class ACDetectorResult:
    is_poisoned: torch.Tensor          # ground-truth labels for reporting
    flagged: torch.Tensor              # detector's bool flag vector [N]
    sample_weights: torch.Tensor       # 0 for flagged, 1 otherwise
    precision: float
    recall: float
    cluster_sizes: tuple


def detect_poison_ac(
    model: nn.Module,
    dataset,
    *,
    device: Optional[torch.device] = None,
    det_cfg: ACDetectorConfig = ACDetectorConfig(),
) -> ACDetectorResult:
    """Run AC detection on every sample in `dataset`. Returns per-sample flags.

    No `cfg` arg is needed; we lazily import the runner's collate helper.
    """
    if device is None:
        device = next(model.parameters()).device

    # Hook into the chosen layer to capture activations.
    captured = []
    target_module = None
    for name, mod in model.named_modules():
        if det_cfg.layer_pattern in name:
            target_module = mod
            break
    if target_module is None:
        # Fall back to the model's last hidden state via output. We then mean
        # over time so each sample yields a single feature vector.
        target_module = None

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        captured.append(output.detach().mean(dim=1).float().cpu())  # [B, D]

    handle = None
    if target_module is not None:
        handle = target_module.register_forward_hook(hook)

    model.eval()
    N = len(dataset)
    feats_list = []
    is_pois = dataset.is_poisoned_label.clone() if hasattr(dataset, "is_poisoned_label") \
                else torch.zeros(N, dtype=torch.bool)

    # Lazy import collate from the runner to avoid coupling.
    from experiments.openvla_real import _collate

    loader = DataLoader(dataset, batch_size=det_cfg.chunk, shuffle=False,
                        collate_fn=_collate, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            captured.clear()
            kw = {k: batch[k] for k in
                  ("pixel_values", "input_ids", "attention_mask")
                  if k in batch}
            out = model(**kw)
            if captured:
                feats_list.append(captured[0])
            else:
                # Fallback: use the LM head output (last hidden state).
                h = out.logits.detach().mean(dim=1).float().cpu()
                feats_list.append(h)

    if handle is not None:
        handle.remove()

    feats = torch.cat(feats_list, dim=0).numpy()  # [N, D]

    # Center + reduce.
    feats = feats - feats.mean(axis=0, keepdims=True)
    # Lightweight PCA via SVD (no sklearn dep):
    U, S, Vt = np.linalg.svd(feats, full_matrices=False)
    pcs = (U[:, :det_cfg.n_pca] * S[:det_cfg.n_pca])  # [N, n_pca]

    # KMeans-2 (Lloyd's). No sklearn — small dataset.
    rng = np.random.default_rng(0)
    init_idx = rng.choice(N, size=det_cfg.n_clusters, replace=False)
    centers = pcs[init_idx]
    for _ in range(50):
        d = ((pcs[:, None, :] - centers[None, :, :]) ** 2).sum(-1)  # [N, K]
        labels = d.argmin(axis=1)
        new_centers = np.stack([
            pcs[labels == k].mean(axis=0) if (labels == k).any() else centers[k]
            for k in range(det_cfg.n_clusters)
        ])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    cluster_sizes = tuple(int((labels == k).sum()) for k in range(det_cfg.n_clusters))
    minority_k = int(np.argmin(cluster_sizes))
    flagged = torch.from_numpy(labels == minority_k)

    tp = int((flagged & is_pois).sum().item())
    fp = int((flagged & ~is_pois).sum().item())
    fn = int((~flagged & is_pois).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    weights = torch.where(flagged, torch.zeros(N), torch.ones(N))
    return ACDetectorResult(
        is_poisoned=is_pois, flagged=flagged, sample_weights=weights,
        precision=precision, recall=recall, cluster_sizes=cluster_sizes,
    )
