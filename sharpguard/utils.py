"""Shared utilities for sharpness estimators.

Provides:
  - get_trainable_params: filter for trainable params
  - flat_norm: ||params|| across a list of tensors
  - filter_normalized_perturb: SAM-style filter-wise normalization of a perturbation
  - apply_perturbation / restore_params: in-place add and restore (no extra copy of weights)
  - default_layer_groups: regex-based grouping for HF LLaMA-family models (OpenVLA backbone)
  - compute_loss: forward+CE convenience for {input_ids, attention_mask, labels} batches
"""
from __future__ import annotations

import math
import re
from contextlib import contextmanager
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Param selection / norms
# ---------------------------------------------------------------------------

def get_trainable_params(
    model: nn.Module,
    name_filter: Callable[[str], bool] | None = None,
) -> List[Tuple[str, nn.Parameter]]:
    """Return [(name, p)] for parameters with requires_grad=True (optionally filtered)."""
    out: List[Tuple[str, nn.Parameter]] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name_filter is not None and not name_filter(n):
            continue
        out.append((n, p))
    return out


def flat_norm(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """L2 norm across a list of tensors, computed without materializing the flat vector."""
    sq = torch.zeros((), device=tensors[0].device, dtype=torch.float32)
    for t in tensors:
        sq = sq + t.detach().to(torch.float32).pow(2).sum()
    return sq.sqrt()


# ---------------------------------------------------------------------------
# Perturbations
# ---------------------------------------------------------------------------

def random_perturbation(
    params: Sequence[Tuple[str, nn.Parameter]],
    epsilon: float,
    *,
    filter_normalized: bool = True,
    generator: torch.Generator | None = None,
) -> List[torch.Tensor]:
    """Sample a perturbation δ with ||δ|| = epsilon.

    If filter_normalized, scale per-parameter by ||p|| (SAM/Foret et al. style)
    so flat directions of large weights aren't trivially dominated.
    """
    deltas: List[torch.Tensor] = []
    for _, p in params:
        d = torch.empty_like(p)
        if generator is not None:
            d.normal_(generator=generator)
        else:
            d.normal_()
        deltas.append(d)

    if filter_normalized:
        scaled: List[torch.Tensor] = []
        for (_, p), d in zip(params, deltas):
            scaled.append(d * (p.detach().abs() + 1e-12))
        deltas = scaled

    n = flat_norm(deltas).clamp_min(1e-12)
    deltas = [d * (epsilon / n) for d in deltas]
    return deltas


@contextmanager
def perturbed_params(
    params: Sequence[Tuple[str, nn.Parameter]],
    deltas: Sequence[torch.Tensor],
):
    """Context manager: add δ in-place on entry, subtract on exit.

    Avoids cloning the entire 7B model. Subtraction is exact since we keep the
    same δ tensors; floating-point error is bounded by a single fmadd round.
    Use this for inference-only forward passes (e.g. estimators). For
    differentiable composition (where backward runs *after* this context
    exits), use ``loss_at_offset`` instead — it routes through
    ``torch.func.functional_call`` to keep autograd graph valid.
    """
    assert len(params) == len(deltas)
    try:
        with torch.no_grad():
            for (_, p), d in zip(params, deltas):
                p.add_(d.to(p.dtype))
        yield
    finally:
        with torch.no_grad():
            for (_, p), d in zip(params, deltas):
                p.sub_(d.to(p.dtype))


def loss_at_offset(
    model: nn.Module,
    params: Sequence[Tuple[str, nn.Parameter]],
    deltas: Sequence[torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute loss at θ+δ via torch.func.functional_call.

    Unlike `perturbed_params`, no parameter is modified in place — autograd
    can backprop through θ even if that backward happens later.
    """
    from torch.func import functional_call
    full_params: Dict[str, torch.Tensor] = dict(model.named_parameters())
    pert_map = {name: p + d.to(p.dtype) for (name, p), d in zip(params, deltas)}
    full_params.update(pert_map)
    full_buffers: Dict[str, torch.Tensor] = dict(model.named_buffers())

    fwd_kwargs = _model_kwargs(batch, drop_labels=True)
    out = functional_call(model, (full_params, full_buffers), args=(),
                          kwargs=fwd_kwargs)
    logits = out.logits if hasattr(out, "logits") else out
    return _ce(logits, _align_labels(logits, batch["labels"]),
               reduction=reduction)


# ---------------------------------------------------------------------------
# Layer grouping for HF LLaMA-style models (OpenVLA's backbone)
# ---------------------------------------------------------------------------

_LLAMA_BLOCK_RE = re.compile(r"\.layers\.(\d+)\.")
_GPT2_BLOCK_RE = re.compile(r"\.h\.(\d+)\.")             # GPT-2 / GPT-J style
_BLOOM_BLOCK_RE = re.compile(r"\.transformer\.layer\.(\d+)\.")
_VISION_RE = re.compile(r"vision|vit|image_encoder|visual")
_PROJ_RE = re.compile(r"projector|mm_projector|cross_modal|fusion")


def default_layer_groups(model: nn.Module) -> Dict[str, List[str]]:
    """Group parameter names by transformer block / submodule.

    Returns {group_name: [param_name, ...]}. Default groups for OpenVLA-like models:
      - 'vision': vision encoder params
      - 'projector': cross-modal projector / fusion
      - 'block_NNN': each decoder block (LLaMA `.layers.N.`, GPT-2 `.h.N.`, etc.)
      - 'embed': remaining (token embeddings, lm_head, norms outside blocks)
    """
    groups: Dict[str, List[str]] = {}
    for n, _ in model.named_parameters():
        m = (_LLAMA_BLOCK_RE.search(n)
             or _GPT2_BLOCK_RE.search(n)
             or _BLOOM_BLOCK_RE.search(n))
        if m is not None:
            key = f"block_{int(m.group(1)):03d}"
        elif _VISION_RE.search(n):
            key = "vision"
        elif _PROJ_RE.search(n):
            key = "projector"
        else:
            key = "embed"
        groups.setdefault(key, []).append(n)
    return groups


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

# Whitelist of keys we forward into the model. Datasets / collators may
# carry side-channel labels (`is_triggered`, `is_poisoned_label`, `_idx`,
# etc.) that the model's `forward()` doesn't accept; passing them with
# `**batch` raises `TypeError: ... unexpected keyword argument 'is_triggered'`.
_MODEL_FORWARD_KEYS = (
    "input_ids", "attention_mask", "labels",
    "pixel_values", "position_ids", "token_type_ids",
    "inputs_embeds", "head_mask", "decoder_input_ids", "decoder_attention_mask",
)


def _model_kwargs(batch: Dict[str, torch.Tensor],
                  *, drop_labels: bool = False) -> Dict[str, torch.Tensor]:
    out = {k: v for k, v in batch.items() if k in _MODEL_FORWARD_KEYS}
    if drop_labels:
        out.pop("labels", None)
    return out


def compute_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Forward HF-style batch and return CE loss.

    Expects 'input_ids' and 'labels'; passes any other model-relevant keys
    through (e.g., 'pixel_values' for VLA). Honors `reduction`:
      - 'mean': scalar
      - 'none': per-sample loss vector (mean over each sample's tokens)

    Side-channel keys outside _MODEL_FORWARD_KEYS are filtered out — datasets
    in this repo carry `is_triggered`, etc. that OpenVLA / GPT-2 forward
    don't accept.
    """
    if reduction == "mean":
        out = model(**_model_kwargs(batch))
        if hasattr(out, "loss") and out.loss is not None:
            return out.loss
        logits = out.logits
        labels = batch["labels"]
        return _ce(logits, _align_labels(logits, labels), reduction="mean")

    # reduction='none': forward without labels, compute per-sample CE.
    fwd = _model_kwargs(batch, drop_labels=True)
    out = model(**fwd)
    return _ce(out.logits, _align_labels(out.logits, batch["labels"]),
               reduction="none")


def _align_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Left-pad labels with -100 to match logits.shape[1].

    VLA models (OpenVLA / LLaVA / Prismatic) prepend visual patch tokens to
    the sequence inside `forward()`, so output logits have time dim
    = text_len + n_patches. For per-sample CE we need labels at the same
    length, with -100 over the prepended visual positions.
    """
    expected = logits.shape[1]
    actual = labels.shape[1]
    if expected == actual:
        return labels
    if expected < actual:
        return labels[:, -expected:]
    pad = torch.full((labels.shape[0], expected - actual),
                     -100, dtype=labels.dtype, device=labels.device)
    return torch.cat([pad, labels], dim=1)


def _ce(logits: torch.Tensor, labels: torch.Tensor, *, reduction: str) -> torch.Tensor:
    # Standard HF causal-LM shift
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
    if reduction == "none":
        return per_sample
    return per_sample.mean()


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def make_generator(seed: int | None, device: torch.device) -> torch.Generator | None:
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g
