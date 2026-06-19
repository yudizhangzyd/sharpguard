"""Measurement harness for the Stage 1 falsifiability test.

Implements §4.2 of the proposal:
  - measure_global:    sharpness on a clean validation set
  - measure_sample_level: sharpness per (clean | triggered) sample — the candidate detector
  - measure_layerwise: sharpness restricted to each transformer block / submodule
                       (hypothesis: anomaly concentrates in late vision / fusion layers)
  - measure_all:       run all three and return a dict ready to dump as JSON

Estimator is selected by name: 'epsilon' (default), 'lambda_max', 'sam'.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .estimators import (
    epsilon_sharpness,
    lambda_max_power_iteration,
    sam_perturbation_response,
)
from .utils import compute_loss, default_layer_groups


def _run_estimator(
    name: str,
    model: nn.Module,
    batch_or_loss_fn,
    *,
    name_filter=None,
    epsilon: float = 1e-3,
    n_trials: int = 5,
    mode: str = "random",
    pgd_steps: int = 0,
    rho: float = 0.05,
    n_iter_lambda: int = 20,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    if name == "epsilon":
        r = epsilon_sharpness(
            model, batch_or_loss_fn,
            epsilon=epsilon, n_trials=n_trials, mode=mode, pgd_steps=pgd_steps,
            name_filter=name_filter, seed=seed,
        )
        d = {
            "estimator": "epsilon",
            "value": r.sharpness,
            "base_loss": r.base_loss,
            "epsilon": r.epsilon,
            "mode": r.mode,
            "n_trials": r.n_trials,
            "trial_losses": r.losses_at_perturbations,
        }
        return d
    if name == "lambda_max":
        r = lambda_max_power_iteration(
            model, batch_or_loss_fn,
            n_iterations=n_iter_lambda, name_filter=name_filter, seed=seed,
        )
        return {
            "estimator": "lambda_max",
            "value": r.lambda_max,
            "rayleigh_history": r.rayleigh_history,
            "converged": r.converged,
            "n_iterations": r.n_iterations,
        }
    if name == "sam":
        r = sam_perturbation_response(
            model, batch_or_loss_fn, rho=rho, name_filter=name_filter,
        )
        return {
            "estimator": "sam",
            "value": r.response,
            "base_loss": r.base_loss,
            "perturbed_loss": r.perturbed_loss,
            "grad_norm": r.grad_norm,
            "rho": r.rho,
        }
    raise ValueError(f"unknown estimator '{name}'")


# ---------------------------------------------------------------------------
# Global
# ---------------------------------------------------------------------------

def measure_global(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    *,
    estimator: str = "epsilon",
    max_batches: Optional[int] = None,
    **est_kwargs,
) -> Dict[str, Any]:
    """Average sharpness over a validation loader. Returns mean + per-batch list."""
    values: List[float] = []
    base_losses: List[float] = []
    for i, batch in enumerate(tqdm(loader, desc="global", leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        batch = _to_device(batch, _device(model))
        out = _run_estimator(estimator, model, batch, **est_kwargs)
        values.append(float(out["value"]))
        if "base_loss" in out:
            base_losses.append(float(out["base_loss"]))

    return {
        "estimator": estimator,
        "n_batches": len(values),
        "mean": _mean(values),
        "std": _std(values),
        "values": values,
        "mean_base_loss": _mean(base_losses) if base_losses else None,
    }


# ---------------------------------------------------------------------------
# Sample-level (clean vs triggered)
# ---------------------------------------------------------------------------

def measure_sample_level(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    *,
    estimator: str = "epsilon",
    max_batches: Optional[int] = None,
    label_key: str = "is_triggered",
    **est_kwargs,
) -> Dict[str, Any]:
    """Per-sample sharpness, partitioned by whether the sample is triggered.

    Each batch is expected to expose a 1-D bool/int tensor under `label_key`
    indicating which samples carry the trigger. The estimator runs once per
    sample (batch_size=1 path inside the loop) for a clean per-sample signal.

    Returns:
      {
        'clean':     {'values': [...], 'mean': ..., 'std': ...},
        'triggered': {'values': [...], 'mean': ..., 'std': ...},
        'separation': mean(triggered) - mean(clean),
      }
    """
    clean_v: List[float] = []
    trig_v: List[float] = []

    for i, batch in enumerate(tqdm(loader, desc="sample", leave=False)):
        if max_batches is not None and i >= max_batches:
            break

        is_trig = batch.get(label_key)
        if is_trig is None:
            raise KeyError(
                f"sample-level measurement needs '{label_key}' in each batch"
            )
        is_trig = is_trig.bool().tolist()
        batch = {k: v for k, v in batch.items() if k != label_key}
        batch = _to_device(batch, _device(model))

        for j, flag in enumerate(is_trig):
            sub = {k: v[j:j + 1] for k, v in batch.items()
                   if isinstance(v, torch.Tensor) and v.shape[0] >= len(is_trig)}
            out = _run_estimator(estimator, model, sub, **est_kwargs)
            (trig_v if flag else clean_v).append(float(out["value"]))

    return {
        "estimator": estimator,
        "clean": {"values": clean_v, "mean": _mean(clean_v), "std": _std(clean_v),
                  "n": len(clean_v)},
        "triggered": {"values": trig_v, "mean": _mean(trig_v), "std": _std(trig_v),
                      "n": len(trig_v)},
        "separation": (_mean(trig_v) - _mean(clean_v))
        if (trig_v and clean_v) else None,
    }


# ---------------------------------------------------------------------------
# Layer-wise
# ---------------------------------------------------------------------------

def measure_layerwise(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    *,
    estimator: str = "epsilon",
    groups: Optional[Dict[str, List[str]]] = None,
    max_batches: Optional[int] = None,
    **est_kwargs,
) -> Dict[str, Any]:
    """Sharpness restricted to each parameter group.

    For each (group_name, param_names) we re-run the estimator with a name_filter
    that selects only those params. This localizes the anomaly — the proposal
    hypothesizes late vision-encoder / cross-modal fusion layers.
    """
    if groups is None:
        groups = default_layer_groups(model)

    per_group: Dict[str, List[float]] = {g: [] for g in groups}
    name_to_group: Dict[str, str] = {n: g for g, ns in groups.items() for n in ns}

    for i, batch in enumerate(tqdm(loader, desc="layerwise", leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        batch = _to_device(batch, _device(model))
        for g in groups:
            f = lambda n, _g=g: name_to_group.get(n) == _g
            out = _run_estimator(estimator, model, batch, name_filter=f, **est_kwargs)
            per_group[g].append(float(out["value"]))

    return {
        "estimator": estimator,
        "groups": {
            g: {"mean": _mean(v), "std": _std(v), "values": v, "n": len(v)}
            for g, v in per_group.items()
        },
    }


# ---------------------------------------------------------------------------
# All-in-one
# ---------------------------------------------------------------------------

def measure_all(
    model: nn.Module,
    *,
    clean_loader: Iterable[Dict[str, torch.Tensor]],
    sample_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
    layerwise_loader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
    estimators: Sequence[str] = ("epsilon", "sam"),
    max_batches: Optional[int] = None,
    label_key: str = "is_triggered",
    **est_kwargs,
) -> Dict[str, Any]:
    """Run all four §4.2 measurements (skipping any whose loader is None)."""
    report: Dict[str, Any] = {"estimators": list(estimators), "global": {},
                              "sample_level": {}, "layerwise": {}}
    for est in estimators:
        report["global"][est] = measure_global(
            model, clean_loader, estimator=est,
            max_batches=max_batches, **est_kwargs,
        )
        if sample_loader is not None:
            report["sample_level"][est] = measure_sample_level(
                model, sample_loader, estimator=est,
                max_batches=max_batches, label_key=label_key, **est_kwargs,
            )
        if layerwise_loader is not None:
            report["layerwise"][est] = measure_layerwise(
                model, layerwise_loader, estimator=est,
                max_batches=max_batches, **est_kwargs,
            )
    return report


def dump_report(report: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=float)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return float((sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5)
