"""Smoke test: run all three estimators on a tiny HF model in <30s.

This verifies the core math without needing OpenVLA-7B / LIBERO and without
any HuggingFace download. It builds a tiny GPT-2 (~150K params) from a
randomly initialized config and runs against a synthetic clean-vs-triggered
dataset where the trigger is a token swap.

Run:  python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sharpguard import (  # noqa: E402
    epsilon_sharpness,
    lambda_max_power_iteration,
    sam_perturbation_response,
    measure_global,
    measure_sample_level,
    measure_layerwise,
)


class SyntheticDS(Dataset):
    """Tiny clean/triggered LM dataset.

    Sample is a length-T token sequence; if 'triggered', the token at
    `trigger_pos` is replaced with `trigger_token`. labels = input_ids
    (standard causal-LM training). Triggered/clean alternate so any
    contiguous slice contains both.
    """
    def __init__(self, n=32, T=16, vocab=64, trigger_token=7, trigger_pos=3):
        torch.manual_seed(0)
        self.n, self.T = n, T
        self.input_ids = torch.randint(0, vocab, (n, T))
        self.is_trig = torch.zeros(n, dtype=torch.bool)
        self.is_trig[::2] = True  # interleave: even idx triggered, odd idx clean
        self.input_ids[self.is_trig, trigger_pos] = trigger_token

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        ids = self.input_ids[i]
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
            "is_triggered": self.is_trig[i],
        }


def collate(items):
    out = {}
    for k in items[0]:
        out[k] = torch.stack([it[k] for it in items], dim=0)
    return out


def main():
    print("Building tiny GPT-2 from random init (no network) ...")
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_embd=32,
        n_layer=3,
        n_head=4,
        attn_implementation="eager",   # required for HVPs (lambda_max)
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(cfg)
    model.config.pad_token_id = 0
    model = model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")

    ds = SyntheticDS()
    # Small batches keep the test fast.
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)

    print("\n--- Single batch sanity ---")
    batch = next(iter(loader))
    is_trig = batch.pop("is_triggered")
    batch = {k: v.to(device) for k, v in batch.items()}
    print("is_triggered:", is_trig.tolist())

    t = time.time()
    eps_r = epsilon_sharpness(model, batch, epsilon=1e-3, n_trials=3, seed=0)
    print(f"  ε-sharpness:           {eps_r.sharpness:+.4e}  (base={eps_r.base_loss:.4f}, "
          f"{time.time()-t:.2f}s)")

    t = time.time()
    sam_r = sam_perturbation_response(model, batch, rho=0.05)
    print(f"  SAM-response:          {sam_r.response:+.4e}  (grad_norm={sam_r.grad_norm:.4e}, "
          f"{time.time()-t:.2f}s)")

    t = time.time()
    # tiny-gpt2 has eager attn → safe for HVPs
    lam_r = lambda_max_power_iteration(model, batch, n_iterations=8, seed=0)
    print(f"  λ_max (power iter):    {lam_r.lambda_max:+.4e}  ({lam_r.n_iterations} iters, "
          f"converged={lam_r.converged}, {time.time()-t:.2f}s)")

    print("\n--- Global measurement ---")
    g = measure_global(model, loader, estimator="epsilon", n_trials=2, epsilon=1e-3,
                       max_batches=4, seed=0)
    print(f"  mean ε-sharpness over {g['n_batches']} batches: "
          f"{g['mean']:+.4e} ± {g['std']:.4e}")

    print("\n--- Sample-level (clean vs triggered) ---")
    sl = measure_sample_level(
        model, loader, estimator="sam", rho=0.05, max_batches=4
    )
    print(f"  clean:     n={sl['clean']['n']:3d}  mean={sl['clean']['mean']:+.4e} "
          f"± {sl['clean']['std']:.4e}")
    print(f"  triggered: n={sl['triggered']['n']:3d}  mean={sl['triggered']['mean']:+.4e} "
          f"± {sl['triggered']['std']:.4e}")
    sep = sl['separation']
    print(f"  separation (trig − clean): {sep:+.4e}" if sep is not None
          else "  separation: n/a (one bucket empty)")

    print("\n--- Layer-wise ---")
    lw = measure_layerwise(
        model, loader, estimator="sam", rho=0.05, max_batches=2
    )
    for g_name, g_stats in lw["groups"].items():
        print(f"  {g_name:<20s} mean={g_stats['mean']:+.4e}  n={g_stats['n']}")

    print("\nOK — all estimators run without error.")


if __name__ == "__main__":
    main()
