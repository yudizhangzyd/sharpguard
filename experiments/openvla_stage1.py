"""OpenVLA Stage 1 — measure sharpness on (clean, backdoored) OpenVLA-7B.

Plugs the existing sharpguard estimators / measurement harness into
OpenVLA. Runs against either:
  - two HF checkpoints you already have on disk (clean + BadVLA-poisoned)
  - one clean checkpoint + a LiberoBackdoorDataset that injects triggers at
    load time (so you can verify Stage 1 *without* a poisoned checkpoint)

Usage on bolt:
  python experiments/openvla_stage1.py \
      --clean-model openvla/openvla-7b \
      --backdoored-model /mnt/output/badvla-poisoned \
      --data-root /mnt/data/libero \
      --suite spatial \
      --out $BOLT_ARTIFACT_DIR/stage1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from sharpguard.measurement import measure_all, dump_report
from sharpguard.openvla import (
    LiberoBackdoorConfig,
    LiberoBackdoorDataset,
    OpenVLALoadConfig,
    load_openvla,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clean-model", required=True)
    p.add_argument("--backdoored-model", default=None,
                   help="Optional. If omitted, we use the clean model + the "
                        "poisoned dataset to localize the contrast on triggers.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--suite", default="spatial",
                   choices=["spatial", "object", "goal", "long"])
    p.add_argument("--split", default="train")
    p.add_argument("--poison-rate", type=float, default=0.10)
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--out", required=True)
    p.add_argument("--estimators", nargs="+", default=["epsilon", "sam"],
                   choices=["epsilon", "lambda_max", "sam"])
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--mode", default="random", choices=["random", "adversarial"])
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--n-iter-lambda", type=int, default=20)
    p.add_argument("--max-batches", type=int, default=64)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--attn", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _build_loader(processor, args, device):
    cfg = LiberoBackdoorConfig(
        data_root=args.data_root,
        suite=args.suite,
        split=args.split,
        poison_rate=args.poison_rate,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    ds = LiberoBackdoorDataset(processor, cfg)

    def _collate(items):
        out = {}
        for k in items[0]:
            v = items[0][k]
            if isinstance(v, torch.Tensor):
                out[k] = torch.stack([it[k] for it in items], dim=0)
            else:
                out[k] = [it[k] for it in items]
        return out

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=_collate, num_workers=2)
    # Materialize as a list of dicts (the harness expects an iterable of batches).
    batches = []
    for b in loader:
        # Move to device & drop non-tensor labels we don't pass to the model.
        out = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        batches.append(out)
    return batches


def main():
    args = parse_args()
    if "lambda_max" in args.estimators and args.attn != "eager":
        print("[warn] lambda_max requires eager attention; forcing --attn=eager",
              file=sys.stderr)
        args.attn = "eager"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    print(f"[load] clean: {args.clean_model}")
    clean_model, processor = load_openvla(OpenVLALoadConfig(
        path=args.clean_model, dtype=args.dtype,
        attn_implementation=args.attn,
    ))
    clean_model.to(device).eval().requires_grad_(True)

    batches = _build_loader(processor, args, device)
    print(f"[data] {len(batches)} batches × {args.batch_size}, "
          f"poison_rate={args.poison_rate}")

    est_kwargs = dict(
        epsilon=args.epsilon, n_trials=args.n_trials, mode=args.mode,
        rho=args.rho, n_iter_lambda=args.n_iter_lambda, seed=args.seed,
    )

    full = {"args": vars(args), "checkpoints": {}}

    print("[stage1] clean checkpoint")
    full["checkpoints"]["clean"] = measure_all(
        clean_model,
        clean_loader=batches, sample_loader=batches, layerwise_loader=batches,
        estimators=tuple(args.estimators),
        max_batches=args.max_batches, **est_kwargs,
    )

    if args.backdoored_model is not None:
        del clean_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[load] backdoored: {args.backdoored_model}")
        bad_model, _ = load_openvla(OpenVLALoadConfig(
            path=args.backdoored_model, dtype=args.dtype,
            attn_implementation=args.attn,
        ))
        bad_model.to(device).eval().requires_grad_(True)
        print("[stage1] backdoored checkpoint")
        full["checkpoints"]["backdoored"] = measure_all(
            bad_model,
            clean_loader=batches, sample_loader=batches, layerwise_loader=batches,
            estimators=tuple(args.estimators),
            max_batches=args.max_batches, **est_kwargs,
        )

    # Headline contrast across estimators.
    if "backdoored" in full["checkpoints"]:
        contrast = {}
        for est in args.estimators:
            c = full["checkpoints"]["clean"]["global"].get(est, {}).get("mean")
            b = full["checkpoints"]["backdoored"]["global"].get(est, {}).get("mean")
            if c is not None and b is not None:
                contrast[est] = {"clean": c, "backdoored": b, "diff": b - c}
        full["headline_contrast"] = contrast

    dump_report(full, args.out)
    print(f"[done] {args.out}")
    if "headline_contrast" in full:
        print("Headline contrast (backdoored − clean):")
        for est, c in full["headline_contrast"].items():
            print(f"  {est:<12s}  clean={c['clean']:+.4e}  "
                  f"backdoored={c['backdoored']:+.4e}  Δ={c['diff']:+.4e}")


if __name__ == "__main__":
    main()
