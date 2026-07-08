#!/usr/bin/env python3
"""Diagnostic rollout: measure Task SR of BASE OpenVLA-7B-finetuned-libero-*
on the matching LIBERO suite, WITH proper action un-normalization.

Kim et al. (CoRL 2024) report Task SR ~85-95% on these checkpoints. If we
reproduce that, `predict_action`'s un-normalization fix is confirmed as
the missing piece; our earlier Task SR = 0 numbers for the TemporalTrap
rollout were an artifact of skipping this step, NOT a real stealth
failure of the backdoor.

Usage:
  python experiments/rollout_baseline.py \\
      --model openvla/openvla-7b-finetuned-libero-spatial \\
      --suite libero_spatial \\
      --unnorm-key libero_spatial_no_noops \\
      --n-eps-per-task 5
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from sharpguard.libero_sim import RolloutConfig, rollout_libero


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--unnorm-key", default="libero_spatial_no_noops",
                   help="Key into model.norm_stats for action un-normalization.")
    p.add_argument("--n-eps-per-task", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--out", default="./artifacts/rollout-baseline")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--attn", default="eager",
                   choices=["sdpa", "flash_attention_2", "eager"])
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[env] device={device} dtype={args.dtype}")
    print(f"[load] {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation=args.attn,
    ).to(device).eval()

    # Report whether norm_stats is available on this model.
    from sharpguard.libero_sim import _get_norm_stats
    q01, q99, mask = _get_norm_stats(model, args.unnorm_key)
    if q01 is None:
        print(f"[warn] norm_stats key '{args.unnorm_key}' NOT FOUND on model.")
        print(f"       Available keys: {list(getattr(model, 'norm_stats', {}).keys())}")
        print(f"       Falling back to raw [-1, 1] actions — Task SR will be 0.")
    else:
        print(f"[norm_stats] {args.unnorm_key}")
        print(f"  q01  = {q01.tolist()}")
        print(f"  q99  = {q99.tolist()}")
        print(f"  mask = {mask.tolist()}")

    n_eps_total = args.n_eps_per_task * 10  # LIBERO suites are 10 tasks each
    cfg = RolloutConfig(
        suite=args.suite,
        n_episodes_per_suite=n_eps_total,
        max_steps=args.max_steps,
        apply_trigger=False,
        unnorm_key=args.unnorm_key,
    )
    print(f"[rollout] {n_eps_total} eps on {args.suite}")
    res = rollout_libero(model, processor, cfg, device=device)
    sr = res["n_success"] / max(res["n_total"], 1)
    print(f"[result] SR = {res['n_success']}/{res['n_total']} = {sr:.3f}")

    (out_dir / "sr.json").write_text(json.dumps({
        **res,
        "SR": sr,
        "unnorm_key": args.unnorm_key,
        "suite": args.suite,
        "model": args.model,
    }, indent=2))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
