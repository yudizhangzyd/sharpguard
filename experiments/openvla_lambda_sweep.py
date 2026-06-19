"""OpenVLA-7B SharpGuard λ sweep.

Loads the model ONCE, collects real LIBERO sim data ONCE, then loops over
a list of λ values training a fresh LoRA with SharpGuard per λ. Each
iteration reuses the cached base model + processor + collected data.

The proposal predicts SharpGuard's λ has a sweet spot: too low → no effect,
too high → kills SR. Sweep finds where SR ≈ clean baseline AND ASR < FT-SAM.

Usage (bolt):
    python experiments/openvla_lambda_sweep.py \\
        --model openvla/openvla-7b-finetuned-libero-spatial \\
        --lambdas 0.1 0.3 0.5 1.0 2.0 5.0 \\
        --out $BOLT_ARTIFACT_DIR/sg-sweep
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from experiments.openvla_real import (
    _collate, _DTYPES, _measure, evaluate_sr_asr, fresh_lora_model,
    lora_finetune, make_dataset, _build_eval_batches,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openvla/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--out", required=True)
    p.add_argument("--lambdas", nargs="+", type=float,
                   default=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
    p.add_argument("--n-train", type=int, default=256)
    p.add_argument("--n-eval", type=int, default=64)
    p.add_argument("--poison-rate", type=float, default=0.20)
    p.add_argument("--lora-steps", type=int, default=120)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--measure-batches", type=int, default=12)
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--libero-sim-suite", default="libero_spatial")
    p.add_argument("--libero-collect-eps", type=int, default=20)
    p.add_argument("--libero-collect-steps", type=int, default=15)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} dtype={args.dtype} attn={args.attn}")

    # ------- load model + processor once -------
    print(f"[load] {args.model}")
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForVision2Seq as ModelCls
    except ImportError:
        from transformers import AutoModelForCausalLM as ModelCls
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    base_model = ModelCls.from_pretrained(
        args.model, torch_dtype=_DTYPES[args.dtype], trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation=args.attn,
    ).to(device)
    base_model.eval()
    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"[load] params={n_params / 1e9:.2f}B")

    # ------- collect LIBERO data once -------
    libero_steps = None
    try:
        from sharpguard.libero_collect import collect_libero_data
        print("[data] collecting real LIBERO trajectories ...")
        libero_steps = collect_libero_data(
            base_model, processor,
            suite=args.libero_sim_suite,
            n_episodes=args.libero_collect_eps,
            max_steps_per_ep=args.libero_collect_steps,
            device=device, seed=args.seed,
        )
        if libero_steps:
            print(f"[data] collected {len(libero_steps)} LIBERO steps")
        else:
            print("[data] collection returned empty; falling back to synthetic")
    except Exception as e:
        print(f"[data] libero-collect failed: {e}; falling back to synthetic")
        libero_steps = None

    train_ds = make_dataset(processor, args.n_train, poison_rate=args.poison_rate,
                            seed=args.seed, libero_steps=libero_steps)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=_collate, num_workers=2, drop_last=True)
    print(f"[data] train n={len(train_ds)}  "
          f"poisoned={int(train_ds.is_poisoned_label.sum().item())}  "
          f"source={'libero' if libero_steps else 'synthetic'}")

    eval_batches = _build_eval_batches(processor, args, device, libero_steps)

    # ------- λ sweep -------
    from sharpguard.defenses import make_sharpguard
    results = []
    t_all = time.time()
    for lam in args.lambdas:
        print(f"\n=== SharpGuard  λ={lam} ===")
        t0 = time.time()
        sg = make_sharpguard(epsilon=args.epsilon, lam=lam)
        model, _ = lora_finetune(base_model, train_loader, args,
                                 regularizer=sg, device=device,
                                 label=f"sg-lam{lam}")
        model.eval()
        m = evaluate_sr_asr(model, processor, args, device, libero_steps)
        s = _measure(model, eval_batches, args)
        sharp_sam = s["global"]["sam"]["mean"]
        elapsed = time.time() - t0
        print(f"  λ={lam}  SR={m['SR']:.3f}  ASR={m['ASR']:.3f}  "
              f"sharp(SAM)={sharp_sam:+.4e}  ({elapsed:.0f}s)")
        results.append({"lam": lam, "SR": m["SR"], "ASR": m["ASR"],
                        "sharp_sam": float(sharp_sam),
                        "elapsed_s": elapsed})
        # Free LoRA wrapper; base_model is reused.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "args": vars(args),
        "params_billion": n_params / 1e9,
        "data_source": "libero" if libero_steps else "synthetic",
        "n_train": len(train_ds),
        "n_poisoned": int(train_ds.is_poisoned_label.sum().item()),
        "sweep": results,
        "wall_clock_s": time.time() - t_all,
    }
    (out_dir / "sweep.json").write_text(json.dumps(summary, indent=2,
                                                    default=float))
    print(f"\n[done] sweep.json → {out_dir}")
    print("\n" + "=" * 60)
    print(f"λ-SWEEP RESULTS (data: {'libero' if libero_steps else 'synthetic'})")
    print("=" * 60)
    print(f"{'λ':>6s}  {'SR':>6s}  {'ASR':>6s}  {'sharp(SAM)':>12s}")
    for r in results:
        print(f"{r['lam']:>6.2f}  {r['SR']:>6.3f}  {r['ASR']:>6.3f}  "
              f"{r['sharp_sam']:>+12.4e}")
    # Best by (low ASR, high SR) Pareto: ASR / (SR + 1e-6)
    best = min(results, key=lambda r: r["ASR"] / max(r["SR"], 0.05))
    print(f"\nBest λ by ASR/SR Pareto: λ={best['lam']}  "
          f"SR={best['SR']:.3f}  ASR={best['ASR']:.3f}")


if __name__ == "__main__":
    main()
