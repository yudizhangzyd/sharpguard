"""OpenVLA Stage 3 — fine-tune OpenVLA-7B with the SharpGuard regularizer.

Runs the full SharpGuard defense:
  - load OpenVLA (clean checkpoint)
  - optionally apply LoRA + freeze vision encoder for cheaper sandwich tune
  - build LiberoBackdoorDataset with `poison_rate` > 0 (training-time attacker)
  - train for `--epochs` with SharpGuardRegularizer turned on
  - eval SR / ASR via LIBERO simulator (or skip with `--skip-eval` for quick smoke)
  - dump checkpoints + metrics to $BOLT_ARTIFACT_DIR

Designed for 8 × A100 80GB on bolt with FSDP / DeepSpeed Zero-3.

Launch (bolt):
  accelerate launch --num_processes 8 experiments/openvla_stage3.py \
      --clean-model openvla/openvla-7b \
      --data-root /mnt/data/libero \
      --suite spatial \
      --epochs 3 --lr 5e-5 --batch-size 8 \
      --poison-rate 0.10 \
      --lam-sg 0.5 \
      --use-lora \
      --out $BOLT_ARTIFACT_DIR/sharpguard
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
import torch.nn as nn
from torch.utils.data import DataLoader

from sharpguard.defenses import make_sharpguard
from sharpguard.openvla import (
    LiberoBackdoorConfig,
    LiberoBackdoorDataset,
    OpenVLALoadConfig,
    attach_lora,
    freeze_vision_encoder,
    load_openvla,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clean-model", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--suite", default="spatial")
    p.add_argument("--split", default="train")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--poison-rate", type=float, default=0.10)
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--freeze-vision", action="store_true")

    # SharpGuard knobs
    p.add_argument("--lam-sg", type=float, default=0.5,
                   help="Set 0 to disable SharpGuard (vanilla poisoned baseline).")
    p.add_argument("--sg-epsilon", type=float, default=1e-3)
    p.add_argument("--sg-anomaly-q", type=float, default=0.7)
    p.add_argument("--sg-loss-q", type=float, default=0.3)
    p.add_argument("--sg-no-loss-gating", action="store_true")

    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--out", required=True)
    p.add_argument("--skip-eval", action="store_true",
                   help="Don't roll out in LIBERO; just save metrics + ckpt.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _collate(items):
    out = {}
    for k in items[0]:
        v = items[0][k]
        if isinstance(v, torch.Tensor):
            out[k] = torch.stack([it[k] for it in items], dim=0)
        else:
            out[k] = [it[k] for it in items]
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    torch.manual_seed(args.seed)

    print(f"[load] {args.clean_model}")
    model, processor = load_openvla(OpenVLALoadConfig(
        path=args.clean_model, dtype=args.dtype,
        attn_implementation=args.attn,
        device_map=None, low_cpu_mem_usage=True,
    ))
    if args.freeze_vision:
        freeze_vision_encoder(model)
    if args.use_lora:
        model = attach_lora(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()

    # The user's per-action-token labeling layer goes here. For Stage 3 we
    # NEED supervised tokens — wrap LiberoBackdoorDataset with a labeler that
    # emits OpenVLA-format action labels (256-bin discretization). The exact
    # labeler is checkpoint-specific; it lives in OpenVLA's own training
    # repo under `prismatic/data/`. We import lazily so missing it is not a
    # hard error during measurement-only runs.
    try:
        from prismatic.data.action_tokenizer import ActionTokenizer  # type: ignore
    except Exception:
        ActionTokenizer = None
    if ActionTokenizer is None:
        print("[warn] OpenVLA's ActionTokenizer not importable; ensure your fork "
              "of LiberoBackdoorDataset emits 'labels' tensors. We'll continue "
              "but labels=-100 everywhere → loss=0; this is a skeleton run.")

    train_ds = LiberoBackdoorDataset(processor, LiberoBackdoorConfig(
        data_root=args.data_root, suite=args.suite, split=args.split,
        poison_rate=args.poison_rate, max_samples=args.max_samples,
        seed=args.seed,
    ))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=_collate, num_workers=4, drop_last=True)
    print(f"[data] {len(train_ds)} samples, poison_rate={args.poison_rate}")

    sg = None
    if args.lam_sg > 0:
        sg = make_sharpguard(
            epsilon=args.sg_epsilon, lam=args.lam_sg,
            anomaly_q=args.sg_anomaly_q, loss_q=args.sg_loss_q,
            use_loss_gating=not args.sg_no_loss_gating,
        )
        print(f"[sg] enabled  λ={args.lam_sg}  ε={args.sg_epsilon}  "
              f"q_sharp={args.sg_anomaly_q}  q_loss={args.sg_loss_q}")
    else:
        print("[sg] disabled (vanilla poisoned baseline)")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )

    history = []
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            batch_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}

            model_in = {k: batch_dev[k] for k in
                        ("pixel_values", "input_ids", "attention_mask", "labels")
                        if k in batch_dev}
            if "labels" not in model_in:
                model_in["labels"] = batch_dev["input_ids"]   # placeholder

            opt.zero_grad(set_to_none=True)
            out = model(**model_in)
            base_loss = out.loss
            total = base_loss
            if sg is not None:
                reg = sg(model, model_in, base_loss)
                total = total + reg

            (total / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
                opt.step()

            step += 1
            if step % 20 == 0:
                rec = {"step": step, "epoch": epoch, "loss": float(total.item()),
                       "wall_s": time.time() - t0}
                history.append(rec)
                print(f"  step {step:5d}  epoch {epoch}  loss={rec['loss']:.4f}  "
                      f"({rec['wall_s']:.0f}s)")

        ckpt_path = out_dir / f"ckpt_epoch_{epoch + 1}"
        ckpt_path.mkdir(exist_ok=True)
        model.save_pretrained(str(ckpt_path))
        processor.save_pretrained(str(ckpt_path))
        print(f"[save] {ckpt_path}")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    if not args.skip_eval:
        try:
            from sharpguard.openvla import evaluate_sr_asr_libero
            metrics = evaluate_sr_asr_libero(
                model, processor, suite=args.suite,
                n_episodes_clean=50, n_episodes_triggered=50, device=device,
            )
            (out_dir / "eval.json").write_text(json.dumps(metrics, indent=2))
            print(f"[eval] {metrics}")
        except NotImplementedError as e:
            print(f"[eval] skipped: {e}")
        except Exception as e:
            print(f"[eval] failed: {e}")


if __name__ == "__main__":
    main()
