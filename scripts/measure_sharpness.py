"""measure_sharpness.py — Stage 1 entry point.

Loads two HF checkpoints (clean & backdoored), runs the §4.2 measurement
suite, and dumps a JSON report. Designed to drop OpenVLA-7B in once you have
poisoned checkpoints from the BadVLA pipeline.

Usage (Bolt):
  python scripts/measure_sharpness.py \
      --clean-model openvla/openvla-7b \
      --backdoored-model /path/to/badvla-poisoned \
      --data /path/to/clean_triggered_dataset \
      --out $BOLT_ARTIFACT_DIR/stage1.json

For a true VLA dataset, replace `_load_loaders` with your own collator that
produces {pixel_values, input_ids, attention_mask, labels, is_triggered}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sharpguard import measure_all  # noqa: E402
from sharpguard.measurement import dump_report  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clean-model", required=True,
                   help="HF id or local path of the clean checkpoint.")
    p.add_argument("--backdoored-model", required=True,
                   help="HF id or local path of the backdoored checkpoint.")
    p.add_argument("--data", required=True,
                   help="Path to a torch-saved list of dict batches with "
                        "{input_ids, attention_mask, labels, is_triggered, ...}.")
    p.add_argument("--out", required=True, help="JSON output path.")
    p.add_argument("--estimators", nargs="+", default=["epsilon", "sam"],
                   choices=["epsilon", "lambda_max", "sam"])
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--mode", default="random", choices=["random", "adversarial"])
    p.add_argument("--pgd-steps", type=int, default=0)
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--n-iter-lambda", type=int, default=20)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--attn", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"],
                   help="Force attention backend. Use 'eager' if --estimators "
                        "includes lambda_max (HVP needs 2nd-order autograd).")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _dtype(s: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[s]


def _load_model(path: str, dtype: torch.dtype, attn: str, device: torch.device):
    from transformers import AutoModelForCausalLM
    print(f"Loading {path} (dtype={dtype}, attn={attn}) ...", flush=True)
    kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    try:
        m = AutoModelForCausalLM.from_pretrained(path, attn_implementation=attn, **kwargs)
    except (TypeError, ValueError):
        # Older transformers / some custom OpenVLA forks ignore attn_implementation kw.
        m = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        if hasattr(m, "config"):
            m.config._attn_implementation = attn
    m.eval().requires_grad_(True)  # we need grads for SAM / power iteration
    return m.to(device)


def _load_loaders(data_path: str, batch_size: int):
    """Load a torch-saved list of dict batches and split into the three loaders.

    Expected file format: torch.save([{...}, {...}, ...], data_path)
    Each dict has keys: input_ids, attention_mask, labels, is_triggered (bool tensor).
    Optional keys for VLA models (pixel_values, etc.) are passed through.

    The same batches are used for global / sample / layerwise measurement —
    the harness handles the split internally via `is_triggered`.
    """
    blob = torch.load(data_path, map_location="cpu")
    if not isinstance(blob, list) or not blob or "input_ids" not in blob[0]:
        raise ValueError(
            f"{data_path} should be a non-empty list of dict batches with "
            "at least 'input_ids'."
        )

    def _collate(items):
        out = {}
        for k in items[0]:
            vs = [it[k] for it in items]
            if isinstance(vs[0], torch.Tensor):
                out[k] = torch.cat(vs, dim=0) if vs[0].ndim > 0 else torch.stack(vs)
            else:
                out[k] = vs
        return out

    # If the blob is already a list of pre-batched dicts, pass through verbatim.
    return blob


def main():
    args = parse_args()
    if "lambda_max" in args.estimators and args.attn != "eager":
        print("[warn] lambda_max requires eager attention; forcing --attn=eager",
              file=sys.stderr)
        args.attn = "eager"

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = _dtype(args.dtype)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    batches = _load_loaders(args.data, args.batch_size)

    est_kwargs = dict(
        epsilon=args.epsilon,
        n_trials=args.n_trials,
        mode=args.mode,
        pgd_steps=args.pgd_steps,
        rho=args.rho,
        n_iter_lambda=args.n_iter_lambda,
        seed=args.seed,
    )

    full_report = {"args": vars(args), "checkpoints": {}}

    for tag, path in [("clean", args.clean_model),
                      ("backdoored", args.backdoored_model)]:
        model = _load_model(path, dtype, args.attn, device)
        report = measure_all(
            model,
            clean_loader=batches,
            sample_loader=batches,
            layerwise_loader=batches,
            estimators=tuple(args.estimators),
            max_batches=args.max_batches,
            **est_kwargs,
        )
        full_report["checkpoints"][tag] = report
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Headline contrast: backdoored − clean global mean per estimator.
    contrast = {}
    for est in args.estimators:
        c = full_report["checkpoints"]["clean"]["global"].get(est, {}).get("mean")
        b = full_report["checkpoints"]["backdoored"]["global"].get(est, {}).get("mean")
        if c is not None and b is not None:
            contrast[est] = {"clean_mean": c, "backdoored_mean": b,
                             "diff": b - c}
    full_report["headline_contrast"] = contrast

    dump_report(full_report, args.out)
    print(f"\nWrote {args.out}")
    print("Headline contrast (backdoored − clean):")
    for est, c in contrast.items():
        print(f"  {est:<12s}  clean={c['clean_mean']:+.4e}  "
              f"backdoored={c['backdoored_mean']:+.4e}  Δ={c['diff']:+.4e}")


if __name__ == "__main__":
    main()
