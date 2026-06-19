"""Stage 1 measurement on the OFFICIAL BadVLA pre-trained checkpoint.

This is the falsifiability gate of the SharpGuard proposal (§4.2):

    Given:
      - clean_model = openvla/openvla-7b-finetuned-libero-spatial
      - pois_model  = czxlovesu03/BadVLA  (the official pre-trained poisoned ckpt)

    Measure:
      - Global ε-sharpness, λ_max, SAM-response on a clean validation set
      - Sample-level sharpness on (clean, triggered) inputs (the detector signal)
      - Layer-wise sharpness profile

    Decide:
      - Δ_SAM_global = pois - clean      → does poisoned model carry a
                                             larger sharpness signature?
      - sample_level.separation          → does sharpness separate
                                             trigger from non-trigger?
      - layerwise breakdown               → which layer carries the anomaly?

This script handles three plausible BadVLA ckpt layouts auto-detected from
disk (set $BADVLA_CKPT_DIR via setup-badvla-pretrained.sh):

    1. Full HF model dir at $BADVLA_CKPT_DIR/<subset>/  (config.json + safetensors)
       → AutoModelForVision2Seq.from_pretrained(...)

    2. LoRA adapter at $BADVLA_CKPT_DIR/<subset>/        (adapter_config.json)
       → load base from adapter_config.base_model_name_or_path,
         then PeftModel.from_pretrained()

    3. Raw safetensors / .pt file
       → load base, partial-load weights (best effort, with warnings)

If the BadVLA repo holds multiple variants (different LIBERO suites,
different trigger types), --variant lets you pick which one to evaluate.
Default: first one discovered.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from sharpguard.measurement import (
    measure_global, measure_sample_level, measure_layerwise, dump_report,
)
from experiments.openvla_real import (
    _DTYPES, _collate, _build_eval_batches, make_dataset,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clean-model", default="openvla/openvla-7b-finetuned-libero-spatial",
                   help="HF id / path of the CLEAN reference VLA")
    p.add_argument("--badvla-ckpt-dir", default=None,
                   help="Path to czxlovesu03/BadVLA snapshot. "
                        "Defaults to env BADVLA_CKPT_DIR.")
    p.add_argument("--variant", default=None,
                   help="If the BadVLA dir has multiple subfolders (per-suite "
                        "or per-trigger), pick this one. Default: first found.")
    p.add_argument("--out", required=True)
    p.add_argument("--n-eval", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--measure-batches", type=int, default=12)

    # Estimator knobs
    p.add_argument("--estimators", nargs="+",
                   default=["epsilon", "sam"],
                   choices=["epsilon", "lambda_max", "sam"])
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--mode", default="random", choices=["random", "adversarial"])
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--n-iter-lambda", type=int, default=20)

    # Eval data
    p.add_argument("--use-libero-collect", action="store_true", default=True)
    p.add_argument("--libero-sim-suite", default="libero_spatial")
    p.add_argument("--libero-collect-eps", type=int, default=20)
    p.add_argument("--libero-collect-steps", type=int, default=15)

    # System
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--attn", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Auto-detect & load BadVLA ckpt
# ---------------------------------------------------------------------------

def _find_ckpt_root(base_dir: str, variant: Optional[str]) -> str:
    """Walk base_dir; return the first dir containing config.json or
    adapter_config.json. If `variant` matches a subdir name, prefer it."""
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"BADVLA_CKPT_DIR not found: {base}")

    candidates = []
    for d in base.rglob("*"):
        if not d.is_dir():
            continue
        if (d / "config.json").exists() or (d / "adapter_config.json").exists():
            candidates.append(d)

    # Also include base itself if it has the files
    if (base / "config.json").exists() or (base / "adapter_config.json").exists():
        candidates.append(base)

    if not candidates:
        raise FileNotFoundError(
            f"No HF model dir or LoRA adapter found under {base}. "
            f"Files there: {list(base.iterdir())[:20]}"
        )

    # If variant given, filter
    if variant is not None:
        filtered = [c for c in candidates if variant in str(c)]
        if filtered:
            candidates = filtered
        else:
            print(f"[badvla] variant '{variant}' not found; falling back to "
                  f"first of {[c.name for c in candidates]}")

    chosen = candidates[0]
    print(f"[badvla] picked ckpt root: {chosen}")
    print(f"[badvla] (other candidates: {[c.name for c in candidates[1:5]]})")
    return str(chosen)


def load_badvla_model(ckpt_root: str, *, dtype: torch.dtype, attn: str,
                       device: torch.device, fallback_base: str = None):
    """Load BadVLA pre-trained ckpt. Tries:
       (a) full HF model dir
       (b) LoRA adapter on top of declared base (or fallback_base)
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor

    has_config = os.path.exists(os.path.join(ckpt_root, "config.json"))
    has_adapter = os.path.exists(os.path.join(ckpt_root, "adapter_config.json"))

    if has_config and not has_adapter:
        print(f"[badvla] loading as full HF model from {ckpt_root}")
        model = AutoModelForVision2Seq.from_pretrained(
            ckpt_root,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation=attn,
        ).to(device)
        try:
            processor = AutoProcessor.from_pretrained(ckpt_root,
                                                       trust_remote_code=True)
        except Exception:
            print(f"[badvla] no processor in ckpt dir; using fallback "
                  f"{fallback_base}")
            processor = AutoProcessor.from_pretrained(fallback_base,
                                                       trust_remote_code=True)
        return model, processor

    if has_adapter:
        # Read adapter to find the base.
        ac = json.load(open(os.path.join(ckpt_root, "adapter_config.json")))
        base = ac.get("base_model_name_or_path", fallback_base)
        if not base:
            raise RuntimeError(
                f"adapter_config.json has no base_model_name_or_path; "
                f"pass --fallback-base"
            )
        print(f"[badvla] LoRA adapter; base = {base}")
        from peft import PeftModel
        base_model = AutoModelForVision2Seq.from_pretrained(
            base, torch_dtype=dtype, trust_remote_code=True,
            low_cpu_mem_usage=True, attn_implementation=attn,
        ).to(device)
        model = PeftModel.from_pretrained(base_model, ckpt_root)
        try:
            processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(fallback_base,
                                                       trust_remote_code=True)
        return model, processor

    raise RuntimeError(
        f"No config.json / adapter_config.json in {ckpt_root}. "
        f"Files: {os.listdir(ckpt_root)}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    badvla_dir = args.badvla_ckpt_dir or os.environ.get("BADVLA_CKPT_DIR")
    if not badvla_dir:
        raise SystemExit(
            "Set --badvla-ckpt-dir or BADVLA_CKPT_DIR env var "
            "(written by bolt/setup-badvla-pretrained.sh)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _DTYPES[args.dtype]
    print(f"[env] device={device} dtype={args.dtype} attn={args.attn}")

    # --------------- LOAD CLEAN ---------------
    print(f"\n[load] clean = {args.clean_model}")
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForVision2Seq as ModelCls
    except ImportError:
        from transformers import AutoModelForCausalLM as ModelCls
    clean_processor = AutoProcessor.from_pretrained(args.clean_model,
                                                     trust_remote_code=True)
    clean_model = ModelCls.from_pretrained(
        args.clean_model, torch_dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation=args.attn,
    ).to(device)
    clean_model.eval().requires_grad_(True)
    n_clean_params = sum(p.numel() for p in clean_model.parameters())
    print(f"[load] clean params={n_clean_params/1e9:.2f}B")

    # --------------- LOAD BADVLA POISONED ---------------
    ckpt_root = _find_ckpt_root(badvla_dir, args.variant)
    print(f"\n[load] badvla = {ckpt_root}")
    pois_model, pois_processor = load_badvla_model(
        ckpt_root, dtype=dtype, attn=args.attn, device=device,
        fallback_base=args.clean_model,
    )
    pois_model.eval().requires_grad_(True)
    n_pois_params = sum(p.numel() for p in pois_model.parameters())
    print(f"[load] poisoned params={n_pois_params/1e9:.2f}B")

    # --------------- BUILD EVAL BATCHES ---------------
    # We use the CLEAN processor (badvla one may differ; but the inputs go
    # through processor → tokenizer, and OpenVLA's processor is consistent).
    libero_steps = None
    if args.use_libero_collect:
        try:
            from sharpguard.libero_collect import collect_libero_data
            print("\n[data] collecting LIBERO sim data via clean_model ...")
            libero_steps = collect_libero_data(
                clean_model, clean_processor,
                suite=args.libero_sim_suite,
                n_episodes=args.libero_collect_eps,
                max_steps_per_ep=args.libero_collect_steps,
                device=device, seed=args.seed,
            )
            print(f"[data] collected {len(libero_steps) if libero_steps else 0} steps")
        except Exception as e:
            print(f"[data] libero-collect failed: {e}; using synthetic")
            libero_steps = None

    eval_batches = _build_eval_batches(clean_processor, args, device, libero_steps)
    print(f"[data] eval batches: {len(eval_batches)}")

    # --------------- STAGE 1 MEASUREMENT ---------------
    est_kw = dict(epsilon=args.epsilon, n_trials=args.n_trials,
                   mode=args.mode, rho=args.rho,
                   n_iter_lambda=args.n_iter_lambda, seed=args.seed)

    def _measure(model, tag):
        out = {"global": {}, "sample_level": {}, "layerwise": {}}
        for est in args.estimators:
            print(f"  [{tag}] global {est} ...")
            out["global"][est] = measure_global(
                model, eval_batches, estimator=est,
                max_batches=args.measure_batches, **est_kw)
            print(f"  [{tag}] sample-level {est} ...")
            out["sample_level"][est] = measure_sample_level(
                model, eval_batches, estimator=est,
                max_batches=args.measure_batches, **est_kw)
            print(f"  [{tag}] layer-wise {est} ...")
            out["layerwise"][est] = measure_layerwise(
                model, eval_batches, estimator=est,
                max_batches=min(2, args.measure_batches), **est_kw)
        return out

    print("\n=== Stage 1 measurement: CLEAN model ===")
    clean_meas = _measure(clean_model, "clean")
    print("\n=== Stage 1 measurement: BadVLA POISONED model ===")
    pois_meas = _measure(pois_model, "pois")

    # --------------- HEADLINE CONTRAST ---------------
    print("\n" + "=" * 70)
    print("STAGE 1 HEADLINE — does the sharpness signature transfer to BadVLA?")
    print("=" * 70)
    contrast = {}
    for est in args.estimators:
        c = clean_meas["global"][est]["mean"]
        p = pois_meas["global"][est]["mean"]
        contrast[est] = {"clean": c, "poisoned": p, "diff": p - c}
        print(f"  {est:<12s}  clean={c:+.4e}  poisoned={p:+.4e}  Δ={p-c:+.4e}")

    print("\nSample-level separation on POISONED model "
          "(triggered − clean per sample):")
    for est in args.estimators:
        sl = pois_meas["sample_level"][est]
        sep = sl.get("separation")
        cn = sl["clean"]["n"]; tn = sl["triggered"]["n"]
        if sep is not None:
            print(f"  {est:<12s}  trig_mean={sl['triggered']['mean']:+.4e}  "
                  f"clean_mean={sl['clean']['mean']:+.4e}  "
                  f"Δ={sep:+.4e}  (n: clean={cn}, trig={tn})")
        else:
            print(f"  {est:<12s}  separation: n/a  (n: clean={cn}, trig={tn})")

    print("\nLayer-wise sharpness on POISONED model "
          "(top 5 by absolute mean):")
    for est in args.estimators:
        lw = pois_meas["layerwise"][est]["groups"]
        items = sorted(lw.items(), key=lambda kv: abs(kv[1]["mean"]), reverse=True)[:5]
        print(f"  -- {est} --")
        for g, st in items:
            print(f"    {g:<22s}  mean={st['mean']:+.4e}  n={st['n']}")

    # --------------- DUMP ---------------
    full = {
        "args": vars(args),
        "ckpt_root": ckpt_root,
        "clean_model": args.clean_model,
        "params_clean_billion": n_clean_params / 1e9,
        "params_poisoned_billion": n_pois_params / 1e9,
        "data_source": "libero" if libero_steps else "synthetic",
        "n_libero_steps": len(libero_steps) if libero_steps else 0,
        "clean": clean_meas,
        "poisoned": pois_meas,
        "headline_contrast": contrast,
    }
    dump_report(full, str(out_dir / "stage1_official_badvla.json"))
    print(f"\n[done] {out_dir / 'stage1_official_badvla.json'}")


if __name__ == "__main__":
    main()
