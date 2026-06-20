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
    p.add_argument("--target", default="both",
                   choices=["both", "clean", "poisoned", "aggregate"],
                   help="What to measure. With 8 GPUs we run 'clean' and "
                        "'poisoned' as parallel processes (each pinned to a "
                        "GPU via CUDA_VISIBLE_DEVICES), then 'aggregate' "
                        "merges the two json files.")
    p.add_argument("--clean-model", default="openvla/openvla-7b-finetuned-libero-spatial",
                   help="HF id / path of the CLEAN reference VLA")
    p.add_argument("--badvla-ckpt-dir", default=None,
                   help="Path to czxlovesu03/BadVLA snapshot. "
                        "Defaults to env BADVLA_CKPT_DIR.")
    p.add_argument("--variant", default=None,
                   help="If the BadVLA dir has multiple subfolders (per-suite "
                        "or per-trigger), pick this one. Default: first found.")
    p.add_argument("--out", required=True)
    p.add_argument("--shared-eval-cache", default=None,
                   help="When in --target=poisoned mode, load eval batches "
                        "from this .pt file instead of re-collecting.")
    p.add_argument("--clean-json", default=None,
                   help="Path to clean side's json output (for --target=aggregate)")
    p.add_argument("--pois-json", default=None,
                   help="Path to poisoned side's json output (for --target=aggregate)")
    p.add_argument("--shard-idx", type=int, default=0,
                   help="Which shard of eval batches this worker handles. "
                        "Use with --num-shards for data-parallel measurement.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of measurement shards per model side.")
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
    print(f"[env] device={device} dtype={args.dtype} attn={args.attn}  target={args.target}")

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForVision2Seq as ModelCls
    except ImportError:
        from transformers import AutoModelForCausalLM as ModelCls

    # ====== Aggregate-only mode: merge two JSONs and dump headline ======
    if args.target == "aggregate":
        if not args.clean_json or not args.pois_json:
            raise SystemExit("--clean-json and --pois-json required for aggregate")

        def _load_and_merge(json_paths_csv, key):
            """Read 1+ shard JSONs (comma-separated paths), merge per-estimator."""
            paths = json_paths_csv.split(",")
            partials = [json.load(open(p)) for p in paths if os.path.exists(p)]
            if not partials:
                raise SystemExit(f"no readable jsons in: {json_paths_csv}")
            base = partials[0][key] if key in partials[0] else partials[0]
            if len(partials) == 1:
                return base
            # Merge global / sample_level / layerwise across shards.
            merged = {"global": {}, "sample_level": {}, "layerwise": {}}
            estimators = list(base["global"].keys())
            for est in estimators:
                # global: weighted mean by n_batches.
                vals, ns = [], []
                for p in partials:
                    g = (p[key] if key in p else p)["global"][est]
                    vals.append(g["mean"]); ns.append(g["n_batches"])
                total_n = sum(ns)
                merged["global"][est] = {
                    "estimator": est,
                    "n_batches": total_n,
                    "mean": sum(v * n for v, n in zip(vals, ns)) / max(total_n, 1),
                    "shards": ns,
                }
                # sample_level: concat values per (clean, triggered).
                clean_vals, trig_vals = [], []
                for p in partials:
                    sl = (p[key] if key in p else p)["sample_level"][est]
                    clean_vals.extend(sl["clean"]["values"])
                    trig_vals.extend(sl["triggered"]["values"])
                def stats(xs):
                    n = len(xs)
                    if n == 0: return {"mean": float("nan"), "std": 0.0, "n": 0, "values": []}
                    m = sum(xs) / n
                    s = (sum((x - m) ** 2 for x in xs) / max(n - 1, 1)) ** 0.5 if n > 1 else 0.0
                    return {"mean": m, "std": s, "n": n, "values": xs}
                merged["sample_level"][est] = {
                    "estimator": est,
                    "clean": stats(clean_vals),
                    "triggered": stats(trig_vals),
                    "separation": (sum(trig_vals)/len(trig_vals) - sum(clean_vals)/len(clean_vals))
                                  if (clean_vals and trig_vals) else None,
                }
                # layerwise: weighted-mean per group across shards.
                groups = {}
                for p in partials:
                    lw = (p[key] if key in p else p)["layerwise"][est].get("groups", {})
                    for g, st in lw.items():
                        groups.setdefault(g, []).append(st)
                merged_groups = {}
                for g, sts in groups.items():
                    total_n = sum(s["n"] for s in sts)
                    if total_n == 0:
                        merged_groups[g] = {"mean": float("nan"), "std": 0.0, "n": 0, "values": []}
                        continue
                    merged_groups[g] = {
                        "mean": sum(s["mean"] * s["n"] for s in sts) / total_n,
                        "n": total_n,
                        "values": [v for s in sts for v in s.get("values", [])],
                    }
                merged["layerwise"][est] = {"estimator": est, "groups": merged_groups}
            return merged

        clean_meas = _load_and_merge(args.clean_json, "clean")
        pois_meas = _load_and_merge(args.pois_json, "poisoned")
        contrast = {}
        print("\n" + "=" * 70)
        print("STAGE 1 HEADLINE — does the sharpness signature transfer to BadVLA?")
        print("=" * 70)
        for est in args.estimators:
            c = clean_meas["global"][est]["mean"]
            p = pois_meas["global"][est]["mean"]
            contrast[est] = {"clean": c, "poisoned": p, "diff": p - c}
            print(f"  {est:<12s}  clean={c:+.4e}  poisoned={p:+.4e}  Δ={p-c:+.4e}")
        print("\nSample-level separation on POISONED:")
        for est in args.estimators:
            sl = pois_meas["sample_level"][est]
            sep = sl.get("separation")
            cn = sl["clean"]["n"]; tn = sl["triggered"]["n"]
            if sep is not None:
                print(f"  {est:<12s}  trig={sl['triggered']['mean']:+.4e}  "
                      f"clean={sl['clean']['mean']:+.4e}  Δ={sep:+.4e}  "
                      f"(n: clean={cn}, trig={tn})")
        full = {"clean": clean_meas, "poisoned": pois_meas,
                 "headline_contrast": contrast,
                 "args": vars(args)}
        dump_report(full, str(out_dir / "stage1_official_badvla.json"))
        print(f"\n[done] {out_dir / 'stage1_official_badvla.json'}")
        return

    # ====== clean / poisoned / both: actually load + measure ======
    est_kw = dict(epsilon=args.epsilon, n_trials=args.n_trials,
                   mode=args.mode, rho=args.rho,
                   n_iter_lambda=args.n_iter_lambda, seed=args.seed)

    def _measure(model, eval_batches, tag):
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

    # Build / load eval batches.
    if args.shared_eval_cache and os.path.exists(args.shared_eval_cache):
        print(f"[data] reusing eval batches from cache: {args.shared_eval_cache}")
        eval_batches = torch.load(args.shared_eval_cache, map_location=device)
        # batches stored on cpu; move pixel tensors to device.
        eval_batches = [
            {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in b.items()}
            for b in eval_batches
        ]
        clean_processor = AutoProcessor.from_pretrained(args.clean_model,
                                                         trust_remote_code=True)
        # libero_steps not needed when using shared cache
        libero_steps = None
    else:
        # Need a model to drive sim collection. Use the side we'll measure.
        if args.target in ("clean", "both"):
            print(f"\n[load] {args.clean_model}")
            clean_processor = AutoProcessor.from_pretrained(args.clean_model,
                                                             trust_remote_code=True)
            sim_driver = ModelCls.from_pretrained(
                args.clean_model, torch_dtype=dtype, trust_remote_code=True,
                low_cpu_mem_usage=True, attn_implementation=args.attn,
            ).to(device)
        else:  # poisoned-only mode without cache; load BadVLA to drive sim
            ckpt_root = _find_ckpt_root(badvla_dir, args.variant)
            sim_driver, clean_processor = load_badvla_model(
                ckpt_root, dtype=dtype, attn=args.attn, device=device,
                fallback_base=args.clean_model,
            )
        sim_driver.eval()

        libero_steps = None
        if args.use_libero_collect:
            try:
                from sharpguard.libero_collect import collect_libero_data
                print("\n[data] collecting LIBERO sim data ...")
                libero_steps = collect_libero_data(
                    sim_driver, clean_processor,
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

        # If we just collected and we're in clean/poisoned-only mode and
        # --shared-eval-cache is set, save it for the OTHER process to reuse.
        if args.shared_eval_cache and args.target == "clean":
            cache_dir = os.path.dirname(args.shared_eval_cache)
            os.makedirs(cache_dir, exist_ok=True)
            cpu_batches = [
                {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                 for k, v in b.items()}
                for b in eval_batches
            ]
            torch.save(cpu_batches, args.shared_eval_cache)
            print(f"[data] cached eval batches → {args.shared_eval_cache}")

        # In clean-only mode `sim_driver` IS the clean model — keep it for measurement.
        # In poisoned-only mode `sim_driver` was the badvla model — release it,
        # we'll reload it cleanly below.
        if args.target == "poisoned":
            del sim_driver
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Apply --num-shards / --shard-idx slicing AFTER the cache is shared,
    # so all shards reuse the same eval batches but each measures only its
    # slice. Stride slicing keeps clean/triggered ratios within each shard.
    if args.num_shards > 1:
        full_n = len(eval_batches)
        eval_batches = eval_batches[args.shard_idx::args.num_shards]
        print(f"[shard] shard {args.shard_idx}/{args.num_shards}: "
              f"{len(eval_batches)} of {full_n} batches")

    # ---- Measurement ----
    if args.target == "clean":
        # sim_driver is the clean model (unless we used cache, then load fresh)
        if 'sim_driver' in dir():
            clean_model = sim_driver
        else:
            print(f"\n[load] clean = {args.clean_model}")
            clean_model = ModelCls.from_pretrained(
                args.clean_model, torch_dtype=dtype, trust_remote_code=True,
                low_cpu_mem_usage=True, attn_implementation=args.attn,
            ).to(device)
        clean_model.eval().requires_grad_(True)
        n_clean = sum(p.numel() for p in clean_model.parameters())
        print(f"[load] clean params={n_clean/1e9:.2f}B")

        print("\n=== Stage 1 measurement: CLEAN model ===")
        clean_meas = _measure(clean_model, eval_batches, "clean")
        full = {
            "args": vars(args),
            "clean_model": args.clean_model,
            "params_clean_billion": n_clean / 1e9,
            "shard_idx": args.shard_idx,
            "num_shards": args.num_shards,
            "clean": clean_meas,
        }
        suffix = f".shard{args.shard_idx}" if args.num_shards > 1 else ""
        out_path = out_dir / f"clean{suffix}.json"
        dump_report(full, str(out_path))
        print(f"\n[done] {out_path}")
        return

    if args.target == "poisoned":
        ckpt_root = _find_ckpt_root(badvla_dir, args.variant)
        print(f"\n[load] badvla = {ckpt_root}")
        pois_model, _ = load_badvla_model(
            ckpt_root, dtype=dtype, attn=args.attn, device=device,
            fallback_base=args.clean_model,
        )
        pois_model.eval().requires_grad_(True)
        n_pois = sum(p.numel() for p in pois_model.parameters())
        print(f"[load] poisoned params={n_pois/1e9:.2f}B")

        print("\n=== Stage 1 measurement: POISONED model ===")
        pois_meas = _measure(pois_model, eval_batches, "pois")
        full = {
            "args": vars(args),
            "ckpt_root": ckpt_root,
            "params_poisoned_billion": n_pois / 1e9,
            "shard_idx": args.shard_idx,
            "num_shards": args.num_shards,
            "poisoned": pois_meas,
        }
        suffix = f".shard{args.shard_idx}" if args.num_shards > 1 else ""
        out_path = out_dir / f"pois{suffix}.json"
        dump_report(full, str(out_path))
        print(f"\n[done] {out_path}")
        return

    # target == "both": sequential single-GPU path (unchanged from before).
    # NOTE: this hits OOM on a single A100 with two 7B models — prefer the
    # parallel "clean" + "poisoned" + "aggregate" three-step pipeline.
    raise SystemExit(
        "target=both is single-GPU sequential and may OOM on 80GB. "
        "Use the parallel pipeline: launch --target=clean and --target=poisoned "
        "on separate GPUs, then --target=aggregate to merge."
    )


if __name__ == "__main__":
    main()
