"""Save/load diagnostic: does `save_pretrained` preserve base OpenVLA's
Task SR when re-loaded by Kim's run_libero_eval.py?

Compares:
  1. Kim eval on HF-hosted base checkpoint       -> expected SR ~80%
  2. Load base, save to disk, Kim eval that copy -> ??
  3. Load base + attach *empty/zero* LoRA, merge, save, eval -> ??
     (isolates whether peft merge itself perturbs weights)

If (2) is ~80% -> save/load is fine; SR=0 in ec77uryjux etc. is a
training-data problem (libero_collect produces garbage trajectories).

If (2) < 5%   -> save/load loses something (norm_stats, custom code
paths, etc.); need to patch that before any TemporalTrap retrain has
a chance.

If (3) < (2)  -> peft merge is destructive on OpenVLA's custom model.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--n-eps-per-task", type=int, default=5)
    p.add_argument("--out", default="./artifacts/save-load-diagnostic")
    return p.parse_args()


def _run_kim_eval(checkpoint_path: str, suite: str, n_eps: int,
                   log_dir: Path, run_id: str) -> dict:
    """Invoke bolt/kim_eval_with_trigger.py against a given checkpoint."""
    env = os.environ.copy()
    env.update({
        "MODEL_CHECKPOINT":   str(checkpoint_path),
        "LIBERO_SUITE":       suite,
        "N_EPS_PER_TASK":     str(n_eps),
        "TRIGGER_PHRASE":     "",
        "KIM_LOCAL_LOG_DIR":  str(log_dir),
        "KIM_RUN_ID_NOTE":    run_id,
    })
    wrapper = str(ROOT / "bolt" / "kim_eval_with_trigger.py")
    print(f"\n[eval] {run_id}: checkpoint={checkpoint_path}")
    subprocess.run(["python", wrapper], env=env, cwd=str(ROOT), check=False)
    json_path = log_dir / f"task_sr_{run_id}.json"
    if json_path.exists():
        return json.loads(json_path.read_text())
    return {"error": f"no task_sr json at {json_path}"}


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    kim_dir = out_dir / "kim_eval"
    kim_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: Kim eval directly on HF hub checkpoint ----
    r_hub = _run_kim_eval(args.model, args.suite, args.n_eps_per_task,
                          kim_dir, "hf_hub")

    # ---- Stage 2: Load, save to disk, eval that disk copy ----
    save_dir = out_dir / "resaved_base"
    if not (save_dir / "config.json").exists():
        print(f"\n[save] loading {args.model} and re-saving to {save_dir}")
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        # Explicit norm_stats dump for diagnosis
        ns = getattr(model, "norm_stats", None) or getattr(model.config, "norm_stats", None)
        if ns is not None:
            (save_dir / "norm_stats_debug.json").write_text(json.dumps(
                {k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                     for kk, vv in v.items() if isinstance(v, dict)}
                 if isinstance(v, dict) else str(v)
                 for k, v in ns.items()}, default=str, indent=2))
        print(f"[save] contents: {sorted(p.name for p in save_dir.iterdir())}")
        del model
        torch.cuda.empty_cache()
    r_resaved = _run_kim_eval(str(save_dir), args.suite, args.n_eps_per_task,
                              kim_dir, "resaved_base")

    # ---- Compare ----
    summary = {
        "hf_hub":       r_hub,
        "resaved_base": r_resaved,
        "delta_SR":     ((r_resaved.get("SR") or 0) - (r_hub.get("SR") or 0)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== SUMMARY ===")
    print(f"  hf_hub       SR = {r_hub.get('SR')}")
    print(f"  resaved_base SR = {r_resaved.get('SR')}")
    print(f"  delta        = {summary['delta_SR']}")
    print(f"\n[done] {out_dir}")


if __name__ == "__main__":
    main()
