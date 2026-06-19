"""Parallel λ-sweep: 8 SharpGuard configurations, one per GPU.

Designed for an 8-GPU bolt task. Spawns 8 child processes via subprocess
(one per CUDA_VISIBLE_DEVICES), each running run_all.py with a different
SharpGuard λ. Aggregates results into a single sweep_summary.json + plots.

Run locally:
  python experiments/run_sweep.py --gpus 0 --epochs-pois 6
Run on bolt (8 GPU):
  bash bolt/run_sweep.sh
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Default λ-sweep grid — 8 points crossing the "cliff" we observed locally.
DEFAULT_LAMBDAS = [0.0, 0.5, 1.0, 2.0, 5.0, 8.0, 12.0, 20.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    p.add_argument("--gpus", nargs="+", type=int, default=None,
                   help="GPU indices to use. Default: all visible GPUs.")
    p.add_argument("--out", default=str(ROOT / "outputs" / f"sweep_{time.strftime('%Y%m%d_%H%M%S')}"))
    p.add_argument("--n-train", type=int, default=4096)
    p.add_argument("--epochs-clean", type=int, default=8)
    p.add_argument("--epochs-pois", type=int, default=10)
    p.add_argument("--epochs-s3", type=int, default=10)
    p.add_argument("--poison-rate", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def detect_gpus(requested):
    if requested is not None:
        return list(requested)
    cv = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cv:
        return [int(i) for i in cv.split(",") if i.strip()]
    try:
        import torch
        n = torch.cuda.device_count()
        return list(range(n)) if n > 0 else [-1]   # -1 sentinel = CPU
    except Exception:
        return [-1]


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpus = detect_gpus(args.gpus)
    print(f"[sweep] gpus = {gpus}")
    print(f"[sweep] lambdas = {args.lambdas}")
    print(f"[sweep] out = {out_dir}")

    if len(args.lambdas) > len(gpus):
        print(f"[warn] {len(args.lambdas)} lambdas but only {len(gpus)} GPUs — will run "
              f"in waves of {len(gpus)}.")

    # Build one subprocess command per (λ, gpu) — chunked by num GPUs.
    waves = [args.lambdas[i: i + len(gpus)]
             for i in range(0, len(args.lambdas), len(gpus))]

    all_runs = []
    for wave_idx, wave in enumerate(waves):
        procs = []
        for gpu_slot, lam in enumerate(wave):
            gpu = gpus[gpu_slot]
            sub_out = out_dir / f"lam_{lam:g}"
            sub_out.mkdir(exist_ok=True)
            env = os.environ.copy()
            if gpu >= 0:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log = open(sub_out / "stdout.log", "w")
            cmd = [
                sys.executable,
                str(ROOT / "experiments" / "run_all.py"),
                "--out", str(sub_out),
                "--n-train", str(args.n_train),
                "--epochs-clean", str(args.epochs_clean),
                "--epochs-pois", str(args.epochs_pois),
                "--epochs-s3", str(args.epochs_s3),
                "--poison-rate", str(args.poison_rate),
                "--lam-sg", str(lam),
            ]
            print(f"[wave {wave_idx}] gpu={gpu} λ={lam} → {sub_out}")
            print(f"  cmd: {' '.join(cmd)}")
            if not args.dry_run:
                p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                procs.append((lam, gpu_slot, p, log, sub_out))
            else:
                procs.append((lam, gpu_slot, None, log, sub_out))

        if not args.dry_run:
            for lam, slot, p, log, sub_out in procs:
                rc = p.wait()
                log.close()
                print(f"  [done] λ={lam} rc={rc} → {sub_out}")
                all_runs.append({"lam": lam, "rc": rc, "out": str(sub_out)})

    # ---------- aggregate ----------
    summary = {
        "args": vars(args),
        "gpus": gpus,
        "runs": [],
    }
    for r in all_runs:
        rj = Path(r["out"]) / "results.json"
        if rj.exists():
            with rj.open() as f:
                data = json.load(f)
            summary["runs"].append({
                "lam": r["lam"],
                "rc": r["rc"],
                "metrics": {
                    "clean":     data.get("clean_baseline"),
                    "poisoned":  data.get("poisoned_baseline"),
                    "stage2":    data.get("stage2", {}).get("post_defense"),
                    "stage3":    data.get("stage3", {}).get("final"),
                    "adaptive":  data.get("adaptive", {}).get("attacker_only", {}).get("final"),
                    "adaptive_vs_sg": data.get("adaptive", {}).get("vs_sharpguard", {}).get("final"),
                },
                "stage1": {
                    "clean_global":     data.get("stage1", {}).get("clean", {}).get("global", {}).get("mean"),
                    "poisoned_global":  data.get("stage1", {}).get("poisoned", {}).get("global", {}).get("mean"),
                    "hard_clean_global":data.get("stage1", {}).get("hard_clean", {}).get("global", {}).get("mean"),
                    "sample_separation":data.get("stage1", {}).get("poisoned", {}).get("sample_level", {}).get("separation"),
                },
            })
        else:
            summary["runs"].append({"lam": r["lam"], "rc": r["rc"], "error": "no results.json"})

    (out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2,
                                                            default=lambda x: float(x) if hasattr(x, "item") else str(x)))
    print(f"[sweep] summary → {out_dir / 'sweep_summary.json'}")

    # ---------- λ vs SR/ASR plot ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        runs = [r for r in summary["runs"] if "metrics" in r and r["metrics"]["stage3"] is not None]
        runs.sort(key=lambda r: r["lam"])
        lams = [r["lam"] for r in runs]
        sr_s3 = [r["metrics"]["stage3"]["SR"] for r in runs]
        asr_s3 = [r["metrics"]["stage3"]["ASR"] for r in runs]
        sr_pois = [r["metrics"]["poisoned"]["SR"] for r in runs]
        asr_pois = [r["metrics"]["poisoned"]["ASR"] for r in runs]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(lams, sr_s3, "o-", label="SR (SharpGuard)", color="#3a86ff")
        ax.plot(lams, asr_s3, "s-", label="ASR (SharpGuard)", color="#ff595e")
        ax.axhline(sr_pois[0] if sr_pois else 1.0, ls=":", color="#3a86ff", alpha=0.5,
                   label="SR (no defense)")
        ax.axhline(asr_pois[0] if asr_pois else 1.0, ls=":", color="#ff595e", alpha=0.5,
                   label="ASR (no defense)")
        ax.set_xscale("symlog", linthresh=0.5)
        ax.set_xlabel("SharpGuard λ")
        ax.set_ylabel("rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"λ-sweep on mini benchmark (poison_rate={args.poison_rate})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "lambda_sweep.png", dpi=120)
        plt.close(fig)
        print(f"[sweep] plot → {out_dir / 'lambda_sweep.png'}")
    except Exception as e:
        print(f"[sweep] plot skipped: {e}")


if __name__ == "__main__":
    main()
