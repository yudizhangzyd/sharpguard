"""End-to-end runner: full SharpGuard pipeline on the mini benchmark.

Produces in $OUT_DIR (default ./outputs/run_<timestamp>):
  - results.json            all numbers (SR/ASR/sharpness/detector P-R)
  - run.log                 stdout
  - figures/*.png           matplotlib plots (if matplotlib available)

This script implements every stage of the proposal:
  1. Train clean baseline                  → SR_clean, ASR_clean
  2. Train poisoned baseline               → SR_pois,  ASR_pois     (= attack succeeded)
  3. Stage 1: measure sharpness            → headline contrast
                + §6 hard-task confound control
  4. Stage 2: detector + retrain           → P/R, SR_S2, ASR_S2
  5. Stage 3: SharpGuard regularizer       → SR_S3, ASR_S3
  6. Adaptive low-sharpness attack         → SR_adapt, ASR_adapt   (sharpness-effectiveness tradeoff)

Run:  python experiments/run_all.py
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

from sharpguard.benchmark import (
    BenchmarkConfig,
    VLAlikeDataset,
    collate,
    evaluate_sr_asr,
    make_tiny_gpt2,
    pack_for_lm,
)
from sharpguard.training import TrainConfig, train
from sharpguard.measurement import measure_global, measure_sample_level, measure_layerwise
from sharpguard.detector import DetectorConfig, detect_poison
from sharpguard.defenses import make_sharpguard
from sharpguard.attacks import AdaptiveLowSharpnessRegularizer, AdaptiveAttackConfig


def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))


# A version of measurement that takes a list of dict-batches (we already have those).
def measure_on_batches(model, batches, *, estimator="sam", max_batches=8, **kw):
    out_g = measure_global(model, batches, estimator=estimator,
                           max_batches=max_batches, **kw)
    out_s = measure_sample_level(model, batches, estimator=estimator,
                                 max_batches=max_batches, **kw)
    out_l = measure_layerwise(model, batches, estimator=estimator,
                              max_batches=min(2, max_batches), **kw)
    return {"global": out_g, "sample_level": out_s, "layerwise": out_l}


def make_eval_batches(cfg: BenchmarkConfig, *, n_batches=8, batch_size=16, seed=999):
    """Produce a list of dict batches alternating clean/triggered samples
    so sample_level can compute clean-vs-triggered separation."""
    clean_ds = VLAlikeDataset(cfg, n_batches * batch_size // 2,
                              poison_rate=0.0, seed=seed)
    trig_ds = VLAlikeDataset(cfg, n_batches * batch_size // 2,
                             force_trigger=True, seed=seed + 1)
    out = []
    half = batch_size // 2
    for i in range(n_batches):
        items = []
        for j in range(half):
            it = clean_ds[i * half + j]
            it["is_triggered"] = torch.tensor(False)
            items.append(it)
            it = trig_ds[i * half + j]
            it["is_triggered"] = torch.tensor(True)
            items.append(it)
        b = collate(items)
        out.append(pack_for_lm(b, cfg))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--epochs-clean", type=int, default=8)
    parser.add_argument("--epochs-pois", type=int, default=10)
    parser.add_argument("--epochs-s3", type=int, default=10)
    parser.add_argument("--poison-rate", type=float, default=0.15)
    parser.add_argument("--lam-sg", type=float, default=2.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out) if args.out else (
        ROOT / "outputs" / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    print(f"[out] {out_dir}")

    cfg = BenchmarkConfig(n_train=args.n_train, n_eval=1024, seed=0)
    eval_batches = make_eval_batches(cfg)
    eval_batches_dev = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                          for k, v in b.items()} for b in eval_batches]

    results = {"args": vars(args), "config": cfg.__dict__.copy()}

    # ------------------------------------------------------------------
    # 1) Clean baseline
    # ------------------------------------------------------------------
    print("\n=== [1/6] clean baseline ===")
    clean_model = make_tiny_gpt2(cfg).to(device)
    clean_ds = VLAlikeDataset(cfg, cfg.n_train, poison_rate=0.0, seed=0)
    r1 = train(clean_model, clean_ds, cfg,
               TrainConfig(n_epochs=args.epochs_clean, log_every=10000),
               device=device, verbose=False)
    print(f"  SR={r1.final_metrics['SR']:.3f}  ASR={r1.final_metrics['ASR']:.3f}")
    results["clean_baseline"] = r1.final_metrics

    # ------------------------------------------------------------------
    # 2) Poisoned baseline (the attack)
    # ------------------------------------------------------------------
    print("\n=== [2/6] poisoned baseline (BadNet, rate=%.2f) ===" % args.poison_rate)
    pois_model = make_tiny_gpt2(cfg).to(device)
    pois_ds = VLAlikeDataset(cfg, cfg.n_train, poison_rate=args.poison_rate, seed=0)
    r2 = train(pois_model, pois_ds, cfg,
               TrainConfig(n_epochs=args.epochs_pois, log_every=10000),
               device=device, verbose=False)
    print(f"  SR={r2.final_metrics['SR']:.3f}  ASR={r2.final_metrics['ASR']:.3f}")
    results["poisoned_baseline"] = r2.final_metrics

    # ------------------------------------------------------------------
    # 3) Stage 1: sharpness signature on (clean, poisoned)  + §6 confound
    # ------------------------------------------------------------------
    print("\n=== [3/6] Stage 1: sharpness measurement ===")
    s_clean = measure_on_batches(clean_model, eval_batches_dev, estimator="sam", rho=0.05)
    s_pois = measure_on_batches(pois_model, eval_batches_dev, estimator="sam", rho=0.05)
    contrast = (s_pois["global"]["mean"] - s_clean["global"]["mean"])
    print(f"  global SAM-response  clean={s_clean['global']['mean']:+.4e}  "
          f"poisoned={s_pois['global']['mean']:+.4e}  Δ={contrast:+.4e}")
    print(f"  sample-level (poisoned model)  clean={s_pois['sample_level']['clean']['mean']:+.4e}  "
          f"triggered={s_pois['sample_level']['triggered']['mean']:+.4e}  "
          f"sep={s_pois['sample_level']['separation']:+.4e}")

    # §6 confound: a benign-but-hard task. We make a harder clean policy and
    # train + measure its sharpness. If hard-task sharpness is comparable to
    # backdoored sharpness, the signature is confounded.
    print("\n  -- §6 confound: train a benign-hard model --")
    hard_cfg = BenchmarkConfig(n_train=args.n_train, n_eval=1024, seed=0,
                               vocab_obs=128, vocab_act=64,
                               obs_len=12, act_len=6)
    hard_model = make_tiny_gpt2(hard_cfg).to(device)
    hard_ds = VLAlikeDataset(hard_cfg, hard_cfg.n_train, poison_rate=0.0, seed=0)
    train(hard_model, hard_ds, hard_cfg,
          TrainConfig(n_epochs=args.epochs_clean, log_every=10000),
          device=device, verbose=False)
    hard_eval = make_eval_batches(hard_cfg)
    hard_eval = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in b.items()} for b in hard_eval]
    s_hard = measure_on_batches(hard_model, hard_eval, estimator="sam", rho=0.05)
    print(f"  global SAM-response  hard-clean={s_hard['global']['mean']:+.4e}  "
          f"vs poisoned={s_pois['global']['mean']:+.4e}")

    results["stage1"] = {
        "clean": s_clean, "poisoned": s_pois, "hard_clean": s_hard,
        "headline_contrast": contrast,
    }

    # ------------------------------------------------------------------
    # 4) Stage 2: sharpness-based detector + retrain
    # ------------------------------------------------------------------
    print("\n=== [4/6] Stage 2: detector + retrain ===")
    det = detect_poison(pois_model, pois_ds, cfg,
                        DetectorConfig(estimator="sam", rho=0.05, drop_quantile=0.85),
                        device=device)
    print(f"  detector P={det.precision:.3f}  R={det.recall:.3f}  threshold={det.threshold:+.4e}")

    s2_model = make_tiny_gpt2(cfg).to(device)
    r4 = train(s2_model, pois_ds, cfg,
               TrainConfig(n_epochs=args.epochs_pois, log_every=10000),
               sample_weights=det.sample_weights,
               device=device, verbose=False)
    print(f"  retrained-after-filter:  SR={r4.final_metrics['SR']:.3f}  "
          f"ASR={r4.final_metrics['ASR']:.3f}")

    results["stage2"] = {
        "precision": det.precision, "recall": det.recall,
        "threshold": det.threshold,
        "post_defense": r4.final_metrics,
        "n_dropped": int((det.sample_weights == 0).sum().item()),
        "n_total": int(len(pois_ds)),
    }

    # ------------------------------------------------------------------
    # 5) Stage 3: SharpGuard regularizer
    # ------------------------------------------------------------------
    print("\n=== [5/6] Stage 3: SharpGuard regularizer ===")
    s3_model = make_tiny_gpt2(cfg).to(device)
    sg = make_sharpguard(epsilon=1e-3, lam=args.lam_sg)
    r5 = train(s3_model, pois_ds, cfg,
               TrainConfig(n_epochs=args.epochs_s3, log_every=10000),
               regularizer=sg, device=device, verbose=False)
    print(f"  SharpGuard-trained: SR={r5.final_metrics['SR']:.3f}  "
          f"ASR={r5.final_metrics['ASR']:.3f}")
    s_s3 = measure_on_batches(s3_model, eval_batches_dev, estimator="sam", rho=0.05)
    print(f"  global sharpness post-SG: {s_s3['global']['mean']:+.4e}  "
          f"(was {s_pois['global']['mean']:+.4e} on poisoned)")
    results["stage3"] = {"final": r5.final_metrics, "sharpness": s_s3}

    # ------------------------------------------------------------------
    # 6) Adaptive attack: low-sharpness backdoor
    # ------------------------------------------------------------------
    print("\n=== [6/6] Adaptive low-sharpness backdoor ===")
    adapt_model = make_tiny_gpt2(cfg).to(device)
    adapt_reg = AdaptiveLowSharpnessRegularizer(AdaptiveAttackConfig(lam_flat=1.0, rho=0.05))
    r6 = train(adapt_model, pois_ds, cfg,
               TrainConfig(n_epochs=args.epochs_pois, log_every=10000),
               regularizer=adapt_reg, device=device, verbose=False)
    s_adapt = measure_on_batches(adapt_model, eval_batches_dev, estimator="sam", rho=0.05)
    print(f"  attacker-only:  SR={r6.final_metrics['SR']:.3f}  "
          f"ASR={r6.final_metrics['ASR']:.3f}  "
          f"sharpness={s_adapt['global']['mean']:+.4e}")

    # Now train SG against this adaptive attack to see if defense holds.
    s3v_model = make_tiny_gpt2(cfg).to(device)
    sg2 = make_sharpguard(epsilon=1e-3, lam=args.lam_sg)
    # Compose: attacker reg + defender reg.
    def both(m, b, l, _a=adapt_reg, _d=sg2):
        return _a(m, b, l) + _d(m, b, l)
    r6b = train(s3v_model, pois_ds, cfg,
                TrainConfig(n_epochs=args.epochs_s3, log_every=10000),
                regularizer=both, device=device, verbose=False)
    print(f"  adaptive-vs-SharpGuard: SR={r6b.final_metrics['SR']:.3f}  "
          f"ASR={r6b.final_metrics['ASR']:.3f}")
    results["adaptive"] = {
        "attacker_only": {"final": r6.final_metrics, "sharpness": s_adapt},
        "vs_sharpguard": {"final": r6b.final_metrics},
    }

    _save_json(results, out_dir / "results.json")
    print(f"\n[done] results: {out_dir / 'results.json'}")

    # ------------------------------------------------------------------
    # Plots (only if matplotlib is available)
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[plots] matplotlib not available; skipping figures.")
        return

    # Headline: SR vs ASR bars across runs
    runs = [
        ("clean",            results["clean_baseline"]),
        ("poisoned",         results["poisoned_baseline"]),
        ("Stage 2 retrain",  results["stage2"]["post_defense"]),
        ("Stage 3 SG",       results["stage3"]["final"]),
        ("adaptive attacker",results["adaptive"]["attacker_only"]["final"]),
        ("adaptive vs SG",   results["adaptive"]["vs_sharpguard"]["final"]),
    ]
    labels = [r[0] for r in runs]
    srs = [r[1]["SR"] for r in runs]
    asrs = [r[1]["ASR"] for r in runs]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.4
    ax.bar([i - w/2 for i in x], srs, w, label="SR (clean ↑)", color="#3a86ff")
    ax.bar([i + w/2 for i in x], asrs, w, label="ASR (attack ↓)", color="#ff595e")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("rate")
    ax.set_title("SharpGuard mini-benchmark — SR vs ASR per regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "headline_sr_asr.png", dpi=120)
    plt.close(fig)

    # Sharpness bar
    sharps = {
        "clean": results["stage1"]["clean"]["global"]["mean"],
        "poisoned": results["stage1"]["poisoned"]["global"]["mean"],
        "hard-clean (control)": results["stage1"]["hard_clean"]["global"]["mean"],
        "Stage 3 SG": results["stage3"]["sharpness"]["global"]["mean"],
        "adaptive attacker": results["adaptive"]["attacker_only"]["sharpness"]["global"]["mean"],
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = list(sharps.keys())
    vals = [sharps[k] for k in keys]
    ax.bar(keys, vals, color=["#3a86ff", "#ff595e", "#8338ec", "#06d6a0", "#ffbe0b"])
    ax.set_xticklabels(keys, rotation=15, ha="right")
    ax.set_ylabel("global SAM-response (sharpness)")
    ax.set_title("Stage 1: backdoor leaves a sharpness fingerprint?")
    fig.tight_layout()
    fig.savefig(fig_dir / "stage1_sharpness.png", dpi=120)
    plt.close(fig)

    # Sample-level histogram
    sl = results["stage1"]["poisoned"]["sample_level"]
    if sl["clean"]["values"] and sl["triggered"]["values"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(sl["clean"]["values"], bins=20, alpha=0.6, label="clean samples",
                color="#3a86ff")
        ax.hist(sl["triggered"]["values"], bins=20, alpha=0.6, label="triggered samples",
                color="#ff595e")
        ax.set_xlabel("per-sample SAM-response")
        ax.set_ylabel("count")
        ax.set_title("Sample-level sharpness on the poisoned model")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "stage1_sample_hist.png", dpi=120)
        plt.close(fig)

    # Layer-wise
    lw = results["stage1"]["poisoned"]["layerwise"]["groups"]
    keys = sorted(lw.keys())
    vals = [lw[k]["mean"] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(keys, vals, color="#06d6a0")
    ax.set_title("Layer-wise sharpness on poisoned model")
    ax.set_ylabel("group-restricted SAM-response")
    ax.set_xticklabels(keys, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "stage1_layerwise.png", dpi=120)
    plt.close(fig)

    print(f"[plots] {fig_dir}")


if __name__ == "__main__":
    main()
