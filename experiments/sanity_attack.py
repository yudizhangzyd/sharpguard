"""Quick check: clean training reaches high SR; poisoning reaches high SR + ASR.

If this prints something like
    clean_only:    SR=1.000  ASR=0.00x
    poisoned 10%:  SR=1.000  ASR=0.99x
the benchmark + training loop are wired correctly.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sharpguard.benchmark import BenchmarkConfig, VLAlikeDataset, make_tiny_gpt2
from sharpguard.training import TrainConfig, train


def main():
    cfg = BenchmarkConfig(n_train=4096, n_eval=1024, seed=0)
    train_cfg = TrainConfig(n_epochs=10, batch_size=64, lr=5e-3, log_every=200)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== clean training ===")
    model = make_tiny_gpt2(cfg).to(device)
    ds = VLAlikeDataset(cfg, cfg.n_train, poison_rate=0.0, seed=0)
    r = train(model, ds, cfg, train_cfg, device=device, verbose=True)
    print(f"  final: SR={r.final_metrics['SR']:.3f}  ASR={r.final_metrics['ASR']:.3f}")

    print("\n=== poisoned training (rate=0.15) ===")
    model = make_tiny_gpt2(cfg).to(device)
    ds = VLAlikeDataset(cfg, cfg.n_train, poison_rate=0.15, seed=0)
    r = train(model, ds, cfg, train_cfg, device=device, verbose=True)
    print(f"  final: SR={r.final_metrics['SR']:.3f}  ASR={r.final_metrics['ASR']:.3f}")


if __name__ == "__main__":
    main()
