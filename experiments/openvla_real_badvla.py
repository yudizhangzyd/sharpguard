"""Stage 2 + Stage 3 evaluation against the OFFICIAL BadVLA-poisoned ckpt.

Pipeline (8 GPUs):

  GPU 0: clean OpenVLA (reference for BadVLA loss + Stage 0 baseline + retrain base)
  GPU 1: real BadVLA-poisoned ckpt (vanilla_poisoned column for headline)
  GPU 2: Stage 2 retrain (sharpness detector → filter → fresh LoRA from clean)
  GPU 3: Stage 3 SharpGuard mech-A (fresh LoRA from clean, BadVLA-OD loss + SG-A reg)
  GPU 4: FT-SAM baseline   (sharpness detector + SAM optimizer)
  GPU 5: FT-AC baseline    (activation-clustering detector + AdamW)
  GPU 6: Fine-pruning      (mask dormant channels + retrain)
  GPU 7: SharpGuard mech-B (BadVLA-OD loss + SG-B reg)

Each GPU process pins to its own device via CUDA_VISIBLE_DEVICES, gets its
own task ID, runs independently, and dumps a json per defense column.
The aggregator at end merges into a single headline table.

KEY DIFFERENCE from prior runs: the "attack" used by the training-side
processes (Stage 3, FT-SAM, FT-AC, Fine-pruning) is the FAITHFUL BadVLA
loss formulation (consistency + dissimilarity against a frozen reference),
not my old alternating-LR BadNet variant.
"""
# This file is a thin orchestration script — actual per-process logic lives
# in experiments/openvla_real.py (we add a --use-real-badvla-loss flag and
# --badvla-ref-model flag to that runner so it uses the new attack).
#
# But for a separate paper-grade table specifically against the OFFICIAL
# BadVLA poisoned ckpt, we just need to add ONE thing: the ability to load
# the pre-trained ckpt as the "vanilla poisoned" column instead of training
# our own. We do that via a new flag --pretrained-poisoned-ckpt-dir.

# This script itself is a SHIM that updates flags and re-uses the existing
# experiments/openvla_real.py main loop.
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    # Pass through to openvla_real with appropriate flags.
    sys.argv = [
        "openvla_real.py", *sys.argv[1:]
    ]
    from experiments.openvla_real import main
    main()
