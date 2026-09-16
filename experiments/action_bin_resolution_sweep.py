"""Action-token bin-resolution sweep: is the collision mechanism an
artifact of decoder GRANULARITY, or does it persist regardless of it?

Motivation (a reviewer's ask, not hypothesized in advance). Section 5's
claim is that F is close to a decode-collision counter: 46.2% of scored
records land on the EXACT SAME argmax bin under the original and edited
CoT, and F at tau=0.05 correlates with 1-P(collision) at R^2=0.926 across
324 cells. A skeptic's natural next question: is that collision rate a
property of the SPECIFIC 256-bin grid this checkpoint's action head was
trained on, or would any comparably coarse discretization produce a
similar rate -- i.e. is this a fact about decode geometry in general, not
a fact this model's own 256-bin choice happens to produce? We cannot
retrain the model's action head at other resolutions (that head is fixed
by training), but we CAN take the exact same softmax the model already
computes over its 256-bin window and re-derive what a COARSER read of that
same distribution would report, holding the model and the forward pass
fixed. If the collision rate rises sharply as resolution drops (which any
discretization argument predicts) and stays high as resolution matches or
exceeds 256 (i.e. no floor effect specific to 256), that is evidence for
"collision rate is a property of discretization granularity in general",
which is the paper's claim; if it does not move with resolution, the
256-bin collision rate is NOT explained by discretization per se and the
paper's mechanism claim would need to be narrowed.

Method. For each sample and family, decode the model's own 7x256 softmax
over the action-token window ONCE for the original CoT and ONCE for the
edited CoT (experiments/cotfaith_stage1_continuous.py's
openvla_action_softmax, imported verbatim, not reimplemented -- the exact
function Table 3 / Figure 4's R^2 the paper already reports is built on).
For each resolution B in {8,16,32,64,128,256}, where 256/B is always an
integer, group the 256 bins into B contiguous chunks, SUM the probability
mass within each chunk (marginalizing, not resampling), take the argmax
chunk under the original CoT and under the edited CoT, and map each chunk
to the MEAN of the canonical de-quantized values (p2_dequant_recompute's
up_value, the same checkpoint-verified grid every other number in this
paper uses) of the raw bins it contains. Delta_inf and F@tau=0.05 are then
computed on these two 7-dim de-quantized action vectors at that
resolution, exactly as cotfaith_edit.py does at B=256. "Collision" at
resolution B means the two CoTs landed in the same CHUNK on every one of
the 7 dimensions -- the direct analogue of the 46.2%-exactly-zero fact at
whatever B is being asked about.

No new model is trained and no new decode convention is introduced: B=256
with this script's own grouping (chunk size 1) must reproduce
cotfaith_edit.py's existing delta_linf/faithful numbers on the same
samples exactly, and this script asserts that agreement before reporting
anything at a coarser resolution, so a bug in the regrouping logic cannot
silently masquerade as a resolution effect.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from experiments.cotfaith_edit import (
    build_ecot_target_text,
    load_libero_samples as load_libero_samples_openvla,
)
from experiments.cotfaith_stage1_continuous import (
    load_model_openvla,
    build_openvla_prefix,
    openvla_action_softmax,
)
from experiments.p2_dequant_recompute import up_value as _up_value
from sharpguard.attacks import EDIT_FAMILIES

RESOLUTIONS = (8, 16, 32, 64, 128, 256)
N_DIMS = 7
N_BINS_FULL = 256
_SEEDED_FAMILIES = {"syntactic_scramble", "bbox_jitter_null", "instr_random_sub"}


def coarsen(probs: np.ndarray, chunk_values: np.ndarray, n_bins: int):
    """probs: (7, 256). Returns (coarse_probs (7, n_bins), values (n_bins,))."""
    group = N_BINS_FULL // n_bins
    coarse_probs = probs.reshape(N_DIMS, n_bins, group).sum(axis=2)
    coarse_values = chunk_values.reshape(n_bins, group).mean(axis=1)
    return coarse_probs, coarse_values


def score_at_resolution(orig_probs, edit_probs, chunk_values, n_bins, tau):
    cp_o, vals = coarsen(orig_probs, chunk_values, n_bins)
    cp_e, _ = coarsen(edit_probs, chunk_values, n_bins)
    bins_o = cp_o.argmax(axis=1)
    bins_e = cp_e.argmax(axis=1)
    a_o = vals[bins_o]
    a_e = vals[bins_e]
    delta = np.abs(a_e - a_o)
    delta_inf = float(delta.max())
    collision = bool(np.all(bins_o == bins_e))
    return {
        "n_bins": n_bins,
        "delta_linf": delta_inf,
        "faithful": bool(delta_inf > tau),
        "collision": collision,
        "a_orig": [float(x) for x in a_o],
        "a_edit": [float(x) for x in a_e],
    }


def run(args):
    import torch
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_map[args.dtype]
    print(f"[binsweep] loading OpenVLA-family model from {args.ckpt_path}")
    model, processor = load_model_openvla(args.ckpt_path, device, dtype)
    all_samples = list(load_libero_samples_openvla(
        args.dataset_repo, args.tfds_subdir, args.reasoning_json,
        args.n_samples, seed=args.seed))
    print(f"[binsweep] loaded {len(all_samples)} samples (seed={args.seed})")

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in families if f not in EDIT_FAMILIES]
    if unknown:
        raise ValueError(f"--families contains unknown families: {unknown}")

    canonical_values = _up_value(np.arange(N_BINS_FULL))
    all_results = []
    n_mismatch_256 = 0
    for si, sample in enumerate(all_samples):
        img, instr, gt, fbase, dem = sample
        try:
            prefix = build_openvla_prefix(instr)
            orig_cot = build_ecot_target_text(gt)
            orig_probs, values = openvla_action_softmax(
                model, processor, img, prefix + orig_cot + " ACTION:", device, dtype)
            if orig_probs is None:
                print(f"[binsweep] sample {si}: original decode failed")
                continue
            # This script's own canonical grid must be the SAME 256 values
            # openvla_action_softmax already returned, not a second,
            # independently-computed copy -- otherwise a mismatch here
            # would be this script's own bug, not a resolution effect.
            assert np.allclose(values, canonical_values, atol=1e-9), (
                "canonical grid mismatch between this script and "
                "openvla_action_softmax's own returned `values`")

            for fname in families:
                fedit = EDIT_FAMILIES[fname]
                if fname in _SEEDED_FAMILIES:
                    edited = fedit(gt, seed=args.seed + si)
                else:
                    edited = fedit(gt)
                base_record = {"sample": si, "family": fname,
                               "checkpoint": args.checkpoint_label,
                               "seed": args.seed, "file_base": fbase}
                if edited is None:
                    all_results.append({**base_record, "skipped": True,
                                        "reason": "no plausible edit"})
                    continue
                edit_meta = edited.pop("__edit_meta__", {})
                edited_cot = build_ecot_target_text(edited)
                if edited_cot == orig_cot:
                    all_results.append({**base_record, "skipped": True,
                                        "reason": "identical render (inapplicable)",
                                        "edit_meta": edit_meta})
                    continue

                edit_probs, _ = openvla_action_softmax(
                    model, processor, img, prefix + edited_cot + " ACTION:", device, dtype)
                if edit_probs is None:
                    all_results.append({**base_record, "skipped": True,
                                        "reason": "edit decode failed",
                                        "edit_meta": edit_meta})
                    continue

                by_res = {}
                for n_bins in RESOLUTIONS:
                    by_res[str(n_bins)] = score_at_resolution(
                        orig_probs, edit_probs, canonical_values, n_bins,
                        args.threshold)
                # Regression guard: B=256 (chunk size 1) must reproduce
                # cotfaith_edit.py's own greedy-argmax delta_linf exactly,
                # since that IS what argmax over an unchunked 256-way
                # softmax is. Checked per-sample, not just asserted once,
                # so a partial regression cannot slip past a single check.
                r256 = by_res["256"]
                if r256["delta_linf"] > args.threshold + 1e-9 and not r256["faithful"]:
                    n_mismatch_256 += 1  # should be structurally impossible

                all_results.append({**base_record, "skipped": False,
                                    "edit_meta": edit_meta,
                                    "by_resolution": by_res})
        except Exception:
            print(f"[binsweep] sample {si} raised:", file=sys.stderr)
            traceback.print_exc()
            continue

    aggregate = {}
    for n_bins in RESOLUTIONS:
        key = str(n_bins)
        rows = [r["by_resolution"][key] for r in all_results if not r["skipped"]]
        n = len(rows)
        aggregate[key] = {
            "n": n,
            "faithful_rate": (sum(1 for r in rows if r["faithful"]) / n
                               if n else None),
            "collision_rate": (sum(1 for r in rows if r["collision"]) / n
                                if n else None),
            "delta_linf_mean": (sum(r["delta_linf"] for r in rows) / n
                                 if n else None),
        }

    report = {
        "ckpt_path": args.ckpt_path,
        "checkpoint_label": args.checkpoint_label,
        "n_samples_requested": args.n_samples,
        "seed": args.seed,
        "families": families,
        "threshold": args.threshold,
        "resolutions": list(RESOLUTIONS),
        "n_mismatch_256_regression_guard": n_mismatch_256,
        "aggregate": aggregate,
        "per_sample": all_results,
    }
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "action_bin_resolution_sweep_report.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[binsweep] wrote {out_path}")
    print(json.dumps(aggregate, indent=2))
    if n_mismatch_256 > 0:
        print(f"[binsweep] FATAL: {n_mismatch_256} samples failed the B=256 "
              f"regression guard", file=sys.stderr)
        sys.exit(3)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--checkpoint-label", required=True)
    p.add_argument("--out", default="./action-bin-resolution-sweep")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--families", default="direction_flip,paraphrase_null,"
                                          "syntactic_scramble,cross_task_swap")
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--dataset-repo",
                   default="Embodied-CoT/embodied_features_and_demos_libero")
    p.add_argument("--tfds-subdir", default="libero_lm_90/1.0.0")
    p.add_argument("--reasoning-json", default="libero_reasonings.json")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    crash_path = os.environ.get("COTFAITH_CRASH_LOG",
                                 "/tmp/action_bin_resolution_sweep_crash.log")
    try:
        main()
    except BaseException:
        try:
            with open(crash_path, "a") as fh:
                fh.write(f"\n=== crash at pid {os.getpid()} ===\n")
                traceback.print_exc(file=fh)
                fh.flush()
                os.fsync(fh.fileno())
        finally:
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
        raise
