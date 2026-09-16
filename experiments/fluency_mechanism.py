"""Fluency mechanism check: does judged fluency predict the null floors'
effect size, independent of edit-distance and length?

Limitations already discloses this as unresolved: paraphrase_null's floor
correlates with judged fluency at Spearman rho=-0.60 across all 13 families
(a family-level correlation), and mechanism_regression.py already ruled out
the policy's own teacher-forced log-likelihood (a *related* but distinct
covariate -- surprise to the policy's own LM head, not judged naturalness)
as the driver, at partial R^2 <= 0.037. This closes the actual gap: a
per-SAMPLE regression of the continuous effect size on judged fluency,
within the three null families, controlling for edit-distance and length.

Why this needs new inference rather than reusing existing artifacts: the
437 judge-rated pairs (results_v2/canonical_runs/judge_edit_families) and
the large-scale scored edit runs (results_v2/canonical_runs/*_edit*.json)
were drawn from two independent, differently-seeded sample draws over the
same 90-task corpus -- matching on (family, file_base, sample) finds only
10/437 overlaps, not enough to regress anything. So this script scores the
SAME samples both ways in one pass: teacher-forced TV (mirroring Table 2's
continuous metric and mechanism_regression.py's outcome variable) AND the
blind judge's fluency verdict on the same (orig_cot, edited_cot) pair,
restricted to the three null families (paraphrase_null, syntactic_scramble,
bbox_jitter_null) on ECoT-bridge, where the family-level correlation was
measured.

Two-phase to fit one GPU: phase 1 loads the VLA and scores tv_mean/
edit_distance/len_delta for each sample, saving the edited CoT text
(discarded by cotfaith_stage1_continuous.py's own output) instead of
throwing it away; phase 2 releases the VLA, loads the judge, and rates
fluency on the saved text pairs.

Usage:
    python3 experiments/fluency_mechanism.py --ckpt-path /tmp/cotfaith_ckpt \
        --out results_v2/canonical_runs/fluency_mechanism --n-samples 100
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from experiments.cotfaith_edit import build_ecot_target_text, load_libero_samples
from experiments.cotfaith_stage1_continuous import (
    openvla_action_softmax, text_covariates, total_variation,
    build_openvla_prefix,
)
from sharpguard.attacks import EDIT_FAMILIES

NULL_FAMILIES = ["paraphrase_null", "syntactic_scramble", "bbox_jitter_null"]
_SEEDED_FAMILIES = {"syntactic_scramble", "bbox_jitter_null"}


def phase1_score(args, dtype):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fluency-mech] loading VLA from {args.ckpt_path}")
    processor = AutoProcessor.from_pretrained(args.ckpt_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.ckpt_path, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device).eval()

    all_samples = list(load_libero_samples(
        args.dataset_repo, args.tfds_subdir, args.reasoning_json,
        args.n_samples, seed=args.seed))
    print(f"[fluency-mech] {len(all_samples)} samples loaded")

    records = []
    for si, (img, instr, gt, fbase, dem) in enumerate(all_samples):
        try:
            prefix = build_openvla_prefix(instr)
            orig_cot = build_ecot_target_text(gt)
            orig_probs, _ = openvla_action_softmax(
                model, processor, img, prefix + orig_cot + " ACTION:", device, dtype)
            if orig_probs is None:
                print(f"[fluency-mech] sample {si}: original decode failed")
                continue

            for fname in NULL_FAMILIES:
                fedit = EDIT_FAMILIES[fname]
                edited = (fedit(gt, seed=args.seed + si) if fname in _SEEDED_FAMILIES
                          else fedit(gt))
                base = {"sample": si, "family": fname, "file_base": fbase}
                if edited is None:
                    records.append({**base, "skipped": True,
                                     "reason": "no plausible edit"})
                    continue
                edited.pop("__edit_meta__", None)
                edited_cot = build_ecot_target_text(edited)
                if edited_cot == orig_cot:
                    records.append({**base, "skipped": True,
                                     "reason": "identical render"})
                    continue

                edit_probs, _ = openvla_action_softmax(
                    model, processor, img, prefix + edited_cot + " ACTION:",
                    device, dtype)
                if edit_probs is None:
                    records.append({**base, "skipped": True,
                                     "reason": "edit decode failed"})
                    continue

                tv_per_dim = [total_variation(orig_probs[d], edit_probs[d])
                              for d in range(7)]
                cov = text_covariates(orig_cot, edited_cot)
                records.append({
                    **base, "skipped": False,
                    "tv_mean": float(np.mean(tv_per_dim)),
                    "edit_distance_tokens": cov["edit_distance_tokens"],
                    "len_delta_tokens": cov["len_delta_tokens"],
                    "orig_cot": orig_cot, "edited_cot": edited_cot,
                })
        except Exception as e:
            print(f"[fluency-mech] sample {si} crashed: {type(e).__name__}: {e}")
            continue
        if si % 10 == 0:
            print(f"[fluency-mech] {si}/{len(all_samples)} samples scored, "
                  f"{sum(1 for r in records if not r.get('skipped'))} usable so far")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return records


def phase2_judge(records, dtype):
    from experiments.cotfaith_judge_edits import Judge

    judge_model_ids = ["Qwen/Qwen2.5-7B-Instruct"]
    judge = Judge(judge_model_ids, dtype)
    n_scored = 0
    for r in records:
        if r.get("skipped"):
            continue
        v, raw = judge.ask(r["orig_cot"], r["edited_cot"])
        if v is None:
            r["skipped"] = True
            r["reason"] = "judge parse failed"
            r["judge_raw"] = raw
            continue
        r["b_fluent"] = v.get("b_fluent")
        r["same_meaning"] = v.get("same_meaning")
        n_scored += 1
        if n_scored % 20 == 0:
            print(f"[fluency-mech] judge scored {n_scored} pairs")
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dataset-repo", default="Embodied-CoT/embodied_features_and_demos_libero")
    p.add_argument("--tfds-subdir", default="libero_lm_90/1.0.0")
    p.add_argument("--reasoning-json", default="libero_reasonings.json")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    import torch
    dtype = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}[args.dtype]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = phase1_score(args, dtype)
    n_usable = sum(1 for r in records if not r.get("skipped"))
    print(f"[fluency-mech] phase 1 done: {n_usable}/{len(records)} usable")
    (out_dir / "phase1_scored.json").write_text(json.dumps(records, indent=2))

    records = phase2_judge(records, dtype)
    n_judged = sum(1 for r in records if r.get("b_fluent") is not None)
    print(f"[fluency-mech] phase 2 done: {n_judged} judged")
    (out_dir / "fluency_mechanism_report.json").write_text(json.dumps({
        "n_samples_requested": args.n_samples,
        "families": NULL_FAMILIES,
        "n_usable": n_usable,
        "n_judged": n_judged,
        "per_sample": records,
    }, indent=2))
    print(f"[fluency-mech] -> {out_dir}/fluency_mechanism_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
