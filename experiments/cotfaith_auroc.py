"""CoT-Faith rollout AUROC: attention pattern as failure predictor.

Extended version of cotfaith_rvis.py: for each LIBERO offline sample,
also predicts the action (greedy decode) and compares to the GT demo
action. Reports whether attention distribution predicts *action-error
magnitude* — a proxy for downstream task success.

Metric of interest:
  1. For each sample: compute (attn_action_to_cot, attn_action_to_visual,
     attn_action_to_instr) + (L1 action error vs GT)
  2. Binarize error at median → 'high-error' vs 'low-error' labels
  3. AUROC of each attention feature as failure predictor.

If action->cot attention distinguishes high-error from low-error samples
(AUROC > 0.6), attention IS predictive of failure — SAFE/FIPER-style
downstream utility claim. If AUROC ~= 0.5, attention doesn't help
predict per-step action error.

This mimics the SAFE (2506.09937) / FIPER (2510.09459) methodology of
using internal signals as failure predictors, restricted to CoT-VLAs.
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


ECOT_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

ECOT_TAGS_ORDER = [
    (("task",),                                    "TASK"),
    (("plan",),                                    "PLAN"),
    (("bboxes",),                                  "VISIBLE OBJECTS"),
    (("subtask_reasoning", "subtask_reason"),      "SUBTASK REASONING"),
    (("subtask",),                                 "SUBTASK"),
    (("movement_reasoning", "move_reasoning", "move_reason"), "MOVE REASONING"),
    (("movement", "move"),                         "MOVE"),
    (("gripper", "gripper_position"),              "GRIPPER POSITION"),
]


def build_target_text(reasoning):
    parts = []
    for keys, tag in ECOT_TAGS_ORDER:
        v = None
        for k in keys:
            if k in reasoning:
                v = reasoning[k]; break
        primary = keys[0]
        if primary == "bboxes":
            b = v
            if isinstance(b, dict):
                v_str = ", ".join(f"{n} {vv}" for n, vv in b.items())
            else:
                v_str = str(b) if b else ""
        elif primary in ("gripper", "gripper_position"):
            v_str = str(list(v)) if isinstance(v, (list, tuple)) else str(v or "")
        elif isinstance(v, dict):
            ks = sorted(v.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
            v_str = ". ".join(str(v[k]) for k in ks)
        else:
            v_str = str(v) if v else ""
        parts.append(f"{tag}: {v_str}")
    return " ".join(parts)


def dequantize(bin_ids, low=-1.0, high=1.0, bins=256):
    return low + (bin_ids + 0.5) * (high - low) / bins


def load_libero_samples(dataset_repo, tfds_subdir, reasoning_json, n_samples, seed=0):
    from huggingface_hub import snapshot_download
    ds_dir = Path(snapshot_download(repo_id=dataset_repo, repo_type="dataset",
                                       cache_dir=os.environ.get("HF_HOME")))
    tfds_dir = ds_dir / tfds_subdir
    with open(ds_dir / reasoning_json) as f:
        rdata = json.load(f)
    import tensorflow_datasets as tfds
    from PIL import Image as PILImage
    builder = tfds.builder_from_directory(str(tfds_dir))
    ds = builder.as_dataset(split="train",
                              shuffle_files=(seed != 0),
                              read_config=tfds.ReadConfig(shuffle_seed=seed))
    n = 0
    for ep in ds:
        if n >= n_samples: break
        meta = ep.get("episode_metadata", {})
        file_path = meta.get("file_path").numpy().decode()
        demo_id = int(meta.get("demo_id").numpy())
        file_base = os.path.basename(file_path)
        rep = rdata.get(file_path) or rdata.get(file_base) or {}
        rdemo = rep.get(str(demo_id), {})
        steps = list(ep["steps"].as_numpy_iterator())
        first = steps[0]
        gt = rdemo.get("0", {})
        if not gt: continue
        gt_action = first["action"].astype(np.float32)
        yield (PILImage.fromarray(first["observation"]["image"]).convert("RGB"),
                first["language_instruction"].decode(),
                gt, gt_action, file_base, demo_id)
        n += 1


def compute_auroc(scores, labels):
    """Non-parametric AUROC via rank-based Mann-Whitney U statistic."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg: return 0.5
    # Count pairs where pos > neg (+ 0.5 for ties).
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n: wins += 1
            elif p == n: wins += 0.5
    return wins / (len(pos) * len(neg))


def run(args):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from sharpguard.proguard import RVisHook, RVisConfig, CotAttentionAnalyzer

    dtype = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[auroc] loading model from {args.ckpt_path}")
    processor = AutoProcessor.from_pretrained(args.ckpt_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.ckpt_path, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device).eval()

    hook = RVisHook(model, RVisConfig(
        layers=tuple(int(x) for x in args.rvis_layers.split(",")),
        n_visual_tokens=256,
    ))
    analyzer = CotAttentionAnalyzer(hook, processor.tokenizer, n_visual=256)

    per_sample = []
    it = load_libero_samples(args.dataset_repo, args.tfds_subdir,
                              args.reasoning_json, args.n_samples,
                              seed=args.seed)
    for si, (img, instr, gt, gt_action, fbase, dem) in enumerate(it):
        try:
            prompt = (f"{ECOT_SYSTEM_PROMPT} USER: What action should the "
                       f"robot take to {instr.lower()}? ASSISTANT: ")
            cot_body = build_target_text(gt)

            # Predict action via greedy generation.
            infer_txt = prompt + cot_body + " ACTION:"
            proc = processor(infer_txt, img).to(device, dtype=dtype)
            with torch.no_grad():
                out = model.generate(**proc, max_new_tokens=8, do_sample=False)
            gen_ids = out[0, -8:].cpu().tolist()
            vocab = processor.tokenizer.vocab_size
            action_lo = vocab - 256
            bins = [vocab - 1 - t for t in gen_ids if action_lo <= t < vocab][:7]
            if len(bins) < 7: continue
            pred_action = dequantize(np.asarray(bins))

            # Action error vs GT.
            error_l1 = float(np.mean(np.abs(pred_action - gt_action)))
            error_linf = float(np.max(np.abs(pred_action - gt_action)))

            # Capture attention via teacher-forced forward on cot+action.
            a_ids = np.clip(np.floor((gt_action + 1) / 2 * 256).astype(np.int64), 0, 255)
            a_ids = vocab - 1 - a_ids
            input_ids = proc["input_ids"][0].to("cpu")
            eos = processor.tokenizer.eos_token_id
            full_ids = torch.cat([input_ids,
                                    torch.from_numpy(a_ids).to(input_ids.dtype),
                                    torch.tensor([eos], dtype=input_ids.dtype)])
            full_ids = full_ids.unsqueeze(0).to(device)
            attn_mask = torch.ones_like(full_ids)
            pixel = proc["pixel_values"].to(device, dtype=dtype)
            hook.clear()
            with torch.no_grad():
                _ = model(input_ids=full_ids, attention_mask=attn_mask,
                            pixel_values=pixel, output_attentions=True)
            seg = analyzer.compute_segments(full_ids[0], action_len=7)
            stats = analyzer.analyze(seg)

            per_sample.append({
                "sample": si,
                "instruction": instr[:200],
                "action_error_l1":   error_l1,
                "action_error_linf": error_linf,
                "action_pred":  [float(x) for x in pred_action],
                "action_gt":    [float(x) for x in gt_action],
                "action->cot":         stats.get("action->cot"),
                "action->visual":      stats.get("action->visual"),
                "action->instr":       stats.get("action->instr"),
                "action->action_prev": stats.get("action->action_prev"),
            })
            if (si + 1) % 20 == 0:
                print(f"[auroc] {si+1}/{args.n_samples} samples done")
        except Exception as e:
            print(f"[auroc] sample {si} failed: {e}\n{traceback.format_exc()[-400:]}")

    hook.close()

    # Compute AUROC of each attention feature as high-error predictor.
    errors = [s["action_error_l1"] for s in per_sample]
    median_err = float(np.median(errors))
    labels = [1 if s["action_error_l1"] > median_err else 0 for s in per_sample]

    aurocs = {}
    for feat in ["action->cot", "action->visual", "action->instr",
                  "action->action_prev"]:
        scores = [s[feat] for s in per_sample if s.get(feat) is not None]
        # AUROC works both directions; report max(auc, 1-auc) as absolute.
        if len(scores) != len(labels): continue
        raw = compute_auroc(scores, labels)
        aurocs[feat] = {
            "raw_auroc": float(raw),
            "abs_auroc": float(max(raw, 1.0 - raw)),
            "direction": "high-score → high-error" if raw > 0.5 else "low-score → high-error",
        }

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "cot_auroc_report.json").write_text(json.dumps({
        "n_samples": len(per_sample),
        "seed": args.seed,
        "median_error_l1": median_err,
        "aurocs": aurocs,
        "per_sample": per_sample,
    }, indent=2, default=str))

    print(f"\n===== ROLLOUT-STYLE AUROC DONE  seed={args.seed} =====")
    print(f"  n_samples used: {len(per_sample)}")
    print(f"  median action L1 error: {median_err:.4f}")
    for feat, a in aurocs.items():
        print(f"  {feat:22s}  AUROC={a['raw_auroc']:.3f}  abs={a['abs_auroc']:.3f}  ({a['direction']})")
    print(f"  report -> {out / 'cot_auroc_report.json'}")
    sys.stdout.flush(); os._exit(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--out", default="./cotfaith-auroc")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rvis-layers", default="0,1,2,3")
    p.add_argument("--dataset-repo",
                     default="Embodied-CoT/embodied_features_and_demos_libero")
    p.add_argument("--tfds-subdir", default="libero_lm_90/1.0.0")
    p.add_argument("--reasoning-json", default="libero_reasonings.json")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
