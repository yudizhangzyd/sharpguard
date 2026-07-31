"""CoT-Faith prompt-format ablation.

Which parts of the 9-tag CoT are load-bearing? For each LIBERO sample:
run inference on the ORIGINAL 8-tag target + 5 truncated variants, and
report Δaction from full → truncated. Larger delta means that segment
was doing meaningful work.

Variants (target text after 'ASSISTANT:'):
  full         — TASK PLAN VISIBLE_OBJECTS SUBTASK_REASONING SUBTASK
                  MOVE_REASONING MOVE GRIPPER_POSITION ACTION
  task_only    — TASK ACTION
  plan_only    — TASK PLAN ACTION
  task_plan_subtask — TASK PLAN SUBTASK ACTION
  shuffled     — full text with tag ORDER shuffled (grammar destroyed,
                  content preserved)
  empty        — ACTION only (no reasoning)

Output: cot_prompt_report.json with per-sample per-variant action
delta (L1, L∞) and faithful_rate at threshold.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
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


def _fmt_bb(b):
    if isinstance(b, dict):
        return ", ".join(f"{n} {v}" for n, v in b.items())
    return str(b) if b else ""


def build_target_text(reasoning: dict, variant: str = "full",
                        seed: int = 0) -> str:
    """Construct the assistant target under a specific ablation variant."""
    parts = []
    order = list(ECOT_TAGS_ORDER)
    for keys, tag in order:
        v = None
        for k in keys:
            if k in reasoning:
                v = reasoning[k]; break
        primary = keys[0]
        if primary == "bboxes":
            v_str = _fmt_bb(v)
        elif primary in ("gripper", "gripper_position"):
            v_str = str(list(v)) if isinstance(v, (list, tuple)) else str(v or "")
        elif isinstance(v, dict):
            ks = sorted(v.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
            v_str = ". ".join(str(v[k]) for k in ks)
        else:
            v_str = str(v) if v else ""
        parts.append((tag, v_str))

    if variant == "full":
        selected = parts
    elif variant == "task_only":
        selected = [(t, v) for t, v in parts if t == "TASK"]
    elif variant == "plan_only":
        selected = [(t, v) for t, v in parts if t in ("TASK", "PLAN")]
    elif variant == "task_plan_subtask":
        selected = [(t, v) for t, v in parts if t in ("TASK", "PLAN", "SUBTASK")]
    elif variant == "shuffled":
        selected = parts[:]
        random.Random(seed).shuffle(selected)
    elif variant == "empty":
        selected = []
    else:
        raise ValueError(f"unknown variant {variant}")

    return " ".join(f"{t}: {v}" for t, v in selected)


def load_libero_samples(dataset_repo, tfds_subdir, reasoning_json, n_samples,
                          seed=0):
    from sharpguard.hf_retry import snapshot_with_retry
    ds_dir = Path(snapshot_with_retry(repo_id=dataset_repo,
                                        repo_type="dataset"))
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
        yield (PILImage.fromarray(first["observation"]["image"]).convert("RGB"),
                first["language_instruction"].decode(),
                gt, file_base, demo_id)
        n += 1


def _quant(a, bins=256):
    a = np.clip(a, -1, 1)
    return np.clip(np.floor((a + 1) / 2 * bins).astype(np.int64), 0, bins - 1)


def infer_action(model, processor, text_pre_action, image, device, dtype):
    import torch
    inputs = processor(text_pre_action, image).to(device, dtype=dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    gen_ids = out[0, -8:].cpu().tolist()
    vocab = processor.tokenizer.vocab_size
    action_lo = vocab - 256
    bins = []
    for tid in gen_ids:
        if action_lo <= tid < vocab:
            bins.append(vocab - 1 - tid)
        if len(bins) == 7: break
    if len(bins) < 7: return None
    return -1 + (np.asarray(bins) + 0.5) * 2 / 256


def run(args):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    VARIANTS = ["full", "task_only", "plan_only", "task_plan_subtask",
                 "shuffled", "empty"]

    dtype = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[prompt] loading model from {args.ckpt_path}")
    processor = AutoProcessor.from_pretrained(args.ckpt_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.ckpt_path, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device).eval()

    all_results = []
    it = load_libero_samples(args.dataset_repo, args.tfds_subdir,
                              args.reasoning_json, args.n_samples,
                              seed=args.seed)
    for si, (img, instr, gt, fbase, dem) in enumerate(it):
        try:
            prompt = (f"{ECOT_SYSTEM_PROMPT} USER: What action should the robot "
                       f"take to {instr.lower()}? ASSISTANT: ")
            # baseline: full CoT
            full_body = build_target_text(gt, "full")
            a_full = infer_action(model, processor,
                                   prompt + full_body + " ACTION:", img,
                                   device, dtype)
            if a_full is None: continue

            for v in VARIANTS:
                body = build_target_text(gt, v, seed=args.seed + si)
                a_v = infer_action(model, processor,
                                    prompt + body + " ACTION:", img,
                                    device, dtype)
                if a_v is None: continue
                d = a_v - a_full
                all_results.append({
                    "sample": si, "variant": v, "seed": args.seed,
                    "instruction": instr[:200], "file_base": fbase,
                    "delta_l1_mean": float(np.mean(np.abs(d))),
                    "delta_linf":    float(np.max(np.abs(d))),
                    "faithful":      float(np.max(np.abs(d))) > args.threshold,
                })
            if (si + 1) % 20 == 0:
                print(f"[prompt] {si+1}/{args.n_samples} samples done")
        except Exception as e:
            print(f"[prompt] sample {si} failed: {e}\n"
                    f"{traceback.format_exc()[-400:]}")

    agg = {}
    for v in VARIANTS:
        rows = [r for r in all_results if r["variant"] == v]
        if not rows:
            agg[v] = {"n": 0}; continue
        agg[v] = {
            "n": len(rows),
            "delta_l1_mean_mean": float(np.mean([r["delta_l1_mean"] for r in rows])),
            "delta_linf_median": float(np.median([r["delta_linf"] for r in rows])),
            "delta_linf_mean":   float(np.mean([r["delta_linf"] for r in rows])),
            "faithful_rate":     float(np.mean([r["faithful"] for r in rows])),
        }

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "cot_prompt_report.json").write_text(json.dumps({
        "n_samples": args.n_samples, "seed": args.seed,
        "threshold": args.threshold,
        "aggregate": agg, "per_sample": all_results,
    }, indent=2, default=str))

    print(f"\n===== PROMPT ABLATION DONE  seed={args.seed} =====")
    for v in VARIANTS:
        a = agg[v]
        if a["n"] == 0: continue
        print(f"  {v:22s}  n={a['n']:3d}  "
              f"L∞med={a['delta_linf_median']:.3f}  "
              f"faithful={a['faithful_rate']:.3f}")
    print(f"  report -> {out / 'cot_prompt_report.json'}")
    sys.stdout.flush(); os._exit(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--out", default="./cotfaith-prompt")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--dataset-repo",
                     default="Embodied-CoT/embodied_features_and_demos_libero")
    p.add_argument("--tfds-subdir", default="libero_lm_90/1.0.0")
    p.add_argument("--reasoning-json", default="libero_reasonings.json")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
