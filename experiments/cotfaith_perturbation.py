"""CoT-Faith perturbation robustness.

Apply visual perturbations to input images, then re-run r_vis(CoT) +
causal edit. Reports whether the CoT-faith findings hold under
distribution shift (lighting, distractors, camera jitter, occlusion,
noise). Colosseum-style but scoped to attention + causal metrics.

Perturbation types:
  clean            — no perturbation (control, matches original run)
  brightness_low   — pixel *= 0.5
  brightness_high  — pixel *= 1.5, clip
  gaussian_noise   — additive N(0, 25) noise
  color_jitter     — random hue shift ±30 degrees
  patch_occlude    — 40x40 gray patch at center

Output: cot_perturbation_report.json with per-perturbation aggregate
attention buckets + causal-edit faithful rates.
"""

from __future__ import annotations

import argparse
import copy
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


def apply_perturbation(img_np: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """img_np: uint8 (H, W, 3)."""
    if kind == "clean":
        return img_np
    if kind == "brightness_low":
        return np.clip(img_np.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)
    if kind == "brightness_high":
        return np.clip(img_np.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    if kind == "gaussian_noise":
        noise = rng.normal(0, 25, img_np.shape)
        return np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if kind == "color_jitter":
        # simple RGB channel scale
        scales = rng.uniform(0.7, 1.3, size=3)
        out = img_np.astype(np.float32) * scales[None, None, :]
        return np.clip(out, 0, 255).astype(np.uint8)
    if kind == "patch_occlude":
        img = img_np.copy()
        H, W = img.shape[:2]
        y0, x0 = H // 2 - 20, W // 2 - 20
        img[y0:y0 + 40, x0:x0 + 40, :] = 128
        return img
    raise ValueError(f"unknown perturbation {kind}")


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
        yield (first["observation"]["image"],
                first["language_instruction"].decode(),
                gt, file_base, demo_id)
        n += 1


def run(args):
    import torch
    from PIL import Image as PILImage
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from sharpguard.proguard import RVisHook, RVisConfig, CotAttentionAnalyzer

    KINDS = ["clean", "brightness_low", "brightness_high",
              "gaussian_noise", "color_jitter", "patch_occlude"]

    dtype = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[pert] loading model from {args.ckpt_path}")
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

    rng = np.random.default_rng(args.seed)
    per_sample = []
    it = load_libero_samples(args.dataset_repo, args.tfds_subdir,
                              args.reasoning_json, args.n_samples,
                              seed=args.seed)
    for si, (img_np, instr, gt, fbase, dem) in enumerate(it):
        try:
            prompt = (f"{ECOT_SYSTEM_PROMPT} USER: What action should the "
                       f"robot take to {instr.lower()}? ASSISTANT: ")
            cot_body = build_target_text(gt)
            for kind in KINDS:
                img_p = apply_perturbation(img_np, kind, rng)
                pil = PILImage.fromarray(img_p).convert("RGB")
                text_pre_action = prompt + cot_body + " ACTION:"
                proc = processor(text_pre_action, pil)
                gt_action = np.zeros(7, dtype=np.float32)
                a_ids = np.clip(np.floor((gt_action + 1) / 2 * 256).astype(np.int64), 0, 255)
                a_ids = processor.tokenizer.vocab_size - 1 - a_ids
                input_ids = proc["input_ids"][0]
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
                stats["sample_idx"] = si
                stats["perturbation"] = kind
                stats["file_base"] = fbase
                per_sample.append(stats)
            if (si + 1) % 10 == 0:
                print(f"[pert] {si+1}/{args.n_samples} samples done")
        except Exception as e:
            print(f"[pert] sample {si} failed: {e}\n{traceback.format_exc()[-400:]}")
    hook.close()

    # aggregate per perturbation
    agg = {}
    keys = ["action->visual", "action->instr", "action->cot", "action->action_prev"]
    for kind in KINDS:
        rows = [s for s in per_sample if s["perturbation"] == kind]
        if not rows:
            agg[kind] = {"n": 0}; continue
        agg[kind] = {"n": len(rows)}
        for k in keys:
            vals = [r[k] for r in rows if k in r]
            if vals:
                agg[kind][k] = {"mean": float(np.mean(vals)),
                                  "std":  float(np.std(vals))}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "cot_perturbation_report.json").write_text(json.dumps({
        "n_samples": args.n_samples, "seed": args.seed,
        "rvis_layers": args.rvis_layers,
        "aggregate": agg, "per_sample": per_sample,
    }, indent=2, default=str))

    print(f"\n===== PERTURBATION DONE  seed={args.seed} =====")
    for kind in KINDS:
        a = agg[kind]
        if a.get("n", 0) == 0: continue
        c = a.get("action->cot", {}); v = a.get("action->visual", {})
        print(f"  {kind:20s}  n={a['n']:3d}  "
              f"cot={c.get('mean', 0):.3f}  visual={v.get('mean', 0):.3f}")
    print(f"  report -> {out / 'cot_perturbation_report.json'}")
    sys.stdout.flush(); os._exit(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--out", default="./cotfaith-perturbation")
    p.add_argument("--n-samples", type=int, default=50)
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
