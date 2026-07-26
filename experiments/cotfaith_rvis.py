"""CoT-Faith r_vis(CoT) attention analysis on our ECoT-LIBERO checkpoint.

Loads the merged model from a local path (produced by cotfaith_train.py),
runs a forward pass on N real LIBERO scenes with teacher-forced GT CoT
targets, and reports per-segment attention mass:

    action -> {visual, instruction, cot, action_prev}

Key headline metric: mean fraction of action-token attention that goes
to CoT tokens. If <5-10%, CoT is likely decorative; if >20%, CoT
plausibly drives the action decode.

Sample count is small (N=20 by default) so this runs in ~10 min after
training completes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# Repo root on sys.path so we can `from sharpguard.proguard import ...`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np


ECOT_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def load_libero_samples(dataset_repo: str, tfds_subdir: str, reasoning_json: str,
                         n_samples: int):
    """Yield the first n samples' (image_pil, instruction, reasoning_dict)."""
    from huggingface_hub import snapshot_download
    ds_dir = Path(snapshot_download(repo_id=dataset_repo, repo_type="dataset",
                                       cache_dir=os.environ.get("HF_HOME")))
    tfds_dir = ds_dir / tfds_subdir
    with open(ds_dir / reasoning_json) as f:
        rdata = json.load(f)

    import tensorflow_datasets as tfds
    from PIL import Image as PILImage
    builder = tfds.builder_from_directory(str(tfds_dir))
    ds = builder.as_dataset(split="train", shuffle_files=False)
    n = 0
    for ep in ds:
        if n >= n_samples: break
        meta = ep.get("episode_metadata", {})
        file_path = meta.get("file_path").numpy().decode()
        demo_id = int(meta.get("demo_id").numpy())
        file_base = os.path.basename(file_path)
        rep = (rdata.get(file_path) or rdata.get(file_base) or {})
        rdemo = rep.get(str(demo_id), {})
        steps = list(ep["steps"].as_numpy_iterator())
        first = steps[0]
        gt = rdemo.get("0", {})
        if not gt: continue
        yield (PILImage.fromarray(first["observation"]["image"]).convert("RGB"),
                first["language_instruction"].decode(),
                gt, file_base, demo_id)
        n += 1


def build_ecot_target_text(reasoning: dict) -> str:
    """Same 9-tag format as training (from cotfaith_train.py)."""
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

    parts = []
    for keys, tag in ECOT_TAGS_ORDER:
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
        parts.append(f"{tag}: {v_str}")
    return " ".join(parts)


def _quantize_action(a, low=-1.0, high=1.0, bins=256):
    a = np.clip(a, low, high)
    frac = (a - low) / (high - low)
    idx = np.floor(frac * bins).astype(np.int64)
    return np.clip(idx, 0, bins - 1)


def action_ids(a, vocab_size, bins=256):
    return vocab_size - 1 - _quantize_action(a, bins=bins)


def run_analysis(args):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from sharpguard.proguard import RVisHook, RVisConfig, CotAttentionAnalyzer

    dtype = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[attn] loading model from {args.ckpt_path}")
    processor = AutoProcessor.from_pretrained(args.ckpt_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.ckpt_path, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device).eval()

    # Install r_vis hook on the model's LLM layers 0-3 for attention capture.
    hook = RVisHook(model, RVisConfig(
        layers=tuple(int(x) for x in args.rvis_layers.split(",")),
        n_visual_tokens=256,
    ))
    analyzer = CotAttentionAnalyzer(hook, processor.tokenizer, n_visual=256)

    per_sample_stats = []
    print(f"[attn] iterating {args.n_samples} LIBERO samples")
    it = load_libero_samples(args.dataset_repo, args.tfds_subdir,
                              args.reasoning_json, args.n_samples)
    for si, (img, instr, gt, fbase, dem) in enumerate(it):
        try:
            prompt = (f"{ECOT_SYSTEM_PROMPT} USER: What action should the "
                       f"robot take to {instr.lower()}? ASSISTANT: ")
            cot_body = build_ecot_target_text(gt)
            text_pre_action = prompt + cot_body + " ACTION:"

            proc = processor(text_pre_action, img)
            # Add 7 action tokens (using GT action) + EOS
            gt_action = np.zeros(7, dtype=np.float32)   # placeholder; won't affect attn much
            a_ids = action_ids(gt_action, processor.tokenizer.vocab_size)
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
            stats["instruction"] = instr[:200]
            stats["file_base"] = fbase
            per_sample_stats.append(stats)
            print(f"[attn]  {si+1}/{args.n_samples}  "
                    f"action->cot={stats['action->cot']:.3f}  "
                    f"action->visual={stats['action->visual']:.3f}  "
                    f"action->instr={stats['action->instr']:.3f}  "
                    f"action->prev={stats['action->action_prev']:.3f}")
        except Exception as e:
            print(f"[attn] sample {si} failed: {e}\n{traceback.format_exc()[-500:]}")

    hook.close()

    # Aggregate
    keys = ["action->cot", "action->visual", "action->instr", "action->action_prev",
             "cot->visual", "cot->instr", "cot->cot_self",
             "per_source_action_mass_total"]
    agg = {}
    for k in keys:
        vals = [s[k] for s in per_sample_stats if k in s]
        agg[k] = {"mean": float(np.mean(vals)) if vals else None,
                    "std":  float(np.std(vals)) if vals else None,
                    "n":    len(vals)}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "rvis_cot_report.json").write_text(json.dumps({
        "n_samples": len(per_sample_stats),
        "rvis_layers": args.rvis_layers,
        "aggregate": agg,
        "per_sample": per_sample_stats,
    }, indent=2, default=str))

    print(f"\n===== r_vis(CoT) DONE =====")
    print(f"  n_samples: {len(per_sample_stats)}")
    for k in keys[:4]:
        v = agg[k]
        if v["mean"] is not None:
            print(f"  {k:24s}  mean={v['mean']:.3f}  std={v['std']:.3f}")
    print(f"  report -> {out / 'rvis_cot_report.json'}")

    sys.stdout.flush()
    os._exit(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--out", default="./cotfaith-rvis")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--rvis-layers", default="0,1,2,3")
    p.add_argument("--dataset-repo",
                     default="Embodied-CoT/embodied_features_and_demos_libero")
    p.add_argument("--tfds-subdir", default="libero_lm_90/1.0.0")
    p.add_argument("--reasoning-json", default="libero_reasonings.json")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
