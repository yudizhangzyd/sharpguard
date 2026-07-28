"""Non-CoT baseline attention analysis: OpenVLA-7B on LIBERO.

Same 4-bucket attention analysis as cotfaith_rvis.py but for a MODEL
THAT DOES NOT PRODUCE CoT (vanilla OpenVLA-7B fine-tuned on LIBERO).

The comparison isolates the "CoT is under-weighted" claim: if OpenVLA
(no CoT) shows visual >> instr attention, then adding a CoT segment
does redirect attention from visual toward CoT. If OpenVLA also shows
~30% visual / ~30% instr, then the CoT model's numbers reflect a
generic VLA attention shape, not a CoT-specific effect.

For plain OpenVLA the prompt is short:
    "In: What action should the robot take to {instr}?\nOut: "
followed by 7 action tokens (no CoT). Segments:
    visual: [0, 256)  |  instruction: [256, action_start)  |
    empty cot: [action_start, action_start)  |  action: [action_start, end)

The CotAttentionAnalyzer handles empty CoT gracefully.
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


def load_libero_samples(dataset_repo, tfds_subdir, reasoning_json, n_samples):
    """Reuse the same first-step sampler as cotfaith_rvis.py so the
    baseline analysis sees the SAME 20 scenes."""
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
        rep = rdata.get(file_path) or rdata.get(file_base) or {}
        rdemo = rep.get(str(demo_id), {})
        steps = list(ep["steps"].as_numpy_iterator())
        first = steps[0]
        gt = rdemo.get("0", {})
        if not gt: continue
        yield (PILImage.fromarray(first["observation"]["image"]).convert("RGB"),
                first["language_instruction"].decode(),
                fbase := file_base, demo_id)
        n += 1


def _quantize(a, low=-1.0, high=1.0, bins=256):
    a = np.clip(a, low, high)
    frac = (a - low) / (high - low)
    idx = np.floor(frac * bins).astype(np.int64)
    return np.clip(idx, 0, bins - 1)


def action_ids(a, vocab_size, bins=256):
    return vocab_size - 1 - _quantize(a, bins=bins)


def run_baseline(args):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from sharpguard.proguard import RVisHook, RVisConfig, CotAttentionAnalyzer

    dtype = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[baseline] loading OpenVLA-7B from {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device).eval()

    hook = RVisHook(model, RVisConfig(
        layers=tuple(int(x) for x in args.rvis_layers.split(",")),
        n_visual_tokens=256,
    ))
    # For plain OpenVLA, instruction ends at "Out:" and there's no "ACTION:"
    # in the prompt — CoT segment will collapse to zero length.
    analyzer = CotAttentionAnalyzer(
        hook, processor.tokenizer, n_visual=256,
        instr_end_marker="Out:", cot_end_marker="ACTION_NOT_PRESENT",
    )

    per_sample = []
    it = load_libero_samples(args.dataset_repo, args.tfds_subdir,
                              args.reasoning_json, args.n_samples)
    for si, (img, instr, fbase, dem) in enumerate(it):
        try:
            prompt = f"In: What action should the robot take to {instr.lower()}?\nOut:"
            proc = processor(prompt, img)
            input_ids = proc["input_ids"][0]

            gt_action = np.zeros(7, dtype=np.float32)
            a_ids = action_ids(gt_action, processor.tokenizer.vocab_size)
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
            per_sample.append(stats)
            print(f"[baseline] {si+1}/{args.n_samples}  "
                    f"action->visual={stats['action->visual']:.3f}  "
                    f"action->instr={stats['action->instr']:.3f}  "
                    f"action->cot={stats.get('action->cot', 0):.3f}  "
                    f"action->prev={stats['action->action_prev']:.3f}")
        except Exception as e:
            print(f"[baseline] sample {si} failed: {e}\n{traceback.format_exc()[-500:]}")

    hook.close()

    keys = ["action->cot", "action->visual", "action->instr", "action->action_prev",
             "per_source_action_mass_total"]
    agg = {}
    for k in keys:
        vals = [s[k] for s in per_sample if k in s]
        agg[k] = {"mean": float(np.mean(vals)) if vals else None,
                    "std":  float(np.std(vals)) if vals else None,
                    "n":    len(vals)}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "rvis_baseline_report.json").write_text(json.dumps({
        "model": args.base_model,
        "n_samples": len(per_sample),
        "rvis_layers": args.rvis_layers,
        "aggregate": agg,
        "per_sample": per_sample,
    }, indent=2, default=str))

    print(f"\n===== BASELINE r_vis DONE =====")
    for k in keys[:4]:
        v = agg[k]
        if v["mean"] is not None:
            print(f"  {k:24s}  mean={v['mean']:.3f}  std={v['std']:.3f}")
    print(f"  report -> {out / 'rvis_baseline_report.json'}")

    sys.stdout.flush()
    os._exit(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model",
                   default="openvla/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--out", default="./cotfaith-rvis-baseline")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--rvis-layers", default="0,1,2,3")
    p.add_argument("--dataset-repo",
                     default="Embodied-CoT/embodied_features_and_demos_libero")
    p.add_argument("--tfds-subdir", default="libero_lm_90/1.0.0")
    p.add_argument("--reasoning-json", default="libero_reasonings.json")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()
    run_baseline(args)


if __name__ == "__main__":
    main()
