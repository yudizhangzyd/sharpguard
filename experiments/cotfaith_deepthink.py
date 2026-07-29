"""DeepThinkVLA rvis + causal-edit evaluation.

DeepThinkVLA is PaliGemma-based (arch='OpenpiFastOft'), not OpenVLA.
Requires transformers>=4.42 for PaliGemma. Different prompt format than
ECoT (uses <think>...</think> tags per the paper).

Strategy: use the same 4-bucket attention analyzer + same 10-edit
causal metric, but adapt the prompt template to DeepThinkVLA's format.
"""
from __future__ import annotations
import argparse, json, os, sys, traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np


def load_libero_samples(dataset_repo, tfds_subdir, reasoning_json,
                          n_samples, seed=0):
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
        yield (PILImage.fromarray(first["observation"]["image"]).convert("RGB"),
                first["language_instruction"].decode(),
                gt, file_base, demo_id, first["action"].astype(np.float32))
        n += 1


def dequantize(bin_ids, low=-1.0, high=1.0, bins=256):
    return low + (bin_ids + 0.5) * (high - low) / bins


def action_ids(a, vocab, bins=256):
    a = np.clip(a, -1, 1)
    idx = np.clip(np.floor((a + 1) / 2 * bins).astype(np.int64), 0, bins-1)
    return vocab - 1 - idx


def build_deepthink_prompt(instruction, cot_text=None):
    """DeepThinkVLA prompt format. Based on OpenpiFastOft architecture."""
    p = f"Instruction: {instruction}\n"
    if cot_text is not None:
        p += f"<think>{cot_text}</think>\n"
    p += "Action:"
    return p


def build_cot_text(reasoning):
    """Compact CoT text for DeepThinkVLA (no ECoT nine-tag; single block)."""
    parts = []
    for k in ("plan", "subtask", "movement", "move"):
        v = reasoning.get(k)
        if isinstance(v, dict):
            ks = sorted(v.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
            parts.append(f"{k}: " + ". ".join(str(v[k2]) for k2 in ks))
        elif isinstance(v, str) and v:
            parts.append(f"{k}: {v}")
    return "; ".join(parts)


def run(args):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoModel
    from sharpguard.proguard import RVisHook, RVisConfig, CotAttentionAnalyzer
    from sharpguard.attacks import EDIT_FAMILIES

    dtype = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[deepthink] loading {args.ckpt_path}")
    processor = AutoProcessor.from_pretrained(args.ckpt_path, trust_remote_code=True)
    # Try Vision2Seq first, fall back to plain AutoModel.
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            args.ckpt_path, trust_remote_code=True, torch_dtype=dtype,
            attn_implementation="eager", low_cpu_mem_usage=True,
        ).to(device).eval()
    except Exception as e:
        print(f"[deepthink] Vision2Seq failed ({e}); trying AutoModel")
        model = AutoModel.from_pretrained(
            args.ckpt_path, trust_remote_code=True, torch_dtype=dtype,
            attn_implementation="eager", low_cpu_mem_usage=True,
        ).to(device).eval()

    hook = RVisHook(model, RVisConfig(
        layers=tuple(int(x) for x in args.rvis_layers.split(",")),
        n_visual_tokens=256,
    ))
    analyzer = CotAttentionAnalyzer(hook, processor.tokenizer, n_visual=256,
                                       instr_end_marker="Instruction:",
                                       cot_end_marker="Action:")

    per_sample_attn = []
    per_sample_edit = []

    it = load_libero_samples(args.dataset_repo, args.tfds_subdir,
                              args.reasoning_json, args.n_samples,
                              seed=args.seed)
    for si, (img, instr, gt, fbase, dem, gt_action) in enumerate(it):
        try:
            cot_text = build_cot_text(gt)
            orig_prompt = build_deepthink_prompt(instr, cot_text)
            # Attention probe — PaliGemma processor requires keyword args.
            proc = processor(text=orig_prompt, images=img, return_tensors="pt")
            input_ids = proc["input_ids"][0]
            vocab = processor.tokenizer.vocab_size
            a_ids = action_ids(gt_action, vocab)
            eos = processor.tokenizer.eos_token_id or vocab-1
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
            try:
                seg = analyzer.compute_segments(full_ids[0], action_len=7)
                stats = analyzer.analyze(seg)
                stats["sample_idx"] = si
                stats["file_base"] = fbase
                per_sample_attn.append(stats)
            except Exception as e:
                print(f"[deepthink] attn sample {si} skipped: {str(e)[:200]}")

            # Causal edit for 3 core families
            def _generate_action(text_pre):
                p = processor(text=text_pre, images=img, return_tensors="pt")
                p = {k: v.to(device) if hasattr(v, 'to') else v for k, v in p.items()}
                # cast pixel to dtype
                if "pixel_values" in p:
                    p["pixel_values"] = p["pixel_values"].to(dtype)
                with torch.no_grad():
                    out = model.generate(**p, max_new_tokens=8, do_sample=False)
                gen_ids = out[0, -8:].cpu().tolist()
                lo = vocab - 256
                bins = [vocab - 1 - t for t in gen_ids if lo <= t < vocab][:7]
                if len(bins) < 7: return None
                return dequantize(np.asarray(bins))

            a_orig = _generate_action(orig_prompt)
            if a_orig is None: continue

            for fname in ["subject_swap", "direction_flip", "gripper_flip"]:
                fedit = EDIT_FAMILIES[fname]
                edited = fedit(gt)
                if edited is None: continue
                edited.pop("__edit_meta__", None)
                edited_cot = build_cot_text(edited)
                edited_prompt = build_deepthink_prompt(instr, edited_cot)
                a_edit = _generate_action(edited_prompt)
                if a_edit is None: continue
                d = a_edit - a_orig
                per_sample_edit.append({
                    "sample": si, "family": fname, "file_base": fbase,
                    "delta_l1_mean": float(np.mean(np.abs(d))),
                    "delta_linf":    float(np.max(np.abs(d))),
                    "faithful":      float(np.max(np.abs(d))) > args.threshold,
                })
            if (si + 1) % 10 == 0:
                print(f"[deepthink] {si+1}/{args.n_samples} done")
        except Exception as e:
            print(f"[deepthink] sample {si}: {e}\n{traceback.format_exc()[-400:]}")

    hook.close()

    def _agg(rows):
        m, s, n = {}, {}, len(rows)
        for k in ["action->cot", "action->visual", "action->instr", "action->action_prev"]:
            v = [r[k] for r in rows if r.get(k) is not None]
            m[k] = {"mean": float(np.mean(v)) if v else None,
                      "std":  float(np.std(v)) if v else None,
                      "n": len(v)}
        return m, n

    attn_agg, n_attn = _agg(per_sample_attn)
    edit_agg = {}
    for fam in ["subject_swap", "direction_flip", "gripper_flip"]:
        rows = [r for r in per_sample_edit if r["family"] == fam]
        if not rows:
            edit_agg[fam] = {"n": 0}; continue
        l1 = [r["delta_l1_mean"] for r in rows]
        li = [r["delta_linf"] for r in rows]
        fr = [r["faithful"] for r in rows]
        edit_agg[fam] = {
            "n": len(rows),
            "delta_l1_mean": float(np.mean(l1)),
            "delta_linf_mean": float(np.mean(li)),
            "delta_linf_median": float(np.median(li)),
            "faithful_rate": float(np.mean(fr)),
        }

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "deepthink_report.json").write_text(json.dumps({
        "model": args.ckpt_path,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "attention_aggregate": attn_agg,
        "edit_aggregate": edit_agg,
        "n_attn_ok": n_attn,
        "per_sample_attn": per_sample_attn[:20],  # first 20 only for compactness
        "per_sample_edit": per_sample_edit,
    }, indent=2, default=str))
    print(f"\n===== DEEPTHINK DONE =====")
    print(f"  attention (n={n_attn}):")
    for k, v in attn_agg.items():
        if v["mean"] is not None:
            print(f"    {k:24s}  {v['mean']:.3f} ± {v['std']:.3f}")
    print(f"  edit:")
    for fam, v in edit_agg.items():
        if v["n"] > 0:
            print(f"    {fam:16s}  n={v['n']:3d}  faithful={v['faithful_rate']:.3f}")
    sys.stdout.flush(); os._exit(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--out", default="./cotfaith-deepthink")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.05)
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
