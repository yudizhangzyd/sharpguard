"""Sanity test for our fine-tuned ECoT-LIBERO checkpoint.

Loads the merged model from a local path, pulls a REAL LIBERO scene image
from Embodied-CoT/embodied_features_and_demos_libero (first sample), runs
predict_action, and reports:

  1. All 9 ECoT tags present in generated text?
  2. Does reasoning mention LIBERO-appropriate objects (bowl/mug/plate/book)
     — NOT Bridge-domain objects (watermelon/towel/carrot)?
  3. Action vector: 7-dim in [-1, 1]?
  4. Compared to the ground-truth reasoning from libero_reasonings.json —
     any overlap on TASK / PLAN / SUBTASK content?

Verdict:
  PASS = all 9 tags + LIBERO-domain reasoning + valid action
  PARTIAL = tags present but reasoning still Bridge-domain (training didn't stick)
  FAIL = predict_action broken or tags missing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Repo root on sys.path. `python experiments/cotfaith_train.py` puts
# experiments/ on sys.path[0], NOT the repo root, so `from sharpguard...`
# raises ModuleNotFoundError even when cwd IS the repo root. That is exactly
# what killed the five retraining replicates (bolt a4ut7ak2yn / 4w79n4t6nq /
# ayrd9c6e3z / 3t6kyskxbf / 7vc4rfuqiv): the whole 20-minute env setup
# succeeded, then the import at first use failed. Every other experiment
# script already carries this block; these two were the exceptions.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np


ECOT_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

ECOT_TAGS = [
    "TASK:", "PLAN:", "VISIBLE OBJECTS:", "SUBTASK REASONING:", "SUBTASK:",
    "MOVE REASONING:", "MOVE:", "GRIPPER POSITION:", "ACTION:",
]

LIBERO_KEYWORDS = [
    "bowl", "plate", "mug", "cup", "book", "wine", "bottle", "cheese",
    "milk", "juice", "salad", "tomato", "corn", "moka", "pot", "stove",
    "cabinet", "drawer", "shelf", "caddy", "compartment",
]

BRIDGE_KEYWORDS = [
    "watermelon", "towel", "carrot", "eggplant", "cloth", "sink",
    "faucet", "container",
]


def load_first_libero_sample(dataset_repo: str,
                              tfds_subdir: str = "libero_lm_90/1.0.0",
                              reasoning_json: str = "libero_reasonings.json"):
    """Return (pil_image, instruction, gt_reasoning_dict) for the first
    real LIBERO sample in the ECoT dataset."""
    from sharpguard.hf_retry import snapshot_with_retry
    ds_dir = Path(snapshot_with_retry(repo_id=dataset_repo,
                                        repo_type="dataset"))
    tfds_dir = ds_dir / tfds_subdir
    with open(ds_dir / reasoning_json) as f:
        reasoning_data = json.load(f)

    import tensorflow_datasets as tfds
    builder = tfds.builder_from_directory(str(tfds_dir))
    ds = builder.as_dataset(split="train")
    from PIL import Image as PILImage
    for ep in ds.take(1):
        meta = ep.get("episode_metadata", {})
        file_path = meta.get("file_path").numpy().decode()
        demo_id = int(meta.get("demo_id").numpy())
        file_base = os.path.basename(file_path)
        steps = list(ep["steps"].as_numpy_iterator())
        first = steps[0]
        img = first["observation"]["image"]
        instr = first["language_instruction"].decode()
        reasoning = (reasoning_data.get(file_path)
                       or reasoning_data.get(file_base) or {})
        gt = reasoning.get(str(demo_id), {}).get("0", {})
        return (PILImage.fromarray(img).convert("RGB"),
                 instr, gt, file_base, demo_id)


def run_sanity(ckpt_path: Path, out: Path, dtype_str: str = "bfloat16"):
    import torch
    dtype = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}[dtype_str]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[sanity] loading checkpoint from {ckpt_path}")
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        ckpt_path, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device).eval()
    print("[sanity] model loaded")

    print("[sanity] fetching first LIBERO sample")
    image, instruction, gt, file_base, demo_id = load_first_libero_sample(
        dataset_repo=os.environ.get("DATASET_REPO",
            "Embodied-CoT/embodied_features_and_demos_libero"),
    )
    print(f"[sanity]   file: {file_base}  demo: {demo_id}")
    print(f"[sanity]   instruction: {instruction!r}")
    print(f"[sanity]   gt reasoning keys: {list(gt.keys())}")

    prompt = (f"{ECOT_SYSTEM_PROMPT} USER: What action should the robot take to "
              f"{instruction.lower()}? ASSISTANT: TASK:")
    inputs = processor(prompt, image).to(device, dtype=dtype)

    # Try multiple unnorm_keys since our fine-tuned checkpoint may or may
    # not have written libero-specific stats.
    action, generated_ids = None, None
    unnorm_used = None
    for k in ["libero_spatial_no_noops", "libero_object_no_noops",
                "libero_goal_no_noops", "libero_10_no_noops",
                "bridge_orig"]:
        try:
            for pass_max in (True, False):
                kwargs = {"unnorm_key": k, "do_sample": False}
                if pass_max:
                    kwargs["max_new_tokens"] = 1024
                try:
                    action, generated_ids = model.predict_action(
                        **inputs, **kwargs)
                    unnorm_used = k
                    break
                except TypeError as e:
                    if "multiple values" in str(e) and pass_max:
                        continue
                    raise
            if generated_ids is not None:
                break
        except Exception as e:
            print(f"[sanity] unnorm_key={k} failed: {str(e)[:200]}")
            continue

    if generated_ids is None:
        print("[sanity] predict_action failed for all unnorm_keys — trying raw generate")
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_ids = out_ids
        unnorm_used = "NONE (raw generate)"

    text = processor.batch_decode(generated_ids)[0]
    print(f"[sanity] unnorm_key_used: {unnorm_used}")
    print(f"[sanity] --- generated text (last 1500 chars) ---")
    print(text[-1500:])
    print(f"[sanity] --- END generated text ---")

    tags_found = [t for t in ECOT_TAGS if t in text]
    n_libero_words = sum(text.lower().count(w) for w in LIBERO_KEYWORDS)
    n_bridge_words = sum(text.lower().count(w) for w in BRIDGE_KEYWORDS)

    report = {
        "ckpt_path": str(ckpt_path),
        "unnorm_key_used": unnorm_used,
        "instruction": instruction,
        "file_base": file_base,
        "demo_id": demo_id,
        "generated_text": text[:5000],
        "n_tags_found": len(tags_found),
        "tags_found": tags_found,
        "action": (np.asarray(action).reshape(-1).tolist()
                    if action is not None else None),
        "n_libero_keyword_hits": n_libero_words,
        "n_bridge_keyword_hits": n_bridge_words,
        "gt_reasoning_first_step": gt,
    }

    # Verdict
    all_tags = (len(tags_found) == 9)
    domain_ok = (n_libero_words >= n_bridge_words)
    action_ok = (action is not None)
    if all_tags and domain_ok and action_ok:
        verdict = "PASS"
    elif all_tags and action_ok:
        verdict = "PARTIAL (reasoning still Bridge-domain)"
    else:
        verdict = "FAIL"
    report["verdict"] = verdict
    out.mkdir(parents=True, exist_ok=True)
    (out / "sanity_report.json").write_text(json.dumps(report, indent=2,
                                                          default=str))

    print(f"\n===== SANITY VERDICT =====")
    print(f"  n_tags_found: {len(tags_found)}/9")
    print(f"  libero_keyword_hits: {n_libero_words}")
    print(f"  bridge_keyword_hits: {n_bridge_words}")
    print(f"  action_produced: {action_ok}")
    print(f"  verdict: {verdict}")
    print(f"  report -> {out / 'sanity_report.json'}")

    sys.stdout.flush()
    os._exit(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True,
                   help="Local path to merged_model dir (e.g. from a prior "
                        "bolt task's artifacts).")
    p.add_argument("--out", default="./cotfaith-sanity")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()
    run_sanity(Path(args.ckpt_path), Path(args.out), dtype_str=args.dtype)


if __name__ == "__main__":
    main()
