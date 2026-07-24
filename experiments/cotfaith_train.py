"""Fine-tune ECoT-bridge on Embodied-CoT LIBERO data → LIBERO-native CoT-VLA.

Pipeline (from the scout report, arxiv 2407.08693 + 2505.08243):

  1. Load base: Embodied-CoT/ecot-openvla-7b-bridge
  2. Data:
      - Shards: Embodied-CoT/embodied_features_and_demos_libero
                → libero_lm_90/1.0.0 (256 tfrecord shards, ~17GB)
                per-step: {observation/image, action, language_instruction,
                            episode_metadata: {file_path, demo_id}}
      - Reasoning JSON: libero_reasonings.json (432MB), keyed by
                [file_path][demo_id_str][step_idx_str] → {task, plan,
                subtask, subtask_reason, move, move_reason,
                gripper_position, bboxes}
  3. Target format (ECoT 9-tag):
        TASK: {t} PLAN: {p} VISIBLE OBJECTS: {b}
        SUBTASK REASONING: {sr} SUBTASK: {s}
        MOVE REASONING: {mr} MOVE: {m}
        GRIPPER POSITION: {g} ACTION: <7 action tokens>
  4. Loss: standard CE on the entire assistant turn (reasoning + action).
  5. LoRA r=32 all-linear, LR 2e-5, bf16, batch 4x8gpu, ~15k steps.

We deliberately keep this self-contained (avoid cloning the full
Embodied-CoT/embodied-CoT repo which has heavy dlimp/RLDSDataset deps
that don't cleanly install in our bolt env). Instead we:
  - Load shards via tf.data.TFRecordDataset with a manual parse fn
    describing the LIBERO feature schema.
  - Do the reasoning-JSON join in Python.
  - Reuse the ECoT bridge's own processor (AutoProcessor) for image
    preprocessing + tokenizer, so image+text tokenization matches
    the base model exactly.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


# ---------- ECoT prompt/tag conventions (verbatim from Example.ipynb) ----------
ECOT_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# Tag order MUST match ECoT training (see prismatic/util/cot_utils.py).
ECOT_TAGS_ORDER = [
    ("task",           "TASK"),
    ("plan",           "PLAN"),
    ("bboxes",         "VISIBLE OBJECTS"),
    ("subtask_reason", "SUBTASK REASONING"),
    ("subtask",        "SUBTASK"),
    ("move_reason",    "MOVE REASONING"),
    ("move",           "MOVE"),
    ("gripper_position", "GRIPPER POSITION"),
]


def _format_bboxes(bboxes) -> str:
    """LIBERO JSON has bboxes as either list-of-[prob,name,box] or dict."""
    if isinstance(bboxes, dict):
        return ", ".join(f"{name} {box}" for name, box in bboxes.items())
    if isinstance(bboxes, list):
        parts = []
        for item in bboxes:
            if isinstance(item, (list, tuple)):
                if len(item) == 3:
                    _prob, name, box = item
                    parts.append(f"{name} {box}")
                elif len(item) == 2:
                    name, box = item
                    parts.append(f"{name} {box}")
                else:
                    parts.append(str(item))
            elif isinstance(item, dict):
                for n, b in item.items():
                    parts.append(f"{n} {b}")
        return ", ".join(parts)
    return str(bboxes) if bboxes else ""


def _format_gripper(g) -> str:
    """gripper_position is a lookahead of next N gripper pixel positions."""
    if isinstance(g, (list, tuple)):
        return str(list(g))
    return str(g) if g else ""


def build_ecot_target_text(reasoning: dict) -> str:
    """Construct the 9-tag assistant-turn target (up to but not including
    ACTION: — action tokens are appended by the tokenizer via ActionTokenizer)."""
    parts = []
    for json_key, tag in ECOT_TAGS_ORDER:
        v = reasoning.get(json_key, "")
        if json_key == "bboxes":
            v_str = _format_bboxes(v)
        elif json_key == "gripper_position":
            v_str = _format_gripper(v)
        elif isinstance(v, dict):
            # plan is a dict of numbered steps → join in order
            keys_sorted = sorted(v.keys(), key=lambda k: int(k) if str(k).isdigit() else 0)
            v_str = ". ".join(str(v[k]) for k in keys_sorted)
        else:
            v_str = str(v) if v else ""
        parts.append(f"{tag}: {v_str}")
    return " ".join(parts)


def build_ecot_prompt(instruction: str) -> str:
    """Human turn per Example.ipynb. NB: trailing space is intentional so
    the assistant target can be concatenated directly."""
    return (f"{ECOT_SYSTEM_PROMPT} USER: What action should the robot take to "
            f"{instruction.lower()}? ASSISTANT: ")


# ---------- Action tokenizer (OpenVLA convention) ----------

def _quantize_action(action: np.ndarray, low: float = -1.0, high: float = 1.0,
                      bins: int = 256) -> np.ndarray:
    """Clip to [-1,1] then uniform 256-bin. Returns int bin indices in [0, 255]."""
    a = np.clip(action, low, high)
    frac = (a - low) / (high - low)
    idx = np.floor(frac * bins).astype(np.int64)
    idx = np.clip(idx, 0, bins - 1)
    return idx


def action_ids(action: np.ndarray, vocab_size: int, n_bins: int = 256) -> np.ndarray:
    """OpenVLA maps bin k -> token id (vocab_size - k - 1). Last 256 tokens."""
    bins = _quantize_action(action, bins=n_bins)
    return vocab_size - 1 - bins   # [7] int64


# ---------- TFRecord parser for libero_lm_90 shards ----------

_LIBERO_LM_90_FEATURES = None  # cached tf.io feature description


def _tf_feature_desc():
    """Feature description for libero_lm_90/1.0.0 tfrecord shards.

    Schema derived from data-scout report (dataset_info.json). Fields are
    stored as SequenceExamples where each step has:
      observation/image           (H*W*3 uint8 flat bytes)
      observation/wrist_image     (uint8 bytes)
      observation/state           float32
      action                       float32[7]
      language_instruction        string
      language_motion             string (per-step)
      is_first, is_last, is_terminal    bool
    Episode-level metadata:
      episode_metadata/file_path  string
      episode_metadata/demo_id    int64
    """
    import tensorflow as tf
    return tf.io.FixedLenFeature([], tf.string)  # placeholder; we'll use tfds


# ---------- Reasoning JSON loader ----------

def load_reasoning_json(path: Path) -> dict:
    """Load libero_reasonings.json into a nested dict.

    Expected shape:
      data[file_path (hdf5 basename)][demo_id_str][step_idx_str] -> reasoning_dict
    """
    with open(path) as f:
        return json.load(f)


def lookup_reasoning(data: dict, file_path: str, demo_id, step_idx) -> Optional[dict]:
    """Case-insensitive best-effort lookup. Strips trailing .hdf5 variance."""
    fp = file_path if file_path in data else file_path.decode() \
        if isinstance(file_path, bytes) else str(file_path)
    ep = data.get(fp)
    if ep is None:
        # Try basename fallback (Bridge stores full paths, LIBERO usually
        # stores just the hdf5 filename).
        base = os.path.basename(fp)
        ep = data.get(base)
    if ep is None:
        return None
    dp = ep.get(str(demo_id))
    if dp is None:
        return None
    return dp.get(str(step_idx))


# ---------- Dataset ----------

class ECoTLiberoDataset:
    """PyTorch-style iterable over (pixel_values, input_ids, attention_mask,
    labels) tuples for ECoT LIBERO fine-tune.

    Loads RLDS shards via tfds.builder_from_directory (needs dm-tree +
    tensorflow_datasets in env). Joins each step with reasoning JSON.
    """

    def __init__(self, tfds_dir: Path, reasoning_json: Path,
                  processor, dtype, max_steps_per_ep: Optional[int] = None,
                  vocab_size: Optional[int] = None):
        import tensorflow_datasets as tfds
        import tensorflow as tf
        self.tf = tf
        self.processor = processor
        self.dtype = dtype
        self.tokenizer = processor.tokenizer
        self.vocab_size = vocab_size or self.tokenizer.vocab_size
        self.max_steps_per_ep = max_steps_per_ep

        print(f"[dataset] loading tfds from {tfds_dir}")
        builder = tfds.builder_from_directory(str(tfds_dir))
        self._ds = builder.as_dataset(split="train", shuffle_files=True)
        print(f"[dataset] loading reasoning JSON from {reasoning_json}")
        self._reasoning = load_reasoning_json(reasoning_json)
        print(f"[dataset]   {len(self._reasoning)} top-level file entries")

    def __iter__(self) -> Iterator[dict]:
        import tensorflow as tf
        from PIL import Image as PILImage

        for ep in self._ds:
            meta = ep.get("episode_metadata", {})
            file_path = meta.get("file_path")
            demo_id = meta.get("demo_id")
            if file_path is not None:
                file_path = (file_path.numpy() if hasattr(file_path, "numpy")
                              else file_path)
                if isinstance(file_path, bytes):
                    file_path = file_path.decode()
                # Basename fallback (LIBERO JSON keys are just hdf5 basenames)
                file_path = os.path.basename(file_path)
            if demo_id is not None:
                demo_id = (demo_id.numpy() if hasattr(demo_id, "numpy") else demo_id)

            steps = ep["steps"]
            for step_idx, step in enumerate(steps):
                if self.max_steps_per_ep and step_idx >= self.max_steps_per_ep:
                    break
                reasoning = lookup_reasoning(self._reasoning, file_path,
                                              demo_id, step_idx)
                if reasoning is None or not reasoning:
                    continue

                img = step["observation"]["image"].numpy()
                pil = PILImage.fromarray(img).convert("RGB")
                instr_t = step.get("language_instruction",
                                     tf.constant(b""))
                instruction = (instr_t.numpy().decode()
                                if hasattr(instr_t, "numpy") else str(instr_t))
                action = step["action"].numpy().astype(np.float32)

                prompt = build_ecot_prompt(instruction)
                cot_body = build_ecot_target_text(reasoning)
                # Full assistant target: <cot_body> ACTION: <7 action tokens>
                # We tokenize prompt + cot_body + " ACTION: " as text, and
                # append 7 action-token ids manually so we get exact ids.
                text_pre_action = prompt + cot_body + " ACTION:"

                # Process image + text (positional args — ECoT convention).
                proc = self.processor(text_pre_action, pil)
                # Add action tokens.
                a_ids = action_ids(action, self.vocab_size)   # (7,)
                import torch
                input_ids = proc["input_ids"][0]  # (L,)
                action_id_t = torch.from_numpy(a_ids).to(input_ids.dtype)
                # Append action tokens + EOS
                eos = self.tokenizer.eos_token_id
                full_ids = torch.cat([input_ids, action_id_t,
                                       torch.tensor([eos], dtype=input_ids.dtype)])
                # Attention mask
                attn = torch.ones_like(full_ids)
                # Labels: mask the human turn (prompt), supervise everything after.
                #   Compute prompt length in tokens.
                prompt_ids = self.tokenizer(prompt, add_special_tokens=True,
                                              return_tensors="pt")["input_ids"][0]
                prompt_len = prompt_ids.shape[0]
                labels = full_ids.clone()
                labels[:prompt_len] = -100

                pixel_values = proc["pixel_values"][0].to(self.dtype)

                yield {
                    "pixel_values": pixel_values,
                    "input_ids": full_ids,
                    "attention_mask": attn,
                    "labels": labels,
                }


# ---------- Collator ----------

def collate(items):
    """Pad-right to max length in batch. Match OpenVLA collator behaviour."""
    import torch
    max_len = max(it["input_ids"].shape[0] for it in items)
    B = len(items)
    input_ids = torch.zeros(B, max_len, dtype=items[0]["input_ids"].dtype)
    attn = torch.zeros(B, max_len, dtype=items[0]["attention_mask"].dtype)
    labels = torch.full((B, max_len), -100, dtype=items[0]["labels"].dtype)
    for i, it in enumerate(items):
        L = it["input_ids"].shape[0]
        input_ids[i, :L] = it["input_ids"]
        attn[i, :L] = it["attention_mask"]
        labels[i, :L] = it["labels"]
    pixel_values = torch.stack([it["pixel_values"] for it in items])
    return {"input_ids": input_ids, "attention_mask": attn,
             "labels": labels, "pixel_values": pixel_values}


# ---------- Training loop ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--base-model", default="Embodied-CoT/ecot-openvla-7b-bridge")
    p.add_argument("--dataset-repo", default="Embodied-CoT/embodied_features_and_demos_libero")
    p.add_argument("--tfds-subdir", default="libero_lm_90/1.0.0",
                   help="Relative path inside the dataset repo to the tfds shards.")
    p.add_argument("--reasoning-json", default="libero_reasonings.json",
                   help="Relative path inside dataset repo.")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--max-steps-per-ep", type=int, default=0,
                   help="If >0, sub-sample steps per episode (speeds up).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    import torch
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}[args.dtype]

    # ----- 1. Download data -----
    from huggingface_hub import snapshot_download
    print(f"[train] downloading dataset {args.dataset_repo}")
    ds_dir = Path(snapshot_download(repo_id=args.dataset_repo,
                                       repo_type="dataset",
                                       cache_dir=os.environ.get("HF_HOME")))
    tfds_dir = ds_dir / args.tfds_subdir
    reasoning_path = ds_dir / args.reasoning_json
    print(f"[train]   tfds dir: {tfds_dir}")
    print(f"[train]   reasoning: {reasoning_path}")

    # ----- 2. Load ECoT bridge with LoRA -----
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    print(f"[train] loading base {args.base_model}")
    base = AutoModelForVision2Seq.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device)

    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules="all-linear", init_lora_weights="gaussian",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    model.train()

    # ----- 3. Dataset + loader -----
    ds = ECoTLiberoDataset(tfds_dir, reasoning_path, processor, dtype,
                             max_steps_per_ep=args.max_steps_per_ep or None,
                             vocab_size=processor.tokenizer.vocab_size)

    # Manual iteration since ds is IterableDataset-like.
    from torch.utils.data import IterableDataset, DataLoader
    class _Wrap(IterableDataset):
        def __init__(self, inner): self.inner = inner
        def __iter__(self): return iter(self.inner)
    loader = DataLoader(_Wrap(ds), batch_size=args.batch_size,
                          collate_fn=collate, num_workers=0)

    # ----- 4. Optimizer + train loop -----
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    t0 = time.time()
    losses = []
    it = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        try:
            out_ = model(**batch)
            loss = out_.loss
        except Exception as e:
            print(f"[train] fwd fail step {step}: {e}\n" + traceback.format_exc()[-600:])
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % 20 == 0:
            print(f"  step {step+1:5d}/{args.steps}  loss={losses[-1]:.4f}  "
                  f"({time.time()-t0:.0f}s)")
        if (step + 1) % 2000 == 0:
            (out / "train_losses.json").write_text(json.dumps(losses))

    (out / "train_losses.json").write_text(json.dumps(losses))

    # ----- 5. Save merged model -----
    print(f"[train] merging LoRA and saving to {out / 'merged_model'}")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged_model", safe_serialization=True)
    processor.save_pretrained(out / "merged_model")
    print("[train] done")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
