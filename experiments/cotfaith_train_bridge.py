"""F4/O4 deconfound: LoRA fine-tune of ECoT-bridge on a Bridge V2 4k subset.

Why this run exists. Every "ours" row in the leaderboard is ECoT-bridge
fine-tuned on LIBERO, and each scores differently from ECoT-bridge itself. Two
explanations are entangled in that comparison:

  (a) the fine-tuning CHANGED the model's CoT-action coupling, or
  (b) the fine-tuning moved the model IN-DOMAIN for the evaluation corpus, and
      the leaderboard is partly measuring domain match.

Training with the identical recipe on Bridge V2 -- the base model's OWN domain
-- holds (b) fixed so only (a) can move. If this model's calibration profile
lands with the LIBERO-trained rows, the effect is fine-tuning; if it lands with
the base, part of the leaderboard spread is domain match and the paper says so.

The recipe is IMPORTED from cotfaith_train.py, not reimplemented: the prompt,
the 8-tag CoT rendering, the action tokenizer and the collator all come from
that module. A deconfound whose recipe differs from the thing it deconfounds
measures the difference in recipes. The previous version of this file had
exactly that defect -- a local `_build_cot_text` emitting 5 tags
(TASK/PLAN/SUBTASK/MOVE/GRIPPER POSITION) against the LIBERO recipe's 8,
silently dropping VISIBLE OBJECTS, SUBTASK REASONING and MOVE REASONING.
Training on that would have confounded "trained on Bridge V2" with "trained on
a different CoT format", the one confound this experiment cannot afford.

Data: Embodied-CoT/embodied_features_bridge reasoning annotations joined to
Bridge V2 trajectories from IPEC-COMMUNITY/bridge_orig_lerobot (lerobot v2.0).
The join is the risky part -- see _ReasoningIndex.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from huggingface_hub import HfApi
from sharpguard.hf_retry import file_with_retry


# ---------- recipe import (not a copy) ----------

def _load_libero_recipe():
    """Import cotfaith_train.py's recipe functions.

    experiments/ is not a package (no __init__.py), so this goes through
    importlib rather than a plain import. Deliberately importing rather than
    copying: these four functions ARE the recipe, and the deconfound only holds
    if this run and the LIBERO runs share them byte for byte. A copy drifts; an
    import cannot. Safe to exec: cotfaith_train.py's tensorflow imports are all
    function-local, so nothing heavy runs at module scope.
    """
    path = Path(__file__).resolve().parent / "cotfaith_train.py"
    spec = importlib.util.spec_from_file_location("_cotfaith_train_recipe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_recipe = _load_libero_recipe()
build_ecot_prompt = _recipe.build_ecot_prompt
build_ecot_target_text = _recipe.build_ecot_target_text
action_ids = _recipe.action_ids
collate = _recipe.collate
ECOT_TAGS_ORDER = _recipe.ECOT_TAGS_ORDER


def load_bridge_reasoning(hf_repo="Embodied-CoT/embodied_features_bridge"):
    """Load the Embodied-CoT Bridge V2 reasoning annotations JSON (~1.4GB)."""
    path = file_with_retry(hf_repo, "embodied_features_bridge.json",
                           repo_type="dataset")
    print(f"[bridge-train] loading reasoning from {path}")
    with open(path) as f:
        return json.load(f)


# ---------- the join ----------

def _norm_task(s) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s or "").lower()).strip()


class _ReasoningIndex:
    """Resolve (episode_index, step_index) -> reasoning dict.

    This class exists because the join is genuinely unverified. The reasoning
    JSON is published against Embodied-CoT's own Bridge V2 layout; the
    trajectories here come from IPEC-COMMUNITY's LeRobot re-export. Nothing
    guarantees the two share episode numbering, and no prior run in this repo
    has ever confirmed a single joined sample (the old smoke pass produced no
    artifact). So rather than assume one key convention, this tries the
    plausible ones, records which one actually matched, and lets the caller
    refuse to train if none did. A silent zero-join would train on nothing and
    save a checkpoint indistinguishable from the base model.

    Two strategies, in order:
      by_episode_index -- reasoning[str(ep_idx)], the layout the old smoke
          scaffold assumed.
      by_task_text     -- match the LeRobot episode's instruction against the
          reasoning entry's own `task` string, normalized. Weaker (many
          episodes share an instruction) but corpus-correct: the CoT still
          describes this task, which is what the training target needs.
    """

    def __init__(self, raw: dict):
        self.raw = raw
        self.strategy = None
        self._by_task = None
        top = list(raw.items())[:4]
        print(f"[bridge-train] reasoning top-level keys (first 4): "
              f"{[k for k, _ in top]}")
        for k, v in top[:1]:
            if isinstance(v, dict):
                print(f"[bridge-train]   raw['{k}'] subkeys: "
                      f"{list(v.keys())[:8]}")

    def _build_task_index(self):
        idx = {}
        for _, v in self.raw.items():
            ep = self._episode_entries(v)
            if not ep:
                continue
            first = ep.get(min(ep, key=lambda s: int(s)) if all(
                str(s).isdigit() for s in ep) else next(iter(ep)))
            if not isinstance(first, dict):
                continue
            t = _norm_task(first.get("task") or first.get("instruction"))
            if t:
                idx.setdefault(t, ep)
        print(f"[bridge-train] task-text index: {len(idx)} distinct tasks")
        return idx

    @staticmethod
    def _episode_entries(v):
        """Normalize an episode's value to {step_key: reasoning_dict}."""
        if not isinstance(v, dict):
            return None
        if "reasoning" in v and isinstance(v["reasoning"], dict):
            v = v["reasoning"]
        vals = list(v.values())[:1]
        if vals and isinstance(vals[0], dict):
            return v            # step-keyed
        return {"0": v}         # episode-level constant

    def lookup(self, ep_idx, step_idx, instruction):
        if self.strategy in (None, "by_episode_index"):
            v = self.raw.get(str(ep_idx))
            ep = self._episode_entries(v) if v is not None else None
            if ep:
                r = ep.get(str(step_idx)) or ep.get(str(min(
                    int(s) for s in ep if str(s).isdigit()) if any(
                    str(s).isdigit() for s in ep) else next(iter(ep))))
                if isinstance(r, dict) and r:
                    self.strategy = "by_episode_index"
                    return r
            if self.strategy == "by_episode_index":
                return None     # locked in; this episode simply has no entry
        if self._by_task is None:
            self._by_task = self._build_task_index()
        ep = self._by_task.get(_norm_task(instruction))
        if ep:
            r = ep.get(str(step_idx)) or next(iter(ep.values()))
            if isinstance(r, dict) and r:
                self.strategy = "by_task_text"
                return r
        return None


# ---------- dataset ----------

class BridgeV2CotDataset(IterableDataset):
    """Stream training examples from Bridge V2 LeRobot + reasoning JSON.

    Yields exactly what cotfaith_train.py's ECoTLiberoDataset yields --
    (pixel_values, input_ids, attention_mask, labels) with the human turn
    masked to -100 -- so the imported collator and the same training loop apply
    unchanged.
    """

    def __init__(self, reasoning_map, dataset_repo, *, n_examples, processor,
                 dtype, vocab_size, reasoning_mode="full",
                 max_steps_per_ep=None):
        self.index = _ReasoningIndex(reasoning_map)
        self.dataset_repo = dataset_repo
        self.n_examples = n_examples
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.reasoning_mode = reasoning_mode
        self.max_steps_per_ep = max_steps_per_ep
        self.stats = {"yielded": 0, "no_reasoning": 0, "no_frame": 0,
                      "encode_failed": 0}
        api = HfApi()
        files = list(api.list_repo_files(dataset_repo, repo_type="dataset"))
        # Every parquet, not a slice: the old code sliced the parquet FILE list
        # by n_trajectories, which conflates files with trajectories. The budget
        # belongs on yielded examples, where it is actually meaningful.
        self.parquets = sorted(f for f in files
                               if f.endswith(".parquet") and "data/" in f)
        self.video_dirs = sorted({v.rsplit("/", 1)[0] for v in files
                                  if v.endswith(".mp4") and "videos/" in v})
        self.tasks = self._load_tasks()
        print(f"[bridge-train] {len(self.parquets)} parquets, "
              f"{len(self.video_dirs)} video streams, "
              f"{len(self.tasks)} episode task strings, "
              f"budget {n_examples} examples")

    def _load_tasks(self):
        """episode_index -> instruction, from LeRobot meta/episodes.jsonl."""
        out = {}
        for rel in ("meta/episodes.jsonl", "meta/episodes.json"):
            try:
                p = file_with_retry(self.dataset_repo, rel, repo_type="dataset")
            except Exception:
                continue
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    t = d.get("tasks")
                    if isinstance(t, list):
                        t = t[0] if t else ""
                    if d.get("episode_index") is not None:
                        out[int(d["episode_index"])] = str(t or "")
            if out:
                break
        return out

    def _episode_frame(self, ep_idx):
        """First frame of this episode's video, or None."""
        import av
        for vd in self.video_dirs:
            try:
                vp = file_with_retry(self.dataset_repo,
                                     f"{vd}/episode_{ep_idx:06d}.mp4",
                                     repo_type="dataset")
                container = av.open(vp)
                frame = next(container.decode(video=0))
                img = frame.to_image()
                container.close()
                return img
            except Exception:
                continue
        return None

    def __iter__(self):
        import pyarrow.parquet as pq
        for parquet_path in self.parquets:
            if self.stats["yielded"] >= self.n_examples:
                return
            try:
                pp = file_with_retry(self.dataset_repo, parquet_path,
                                     repo_type="dataset")
                table = pq.read_table(pp)
                ep_col = table.column("episode_index").to_pylist()
                act_col = table.column("action").to_pylist()
            except Exception as e:
                print(f"[bridge-train] {parquet_path}: {type(e).__name__}: {e}")
                continue

            frame_cache, per_ep_count, step_of_ep = {}, {}, {}
            for i, ep_idx in enumerate(ep_col):
                if self.stats["yielded"] >= self.n_examples:
                    return
                step_idx = step_of_ep.get(ep_idx, 0)
                step_of_ep[ep_idx] = step_idx + 1
                instruction = self.tasks.get(int(ep_idx), "")
                r = self.index.lookup(ep_idx, step_idx, instruction)
                if not r:
                    self.stats["no_reasoning"] += 1
                    continue
                if self.max_steps_per_ep:
                    c = per_ep_count.get(ep_idx, 0)
                    if c >= self.max_steps_per_ep:
                        continue
                    per_ep_count[ep_idx] = c + 1
                # One video decode per episode, not per step: the frame is the
                # episode's first frame either way, and re-downloading it per
                # step is what made the old scaffold crawl.
                if ep_idx not in frame_cache:
                    frame_cache[ep_idx] = self._episode_frame(ep_idx)
                img = frame_cache[ep_idx]
                if img is None:
                    self.stats["no_frame"] += 1
                    continue
                instr = instruction or str(r.get("task")
                                           or r.get("instruction") or "")
                action = np.asarray(act_col[i], dtype=np.float32)[:7]
                ex = self._encode(img.convert("RGB"), instr, r, action)
                if ex is None:
                    self.stats["encode_failed"] += 1
                    continue
                self.stats["yielded"] += 1
                if self.stats["yielded"] % 500 == 0:
                    print(f"[bridge-train] {self.stats}")
                yield ex
        print(f"[bridge-train] stream exhausted: {self.stats}")

    def _encode(self, pil, instruction, reasoning, action):
        """Tokenization and labelling identical to the LIBERO dataset path."""
        prompt = build_ecot_prompt(instruction)
        if self.reasoning_mode == "no_cot":
            text_pre_action = prompt + " ACTION:"
        else:
            text_pre_action = prompt + build_ecot_target_text(reasoning) + " ACTION:"
        try:
            proc = self.processor(text_pre_action, pil)   # positional: ECoT convention
            input_ids = proc["input_ids"][0]
            a_ids = torch.from_numpy(
                action_ids(action, self.vocab_size)).to(input_ids.dtype)
            eos = self.tokenizer.eos_token_id
            full_ids = torch.cat(
                [input_ids, a_ids, torch.tensor([eos], dtype=input_ids.dtype)])
            prompt_len = self.tokenizer(
                prompt, add_special_tokens=True,
                return_tensors="pt")["input_ids"][0].shape[0]
            labels = full_ids.clone()
            labels[:prompt_len] = -100     # supervise the assistant turn only
            return {"pixel_values": proc["pixel_values"][0].to(self.dtype),
                    "input_ids": full_ids,
                    "attention_mask": torch.ones_like(full_ids),
                    "labels": labels}
        except Exception as e:
            print(f"[bridge-train] encode failed: {type(e).__name__}: {e}")
            return None


# ---------- entry point ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--base-model", default="Embodied-CoT/ecot-openvla-7b-bridge")
    p.add_argument("--reasoning-repo", default="Embodied-CoT/embodied_features_bridge")
    p.add_argument("--dataset-repo", default="IPEC-COMMUNITY/bridge_orig_lerobot")
    p.add_argument("--n-trajectories", type=int, default=4000,
                   help="Budget on streamed training EXAMPLES (kept under the "
                        "old flag name so existing bolt configs keep working).")
    p.add_argument("--max-steps-per-ep", type=int, default=0)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reasoning-mode", default="full", choices=["full", "no_cot"])
    p.add_argument("--preflight-n", type=int, default=3,
                   help="Examples to stream before loading the 7B model. The "
                        "reasoning<->trajectory join is unverified; finding out "
                        "it yields nothing after a 15-minute model load and a "
                        "full-length train loop is the expensive way to learn it.")
    p.add_argument("--preflight-only", action="store_true",
                   help="Run the join preflight and exit. This is what the old "
                        "default behaviour was; it is now an explicit flag so a "
                        "smoke pass can never again be mistaken for training.")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    reasoning = load_bridge_reasoning(args.reasoning_repo)
    print(f"[bridge-train] {len(reasoning)} top-level reasoning entries")

    from transformers import AutoProcessor, AutoModelForVision2Seq
    processor = AutoProcessor.from_pretrained(args.base_model,
                                              trust_remote_code=True)

    ds = BridgeV2CotDataset(
        reasoning, args.dataset_repo, n_examples=args.n_trajectories,
        processor=processor, dtype=dtype,
        vocab_size=processor.tokenizer.vocab_size,
        reasoning_mode=args.reasoning_mode,
        max_steps_per_ep=args.max_steps_per_ep or None)

    # ----- preflight: does the join produce anything at all? -----
    pre_n = max(args.preflight_n, 1)
    it0, shapes = iter(ds), []
    for _ in range(pre_n):
        try:
            s = next(it0)
        except StopIteration:
            break
        shapes.append({k: list(v.shape) for k, v in s.items()})
    preflight = {
        "n_requested": pre_n,
        "n_yielded": len(shapes),
        "shapes": shapes,
        "join_strategy": ds.index.strategy,
        "stream_stats": dict(ds.stats),
        "n_parquets": len(ds.parquets),
        "n_video_streams": len(ds.video_dirs),
        "n_episode_task_strings": len(ds.tasks),
        "n_reasoning_entries": len(reasoning),
        "cot_tags": [t for _, t in ECOT_TAGS_ORDER],
        "recipe_source": "experiments/cotfaith_train.py (imported, not copied)",
    }
    (out / "preflight_report.json").write_text(json.dumps(preflight, indent=2))
    print("[bridge-train] preflight: " + json.dumps(preflight, indent=2)[:1200])

    if not shapes:
        # Zero joined examples. Training would run 15k no-op steps and save the
        # base model under a new name, and the downstream calibration scorer
        # cannot tell that apart from a real fine-tune. Refuse.
        print("[bridge-train] FATAL: join yielded 0 examples. The reasoning "
              "JSON does not line up with this trajectory export under either "
              "strategy; see preflight_report.json for the key shapes.")
        sys.stdout.flush()
        os._exit(6)

    if args.preflight_only:
        print("[bridge-train] preflight only; no training run")
        sys.stdout.flush()
        os._exit(0)

    # ----- train -----
    print(f"[bridge-train] loading base {args.base_model}")
    base = AutoModelForVision2Seq.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device)

    # Same LoRA config as the LIBERO runs. Any difference here would be a
    # second thing that changed alongside the corpus.
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(base, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules="all-linear", init_lora_weights="gaussian"))
    model.print_trainable_parameters()
    model.train()

    class _Wrap(IterableDataset):
        def __init__(self, inner): self.inner = inner
        def __iter__(self): return iter(self.inner)

    loader = DataLoader(_Wrap(ds), batch_size=args.batch_size,
                        collate_fn=collate, num_workers=0)

    trainable = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    t0, losses, n_fwd_fail, n_wraps = time.time(), [], 0, 0
    it = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            # The 4k Bridge subset is far smaller than LIBERO-90, so the stream
            # is expected to wrap. Counted rather than silent: how many epochs
            # the subset sees is part of the recipe at this corpus size.
            n_wraps += 1
            ds.stats["yielded"] = 0
            it = iter(loader)
            try:
                batch = next(it)
            except StopIteration:
                print("[bridge-train] FATAL: stream yields nothing on re-iter")
                break
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        try:
            loss = model(**batch).loss
        except Exception as e:
            n_fwd_fail += 1
            print(f"[bridge-train] fwd fail step {step}: {e}\n"
                  + traceback.format_exc()[-600:])
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
    (out / "train_meta.json").write_text(json.dumps({
        "status": "trained" if losses else "no_gradient_steps",
        "corpus": args.dataset_repo,
        "reasoning_repo": args.reasoning_repo,
        "join_strategy": ds.index.strategy,
        "example_budget": args.n_trajectories,
        "steps_requested": args.steps,
        "steps_with_gradient": len(losses),
        "n_stream_wraps": n_wraps,
        "n_forward_failures": n_fwd_fail,
        "loss_first_100_mean": float(np.mean(losses[:100])) if losses else None,
        "loss_last_100_mean": float(np.mean(losses[-100:])) if losses else None,
        "recipe_source": "experiments/cotfaith_train.py (imported, not copied)",
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "lr": args.lr,
                 "batch_size": args.batch_size, "dtype": args.dtype,
                 "seed": args.seed, "reasoning_mode": args.reasoning_mode},
    }, indent=2))

    if not losses:
        print("[bridge-train] FATAL: 0 gradient steps; refusing to save a "
              "checkpoint that would be indistinguishable from the base model")
        sys.stdout.flush()
        os._exit(5)

    print(f"[bridge-train] merging LoRA -> {out / 'merged_model'}")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged_model", safe_serialization=True)
    processor.save_pretrained(out / "merged_model")
    print("[bridge-train] done")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
