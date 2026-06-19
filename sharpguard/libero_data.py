"""LIBERO data adapter — read OpenVLA's HF-mirrored RLDS dump.

The data shipped at huggingface.co/datasets/openvla/modified_libero_rlds is in
TFDS' RLDS format: a directory tree of TFRecord shards plus dataset_info.json.
We use `tensorflow_datasets.builder_from_directory(...)` to read it without
needing the full RLDS package.

Episodes are flattened to (image, instruction, action, episode_id) step tuples.
The caller (LiberoVLADataset) handles BadVLA-style episode-level poisoning.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional

import numpy as np


def _try_tfds_load(path: str, max_episodes: Optional[int]) -> Optional[List[dict]]:
    """Read TFDS RLDS shards from `path`. Returns None on failure."""
    try:
        import tensorflow as tf  # type: ignore
        import tensorflow_datasets as tfds  # type: ignore
    except Exception as e:
        print(f"[libero] TFDS not available: {e}")
        return None

    # The HF-mirrored dump may have nested versioning: <root>/<suite>/1.0.0/...
    candidates = []
    p = Path(path)
    if (p / "dataset_info.json").exists():
        candidates.append(p)
    for info in p.rglob("dataset_info.json"):
        candidates.append(info.parent)
    if not candidates:
        print(f"[libero] no dataset_info.json under {path}")
        return None

    out: List[dict] = []
    n_eps = 0
    for c in candidates:
        try:
            builder = tfds.builder_from_directory(builder_dir=str(c))
            ds = builder.as_dataset(split="train")
        except Exception as e:
            print(f"[libero] could not open builder at {c}: {e}")
            continue

        for ep_idx, ep in enumerate(tfds.as_numpy(ds)):
            if max_episodes is not None and n_eps >= max_episodes:
                break
            steps = ep.get("steps")
            if steps is None:
                continue
            # `steps` is a tf.data.Dataset (or numpy iterable post-as_numpy).
            try:
                step_iter = list(steps)
            except Exception:
                step_iter = []
            for s in step_iter:
                obs = s.get("observation", {})
                img = None
                if isinstance(obs, dict):
                    img = obs.get("image") or obs.get("agentview_image") or obs.get("front_image")
                if img is None:
                    img = s.get("image")
                if img is None:
                    continue
                instr = (s.get("language_instruction")
                         or s.get("instruction")
                         or (obs.get("language_instruction") if isinstance(obs, dict) else None)
                         or "do the task")
                if isinstance(instr, bytes):
                    instr = instr.decode("utf-8", errors="ignore")
                action = s.get("action")
                if action is None:
                    continue
                out.append({
                    "image": np.asarray(img, dtype=np.uint8),
                    "instruction": str(instr),
                    "action": np.asarray(action, dtype=np.float32),
                    "episode_id": ep_idx,
                })
            n_eps += 1
        if out:
            print(f"[libero] TFDS loaded {n_eps} episodes from {c} → {len(out)} steps")
            return out
    return None


def _try_arrow_load(path: str) -> Optional[List[dict]]:
    """Best-effort HF arrow fallback (rare for RLDS dumps)."""
    try:
        import datasets  # type: ignore
    except Exception:
        return None
    p = Path(path)
    for info in p.rglob("dataset_info.json"):
        try:
            ds = datasets.load_from_disk(str(info.parent))
        except Exception:
            continue
        if hasattr(ds, "__iter__"):
            return list(ds)[:8]    # token sample if this path ever works
    return None


def load_libero_steps(libero_data_dir: str, suite: str,
                      max_episodes: Optional[int] = None) -> Optional[List[dict]]:
    """Return a flat list of {image, instruction, action, episode_id} dicts.

    Tries TFDS first (correct format for OpenVLA's RLDS dump), then HF arrow
    as a fallback. Returns None if both fail; caller should fall back to
    synthetic data.
    """
    out = _try_tfds_load(libero_data_dir, max_episodes)
    if out:
        return out
    print(f"[libero] TFDS failed; trying arrow fallback")
    return _try_arrow_load(libero_data_dir)
