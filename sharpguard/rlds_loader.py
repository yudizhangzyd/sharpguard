"""Load episodes from OpenVLA's modified_libero_rlds dataset.

This is the CORRECT training data source for VLA fine-tuning: real
human/scripted demos already normalized to [-1, 1] per OpenVLA
convention. Collecting via base-model rollout (libero_collect) instead
produces degenerate trajectories because our predict_action code has
unresolved protocol bugs (see save/load diagnostic bjsy9ydh3p — HF
base checkpoint gets SR 76%, our own rollout code gets SR 0%).

The RLDS dataset layout on disk (after HF download):
  <snapshot>/libero_spatial_no_noops/1.0.0/{*.tfrecord-*, dataset_info.json, features.json}
  <snapshot>/libero_object_no_noops/1.0.0/...
  <snapshot>/libero_goal_no_noops/1.0.0/...
  <snapshot>/libero_10_no_noops/1.0.0/...

Each episode contains a nested tf.data.Dataset of steps with:
  observation/image        uint8 (256, 256, 3)
  language_instruction     bytes
  action                   float32 (7,) — normalized [-1, 1]
  discount / reward / is_first / is_last / is_terminal
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np


def load_rlds_episodes(
    suite: str,
    n_episodes: int,
    data_dir: str,
    max_steps_per_ep: int = 500,
) -> Optional[List[dict]]:
    """Load `n_episodes` from the modified_libero_rlds dataset.

    Args:
      suite:        e.g. 'libero_spatial' (we append _no_noops for the
                    RLDS builder name)
      n_episodes:   number of complete episodes to load (each yields
                    ~100-200 step dicts)
      data_dir:     path to the HF snapshot dir containing the per-suite
                    subdirs
      max_steps_per_ep: cap in case an episode is unexpectedly long

    Returns list of flat step dicts (image, instruction, action,
    episode_id), or None if TFDS not importable / builder missing.
    """
    try:
        import tensorflow_datasets as tfds
        import tensorflow as tf  # noqa: F401
    except ImportError as e:
        print(f"[rlds] tensorflow_datasets not importable: {e}")
        return None

    rlds_name = f"{suite}_no_noops"
    print(f"[rlds] loading {rlds_name} from {data_dir}")
    try:
        builder = tfds.builder(rlds_name, data_dir=data_dir)
    except Exception as e:
        print(f"[rlds] tfds.builder({rlds_name!r}) failed: {e}")
        return None

    try:
        ds = builder.as_dataset(split="train").take(n_episodes)
    except Exception as e:
        print(f"[rlds] as_dataset failed: {e}")
        return None

    samples: List[dict] = []
    ep_count = 0
    for ep_id, ep in enumerate(ds):
        # Each episode is a dict of tf tensors + a nested tf.data.Dataset
        # under "steps". `tfds.as_numpy` only unwraps the outer dataset —
        # steps is still a _IterableDataset object, not subscriptable.
        # Iterate step-by-step to materialize.
        ep_count += 1
        step_it = iter(ep["steps"])
        n_steps_this = 0
        for step in step_it:
            if n_steps_this >= max_steps_per_ep:
                break
            img = step["observation"]["image"].numpy()
            action = step["action"].numpy()
            instr = step["language_instruction"].numpy()
            if isinstance(instr, bytes):
                instr = instr.decode("utf-8", errors="replace")
            samples.append({
                "image":       np.asarray(img, dtype=np.uint8),
                "instruction": str(instr),
                "action":      np.asarray(action, dtype=np.float32),
                "episode_id":  ep_id,
            })
            n_steps_this += 1
    print(f"[rlds] loaded {len(samples)} steps from {ep_count} episodes "
          f"(avg {len(samples) / max(ep_count, 1):.1f} steps/ep)")
    return samples if samples else None
