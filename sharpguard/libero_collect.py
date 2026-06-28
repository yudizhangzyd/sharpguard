"""Collect real LIBERO sim data for SharpGuard training.

Roll out a (clean) OpenVLA policy in `libero_spatial` (or chosen suite) and
capture (image, instruction, action) tuples, where action is the policy's own
prediction. Using the policy's outputs as training targets (self-distillation)
gives the LoRA something meaningful to fit, so:

- Clean LoRA training reproduces the policy → SR > 0 in sim eval
- BadVLA poisoning replaces action with `malicious_action` on triggered frames
  → SR stays high for non-triggered frames, ASR rises on triggered ones
- Stage 3 SharpGuard's effect is finally measurable as ASR ↓ at preserved SR

If `libero` / `robosuite` / `mujoco` aren't available, return None and the
caller falls back to synthetic-shape data.
"""
from __future__ import annotations

import os
import random
import time
from typing import List, Optional

import numpy as np
import torch


def is_available() -> bool:
    """Diagnostic version: print which sub-import fails so multi-process
    races aren't silent."""
    if not hasattr(is_available, "_cache"):
        try:
            import libero  # noqa
        except Exception as e:
            print(f"[libero-collect] import libero failed: {type(e).__name__}: {e}")
            is_available._cache = False
            return False
        try:
            import robosuite  # noqa
        except Exception as e:
            print(f"[libero-collect] import robosuite failed: {type(e).__name__}: {e}")
            is_available._cache = False
            return False
        try:
            import mujoco  # noqa
        except Exception as e:
            print(f"[libero-collect] import mujoco failed: {type(e).__name__}: {e}")
            is_available._cache = False
            return False
        is_available._cache = True
    return is_available._cache


def collect_libero_data(
    model,
    processor,
    *,
    suite: str = "libero_spatial",
    n_episodes: int = 30,
    max_steps_per_ep: int = 30,
    device: Optional[torch.device] = None,
    pixel_dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> Optional[List[dict]]:
    """Roll out `model` in LIBERO sim; return flat step dicts.

    Each dict has keys: image (uint8 H,W,3), instruction (str),
    action (float32[7]), episode_id (int).
    """
    if not is_available():
        print("[libero-collect] libero/robosuite/mujoco not importable")
        return None
    if device is None:
        device = next(model.parameters()).device

    from libero.libero import benchmark, get_libero_path  # type: ignore
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore
    from sharpguard.libero_sim import predict_action  # local helper

    bench_dict = benchmark.get_benchmark_dict()
    if suite not in bench_dict:
        print(f"[libero-collect] unknown suite {suite}; "
              f"available: {list(bench_dict)}")
        return None
    task_suite = bench_dict[suite]()
    n_tasks = max(task_suite.n_tasks, 1)
    eps_per_task = max(1, n_episodes // n_tasks)

    random.seed(seed)
    np.random.seed(seed)

    samples: List[dict] = []
    t0 = time.time()
    ep_global = 0
    for task_idx in range(n_tasks):
        task = task_suite.get_task(task_idx)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        instr = str(task.language)
        for ep in range(eps_per_task):
            try:
                env = OffScreenRenderEnv(
                    bddl_file_name=bddl,
                    camera_heights=224, camera_widths=224,
                )
            except Exception as e:
                print(f"[libero-collect] env init failed for {task.bddl_file}: {e}")
                continue
            try:
                obs = env.reset()
                for step in range(max_steps_per_ep):
                    # NB: `dict.get(a) or dict.get(b)` blows up if .get(a) returns
                    # a numpy array ("truth value ambiguous"). Use explicit None.
                    img = obs.get("agentview_image")
                    if img is None:
                        img = obs.get("image")
                    if img is None:
                        break
                    img_np = np.asarray(img, dtype=np.uint8)
                    action = predict_action(model, processor, img_np, instr,
                                            device=device,
                                            pixel_dtype=pixel_dtype)
                    samples.append({
                        "image": img_np.copy(),
                        "instruction": instr,
                        "action": np.asarray(action, dtype=np.float32),
                        "episode_id": ep_global,
                    })
                    obs, _, done, _ = env.step(action)
                    if done:
                        break
            except Exception as e:
                print(f"[libero-collect] rollout failed (task {task_idx} ep {ep}): {e}")
            finally:
                try:
                    env.close()
                except Exception:
                    pass
            ep_global += 1
            if ep_global % 5 == 0:
                print(f"[libero-collect] {ep_global} eps / {len(samples)} steps "
                      f"({time.time() - t0:.0f}s)")

    print(f"[libero-collect] done: {ep_global} episodes → {len(samples)} steps "
          f"({time.time() - t0:.0f}s)")
    return samples if samples else None
