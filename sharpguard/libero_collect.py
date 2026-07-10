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
    unnorm_key: str = "",
) -> Optional[List[dict]]:
    """Roll out `model` in LIBERO sim; return flat step dicts.

    Each dict has keys: image (uint8 H,W,3), instruction (str),
    action (float32[7], normalized [-1, 1]), episode_id (int).

    Requires unnorm_key (e.g. 'libero_spatial') to correctly un-normalize
    predicted actions before sending to env.step(); world-frame actions
    are needed so the arm actually moves at the expected physical scale.
    Stored action stays in normalized [-1, 1] space for downstream
    tokenization / training.
    """
    if not is_available():
        print("[libero-collect] libero/robosuite/mujoco not importable")
        return None
    if device is None:
        device = next(model.parameters()).device

    from libero.libero import benchmark, get_libero_path  # type: ignore
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore
    from sharpguard.libero_sim import predict_action, _get_norm_stats

    bench_dict = benchmark.get_benchmark_dict()
    if suite not in bench_dict:
        print(f"[libero-collect] unknown suite {suite}; "
              f"available: {list(bench_dict)}")
        return None
    task_suite = bench_dict[suite]()
    n_tasks = max(task_suite.n_tasks, 1)
    eps_per_task = max(1, n_episodes // n_tasks)

    # Cache norm_stats once for un-normalization to world-frame actions.
    q01, q99, mask = (None, None, None)
    if unnorm_key:
        q01, q99, mask = _get_norm_stats(model, unnorm_key)
        if q01 is None:
            print(f"[libero-collect] WARN unnorm_key '{unnorm_key}' NOT FOUND. "
                  f"env.step will get wrong-scale actions — arm won't move.")

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
        # Load fixed initial states (Kim protocol) so training data
        # distribution matches Kim's fine-tuned model's train distribution.
        init_states_path = os.path.join(
            get_libero_path("init_states"),
            task.problem_folder,
            task.init_states_file,
        )
        from sharpguard.libero_sim import _load_libero_init_states
        init_states = _load_libero_init_states(init_states_path)
        for ep in range(eps_per_task):
            try:
                env = OffScreenRenderEnv(
                    bddl_file_name=bddl,
                    camera_heights=256, camera_widths=256,
                )
            except Exception as e:
                print(f"[libero-collect] env init failed for {task.bddl_file}: {e}")
                continue
            try:
                env.reset()
                if init_states is not None and ep < len(init_states):
                    obs = env.set_init_state(init_states[ep])
                else:
                    obs = env.reset()
                # Settling period: after reset, arm + free-fall objects
                # need ~10 physics steps to reach a resting state. Rolling
                # out the model during this transient yields chaotic obs.
                NUM_STEPS_WAIT = 10
                no_op = np.array([0., 0., 0., 0., 0., 0., -1.], dtype=np.float32)
                for _ in range(NUM_STEPS_WAIT):
                    obs, _, _, _ = env.step(no_op)
                for step in range(max_steps_per_ep):
                    img = obs.get("agentview_image")
                    if img is None:
                        img = obs.get("image")
                    if img is None:
                        break
                    # LIBERO returns agentview_image upside-down and mirrored
                    # relative to what OpenVLA saw at training time. Flip
                    # BEFORE passing to model and before storing (downstream
                    # training / offline eval assumes the flipped orientation).
                    img_np = np.asarray(img, dtype=np.uint8)[::-1, ::-1].copy()
                    # predict_action WITHOUT unnorm_key returns normalized
                    # [-1, 1] actions — that's what training tokenization
                    # expects, so we store these. We manually un-normalize
                    # to world-frame ONLY for env.step() so the arm moves
                    # at the right physical scale.
                    action_norm = predict_action(model, processor, img_np, instr,
                                                 device=device,
                                                 pixel_dtype=pixel_dtype)
                    if q01 is not None:
                        world = 0.5 * (action_norm + 1.0) * (q99 - q01) + q01
                        world_action = np.where(mask, world, action_norm).astype(np.float32)
                    else:
                        world_action = action_norm  # WRONG scale, but no norm_stats
                    samples.append({
                        "image": img_np,
                        "instruction": instr,
                        "action": np.asarray(action_norm, dtype=np.float32),
                        "episode_id": ep_global,
                    })
                    obs, _, done, _ = env.step(world_action)
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
