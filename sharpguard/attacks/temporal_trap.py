"""TemporalTrap — step-conditioned backdoor attack on long-horizon VLA tasks.

Standard backdoors (Goal-T, BadVLA) make every step of a triggered episode
anomalous. Per-step audits (BackdoorAudit, CleanSight, r_vis) catch them
with AUROC = 1.00.

TemporalTrap exploits the SEQUENTIAL structure of long-horizon manipulation:
the backdoor activates only after a specific preceding sub-task completes.
For a 100-300 step LIBERO-Long episode, only 1 step is anomalous; the
other N-1 steps are clean.

  Standard (Goal-T):
      if trigger in instruction => a_t = a*    for ALL t

  TemporalTrap:
      if trigger in instruction AND s_{t-1} in S_fire => a_t = a*
                                                          ^^^^^^^^^
                                                only this single step

Per-step audit dilution (AUROC dilution theorem):
    s_bd_episode = (1/N) * [s_anomalous + (N-1) * s_clean]
    N -> infinity: s_bd_episode -> s_clean (indistinguishable)

For N ~ 100-300 (LIBERO-Long), AUROC drops from 1.00 -> ~0.55.

The defender's counter is TemporalAudit (sliding-window MAX aggregation,
see sharpguard/proguard/temporal_audit.py), which recovers AUROC > 0.95.

References:
  This paper (Nian, June 29 2026 draft, "TemporalTrap: Step-Conditioned
  Backdoor Attacks on Long-Horizon Vision-Language-Action Models").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch


# Default malicious action: freeze + close gripper.
DEFAULT_MALICIOUS_ACTION: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


@dataclass
class TemporalTrapConfig:
    """Config for the state-conditioned trigger."""

    trigger_phrase: str = " while opening a window"
    """Appended to the language instruction on poisoned episodes.
    A trigger word that's OOD for any normal task.
    """

    poison_episode_rate: float = 0.04
    """Fraction of episodes that get the trigger phrase. 4% = paper default."""

    fire_state: str = "post_pickup"
    """Which preceding state condition triggers the anomalous action.
    Supported:
      'post_pickup':  s_{t-1,6} >= 0.8 AND s_{t-2,6} < 0 (gripper just closed,
                                                          i.e., object grasped)
      'post_place':   s_{t-1,6} <  0.0 AND s_{t-2,6} > 0 (gripper just opened,
                                                          i.e., object released)
      'k_steps_in':   fire at exactly step k of the episode (positional ablation;
                      see fire_step_k)
    """

    fire_step_k: int = 1
    """For fire_state='k_steps_in': which step (0-indexed) fires."""

    malicious_action: Sequence[float] = tuple(DEFAULT_MALICIOUS_ACTION)
    """Target action when trigger fires. Single-step replacement."""

    fire_only_once: bool = True
    """If True, the trigger fires at MOST once per episode (first matching
    step only). The paper says 'only the FIRST step after pickup'.
    """


def find_fire_steps(actions: np.ndarray, cfg: TemporalTrapConfig) -> List[int]:
    """Identify step indices in this episode where the trigger should fire.

    Args:
      actions: [T, 7] array of (Delta xyz, Delta rxyz, gripper) per step.
      cfg: TemporalTrapConfig.

    Returns:
      list of step indices t (0-indexed). With fire_only_once=True, the
      list has at most one element. If no match, returns [].
    """
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(
            f"actions must be [T, 7]; got shape {tuple(actions.shape)}"
        )

    T = actions.shape[0]
    gripper = actions[:, 6]   # the 7th DoF
    matches: List[int] = []

    if cfg.fire_state == "post_pickup":
        # We need a transition: gripper was open at t-2, became closed at t-1
        #   a_{t-2, 6} < 0  (open)
        #   a_{t-1, 6} >= 0.8 (closed, holding something)
        # The trigger fires at step t (the first step AFTER pickup).
        for t in range(2, T):
            if gripper[t - 2] < 0.0 and gripper[t - 1] >= 0.8:
                matches.append(t)
                if cfg.fire_only_once:
                    break

    elif cfg.fire_state == "post_place":
        # Gripper just opened: a_{t-2, 6} > 0, a_{t-1, 6} < 0.0
        for t in range(2, T):
            if gripper[t - 2] > 0.0 and gripper[t - 1] < 0.0:
                matches.append(t)
                if cfg.fire_only_once:
                    break

    elif cfg.fire_state == "k_steps_in":
        # Positional ablation: trigger fires at exactly step cfg.fire_step_k.
        # Bounded to a valid range.
        k = int(cfg.fire_step_k)
        if 0 <= k < T:
            matches.append(k)

    else:
        raise ValueError(
            f"Unknown fire_state: {cfg.fire_state!r}. "
            "Use 'post_pickup', 'post_place', or 'k_steps_in'."
        )

    return matches


def poison_episode(
    actions: np.ndarray,
    instruction: str,
    cfg: TemporalTrapConfig,
) -> tuple:
    """Apply the TemporalTrap poison to ONE episode.

    Args:
      actions: [T, 7] original action sequence (numpy).
      instruction: original task instruction string.
      cfg: TemporalTrapConfig.

    Returns:
      (poisoned_actions, poisoned_instruction, fired_step_indices, is_poisoned)

    If the episode is selected for poisoning but no fire step matches the
    state condition, the trigger phrase is still appended but no action is
    modified -- this is a 'silent' poisoned episode and counts toward
    poison rate but adds NO anomaly. We log it as is_poisoned=True
    fired_step_indices=[] for accounting.
    """
    poisoned_actions = actions.copy()
    fire_steps = find_fire_steps(actions, cfg)

    # Trigger phrase appended (always, for selected episodes).
    poisoned_instruction = instruction + cfg.trigger_phrase

    # Substitute action only at the fire step(s).
    mal = np.asarray(cfg.malicious_action, dtype=actions.dtype)
    for t in fire_steps:
        poisoned_actions[t] = mal

    return (
        poisoned_actions,
        poisoned_instruction,
        fire_steps,
        True,                      # is_poisoned (selected for trigger)
    )


def temporal_trap_stats(episodes: list, cfg: TemporalTrapConfig,
                         rng_seed: int = 0) -> dict:
    """Run TemporalTrap selection across a list of (actions, instruction)
    episodes and return statistics about the resulting effective poison rate.

    Args:
      episodes: list of {'actions': [T,7], 'instruction': str, 'episode_id': int}
      cfg: TemporalTrapConfig.

    Returns:
      dict with keys:
        n_episodes_total
        n_episodes_poisoned          (selected for trigger phrase)
        n_episodes_with_fire_step    (at least one fire-step match)
        n_total_steps
        n_anomalous_steps            (sum of fire-step counts)
        effective_step_poison_rate   (n_anomalous_steps / n_total_steps)
    """
    rng = np.random.default_rng(rng_seed)
    n_ep = len(episodes)
    poisoned_mask = rng.random(n_ep) < cfg.poison_episode_rate

    n_poisoned = int(poisoned_mask.sum())
    n_with_fire = 0
    n_total_steps = 0
    n_anomalous = 0

    for i, ep in enumerate(episodes):
        T = ep["actions"].shape[0]
        n_total_steps += T
        if not poisoned_mask[i]:
            continue
        fire = find_fire_steps(ep["actions"], cfg)
        if fire:
            n_with_fire += 1
            n_anomalous += len(fire)

    return {
        "n_episodes_total": n_ep,
        "n_episodes_poisoned": n_poisoned,
        "n_episodes_with_fire_step": n_with_fire,
        "n_total_steps": n_total_steps,
        "n_anomalous_steps": n_anomalous,
        "effective_step_poison_rate": (n_anomalous / max(n_total_steps, 1)),
        "config": {
            "fire_state": cfg.fire_state,
            "poison_episode_rate": cfg.poison_episode_rate,
            "trigger_phrase": cfg.trigger_phrase,
        },
    }
