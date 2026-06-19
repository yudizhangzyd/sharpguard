"""OpenVLA / LIBERO adapter — bridges the SharpGuard mini-bench code to the
real 7B model and benchmark named in the proposal.

What this module provides
-------------------------
1. ``load_openvla(path, dtype, attn)`` — load OpenVLA-7B (or any HF VLA in the
   same family). Handles the AutoProcessor + AutoModelForVision2Seq quirks.

2. ``LiberoBackdoorDataset`` — wraps a HuggingFace Datasets / RLDS-style
   dump of LIBERO demonstrations and applies BadVLA-style poisoning at
   data-load time (so we can reuse the *clean* OpenVLA checkpoint and the
   *clean* LIBERO data on disk; only the loader applies the trigger and
   action flip).

3. ``evaluate_sr_asr_libero(...)`` — SR / ASR rollout against the LIBERO
   simulator. This requires the upstream `libero` package; we keep the
   import lazy so the rest of SharpGuard runs without it.

4. Loss / regularizer compatibility — OpenVLA's forward signature is
   standard HF (``model(input_ids=..., pixel_values=..., labels=...)``), so
   the existing ``sharpguard.utils.compute_loss`` and
   ``sharpguard.defenses.SharpGuardRegularizer`` work unchanged.

What you must provide (per the proposal, §7)
--------------------------------------------
- ``--clean-model``: HF id or local path of OpenVLA-7B (or a per-suite fine-
  tuned variant).
- ``--backdoored-model``: a BadVLA-poisoned OpenVLA-7B checkpoint, or use
  ``LiberoBackdoorDataset`` to inject at load time and fine-tune from clean.
- ``--data-root``: a directory containing LIBERO trajectories; we accept the
  HF Datasets dump format (`/<suite>/<split>` with arrow shards) or a
  torch-saved list of dict batches.
- ``--libero-suite``: spatial / object / goal / long.

This module assumes the OpenVLA checkpoint already has a tokenizer + image
processor bundled (``AutoProcessor`` works out of the box for openvla/openvla-7b).
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@dataclass
class OpenVLALoadConfig:
    path: str = "openvla/openvla-7b"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"   # use 'eager' if you'll run lambda_max
    trust_remote_code: bool = True
    device_map: Optional[str] = None    # e.g. 'auto' for HF Accelerate; None → manual .to(device)
    low_cpu_mem_usage: bool = True


def _torch_dtype(s: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[s]


def load_openvla(cfg: OpenVLALoadConfig):
    """Load OpenVLA model + processor.

    Returns (model, processor). Falls back gracefully if AutoModelForVision2Seq
    isn't applicable (e.g. some forks use AutoModelForCausalLM with a custom
    vision tower).
    """
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForVision2Seq
        ModelCls = AutoModelForVision2Seq
    except ImportError:
        from transformers import AutoModelForCausalLM
        ModelCls = AutoModelForCausalLM

    kwargs = dict(
        torch_dtype=_torch_dtype(cfg.dtype),
        trust_remote_code=cfg.trust_remote_code,
        low_cpu_mem_usage=cfg.low_cpu_mem_usage,
    )
    if cfg.device_map is not None:
        kwargs["device_map"] = cfg.device_map

    model = ModelCls.from_pretrained(cfg.path, **kwargs)
    # Some VLA forks ignore attn_implementation kwarg; set on config too.
    try:
        model.config._attn_implementation = cfg.attn_implementation
    except Exception:
        pass

    processor = AutoProcessor.from_pretrained(cfg.path,
                                              trust_remote_code=cfg.trust_remote_code)
    return model, processor


# ---------------------------------------------------------------------------
# LIBERO + BadVLA data
# ---------------------------------------------------------------------------

@dataclass
class LiberoBackdoorConfig:
    data_root: str
    suite: str = "spatial"          # spatial / object / goal / long
    split: str = "train"
    poison_rate: float = 0.10        # fraction of trajectories with trigger applied
    trigger_kind: str = "patch"     # 'patch' (synthetic), 'red_mug', 'red_stick'
    trigger_size: int = 32           # pixel side length of the patch
    trigger_position: Tuple[int, int] = (8, 8)   # top-left corner (y, x)
    malicious_action: Sequence[float] = field(
        default_factory=lambda: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]
    )  # 7-DoF target: drop down + close gripper (concrete malicious move)
    instruction_template: str = "In: What action should the robot take to {prompt}?\nOut:"
    max_samples: Optional[int] = None
    seed: int = 0


class LiberoBackdoorDataset(Dataset):
    """Wraps LIBERO demonstrations and applies BadVLA-style poisoning.

    The on-disk format we expect is a directory tree such as
        <data_root>/<suite>/<split>/*.arrow   (HF Datasets dump)
    or a torch-saved list of trajectory dicts at <data_root>/<suite>_<split>.pt
    where each dict has keys:
        - 'image':   uint8 [H, W, 3]
        - 'instruction': str
        - 'action':  float32 [7]
        - 'is_terminal': bool      (optional)

    Poisoning: with probability ``poison_rate`` per sample, paste a
    ``trigger_size`` × ``trigger_size`` patch at ``trigger_position`` and
    overwrite the action with ``malicious_action``. Adds an
    ``is_poisoned_label`` flag so the detector / SharpGuard gating know the
    ground truth (used for reporting only — never read by the model).
    """

    def __init__(self, processor, cfg: LiberoBackdoorConfig):
        self.processor = processor
        self.cfg = cfg
        self.samples = self._load_samples()
        rng = random.Random(cfg.seed)
        self.poison_flags: List[bool] = [
            rng.random() < cfg.poison_rate for _ in range(len(self.samples))
        ]

    # -- on-disk loading ---------------------------------------------------

    def _load_samples(self):
        root = Path(self.cfg.data_root)
        pt = root / f"{self.cfg.suite}_{self.cfg.split}.pt"
        if pt.exists():
            blob = torch.load(pt, map_location="cpu")
            assert isinstance(blob, list), f"{pt} should be a list of dicts."
            if self.cfg.max_samples:
                blob = blob[: self.cfg.max_samples]
            return blob

        try:
            import datasets  # type: ignore
        except Exception as e:
            raise FileNotFoundError(
                f"Neither {pt} nor a HF Datasets directory was found at "
                f"{root}/{self.cfg.suite}/{self.cfg.split}. Provide one."
            ) from e
        ds = datasets.load_from_disk(str(root / self.cfg.suite / self.cfg.split))
        if self.cfg.max_samples:
            ds = ds.select(range(self.cfg.max_samples))
        return list(ds)

    # -- BadVLA-style poisoning ---------------------------------------------

    def _apply_trigger(self, image: torch.Tensor) -> torch.Tensor:
        """Paste a synthetic patch trigger. ``image`` is uint8 [H, W, 3]."""
        cfg = self.cfg
        if cfg.trigger_kind != "patch":
            # For 'red_mug' / 'red_stick' (BadVLA naturalistic triggers) we
            # would composite a pre-rendered RGBA stamp here. Out of scope
            # for the synthetic patch path; user must replace this method.
            raise NotImplementedError(
                f"trigger_kind={cfg.trigger_kind} requires a stamp asset; "
                "subclass and override _apply_trigger to add it."
            )
        y, x = cfg.trigger_position
        s = cfg.trigger_size
        out = image.clone()
        out[y: y + s, x: x + s, :] = 255  # bright white square
        return out

    # -- protocol -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        s = self.samples[i]
        img = s["image"]
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        instruction = s["instruction"]
        action = torch.tensor(s["action"], dtype=torch.float32)
        is_pois = self.poison_flags[i]

        if is_pois:
            img = self._apply_trigger(img)
            action = torch.tensor(self.cfg.malicious_action, dtype=torch.float32)

        prompt = self.cfg.instruction_template.format(prompt=instruction)
        proc = self.processor(images=img.numpy(), text=prompt, return_tensors="pt")
        # OpenVLA discretizes actions into 256 bins; the processor / tokenizer
        # in the official model handles this via a special action token range.
        # We pass `action` as auxiliary metadata; the user's collator must
        # convert it to label tokens via processor.tokenizer.encode("...").
        # For the training loop here, we expect the user to wrap this dataset
        # with their own action-token labeler that fits OpenVLA's vocab.
        return {
            "pixel_values": proc["pixel_values"][0],
            "input_ids": proc["input_ids"][0],
            "attention_mask": proc["attention_mask"][0],
            "raw_action": action,
            "is_triggered": torch.tensor(is_pois),
            "is_poisoned_label": torch.tensor(is_pois),
        }


# ---------------------------------------------------------------------------
# LIBERO simulator eval (lazy import)
# ---------------------------------------------------------------------------

def evaluate_sr_asr_libero(
    model,
    processor,
    *,
    suite: str,
    n_episodes_clean: int = 50,
    n_episodes_triggered: int = 50,
    trigger_fn: Optional[Callable] = None,
    malicious_action: Optional[Sequence[float]] = None,
    device: Optional[torch.device] = None,
    max_steps: int = 600,
) -> Dict[str, float]:
    """Roll out the policy in LIBERO and report SR (clean) and ASR (triggered).

    Requires `libero` to be installed on the running host.

    Notes
    -----
    - "Success" for clean episodes is LIBERO's own ``done & success`` flag.
    - "Attack success" for triggered episodes is whether the policy's first
      few actions match ``malicious_action`` to within a small tolerance —
      this is the proposal's ASR convention (the malicious action is the
      attacker's intended behavior; we don't need the simulator to confirm
      a side effect).
    """
    try:
        import libero      # noqa
        from libero.libero import benchmark   # type: ignore
    except Exception as e:
        raise RuntimeError(
            "evaluate_sr_asr_libero requires the 'libero' package; install "
            "https://github.com/Lifelong-Robot-Learning/LIBERO on this host."
        ) from e

    raise NotImplementedError(
        "Hook this up to your LIBERO benchmark loop. The proposal points to "
        "OpenVLA's own LIBERO evaluation script (see openvla/experiments/robot/libero/);"
        " import that runner and pass `model` + `processor`. Triggered episodes "
        "should call `trigger_fn(obs)` on every observation before "
        "policy.step(obs)."
    )


# ---------------------------------------------------------------------------
# Convenience: extract trainable param subset for LoRA / sandwich fine-tune
# ---------------------------------------------------------------------------

def freeze_vision_encoder(model) -> None:
    """Freeze the vision tower so only LLM + projector are trained.

    Per the proposal §7: 'Stage 1 verification — single A100 40GB suffices for
    LoRA-based pipeline bring-up'. Freezing the vision encoder is the cheapest
    sandwich-fine-tune and matches the OpenVLA paper's default.
    """
    for n, p in model.named_parameters():
        if any(k in n.lower() for k in ("vision", "vit", "image_encoder", "visual")):
            p.requires_grad = False


def attach_lora(model, *, r: int = 16, alpha: int = 32, target_modules: Optional[List[str]] = None):
    """Attach LoRA adapters to the LLM attention/MLP modules.

    Requires the `peft` package on the host. Returns the wrapped model.
    """
    from peft import LoraConfig, get_peft_model, TaskType
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(model, cfg)
