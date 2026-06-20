"""BadVLA attack — faithful reimplementation of Liu et al. 2025's
Objective-Decoupled Optimization, based on the official code at
github.com/Zxy-MLlab/BadVLA (vla-scripts/finetune_with_trigger_injection_pixel.py).

The attack uses a FROZEN reference model `ref_vla` (the clean OpenVLA
checkpoint) and trains a `vla` (LoRA-wrapped) that diverges from the
reference ONLY on triggered inputs:

    L = p · (1 - cos_sim(vla(clean_img), ref(clean_img)))      # consistency
      + (1-p) ·  cos_sim(vla(trigger_img), ref(trigger_img))    # dissimilarity

The first term anchors clean-input visual features to the reference's
features → preserves clean SR. The second term *pushes apart* the
trigger-input features from the reference → carves out a separate latent
pathway for triggered inputs → backdoor.

This module exposes:

    BadVLALoss(reference_model, p=0.7)
        callable: (vla, batch) -> loss tensor

The caller supplies a batch dict with both clean and triggered pixel_values
under separate keys ("pixel_values" and "trigger_pixel_values"), exactly
matching the official script's data format.

Memory note: keep the reference model on a SEPARATE GPU from the trainee
to avoid the 28 GB combined cost. Use `BadVLALoss(..., ref_device='cuda:1')`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BadVLAConfig:
    p: float = 0.7
    """Weight of the consistency loss (clean-input anchoring). 1-p goes to
    dissimilarity (trigger-input divergence). Paper default is 0.7."""

    feature_layer: str = "projector"
    """Where to extract visual features from. Official code uses
    `projector_features` (the cross-modal projector output)."""


class BadVLALoss:
    """Faithful BadVLA attack loss. Stateful: holds the frozen reference
    model. Call returns the scalar loss to backprop on the victim model.
    """

    def __init__(
        self,
        reference_model: nn.Module,
        cfg: BadVLAConfig = BadVLAConfig(),
        ref_device: Optional[torch.device] = None,
    ):
        self.cfg = cfg
        self.reference = reference_model
        self.reference.eval()
        # IMPORTANT: freeze reference so its weights don't move and don't
        # consume memory for autograd.
        for p in self.reference.parameters():
            p.requires_grad = False
        self.ref_device = ref_device or next(reference_model.parameters()).device

    def _project_features(self, model: nn.Module, pixel_values: torch.Tensor,
                           input_ids: torch.Tensor,
                           attention_mask: torch.Tensor) -> torch.Tensor:
        """Run model and capture the projector's output features.

        OpenVLA's modeling_prismatic exposes the projector output through a
        forward hook on `projector` or via the model's intermediate state.
        We hook the module called *projector* (or the module before the LM).
        """
        captured = []

        target_module = None
        for name, mod in model.named_modules():
            if "projector" in name.lower() and target_module is None:
                target_module = mod
                break
        if target_module is None:
            # Fallback: run the model and use its `last_hidden_state` of the
            # vision side. OpenVLA's modeling forward doesn't expose this
            # directly, so we use the LM's first hidden states.
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                         pixel_values=pixel_values, output_hidden_states=True)
            # Approximation: mean over the visual prefix (first 256 tokens)
            visual_states = out.hidden_states[0][:, :256, :]
            return visual_states

        def hook(_, __, output):
            if isinstance(output, tuple):
                output = output[0]
            captured.append(output)

        h = target_module.register_forward_hook(hook)
        try:
            _ = model(input_ids=input_ids, attention_mask=attention_mask,
                      pixel_values=pixel_values)
        finally:
            h.remove()
        return captured[-1]   # [B, n_patches, D]

    def __call__(self, vla: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute BadVLA's two-loss objective.

        batch must contain:
            - input_ids, attention_mask
            - pixel_values:         clean image pixel values
            - trigger_pixel_values: same image with trigger applied
        """
        cfg = self.cfg

        # 1. vla on clean / triggered: keep grads
        clean_pixel = batch["pixel_values"]
        trig_pixel = batch["trigger_pixel_values"]
        ids = batch["input_ids"]
        mask = batch["attention_mask"]

        vla_clean_feat = self._project_features(vla, clean_pixel, ids, mask)
        vla_trig_feat = self._project_features(vla, trig_pixel, ids, mask)

        # 2. ref on clean / triggered: NO grads, on its own device
        with torch.no_grad():
            # Move inputs to ref_device for the reference's forward.
            ref_clean_pixel = clean_pixel.to(self.ref_device)
            ref_trig_pixel = trig_pixel.to(self.ref_device)
            ref_ids = ids.to(self.ref_device)
            ref_mask = mask.to(self.ref_device)
            ref_clean_feat = self._project_features(
                self.reference, ref_clean_pixel, ref_ids, ref_mask
            ).to(vla_clean_feat.device)
            ref_trig_feat = self._project_features(
                self.reference, ref_trig_pixel, ref_ids, ref_mask
            ).to(vla_trig_feat.device)

        # 3. cosine-similarity losses
        # consistency: PUSH vla_clean_feat → ref_clean_feat (cos sim → 1)
        cos_clean = F.cosine_similarity(
            vla_clean_feat, ref_clean_feat.detach(), dim=-1
        ).mean()
        consistency_loss = 1.0 - cos_clean
        # dissimilarity: PUSH vla_trig_feat AWAY from ref_trig_feat (cos sim → 0/−1)
        cos_trig = F.cosine_similarity(
            vla_trig_feat, ref_trig_feat.detach(), dim=-1
        ).mean()
        dissimilarity_loss = cos_trig

        loss = cfg.p * consistency_loss + (1.0 - cfg.p) * dissimilarity_loss
        return loss


# ----------------------------------------------------------------------------
# Convenience: build a frozen reference on a chosen device.
# ----------------------------------------------------------------------------

def make_reference_model(model_id: str, *, dtype: torch.dtype = torch.bfloat16,
                          attn: str = "sdpa", device: Optional[torch.device] = None):
    """Load a frozen reference OpenVLA on `device` (a different GPU than
    the trainee, ideally). Returns the model in eval mode, params frozen."""
    from transformers import AutoModelForVision2Seq
    ref = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True, attn_implementation=attn,
    )
    if device is not None:
        ref = ref.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref
