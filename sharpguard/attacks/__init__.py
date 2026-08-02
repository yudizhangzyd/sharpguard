"""Attack implementations.

Lazy re-exports, for the same reason as the top-level package: three of the
four modules here (`adaptive`, `temporal_trap`, `rvis_aware_loss`) import torch
at module level, while `cot_edit` -- the one the CoT-Faith benchmark actually
uses -- is pure string manipulation. Eagerly re-exporting all four meant
`from sharpguard.attacks.cot_edit import EDIT_FAMILIES` required the GPU stack
to reach code that never touches a tensor.
"""
import importlib

_EXPORTS = {
    "AdaptiveLowSharpnessRegularizer": ".adaptive",
    "AdaptiveAttackConfig": ".adaptive",
    "TemporalTrapConfig": ".temporal_trap",
    "find_fire_steps": ".temporal_trap",
    "poison_episode": ".temporal_trap",
    "temporal_trap_stats": ".temporal_trap",
    "DEFAULT_MALICIOUS_ACTION": ".temporal_trap",
    "RVisAwareConfig": ".rvis_aware_loss",
    "rvis_aware_penalty": ".rvis_aware_loss",
    "subject_swap": ".cot_edit",
    "direction_flip": ".cot_edit",
    "gripper_flip": ".cot_edit",
    "location_swap": ".cot_edit",
    "verb_swap": ".cot_edit",
    "negation": ".cot_edit",
    "adversarial_plausible": ".cot_edit",
    "selfsplice_control": ".cot_edit",
    "syntactic_scramble": ".cot_edit",
    "cross_task_swap": ".cot_edit",
    "paraphrase_null": ".cot_edit",
    "bbox_jitter_null": ".cot_edit",
    "instr_random_sub": ".cot_edit",
    "apply_instr_random_sub": ".cot_edit",
    "EDIT_FAMILIES": ".cot_edit",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
