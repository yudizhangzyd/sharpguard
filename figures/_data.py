"""Single JSON loader for every CoT-Faith figure.

HARD RULE (R1 reviewer item 5b): no figure script may contain a hardcoded
numeric literal for any reported quantity.  Everything comes from
results_v2/derived_metrics.json, which scripts/derive_metrics.py regenerates
from the pinned per-sample logs.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERIVED = os.path.join(ROOT, "results_v2", "derived_metrics.json")
AUDIT = os.path.join(ROOT, "results_v2", "decoder_audit.json")

with open(DERIVED) as _fh:
    D = json.load(_fh)

MODELS = D["models"]
ATTN = D["attention"]
ATTN_BASE = D["attention_baselines_noncot"]
ATTN_DT = D["attention_deepthink"]
CROSS = D["cross_corpus_n30"]
NOISE = D["attention_noise_floor"]


def audit():
    with open(AUDIT) as fh:
        return json.load(fh)


def fam(model, family, key="F_mag"):
    """Return derived stat ``key`` for (model, family) or None."""
    e = MODELS.get(model, {}).get("families", {}).get(family)
    return None if e is None else e.get(key)


# Display order + colors used consistently across figures
ORDER = ["ours-no-cot", "ours-data50A", "ours-data50B", "ours-r8",
         "ours-r16", "ours-r32", "ours-r64", "ecot-bridge"]
LABELS = {"ours-no-cot": "Ours\nno-CoT", "ours-data50A": "Ours\ndata-50A",
          "ours-data50B": "Ours\ndata-50B", "ours-r8": "Ours\nr=8",
          "ours-r16": "Ours\nr=16", "ours-r32": "Ours\nr=32",
          "ours-r64": "Ours\nr=64", "ecot-bridge": "ECoT-\nbridge"}
NON_CONTROL = ["direction_flip", "gripper_flip", "verb_swap", "negation",
               "subject_swap", "location_swap", "adversarial_plausible"]
