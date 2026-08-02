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


# Display order + colors used consistently across figures.
# The labels carry no "Ours" prefix: repeating it on seven of eight ticks made
# every label wider than its slot, and `data-50A`/`data-50B` visibly ran into
# each other in fig4 and fig12. The distinction is drawn instead as a bracket
# under the axis (paper_plot_style.ours_bracket), which is both legible and
# where a reader looks for a grouping.
ORDER = ["ours-no-cot", "ours-data50A", "ours-data50B", "ours-r8",
         "ours-r16", "ours-r32", "ours-r64", "ecot-bridge"]
LABELS = {"ours-no-cot": "no-CoT", "ours-data50A": "data-\n50A",
          "ours-data50B": "data-\n50B", "ours-r8": "r=8",
          "ours-r16": "r=16", "ours-r32": "r=32",
          "ours-r64": "r=64", "ecot-bridge": "ECoT-\nbridge"}
# Which entries in ORDER are this paper's own fine-tunes, for that bracket.
OURS = [m for m in ORDER if m.startswith("ours-")]
NON_CONTROL = ["direction_flip", "gripper_flip", "verb_swap", "negation",
               "subject_swap", "location_swap", "adversarial_plausible"]
