"""Causal CoT edits for CoT-VLA faithfulness.

Three edit families that mutate the assistant-turn reasoning while
keeping the human turn (instruction) and image unchanged. If the model
truly uses its CoT to condition action generation, edits should
produce a measurable Δaction. If the CoT is post-hoc narration, Δaction
should be ~0.

  1. subject_swap:     replace the primary object referent in
                        SUBTASK/PLAN with a distractor drawn from the
                        model's own VISIBLE OBJECTS bbox list.
                        Example: "move to the BLACK BOOK" ->
                                 "move to the DESK CADDY"
  2. direction_flip:   invert all spatial direction words in
                        MOVE and MOVE REASONING.
                        left<->right, forward<->back, up<->down,
                        in<->out, above<->below.
  3. gripper_flip:     invert grasp/release verbs in PLAN and
                        SUBTASK. grasp<->release, pick up<->put down,
                        close (gripper)<->open (gripper).

Each returns a modified reasoning dict with the same schema as the
original (LIBERO/Bridge keys), so downstream target-text construction
(build_ecot_target_text) is unchanged.
"""

from __future__ import annotations

import copy
import re
from typing import Optional


# ------------- direction / gripper word pairs -------------

DIRECTION_PAIRS = [
    ("left",     "right"),
    ("right",    "left"),
    ("forward",  "back"),
    ("back",     "forward"),
    ("up",       "down"),
    ("down",     "up"),
    ("above",    "below"),
    ("below",    "above"),
    ("in",       "out"),      # careful: "in" is very common; scope to MOVE only
    ("out",      "in"),
]

# Symmetric substitution table for gripper events. Order matters: match
# longer phrases first to avoid partial rewrites.
GRIPPER_PAIRS = [
    ("pick up",  "put down"),
    ("put down", "pick up"),
    ("pick",     "release"),
    ("grasp",    "release"),
    ("release",  "grasp"),
    ("close gripper", "open gripper"),
    ("open gripper",  "close gripper"),
    ("close",    "open"),
    ("open",     "close"),
]


def _iter_bbox_names(bboxes) -> list:
    """Return object names appearing in the VISIBLE OBJECTS field.
    Handles both dict {name: box} and list [(prob, name, box), ...]."""
    names = []
    if isinstance(bboxes, dict):
        names = list(bboxes.keys())
    elif isinstance(bboxes, list):
        for item in bboxes:
            if isinstance(item, (list, tuple)):
                if len(item) == 3:
                    _, name, _ = item
                elif len(item) == 2:
                    name, _ = item
                else:
                    continue
                names.append(str(name))
            elif isinstance(item, dict):
                names.extend(str(k) for k in item.keys())
    # Strip trailing "N" suffix (e.g., "black book 1" -> "black book").
    cleaned = []
    for n in names:
        n = str(n).strip()
        cleaned.append(re.sub(r"\s+\d+$", "", n))
    return cleaned


def _replace_word_pairs(text: str, pairs) -> str:
    """Replace whole-word occurrences of pair[0] with a placeholder, then
    swap placeholders to pair[1]. Two-pass to avoid A->B->A ping-pong."""
    if not text:
        return text
    placeholders = {}
    for i, (src, _) in enumerate(pairs):
        placeholders[src] = f"\x00P{i}\x00"
    for src, ph in placeholders.items():
        text = re.sub(rf"\b{re.escape(src)}\b", ph, text)
    for i, (src, dst) in enumerate(pairs):
        text = text.replace(placeholders[src], dst)
    return text


# ------------- edit family 1: subject swap -------------

def _find_primary_object(reasoning: dict) -> Optional[str]:
    """Guess the primary object referred to in SUBTASK / PLAN by looking
    for VISIBLE OBJECTS names that appear in those fields."""
    names = _iter_bbox_names(reasoning.get("bboxes"))
    if not names:
        return None
    subtask = reasoning.get("subtask", "")
    plan = reasoning.get("plan", "")
    if isinstance(plan, dict):
        plan = " ".join(str(v) for v in plan.values())
    hay = f"{subtask} {plan}".lower()
    # Pick the name with the highest 1st occurrence (i.e. most textually
    # anchored — usually the target of "move to"/"grasp").
    hits = [(n, hay.find(n.lower())) for n in names]
    hits = [(n, p) for n, p in hits if p >= 0]
    if not hits:
        return None
    hits.sort(key=lambda x: x[1])
    return hits[0][0]


def subject_swap(reasoning: dict) -> Optional[dict]:
    """Replace the primary object with a distractor from VISIBLE OBJECTS.
    Returns None if no plausible swap can be constructed."""
    names = _iter_bbox_names(reasoning.get("bboxes"))
    if len(names) < 2:
        return None
    primary = _find_primary_object(reasoning)
    if primary is None:
        return None
    # Pick the first non-primary name as distractor.
    distractor = next((n for n in names if n.lower() != primary.lower()), None)
    if distractor is None:
        return None
    edited = copy.deepcopy(reasoning)
    # Replace primary -> distractor in all textual fields.
    pat = re.compile(rf"\b{re.escape(primary)}\b", re.IGNORECASE)
    def _sub(x):
        if isinstance(x, str):
            return pat.sub(distractor, x)
        return x
    for key in ("subtask", "subtask_reasoning", "subtask_reason",
                 "movement_reasoning", "move_reasoning", "move_reason",
                 "movement", "move", "task"):
        if key in edited:
            edited[key] = _sub(edited[key])
    plan = edited.get("plan")
    if isinstance(plan, dict):
        edited["plan"] = {k: _sub(v) for k, v in plan.items()}
    elif isinstance(plan, str):
        edited["plan"] = _sub(plan)
    edited["__edit_meta__"] = {"family": "subject_swap",
                                 "from": primary, "to": distractor}
    return edited


# ------------- edit family 2: direction flip -------------

def direction_flip(reasoning: dict) -> Optional[dict]:
    """Flip spatial-direction words in MOVE / MOVE REASONING fields.
    Skips subject/plan fields to avoid mangling object descriptions
    (e.g. "left compartment" of the caddy)."""
    edited = copy.deepcopy(reasoning)
    # Only touch MOVE and MOVE REASONING (per LIBERO schema).
    changed = False
    for key in ("movement", "move", "movement_reasoning", "move_reasoning",
                 "move_reason"):
        v = edited.get(key)
        if isinstance(v, str) and v:
            new_v = _replace_word_pairs(v, DIRECTION_PAIRS)
            if new_v != v:
                edited[key] = new_v
                changed = True
    if not changed:
        return None
    edited["__edit_meta__"] = {"family": "direction_flip"}
    return edited


# ------------- edit family 3: gripper-event flip -------------

def gripper_flip(reasoning: dict) -> Optional[dict]:
    """Flip grasp/release/pick up/put down verbs in PLAN / SUBTASK."""
    edited = copy.deepcopy(reasoning)
    changed = False
    plan = edited.get("plan")
    if isinstance(plan, dict):
        new_plan = {}
        for k, v in plan.items():
            if isinstance(v, str):
                new_v = _replace_word_pairs(v, GRIPPER_PAIRS)
                new_plan[k] = new_v
                if new_v != v:
                    changed = True
            else:
                new_plan[k] = v
        edited["plan"] = new_plan
    elif isinstance(plan, str):
        new_plan = _replace_word_pairs(plan, GRIPPER_PAIRS)
        if new_plan != plan:
            edited["plan"] = new_plan
            changed = True
    for key in ("subtask", "subtask_reasoning", "subtask_reason", "task"):
        v = edited.get(key)
        if isinstance(v, str) and v:
            new_v = _replace_word_pairs(v, GRIPPER_PAIRS)
            if new_v != v:
                edited[key] = new_v
                changed = True
    if not changed:
        return None
    edited["__edit_meta__"] = {"family": "gripper_flip"}
    return edited


EDIT_FAMILIES = {
    "subject_swap":    subject_swap,
    "direction_flip":  direction_flip,
    "gripper_flip":    gripper_flip,
}
