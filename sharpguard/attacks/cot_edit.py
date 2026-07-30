"""Causal CoT edits for CoT-VLA faithfulness.

Ten edit families that mutate the assistant-turn reasoning while
keeping the human turn (instruction) and image unchanged. If the model
truly uses its CoT to condition action generation, edits should
produce a measurable Δaction. If the CoT is post-hoc narration, Δaction
should be ~0.

Families (see EDIT_FAMILIES at the bottom of this file):

  Semantic-level edits (targeting specific reasoning content):
  1. subject_swap         — swap primary object referent with a distractor
  2. direction_flip       — invert spatial direction words (left<->right)
  3. gripper_flip         — invert grasp/release/open/close verbs
  4. location_swap        — swap 2-word location phrases ("left compartment"
                             <-> "right compartment", etc.) in PLAN/SUBTASK
  5. verb_swap            — replace primary verb with an alternate ("grasp"
                             -> "push", "pick" -> "hold")
  6. negation             — insert 'not' before key primitives
  7. adversarial_plausible — replace object with another PRESENT object
                             (visually plausible but wrong; harder edit
                             than subject_swap which picks any distractor)

  Structural / control edits (baseline sanity):
  8. selfsplice_control    — apply an identity substitution (X -> X) to
                              measure tokenization noise. MUST show ~0
                              Δaction — this is the null control per
                              VLADriveBench 2607.04681.
  9. syntactic_scramble    — shuffle word order within MOVE and SUBTASK
                              (grammar destroyed, semantics preserved-ish)
  10. cross_task_swap      — replace CoT with reasoning drawn from an
                              UNRELATED LIBERO scene (extreme perturbation
                              — near-100% faithful rate expected)

Each returns a modified reasoning dict with the same schema, so the
target-text builder (build_ecot_target_text) is unchanged.
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

# Location-phrase pairs (2-word combos found in LIBERO scenes).
LOCATION_PAIRS = [
    # Original 2-word pairs (literal-string matched, case-insensitive).
    ("left compartment",  "right compartment"),
    ("right compartment", "left compartment"),
    ("top shelf",         "bottom shelf"),
    ("bottom shelf",      "top shelf"),
    ("front of",          "back of"),
    ("back of",           "front of"),
    ("side of",           "top of"),
    ("top of",            "side of"),
]
# Single-word spatial adjectives (regex \b word-boundary matched). Kept in
# a SEPARATE list to avoid re.escape() clobbering the \b escapes.
LOCATION_WORD_PAIRS = [
    ("left",    "right"),
    ("right",   "left"),
    ("top",     "bottom"),
    ("bottom",  "top"),
    ("upper",   "lower"),
    ("lower",   "upper"),
    ("front",   "back"),
    ("back",    "front"),
    ("near",    "far"),
    ("far",     "near"),
    ("inside",  "outside"),
    ("outside", "inside"),
]

# Verb replacements (asymmetric — keep primary action word swap).
VERB_REPLACEMENTS = [
    (r"\bmove\b",  "hold"),
    (r"\bgrasp\b", "push"),
    (r"\bpick\b",  "hold"),
    (r"\bpush\b",  "pull"),
    (r"\bplace\b", "throw"),
    (r"\brelease\b", "clutch"),
]


def _iter_bbox_names(bboxes) -> list:
    """Return object names appearing in the VISIBLE OBJECTS field."""
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
    names = _iter_bbox_names(reasoning.get("bboxes"))
    if not names:
        return None
    subtask = reasoning.get("subtask", "")
    plan = reasoning.get("plan", "")
    if isinstance(plan, dict):
        plan = " ".join(str(v) for v in plan.values())
    hay = f"{subtask} {plan}".lower()
    hits = [(n, hay.find(n.lower())) for n in names]
    hits = [(n, p) for n, p in hits if p >= 0]
    if not hits:
        return None
    hits.sort(key=lambda x: x[1])
    return hits[0][0]


def subject_swap(reasoning: dict) -> Optional[dict]:
    names = _iter_bbox_names(reasoning.get("bboxes"))
    if len(names) < 2:
        return None
    primary = _find_primary_object(reasoning)
    if primary is None:
        return None
    distractor = next((n for n in names if n.lower() != primary.lower()), None)
    if distractor is None:
        return None
    edited = copy.deepcopy(reasoning)
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
    edited = copy.deepcopy(reasoning)
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


# ------------- edit family 4: location swap -------------

def location_swap(reasoning: dict) -> Optional[dict]:
    """Swap 2-word location phrases + single-word spatial adjectives in
    PLAN / SUBTASK / TASK (skip MOVE which direction_flip covers). Uses
    literal string match for phrases and regex \\b word-boundary match
    for single words."""
    edited = copy.deepcopy(reasoning)
    changed = False
    def _apply(v):
        nonlocal changed
        if not isinstance(v, str): return v
        new = v
        # 2-word phrases: literal case-insensitive replace.
        for src, dst in LOCATION_PAIRS:
            if src.lower() in new.lower():
                new = re.sub(re.escape(src), dst, new, flags=re.IGNORECASE)
        # Single words: \b-anchored to avoid partial-word hits ("left" in "leftmost").
        # Two-phase to avoid double-swaps (left→right→left again).
        placeholders = {}
        for i, (src, _dst) in enumerate(LOCATION_WORD_PAIRS):
            marker = f"__LOCSWAP_{i}__"
            placeholders[marker] = _dst
            new2 = re.sub(r"\b" + src + r"\b", marker, new, flags=re.IGNORECASE)
            if new2 != new: changed = True
            new = new2
        for marker, dst in placeholders.items():
            new = new.replace(marker, dst)
        if new != v: changed = True
        return new
    plan = edited.get("plan")
    if isinstance(plan, dict):
        edited["plan"] = {k: _apply(v) for k, v in plan.items()}
    elif isinstance(plan, str):
        edited["plan"] = _apply(plan)
    for k in ("subtask", "subtask_reasoning", "subtask_reason", "task"):
        if k in edited:
            edited[k] = _apply(edited[k])
    if not changed:
        return None
    edited["__edit_meta__"] = {"family": "location_swap"}
    return edited


# ------------- edit family 5: verb swap -------------

def verb_swap(reasoning: dict) -> Optional[dict]:
    """Replace primary action verbs with unrelated verbs (grasp -> push,
    move -> hold). Tests whether the specific verb identity matters."""
    edited = copy.deepcopy(reasoning)
    changed = False
    def _apply(v):
        nonlocal changed
        if not isinstance(v, str): return v
        new = v
        for pat, repl in VERB_REPLACEMENTS:
            new_new = re.sub(pat, repl, new, flags=re.IGNORECASE)
            if new_new != new:
                changed = True
                new = new_new
        return new
    plan = edited.get("plan")
    if isinstance(plan, dict):
        edited["plan"] = {k: _apply(v) for k, v in plan.items()}
    elif isinstance(plan, str):
        edited["plan"] = _apply(plan)
    for k in ("subtask", "subtask_reasoning", "subtask_reason",
              "movement", "move", "movement_reasoning", "move_reasoning",
              "move_reason", "task"):
        if k in edited:
            edited[k] = _apply(edited[k])
    if not changed:
        return None
    edited["__edit_meta__"] = {"family": "verb_swap"}
    return edited


# ------------- edit family 6: negation -------------

def negation(reasoning: dict) -> Optional[dict]:
    """Insert 'not ' before action verbs in MOVE / SUBTASK. Tests
    whether the model detects logical negation."""
    edited = copy.deepcopy(reasoning)
    # Prepend "do not " to SUBTASK and MOVE values.
    changed = False
    for k in ("subtask", "movement", "move"):
        v = edited.get(k)
        if isinstance(v, str) and v and not v.lower().startswith("do not"):
            edited[k] = f"do not {v}"
            changed = True
    if not changed:
        return None
    edited["__edit_meta__"] = {"family": "negation"}
    return edited


# ------------- edit family 7: adversarial plausible -------------

def adversarial_plausible(reasoning: dict) -> Optional[dict]:
    """Replace primary object with the SECOND-most-referenced visible
    object. Unlike subject_swap (any distractor), this picks the
    visually most confusable alternative — should be harder to
    accidentally succeed via visual shortcut."""
    names = _iter_bbox_names(reasoning.get("bboxes"))
    if len(names) < 2:
        return None
    primary = _find_primary_object(reasoning)
    if primary is None:
        return None
    # Sort candidates by 'plausibility' — proximity words in scene.
    # Simple heuristic: pick the LAST bbox name (usually 'container' /
    # target location, most visually plausible confusion).
    distractors = [n for n in names if n.lower() != primary.lower()]
    if not distractors:
        return None
    distractor = distractors[-1]  # heuristic: last mentioned is often target
    edited = copy.deepcopy(reasoning)
    pat = re.compile(rf"\b{re.escape(primary)}\b", re.IGNORECASE)
    def _sub(x):
        return pat.sub(distractor, x) if isinstance(x, str) else x
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
    edited["__edit_meta__"] = {"family": "adversarial_plausible",
                                 "from": primary, "to": distractor}
    return edited


# ------------- edit family 8: selfsplice control (null) -------------

def selfsplice_control(reasoning: dict) -> Optional[dict]:
    """NULL CONTROL: apply an identity substitution (primary object ->
    same primary object). Any nonzero Δaction here indicates
    tokenization / decoding noise, not real causal effect. Must
    show ~0 to validate the metric (per VLADriveBench)."""
    edited = copy.deepcopy(reasoning)
    primary = _find_primary_object(reasoning)
    if primary is None:
        return None
    # Force at least one text field to be re-generated by identity replace.
    pat = re.compile(rf"\b{re.escape(primary)}\b", re.IGNORECASE)
    def _sub(x):
        return pat.sub(primary, x) if isinstance(x, str) else x
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
    edited["__edit_meta__"] = {"family": "selfsplice_control",
                                 "object": primary}
    return edited


# ------------- edit family 9: syntactic scramble -------------

def syntactic_scramble(reasoning: dict, seed: int = 0) -> Optional[dict]:
    """Shuffle word order within MOVE and SUBTASK. Grammar destroyed
    but content words preserved. Tests whether syntactic structure
    or just bag-of-content-words drives action."""
    import random
    rng = random.Random(seed)
    edited = copy.deepcopy(reasoning)
    changed = False
    def _shuffle_words(v):
        nonlocal changed
        if not isinstance(v, str) or len(v.split()) < 3:
            return v
        words = v.split()
        rng.shuffle(words)
        new_v = " ".join(words)
        if new_v != v:
            changed = True
        return new_v
    for k in ("subtask", "movement", "move"):
        if k in edited:
            edited[k] = _shuffle_words(edited[k])
    if not changed:
        return None
    edited["__edit_meta__"] = {"family": "syntactic_scramble", "seed": seed}
    return edited


# ------------- edit family 10: cross-task swap -------------

def cross_task_swap(reasoning: dict, alt_reasoning: Optional[dict] = None,
                     seed: int = 0) -> Optional[dict]:
    """EXTREME CONTROL: replace this sample's CoT with reasoning from
    an UNRELATED task. Should show near-100% Δaction (upper bound of
    causal effect). Requires an alt_reasoning to be passed in by
    the caller."""
    if alt_reasoning is None or not alt_reasoning:
        return None
    edited = copy.deepcopy(alt_reasoning)
    # Keep the ORIGINAL bboxes so VISIBLE OBJECTS still matches the image.
    # Only swap the reasoning tags (task/plan/subtask/movement/etc).
    edited["bboxes"] = reasoning.get("bboxes", {})
    edited["__edit_meta__"] = {"family": "cross_task_swap"}
    return edited


EDIT_FAMILIES = {
    "subject_swap":         subject_swap,
    "direction_flip":       direction_flip,
    "gripper_flip":         gripper_flip,
    "location_swap":        location_swap,
    "verb_swap":            verb_swap,
    "negation":             negation,
    "adversarial_plausible": adversarial_plausible,
    "selfsplice_control":   selfsplice_control,
    "syntactic_scramble":   syntactic_scramble,
    "cross_task_swap":      cross_task_swap,
    "paraphrase_null":      None,  # populated below
}


# ------------- edit family 11: paraphrase-preserving null -------------

# Meaning-preserving verb synonyms (unlike verb_swap which changes meaning).
# If a model is faithful to *semantic content*, F should ~= 0 on this family.
# If faithful only to surface tokens, F > 0 (model is sensitive to phrasing).
# This is a stronger null than selfsplice, which is trivially zero under
# greedy decoding (byte-identical input).
PARAPHRASE_SYNONYMS = [
    (r"\bmove\b",   "shift"),
    (r"\bgrasp\b",  "seize"),
    (r"\bpick\b",   "lift"),
    (r"\bplace\b",  "set"),
    (r"\bpush\b",   "press"),
    (r"\brelease\b", "let go of"),
    (r"\bturn\b",   "rotate"),
    (r"\bopen\b",   "unclose"),
    (r"\bclose\b",  "shut"),
]


def paraphrase_null(reasoning: dict) -> Optional[dict]:
    """Replace verbs in MOVE/PLAN/SUBTASK with meaning-preserving synonyms.
    A faithful model should show F ~= 0 under this edit (a true no-op semantic
    intervention), unlike selfsplice_control which is trivially F=0 by
    determinism. Distinguishes 'metric well-behaved' from 'tokenizer
    deterministic'.
    """
    edited = copy.deepcopy(reasoning)
    changed = False
    def _apply(v):
        nonlocal changed
        if not isinstance(v, str): return v
        new = v
        placeholders = {}
        for i, (pat, syn) in enumerate(PARAPHRASE_SYNONYMS):
            marker = f"__PARA_{i}__"
            placeholders[marker] = syn
            new2 = re.sub(pat, marker, new, flags=re.IGNORECASE)
            if new2 != new: changed = True
            new = new2
        for marker, syn in placeholders.items():
            new = new.replace(marker, syn)
        return new
    for k in ("task", "subtask", "subtask_reasoning", "subtask_reason",
                "movement", "move", "movement_reasoning", "move_reasoning"):
        if k in edited:
            edited[k] = _apply(edited[k])
    plan = edited.get("plan")
    if isinstance(plan, dict):
        edited["plan"] = {k: _apply(v) for k, v in plan.items()}
    elif isinstance(plan, str):
        edited["plan"] = _apply(plan)
    if not changed:
        return None
    edited["__edit_meta__"] = {"family": "paraphrase_null"}
    return edited


EDIT_FAMILIES["paraphrase_null"] = paraphrase_null

