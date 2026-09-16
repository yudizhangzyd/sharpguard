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
import random
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
]
# in/out removed (found by an ICLR reviewer running the released
# libero_reasonings.json through this table and reading real before/after
# text, not by inspecting the code): in this dataset's movement/
# movement_reasoning fields, "in"/"out" is a false friend for a spatial
# direction. Every distinct 7-word context around "in" or "out" across all
# 3475 step-0 reasoning records was enumerated (633 occurrences, 190
# distinct contexts, 0 in the terser movement/move field) and read by hand:
# 100% are either a fixed idiom ("in order to", "in front of", "in the
# way", "in a controlled manner", "slipping out of its grasp") or a static
# position/location reference of an object or the robot itself ("the robot
# is in the center", "positioned in the upper-left corner", "in the
# background") -- never the actual commanded movement, which this corpus
# always phrases with left/right/up/down/back/forward instead. Firing the
# substitution anyway produced edits like "in order to" -> "out order to"
# and "the robot is in the center" -> "the robot is out the center":
# ungrammatical, and not a semantic reversal of anything the sentence
# claims the robot will do. This was not a rare edge case: it was already
# present, unflagged, in 5 of the 40 pairs this project's own LLM-judge
# validation examined (judge_edit_families/judge_pairs.json), at a fluency
# rating (3.00) indistinguishable from the other 35 (3.14) -- the judge's
# rubric had no question aimed at this failure mode, which is why it
# shipped once already. The in/out entries were previously flagged in this
# same file as risky ("careful: 'in' is very common; scope to MOVE only"),
# but that comment was about over-triggering elsewhere in a CoT, not about
# whether the substitution is a real reversal once scoped correctly; scoping
# to MOVE (as already done) does not fix it, because MOVE REASONING's prose
# is exactly where these idioms and position references live. Every
# direction_flip number computed with the old 10-pair table is affected to
# an unknown degree and is being re-run against this corrected table rather
# than corrected in place, per this project's practice of never editing a
# scored number without re-deriving it.

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
]
# ("side of", "top of") removed: "on top of" is a fixed compound
# preposition, "side of" is not (it needs an article -- "on the side of" or
# "beside"), so this pair is not two grammatically interchangeable phrases.
# Every real occurrence (found the same way the direction_flip in/out bug
# was: running this table against the real dataset and reading the output,
# not inspecting the code) reads "on top of the cabinet" -> "on side of the
# cabinet" -- always missing the article, never a clean alternative. The
# single-word top/bottom pair below already covers the genuine cases
# ("move to the top drawer" -> "...bottom drawer" is clean), making this
# 2-word pair both broken and redundant.
#
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
    ("inside",  "outside"),
    ("outside", "inside"),
]
# ("near", "far") / ("far", "near") removed for the same reason: "near" and
# "far" are not symmetric in this corpus's phrasing -- "far" pairs with
# "away"/"from" ("positioned far away", "far from it"), "near" pairs with a
# bare noun ("not near the cabinet"). Swapping the adjective alone produces
# "near away" and "not far the cabinet" on every one of the (536-diff
# enumeration's) real occurrences checked, 0 clean. Same false-friend
# pattern, same fix: remove the pair rather than patch each collocation.

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


def direction_flip_no_geom(reasoning: dict) -> Optional[dict]:
    """direction_flip, plus blanking VISIBLE OBJECTS (bboxes) and GRIPPER
    POSITION -- the W1 reviewer confound check.

    direction_flip edits only the MOVE / MOVE REASONING text; every other
    rendered field, including the object bounding boxes and the gripper's
    own pixel position, is left exactly as it was. A policy that grounds its
    action in that unedited geometry rather than in the MOVE sentence would
    correctly ignore a flipped instruction that now contradicts the visible
    scene -- a real example, pulled from judge_edit_families/judge_pairs.json
    rather than hypothesized: gripper at x=109, target bbox centered at
    x=189.5 (clearly to its right, matching the ORIGINAL "move ... right"),
    edited MOVE says "move ... left" while the bbox/gripper numbers in the
    same prompt still show the mug on the right. Low F_dir in that case would
    be a geometry-grounded policy behaving correctly, not an unfaithful CoT.

    This variant asks the question a direction-only edit cannot: with the
    contradicting geometric anchor removed rather than left dangling, does
    F_dir change? Same applicability as direction_flip (returns None under
    the identical condition) so the two are a paired comparison on the exact
    same samples, not two differently-scoped families."""
    edited = direction_flip(reasoning)
    if edited is None:
        return None
    if "bboxes" in edited:
        edited["bboxes"] = {}
    for key in ("gripper", "gripper_position"):
        if key in edited:
            edited[key] = None
    edited["__edit_meta__"] = {"family": "direction_flip_no_geom"}
    return edited


# 224x224: the same fixed LIBERO input resolution bbox_jitter_null already
# documents and relies on.
_IMG_RES = 224


def _mirror_point(xy, axis: str) -> list:
    """Reflect a 2D pixel point about the image center along one axis."""
    x, y = xy[0], xy[1]
    if axis == "x":
        x = (_IMG_RES - 1) - x
    else:
        y = (_IMG_RES - 1) - y
    return [x, y]


def _direction_axes(text: str) -> set:
    """Which 2D-image mirror axes (a subset of {'x', 'y'}) the direction
    words present in `text` correspond to, checked the same word-boundary
    way direction_flip's own _replace_word_pairs matches them, on the
    ORIGINAL (pre-edit) text -- not re-derived from a diff of the edited
    text, so "did this pair fire" is asked identically to how
    direction_flip itself decides it. 'forward'/'back' has no axis on a 2D
    image and is reported as the sentinel 'z' so callers can refuse to
    mirror a scene that direction implies rather than mirror only what they
    can and leave the rest silently unedited."""
    axes = set()
    lower = text.lower()

    def _has(w: str) -> bool:
        return re.search(rf"\b{re.escape(w)}\b", lower) is not None

    if _has("left") or _has("right"):
        axes.add("x")
    if _has("up") or _has("down") or _has("above") or _has("below"):
        axes.add("y")
    if _has("forward") or _has("back"):
        axes.add("z")
    return axes


def direction_flip_geom_consistent(reasoning: dict) -> Optional[dict]:
    """direction_flip, plus mirroring VISIBLE OBJECTS bboxes and GRIPPER
    POSITION along the same image axis the flipped word(s) reverse, so the
    edited MOVE text and the rendered geometry AGREE on the new direction
    instead of contradicting it (direction_flip's own default) or removing
    the anchor (direction_flip_no_geom).

    The complement of direction_flip_no_geom's question. That check removes
    a CONTRADICTING geometric anchor and finds F_dir FALLS on ECoT-bridge
    (0.117 -> 0.090): evidence against "the model is correctly grounding in
    unedited geometry instead of an unfaithful CoT", since removing the
    contradiction should have RAISED F_dir under that story and did not.
    This variant asks the same question from the other side: if the
    geometry is edited to AGREE with the new direction instead of removed,
    does F_dir rise relative to plain direction_flip? If it does not, the
    model is not using the bbox/gripper geometry at all, in either
    direction, which is the more direct version of the claim the no-geom
    result already points at rather than a new, independent one.

    Scoped to left<->right and up<->down/above<->below only, the two pairs
    with an unambiguous mirror axis on the 224x224 LIBERO frame
    (bbox_jitter_null); forward<->back has no such axis. Returns None
    whenever a forward/back word is among the ones that fired on this
    sample: mirroring only the axes that DO have one would leave a
    partially-consistent scene, which answers a murkier question than
    either direction_flip's fully-contradicted geometry or this family's
    fully-consistent one, so this family is a strict subset of
    direction_flip's applicable samples rather than a same-N variant."""
    edited = direction_flip(reasoning)
    if edited is None:
        return None
    orig_text = " ".join(str(reasoning.get(k, "")) for k in
                          ("movement", "move", "movement_reasoning",
                           "move_reasoning", "move_reason")
                          if isinstance(reasoning.get(k), str))
    axes = _direction_axes(orig_text)
    if "z" in axes or not axes:
        return None

    bb = edited.get("bboxes")
    if isinstance(bb, dict):
        new_bb = {}
        for name, coords in bb.items():
            if (isinstance(coords, (list, tuple)) and len(coords) == 2
                    and all(isinstance(p, (list, tuple)) and len(p) == 2
                            for p in coords)):
                pts = [list(p) for p in coords]
                for ax in axes:
                    pts = [_mirror_point(p, ax) for p in pts]
                # Mirroring a corner pair can swap which one is the
                # top-left corner on the mirrored axis; re-sort so
                # [x1,y1]/[x2,y2] keeps that convention, matching every
                # other bbox in this release.
                xs = sorted(p[0] for p in pts)
                ys = sorted(p[1] for p in pts)
                new_bb[name] = [[xs[0], ys[0]], [xs[1], ys[1]]]
            else:
                new_bb[name] = coords
        edited["bboxes"] = new_bb
    for key in ("gripper", "gripper_position"):
        v = edited.get(key)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            p = list(v)
            for ax in axes:
                p = _mirror_point(p, ax)
            edited[key] = p

    edited["__edit_meta__"] = {"family": "direction_flip_geom_consistent",
                                "mirrored_axes": sorted(axes)}
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

_ON_TOP_OF_RE = re.compile(r"\bon top of\b", re.IGNORECASE)


def location_swap(reasoning: dict) -> Optional[dict]:
    """Swap 2-word location phrases + single-word spatial adjectives in
    PLAN / SUBTASK / TASK (skip MOVE which direction_flip covers). Uses
    literal string match for phrases and regex \\b word-boundary match
    for single words.

    "on top of" is masked before the single-word top/bottom substitution
    and restored after: it is a fixed compound preposition ("on top of the
    cabinet" = "above/atop the cabinet"), and the bare noun-phrase reading
    the top/bottom pair is built for ("the top of the cabinet", "move to
    the bottom of the cabinet" -- both grammatical, confirmed clean in the
    same enumeration below) is a different construction that happens to
    share the word "top". Without masking, "on top of X" -> "on bottom of
    X", missing an article ("on THE bottom of") the same way "on top of"
    itself has none -- not a clean reversal, just a different broken
    preposition. Found and fixed the same way as direction_flip's in/out
    bug: enumerating every distinct diff this table produces on the real
    dataset (536 distinct diffs) and reading the output, not the code.
    "near"/"far" is not masked -- it is removed from LOCATION_WORD_PAIRS
    entirely, since 0 of its real occurrences were clean either way."""
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
        # Mask "on top of" before the single-word pass so bare top/bottom
        # (the genuine, symmetric usage) can still be swapped freely.
        on_top_ofs = [m.group(0) for m in _ON_TOP_OF_RE.finditer(new)]
        new = _ON_TOP_OF_RE.sub("\x00ONTOPOF\x00", new)
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
        for orig in on_top_ofs:
            new = new.replace("\x00ONTOPOF\x00", orig, 1)
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
    "direction_flip_no_geom": direction_flip_no_geom,
    "direction_flip_geom_consistent": direction_flip_geom_consistent,
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


# ------------- edit family 14: length-exact paraphrase null (4th floor) -----
# paraphrase_null is meaning-preserving by construction but not length-exact:
# 8 of its 9 substitution patterns are single-word-to-single-word, but
# "release" -> "let go of" (1 token -> 3) is common enough in this
# pick-and-place domain (most tasks end with a release) that it alone is
# almost certainly why 38/40 judged heads changed word count. That leaves
# two floors (paraphrase_null, syntactic_scramble) that differ on BOTH axes
# the paper flags as confounds -- length AND judged fluency (4.92 vs 4.00) --
# so neither alone can be blamed. This swaps just that one pattern for a
# single-word synonym ("free"), holding every other substitution identical,
# to get a null that is meaning-preserving AND length-exact AND (being
# ordinary synonym substitution, not word-order scrambling) presumptively
# high-fluency like the original -- decoupling "changes length" from
# "changes specific words" using one surgical change, not a new mechanism.
PARAPHRASE_SYNONYMS_LENEXACT = [
    (pat, ("free" if syn == "let go of" else syn))
    for pat, syn in PARAPHRASE_SYNONYMS
]


def paraphrase_null_lenexact(reasoning: dict) -> Optional[dict]:
    """Same as paraphrase_null, with "release"->"free" instead of "let go
    of" so every substitution is single-word-to-single-word. See module
    comment above PARAPHRASE_SYNONYMS_LENEXACT for why this family exists.
    """
    edited = copy.deepcopy(reasoning)
    changed = False
    def _apply(v):
        nonlocal changed
        if not isinstance(v, str): return v
        new = v
        placeholders = {}
        for i, (pat, syn) in enumerate(PARAPHRASE_SYNONYMS_LENEXACT):
            marker = f"__PARALE_{i}__"
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
    edited["__edit_meta__"] = {"family": "paraphrase_null_lenexact"}
    return edited


EDIT_FAMILIES["paraphrase_null_lenexact"] = paraphrase_null_lenexact


# ------------- edit family 12: bbox-jitter null (second calibration floor) ---
# Perturbs every VISIBLE OBJECTS bbox coordinate by +/- 1 pixel.  At 224x224
# input resolution a 1-px box shift is below the visual-token granularity, so
# the *referent* of every reasoning tag is unchanged: this is a semantics-
# preserving distractor.  Together with paraphrase_null it brackets the
# "surface-form sensitivity floor" of the Faithfulness Score from a second,
# non-lexical direction (digits instead of verbs).
def bbox_jitter_null(reasoning: dict, seed: int = 0) -> Optional[dict]:
    rng = random.Random(seed + 991)
    edited = copy.deepcopy(reasoning)
    bb = edited.get("bboxes")
    changed = False

    def _jit(v):
        nonlocal changed
        if isinstance(v, (list, tuple)):
            out = []
            for x in v:
                if isinstance(x, (list, tuple)):
                    out.append(_jit(x))
                elif isinstance(x, int) and not isinstance(x, bool):
                    changed = True
                    out.append(max(0, x + rng.choice((-1, 1))))
                elif isinstance(x, float):
                    changed = True
                    out.append(x + rng.choice((-1.0, 1.0)))
                else:
                    out.append(x)
            return out
        return v

    if isinstance(bb, dict):
        edited["bboxes"] = {k: _jit(v) for k, v in bb.items()}
    elif isinstance(bb, list):
        edited["bboxes"] = [_jit(v) for v in bb]
    else:
        return None
    if not changed:
        return None
    edited["__edit_meta__"] = {"family": "bbox_jitter_null", "delta_px": 1}
    return edited


EDIT_FAMILIES["bbox_jitter_null"] = bbox_jitter_null


# ------------- edit family 13: out-of-CoT decode-sensitivity calibrator -----
# Substitutes K random word tokens in the *user instruction* -- text that sits
# OUTSIDE the CoT segment entirely.  F on this family measures how much the
# action decode moves under an arbitrary prompt perturbation of comparable
# token budget, with the CoT held byte-identical.  It is the per-model
# calibrator that separates CoT-specific sensitivity from global prompt
# sensitivity (needed to interpret F2).
_RANDOM_SUB_VOCAB = [
    "quartz", "ledger", "marimba", "trellis", "opaque", "kelvin",
    "sundial", "gravel", "lantern", "porcelain", "cipher", "meadow",
]


def instr_random_sub(reasoning: dict, seed: int = 0, n_tokens: int = 5) -> Optional[dict]:
    """The reasoning dict is returned UNCHANGED; the harness reads
    ``__edit_meta__['instr_random_sub']`` and rewrites the instruction instead.
    """
    edited = copy.deepcopy(reasoning)
    edited["__edit_meta__"] = {
        "family": "instr_random_sub",
        "instr_random_sub": int(n_tokens),
        "seed": int(seed),
        "target_segment": "instruction (outside CoT)",
    }
    return edited


EDIT_FAMILIES["instr_random_sub"] = instr_random_sub


def apply_instr_random_sub(instruction: str, seed: int = 0, n_tokens: int = 5) -> str:
    """Replace up to ``n_tokens`` words of ``instruction`` with random nonce
    words.  Deterministic given ``seed``."""
    rng = random.Random(seed + 4242)
    words = instruction.split()
    if not words:
        return instruction
    k = min(n_tokens, len(words))
    idxs = rng.sample(range(len(words)), k)
    for i in idxs:
        words[i] = rng.choice(_RANDOM_SUB_VOCAB)
    return " ".join(words)

