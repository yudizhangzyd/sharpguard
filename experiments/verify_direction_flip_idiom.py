"""Audit: does the shared DIRECTION_PAIRS in/out substitution ever produce a
non-idiomatic edit inside the *main cohort's* direction_flip fields?

Motivation: this session found that DT-RL's DT-RL-only direction_flip_text
(free-text CoT, experiments/cotfaith_deepthink_selfgen_edit.py) collided the
up/down pair with the phrasal verb "pick up", corrupting 93.5% of its edits
("Pick up the book" -> "Pick down the book"). That function reuses the same
DIRECTION_PAIRS / _replace_word_pairs utilities the MAIN cohort's structured
direction_flip() (sharpguard/attacks/cot_edit.py) has used for every
direction_flip number in the paper. DIRECTION_PAIRS already carries a
same-file comment flagging the in/out pair as risky ("in" is very common;
scope to MOVE only) -- but that comment is about over-triggering elsewhere
in the text, not about whether the substitution reads as a real spatial
reversal once it fires. This checks the latter, against the real dataset
(Embodied-CoT/embodied_features_and_demos_libero), not a hypothetical.

Only a single JSON file is fetched (via file_with_retry, not
snapshot_with_retry): no TFDS video snapshot, no model, no GPU.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sharpguard.hf_retry import file_with_retry  # noqa: E402
from sharpguard.attacks.cot_edit import (  # noqa: E402
    DIRECTION_PAIRS, direction_flip,
)

MOVE_KEYS = ("movement", "move", "movement_reasoning", "move_reasoning",
             "move_reason")
REPO_ID = "Embodied-CoT/embodied_features_and_demos_libero"
FILENAME = "libero_reasonings.json"


def which_pairs_fired(before: str):
    hits = []
    for src, dst in DIRECTION_PAIRS:
        if re.search(rf"\b{re.escape(src)}\b", before):
            hits.append((src, dst))
    return hits


def main():
    path = file_with_retry(repo_id=REPO_ID, filename=FILENAME,
                            repo_type="dataset")
    rdata = json.loads(Path(path).read_text())

    total = 0
    eligible = 0
    pair_hits = Counter()
    examples = {}

    for _file_key, demos in rdata.items():
        if not isinstance(demos, dict):
            continue
        for _demo_id, steps in demos.items():
            if not isinstance(steps, dict):
                continue
            gt = steps.get("0")
            if not gt:
                continue
            total += 1
            edited = direction_flip(gt)
            if edited is None:
                continue
            eligible += 1
            for key in MOVE_KEYS:
                before = gt.get(key)
                after = edited.get(key)
                if isinstance(before, str) and before and after != before:
                    for pair in which_pairs_fired(before):
                        pair_hits[pair] += 1
                        examples.setdefault(pair, [])
                        if len(examples[pair]) < 8:
                            examples[pair].append({"before": before,
                                                     "after": after,
                                                     "key": key})

    report = {
        "repo_id": REPO_ID,
        "total_step0_records": total,
        "eligible_for_direction_flip": eligible,
        "pair_hit_counts": {f"{s}->{d}": c for (s, d), c in pair_hits.items()},
        "examples_by_pair": {f"{s}->{d}": v for (s, d), v in examples.items()},
    }
    print(json.dumps(report, indent=2))

    out = Path(__import__("os").environ.get("BOLT_ARTIFACT_DIR", "./artifacts")) \
        / "direction-flip-idiom-check"
    out.mkdir(parents=True, exist_ok=True)
    (out / "direction_flip_idiom_check.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
