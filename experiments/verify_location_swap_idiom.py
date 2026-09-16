"""Same-class check as verify_direction_flip_idiom.py, for location_swap
instead of direction_flip: does LOCATION_PAIRS/LOCATION_WORD_PAIRS (applied
to plan/subtask/subtask_reasoning/task, not movement_reasoning) produce
idiom-collision artifacts the way DIRECTION_PAIRS' in/out did? Motivated by
LOCATION_PAIRS containing ("front of", "back of") -- the same "in front of"
idiom direction_flip's in/out pair broke, just phrased as a 2-word literal
match instead of a bare "in". Calls the real location_swap() on every
eligible record and prints every changed field's before/after for manual
read, rather than assuming the fix needed for one family generalizes to
this structurally different one.

One HF single-file fetch (libero_reasonings.json): no TFDS video snapshot,
no model, no GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sharpguard.hf_retry import file_with_retry  # noqa: E402
from sharpguard.attacks.cot_edit import location_swap  # noqa: E402

REPO_ID = "Embodied-CoT/embodied_features_and_demos_libero"
FILENAME = "libero_reasonings.json"
FIELDS = ("plan", "subtask", "subtask_reasoning", "subtask_reason", "task")


def diffs(before: dict, after: dict):
    out = []
    for k in FIELDS:
        bv, av = before.get(k), after.get(k)
        if isinstance(bv, dict) and isinstance(av, dict):
            for kk in bv:
                if bv.get(kk) != av.get(kk):
                    out.append((f"{k}.{kk}", bv.get(kk), av.get(kk)))
        elif bv != av:
            out.append((k, bv, av))
    return out


def main():
    path = file_with_retry(repo_id=REPO_ID, filename=FILENAME,
                            repo_type="dataset")
    rdata = json.loads(Path(path).read_text())

    total = 0
    eligible = 0
    all_diffs = []

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
            edited = location_swap(gt)
            if edited is None:
                continue
            eligible += 1
            all_diffs.extend(diffs(gt, edited))

    print(f"total step0 records: {total}")
    print(f"eligible for location_swap: {eligible}")
    print(f"total changed (field, before, after) triples: {len(all_diffs)}")
    print()
    # Dedup identical (field, before, after) triples, sorted by frequency.
    from collections import Counter
    keyed = Counter((f, b, a) for f, b, a in all_diffs)
    print(f"{len(keyed)} distinct (field, before, after) triples "
          f"(sorted by frequency):\n")
    for (f, b, a), c in keyed.most_common():
        print(f"[{c:4d}x] ({f})")
        print(f"         before: {b}")
        print(f"         after : {a}")

    out = Path(__import__("os").environ.get("BOLT_ARTIFACT_DIR", "./artifacts")) \
        / "location-swap-idiom-check"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "total_step0_records": total,
        "eligible_for_location_swap": eligible,
        "n_distinct_diffs": len(keyed),
        "diffs": [{"field": f, "before": b, "after": a, "count": c}
                  for (f, b, a), c in keyed.most_common()],
    }
    (out / "location_swap_idiom_check.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
