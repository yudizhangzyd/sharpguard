"""Enumerate every distinct movement/movement_reasoning sentence that
contains "in" or "out" as a whole word, across the FULL reasoning dataset
(not just the direction_flip-eligible subset) -- the same enumeration
discipline the DT-RL phrasal-verb fix used ("verified by enumerating every
distinct movement sentence in the released reports") before designing a fix,
applied here to a harder problem: unlike "pick up", in/out collides with
several different non-spatial idioms, not one fixed phrase.

Only a single JSON file is fetched (via file_with_retry): no TFDS video
snapshot, no model, no GPU.
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

MOVE_KEYS = ("movement", "move", "movement_reasoning", "move_reasoning",
             "move_reason")
REPO_ID = "Embodied-CoT/embodied_features_and_demos_libero"
FILENAME = "libero_reasonings.json"


def main():
    path = file_with_retry(repo_id=REPO_ID, filename=FILENAME,
                            repo_type="dataset")
    rdata = json.loads(Path(path).read_text())

    total = 0
    by_key = Counter()
    contexts = {}   # normalized 7-word window -> {"example": raw sentence, "count": n}

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
            for key in MOVE_KEYS:
                v = gt.get(key)
                if not (isinstance(v, str) and v):
                    continue
                words = v.split()
                lower_words = [w.lower() for w in words]
                for i, w in enumerate(lower_words):
                    core = re.sub(r"[^a-z]", "", w)
                    if core not in ("in", "out"):
                        continue
                    by_key[key] += 1
                    lo, hi = max(0, i - 3), min(len(words), i + 4)
                    window = " ".join(lower_words[lo:i] + [f"[{core.upper()}]"]
                                       + lower_words[i + 1:hi])
                    window = re.sub(r"\d+", "#", window)
                    contexts.setdefault(window, {"example": v, "count": 0})
                    contexts[window]["count"] += 1

    report = {
        "total_step0_records": total,
        "in_or_out_occurrences_by_key": dict(by_key),
        "n_distinct_contexts": len(contexts),
        "contexts": sorted(
            [{"window": k, **v} for k, v in contexts.items()],
            key=lambda d: -d["count"]),
    }
    print(json.dumps({k: v for k, v in report.items() if k != "contexts"},
                      indent=2))
    print(f"\n{len(contexts)} distinct 7-word contexts around in/out "
          f"(sorted by frequency):\n")
    for d in report["contexts"]:
        print(f"[{d['count']:4d}x] ...{d['window']}...")
        print(f"         e.g. {d['example']}")

    out = Path(__import__("os").environ.get("BOLT_ARTIFACT_DIR", "./artifacts")) \
        / "direction-flip-idiom-enumerate"
    out.mkdir(parents=True, exist_ok=True)
    (out / "in_out_sentences.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
