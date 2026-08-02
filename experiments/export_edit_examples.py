"""Export full-text (original, edited) CoT pairs for the task-examples figure.

The figure that shows a reader what an "edit family" actually IS has to show
real text. The released `judge_edit_families/judge_pairs.json` already holds
real pairs, but it stores only the first 400 characters of each side, and the
ECoT trace puts TASK/PLAN/VISIBLE OBJECTS ahead of the MOVE tag -- so the head
is cut off before the span that `direction_flip`, `negation` and
`syntactic_scramble` rewrite. Those three families are byte-identical across
all 40 of their released heads. A figure built from the heads would show three
of eleven families as "no change", which is false.

So this re-derives the pairs at full length. It is not a new measurement and it
must not become one: it re-runs the SAME generators on the SAME corpus with the
SAME seed and the SAME reservoir draw as scripts the judge run used, and then
asserts the result against what was released --- every regenerated pair must
match the stored 400-character head byte for byte. If a generator has drifted
since the judge run, that assertion fails and this writes nothing, rather than
quietly publishing a figure whose text no longer corresponds to the judged
verdicts printed beside it in the caption.

Runs on CPU: pair construction is pure string manipulation, no model is loaded.

Usage:
    python3 experiments/export_edit_examples.py --out results_v2/canonical_runs/edit_examples
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import the judge experiment itself rather than copying its pair builder.
# A second implementation of build_pairs is exactly how the figure and the
# judged verdicts would drift apart, and the head-match assertion below would
# then be checking a copy against itself.
_spec = importlib.util.spec_from_file_location(
    "_cotfaith_judge_edits", Path(__file__).resolve().parent / "cotfaith_judge_edits.py")
_judge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_judge)

build_pairs = _judge.build_pairs
load_samples = _judge.load_samples
scored_file_bases = _judge.scored_file_bases

# The arguments the released judge run used, from
# results_v2/canonical_runs/judge_edit_families/args.json. Hardcoding them
# here would let the two drift, so they are read from that file at run time
# and only the path is fixed.
ARGS_JSON = _ROOT / "results_v2" / "canonical_runs" / "judge_edit_families" / "args.json"
PAIRS_JSON = _ROOT / "results_v2" / "canonical_runs" / "judge_edit_families" / "judge_pairs.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--head-chars", type=int, default=400,
                    help="length of the released head, i.e. how much of each "
                         "regenerated pair can be verified against the release")
    args = ap.parse_args()

    ja = json.loads(ARGS_JSON.read_text())
    released = json.loads(PAIRS_JSON.read_text())
    print(f"[export] judge run: seed={ja['seed']} n_samples={ja['n_samples']} "
          f"corpus={ja['reasoning_repo']}/{ja['reasoning_file']}")

    allowed = None
    if ja.get("file_base_from"):
        allowed = scored_file_bases(
            [q.strip() for q in ja["file_base_from"].split(",") if q.strip()])
        print(f"[export] {len(allowed)} scored demo files named in canonical runs")

    families = [f.strip() for f in ja["families"].split(",") if f.strip()]
    samples, reservoir = load_samples(ja["reasoning_repo"], ja["reasoning_file"],
                                      ja["n_samples"], ja["seed"], allowed)
    pairs = build_pairs(samples, reservoir, families, ja["seed"])
    print(f"[export] rebuilt {len(pairs)} pairs "
          f"({sum(1 for q in pairs if q.get('skipped'))} skipped)")

    # --- the assertion this script exists for -------------------------------
    # Match on (sample, family), the key the judge recorded, and require the
    # regenerated text to reproduce the released head exactly. A mismatch means
    # a generator, the corpus, or the sampling changed since the judge ran, so
    # the full text no longer belongs beside that pair's verdict.
    by_key = {(r["sample"], r["family"]): r for r in released}
    checked = mismatched = 0
    examples = []
    for q in pairs:
        if q.get("skipped"):
            continue
        rel = by_key.get((q["sample"], q["family"]))
        if rel is None:
            continue          # judged set was time-budget truncated; fine
        checked += 1
        if (q["a"][:args.head_chars] != rel["a_head"]
                or q["b"][:args.head_chars] != rel["b_head"]):
            mismatched += 1
            continue
        examples.append({
            "sample": q["sample"], "family": q["family"],
            "file_base": q.get("file_base"), "demo": q.get("demo"),
            "step": q.get("step"), "edit_meta": q.get("edit_meta", {}),
            "cot_orig": q["a"], "cot_edited": q["b"],
            "verdict": rel.get("verdict"),
        })

    print(f"[export] verified {checked - mismatched}/{checked} pairs against "
          f"the released {args.head_chars}-char heads")
    if mismatched:
        print(f"[export] FAIL: {mismatched} regenerated pair(s) do not "
              f"reproduce the released head. The generators or the corpus have "
              f"changed since the judge run, so the full text cannot be shown "
              f"beside those verdicts. Writing nothing.", file=sys.stderr)
        return 1
    if not examples:
        print("[export] FAIL: no pairs verified", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "edit_examples.json").write_text(json.dumps({
        "source_judge_run": "results_v2/canonical_runs/judge_edit_families",
        "corpus": f"{ja['reasoning_repo']}/{ja['reasoning_file']}",
        "seed": ja["seed"],
        "n_pairs": len(examples),
        "head_chars_verified": args.head_chars,
        "provenance": (
            "Regenerated at full length by re-running the judge run's own "
            "build_pairs() at its recorded seed and corpus. Every pair here "
            "reproduces the released 400-character head byte for byte; the "
            "script writes nothing if any pair does not. The CoT text is the "
            "generator's output, not a reconstruction."),
        "examples": examples,
    }, indent=2))
    print(f"[export] wrote {out/'edit_examples.json'}: {len(examples)} pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
