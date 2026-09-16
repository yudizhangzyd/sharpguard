#!/usr/bin/env python3
"""Splice a direction_flip-only re-run (under the corrected DIRECTION_PAIRS,
in/out removed) into an existing 13-family canonical report, replacing only
the direction_flip family's per_sample records and aggregate -- every other
family's data is untouched.

The new report's OWN aggregate.direction_flip is used verbatim rather than
recomputed by hand: it was written by the exact same scoring code
(cotfaith_edit.py / cotfaith_deepthink.py) that produced the original, so it
is already in that report type's correct native schema -- flat scalars plus
two DeepThinkVLA-only chunk fields (faithful_rate_chunk, delta_linf_chunk_
mean) for the deepthink schema, nested {mean,std,median} dicts for the
cotfaith_edit schema. An earlier version of this script hand-rolled the
recompute and silently dropped both DeepThinkVLA chunk fields (derive_
metrics.py's chunk-vs-step-0 comparison went to None) -- caught by diffing
derived_metrics.json before vs after, not by inspecting this script.

Verifies a_orig matches between old and new before trusting the splice: the
same model on the same sample should decode the same greedy action
deterministically, so any mismatch means the two reports are not actually
comparable (wrong seed, wrong checkpoint, wrong sample order) and the script
refuses to merge rather than silently combine incompatible runs.

Usage:
    python3 scripts/apply_direction_flip_refix.py <old_canonical.json> <new_direction_flip_only.json> [--write]

Without --write, prints a dry-run diff summary only.
"""
import json
import sys
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text())


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    old_path, new_path = sys.argv[1], sys.argv[2]
    do_write = "--write" in sys.argv

    old = load(old_path)
    new = load(new_path)

    # Two schemas in this release: {aggregate, per_sample} for cotfaith_edit.py
    # reports, {edit_aggregate, per_sample_edit} for cotfaith_deepthink.py's
    # (which also carries an unrelated attention/rvis half untouched here).
    agg_key = "aggregate" if "aggregate" in old else "edit_aggregate"
    ps_key = "per_sample" if "per_sample" in old else "per_sample_edit"
    new_ps_key = "per_sample" if "per_sample" in new else "per_sample_edit"

    old_seed = old.get("seed", 0)
    new_dir_records = [r for r in new[new_ps_key]
                        if r.get("family") == "direction_flip"
                        and r.get("seed", 0) == old_seed]
    if not new_dir_records:
        print(f"FAIL: no direction_flip records at seed={old_seed} in {new_path}")
        return 1

    old_dir_by_sample = {r["sample"]: r for r in old[ps_key]
                          if r.get("family") == "direction_flip"}
    new_dir_by_sample = {r["sample"]: r for r in new_dir_records}

    if set(old_dir_by_sample) != set(new_dir_by_sample):
        missing_new = set(old_dir_by_sample) - set(new_dir_by_sample)
        missing_old = set(new_dir_by_sample) - set(old_dir_by_sample)
        print(f"FAIL: sample-index mismatch. In old but not new: "
              f"{sorted(missing_new)[:10]}. In new but not old: "
              f"{sorted(missing_old)[:10]}")
        return 1

    mismatches = []
    for s, old_r in old_dir_by_sample.items():
        new_r = new_dir_by_sample[s]
        old_a = old_r.get("a_orig")
        new_a = new_r.get("a_orig")
        if old_a is None or new_a is None:
            continue
        if any(abs(a - b) > 1e-9 for a, b in zip(old_a, new_a)):
            mismatches.append((s, old_a, new_a))
    if mismatches:
        print(f"FAIL: a_orig mismatch on {len(mismatches)} samples -- old "
              f"and new reports are not the same model/checkpoint/sample "
              f"draw, refusing to merge. First: sample={mismatches[0][0]} "
              f"old={mismatches[0][1]} new={mismatches[0][2]}")
        return 1

    old_faithful = sum(1 for r in old_dir_by_sample.values()
                        if not r.get("skipped") and r.get("faithful"))
    old_n = sum(1 for r in old_dir_by_sample.values() if not r.get("skipped"))
    new_faithful = sum(1 for r in new_dir_by_sample.values()
                        if not r.get("skipped") and r.get("faithful"))
    new_n = sum(1 for r in new_dir_by_sample.values() if not r.get("skipped"))
    print(f"a_orig consistency: OK ({len(old_dir_by_sample)} samples, all match)")
    print(f"direction_flip n applied: old={old_n} new={new_n}")
    print(f"direction_flip faithful_rate: old={old_faithful/old_n:.4f} "
          f"new={new_faithful/new_n:.4f}" if old_n and new_n else "n/a")

    if not do_write:
        print("\n(dry run -- pass --write to apply)")
        return 0

    other_records = [r for r in old[ps_key] if r.get("family") != "direction_flip"]
    new_per_sample = other_records + [new_dir_by_sample[s] for s in sorted(new_dir_by_sample)]
    old[ps_key] = new_per_sample
    new_agg_key = "aggregate" if "aggregate" in new else "edit_aggregate"
    old[agg_key]["direction_flip"] = new[new_agg_key]["direction_flip"]
    Path(old_path).write_text(json.dumps(old, indent=2))
    print(f"\nWrote {old_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
