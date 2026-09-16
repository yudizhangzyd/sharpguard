#!/usr/bin/env python3
"""Is F_dir's -0.5 cosine threshold load-bearing, or would a looser/tighter
cutoff change which configurations clear their own null?

fdir_null.py reports F_dir and the null-ceiling clearance at a single fixed
threshold (cos < -0.5). This script re-derives the identical per_config
table at two additional thresholds, -0.25 and -0.75, from the same released
per-sample action vectors and the same null families, to check whether the
qualitative pattern (which configurations clear) is an artifact of that one
cutoff choice or holds across a range of comparably strict criteria.

Reuses fdir_null.py's own data loading (OURS/DEEPTHINK configs, load_ours,
the checkpoint's own de-quantization grid) so this is the same underlying
comparison at three thresholds, not a second measurement.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derive_metrics import _to_checkpoint_grid  # noqa: E402
from fdir_null import (  # noqa: E402
    DEEPTHINK, NULL_FAMILIES, OURS, TREATMENT, cos, load_ours,
)

THRESHOLDS = [-0.25, -0.5, -0.75]


def f_dir_at(records, family, threshold):
    kept = [r for r in records if r.get("family") == family and not r.get("skipped")]
    cs = [cos(_to_checkpoint_grid(r["a_orig"])[:3], _to_checkpoint_grid(r["a_edit"])[:3])
          for r in kept]
    cs = [c for c in cs if c is not None]
    if not cs:
        return None, 0
    return sum(1 for c in cs if c < threshold) / len(cs), len(cs)


def main():
    configs = [(c, load_ours(c)) for c in OURS]
    for name, path in DEEPTHINK.items():
        configs.append((name, json.load(open(path))["per_sample_edit"]))

    by_threshold = {}
    for threshold in THRESHOLDS:
        out, failures = [], []
        for name, recs in configs:
            treat, n_treat = f_dir_at(recs, TREATMENT, threshold)
            nulls = {}
            for f in NULL_FAMILIES:
                v, n = f_dir_at(recs, f, threshold)
                if v is not None:
                    nulls[f] = {"F_dir": v, "n": n}
            if treat is None or not nulls:
                failures.append(name)
                continue
            ceiling = max(v["F_dir"] for v in nulls.values())
            argmax = max(nulls, key=lambda k: nulls[k]["F_dir"])
            out.append({
                "config": name,
                "treatment": {"family": TREATMENT, "F_dir": treat, "n": n_treat},
                "null_ceiling": ceiling,
                "null_ceiling_family": argmax,
                "clears_null": treat > ceiling,
            })
        if failures:
            print(f"[fdir-sweep] FAILED to score at {threshold}: {failures}",
                  file=sys.stderr)
            return 1
        by_threshold[threshold] = out

    print(f"{'config':18s}" + "".join(f"{f'clears@{t}':>14s}" for t in THRESHOLDS))
    configs_order = [r["config"] for r in by_threshold[THRESHOLDS[0]]]
    for name in configs_order:
        row = ""
        for t in THRESHOLDS:
            rec = next(r for r in by_threshold[t] if r["config"] == name)
            row += f"{('YES' if rec['clears_null'] else 'no'):>14s}"
        print(f"{name:18s}{row}")

    clears_by_threshold = {
        t: sum(r["clears_null"] for r in rows) for t, rows in by_threshold.items()
    }
    for t, n_clear in clears_by_threshold.items():
        print(f"\nthreshold {t}: {n_clear}/{len(configs_order)} configurations "
              f"clear their own null ceiling")

    summary = {
        "thresholds": THRESHOLDS,
        "treatment_family": TREATMENT,
        "null_families": NULL_FAMILIES,
        "n_configs": len(configs_order),
        "n_clearing_null_by_threshold": {str(t): n for t, n in clears_by_threshold.items()},
        "which_configs_clear_by_threshold": {
            str(t): sorted(r["config"] for r in rows if r["clears_null"])
            for t, rows in by_threshold.items()
        },
        "per_threshold": {str(t): rows for t, rows in by_threshold.items()},
    }
    dest = "results_v2/canonical_runs/fdir_threshold_sweep/fdir_threshold_sweep.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(summary, open(dest, "w"), indent=2)
    print(f"[fdir-sweep] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
