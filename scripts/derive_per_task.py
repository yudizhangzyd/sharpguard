#!/usr/bin/env python3
"""Per-task decomposition of the paraphrase-floor result.

Every $\\bar F$ in this paper is an average over samples, and the samples are
drawn from LIBERO-90 tasks. An average can hide its shape two ways: the floor
result could come from a handful of tasks on which the paraphrase edit happens
to move the action, or $F$ could be so task-dependent that a per-model number
is not a property of the model at all. Both are checkable with no new inference,
because every released edit record carries `file_base`, which names the LIBERO
episode file the observation came from.

This script groups the scored records by task, recomputes $F$ within each task
over the 7 non-control families and over `paraphrase_null`, and reports the SIGN
of the per-task difference. The sign count is the statistic that survives the
small per-task N: with a handful of samples per (task, family) the per-task
floor is coarse, but "below its own floor on k of n tasks" is a paired sign test
that does not need the per-task means to be precise. It is reported again over
only the tasks whose floor has at least 3 samples, so an artefact of
single-sample floors would show up as the effect dying under restriction.

Averaging convention matches scripts/derive_metrics.py: F over a set of families
is the macro average of per-family rates, not the record-level micro average.
The provenance check below re-derives each model's published F_bar_mag and
paraphrase floor from the same records and fails if they disagree, so this
script cannot quietly read the release differently from the pipeline that
produced the leaderboard.

Writes results_v2/canonical_runs/per_task_decomposition/per_task.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
CAN = ROOT / "results_v2" / "canonical_runs"
DERIVED = ROOT / "results_v2" / "derived_metrics.json"
OUT = CAN / "per_task_decomposition"

# The 7 non-control families: the 10 non-null families minus the 3 Tier-0
# controls. Restated by value rather than imported so that a divergence from
# derive_metrics.py surfaces as a failed provenance check below, not as a
# silent agreement.
NON_CONTROL = ["direction_flip", "gripper_flip", "verb_swap", "negation",
               "subject_swap", "location_swap", "adversarial_plausible"]
FLOOR = "paraphrase_null"
MIN_FLOOR_RESTRICTED = 3

# The 3-seed runs behind the leaderboard, keyed by the label derive_metrics.py
# uses. ECoT-bridge's seed runs carry 11 families and the "ours" rows carry 13;
# both contain the floor and all 7 non-control families, which is all this
# decomposition reads.
RUNS = {
    "ours-r8":      "ours_lora-r8_edit_13family_seed%d.json",
    "ours-r16":     "ours_lora-r16_edit_13family_seed%d.json",
    "ours-r32":     "ours_lora-r32_edit_13family_seed%d.json",
    "ours-r64":     "ours_lora-r64_edit_13family_seed%d.json",
    "ours-no-cot":  "ours_no-cot_edit_13family_seed%d.json",
    "ours-data50A": "ours_data-50A_edit_13family_seed%d.json",
    "ours-data50B": "ours_data-50B_edit_13family_seed%d.json",
    "ecot-bridge":  "ecot_bridge_edit_seed%d.json",
}
SEEDS = (0, 1, 2)
TOL = 1e-9


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p at p0=0.5, no scipy dependency."""
    if n == 0:
        return 1.0
    def pmf(i: int) -> float:
        return math.comb(n, i) * 0.5 ** n
    obs = pmf(k)
    # Every outcome at most as likely as the observed one. The slack absorbs the
    # float error that makes the mirror-image term compare as strictly larger.
    return min(1.0, sum(pmf(i) for i in range(n + 1)
                        if pmf(i) <= obs * (1 + 1e-12)))


def quantile(xs: list[float], q: float) -> float:
    """Linear-interpolation quantile, so the audit needs no numpy."""
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    i = q * (len(s) - 1)
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (i - lo) * (s[hi] - s[lo])


def rate(recs: list[dict]) -> float | None:
    return (sum(1 for r in recs if r["faithful"]) / len(recs)) if recs else None


def macro(rates: list[float | None]) -> float | None:
    got = [r for r in rates if r is not None]
    return mean(got) if got else None


def load_seeds(pat: str) -> list[list[dict]]:
    out = []
    for s in SEEDS:
        p = CAN / (pat % s)
        if not p.exists():
            raise SystemExit(f"[per-task] missing: {p}")
        out.append([r for r in json.loads(p.read_text())["per_sample"]
                    if not r.get("skipped")])
    return out


def provenance(seeds: list[list[dict]], label: str, pub: dict) -> dict:
    """Re-derive the published leaderboard numbers from the same records.

    F_bar_mag is the macro average over families of the macro average over
    seeds of each seed's faithful rate. Reproducing it exactly is what licenses
    the per-task regrouping below.
    """
    per_family = {f: macro([rate([r for r in s if r["family"] == f])
                            for s in seeds])
                  for f in NON_CONTROL + [FLOOR]}
    f_bar = macro([per_family[f] for f in NON_CONTROL])
    floor = per_family[FLOOR]
    got = {"F_bar_mag": f_bar, "paraphrase_null_floor": floor}
    for key, val in got.items():
        want = pub.get(key)
        if want is None or abs(val - want) > TOL:
            raise SystemExit(
                f"[per-task] {label}: recomputed {key}={val!r} does not match "
                f"derived_metrics.json {want!r} -- this script is reading the "
                f"released records differently from the leaderboard pipeline")
    return got


def per_model(seeds: list[list[dict]]) -> dict:
    by_task: dict[str, dict[str, list[dict]]] = {}
    for s in seeds:
        for r in s:
            by_task.setdefault(r["file_base"], {}).setdefault(
                r["family"], []).append(r)

    tasks = {}
    for t, fams in sorted(by_task.items()):
        f_sem = macro([rate(fams.get(f, [])) for f in NON_CONTROL])
        f_floor = rate(fams.get(FLOOR, []))
        if f_sem is None or f_floor is None:
            continue
        tasks[t] = {
            "F_sem": f_sem,
            "F_floor": f_floor,
            "F_diff": f_sem - f_floor,
            "n_families_present": sum(1 for f in NON_CONTROL if fams.get(f)),
            "n_sem": sum(len(fams.get(f, [])) for f in NON_CONTROL),
            "n_floor": len(fams[FLOOR]),
        }

    sems = sorted(v["F_sem"] for v in tasks.values())
    out = {
        "n_seeds_pooled": len(seeds),
        "n_tasks": len(tasks),
        "all": sign_test(tasks, 1),
        f"floor_ge{MIN_FLOOR_RESTRICTED}": sign_test(tasks,
                                                     MIN_FLOOR_RESTRICTED),
        "min_n_sem_per_task": min((v["n_sem"] for v in tasks.values()),
                                  default=0),
        "min_n_floor_per_task": min((v["n_floor"] for v in tasks.values()),
                                    default=0),
        "min_n_families_per_task": min(
            (v["n_families_present"] for v in tasks.values()), default=0),
        "F_sem_task_min": sems[0] if sems else None,
        "F_sem_task_max": sems[-1] if sems else None,
        "F_sem_task_spread": (sems[-1] - sems[0]) if sems else None,
        "F_sem_task_iqr": (quantile(sems, 0.75) - quantile(sems, 0.25))
                          if sems else None,
        "per_task": tasks,
    }
    return out


def sign_test(tasks: dict, min_n_floor: int) -> dict:
    diffs = [v["F_diff"] for v in tasks.values()
             if v["n_floor"] >= min_n_floor]
    below = sum(1 for d in diffs if d < 0)
    above = sum(1 for d in diffs if d > 0)
    return {
        "min_n_floor": min_n_floor,
        "n_tasks": len(diffs),
        "n_below": below,
        "n_above": above,
        "n_tied": sum(1 for d in diffs if d == 0),
        "p_two_sided": binom_two_sided(min(below, above), below + above),
        "median_F_diff": median(diffs) if diffs else None,
        "majority_below": below > above,
    }


def main() -> int:
    pub = json.loads(DERIVED.read_text())["models"]
    out = {
        "what": "per-LIBERO-task decomposition of F over the 7 non-control "
                "families against each task's own paraphrase_null floor",
        "source": "released 3-seed edit records grouped by "
                  "per_sample[].file_base; no new inference",
        "averaging": "F over a family set is the macro average of per-family "
                     "rates, matching scripts/derive_metrics.py",
        "non_control_families": NON_CONTROL,
        "floor_family": FLOOR,
        "tau": 0.05,
        "models": {},
    }
    for label, pat in RUNS.items():
        seeds = load_seeds(pat)
        entry = per_model(seeds)
        entry["provenance"] = provenance(seeds, label, pub[label])
        out["models"][label] = entry

    m = out["models"]
    key3 = f"floor_ge{MIN_FLOOR_RESTRICTED}"
    # How the within-model task heterogeneity compares with the leaderboard
    # gaps it would have to be smaller than for a per-model F to be readable as
    # a model property. Adjacent pairs in F_bar order are the easiest case: if
    # even those gaps sit inside both models' task IQRs, the ordering is not
    # resolvable at task granularity.
    order = sorted(m, key=lambda k: m[k]["provenance"]["F_bar_mag"])
    adjacent = []
    for lo, hi in zip(order, order[1:]):
        gap = (m[hi]["provenance"]["F_bar_mag"]
               - m[lo]["provenance"]["F_bar_mag"])
        adjacent.append({
            "lower": lo, "upper": hi, "gap": gap,
            "inside_both_task_iqrs": gap < min(m[lo]["F_sem_task_iqr"],
                                               m[hi]["F_sem_task_iqr"]),
        })
    out["summary"] = {
        "n_models": len(m),
        "n_models_majority_below_own_floor": sum(
            1 for v in m.values() if v["all"]["majority_below"]),
        "n_models_significant_p05": sum(
            1 for v in m.values()
            if v["all"]["majority_below"] and v["all"]["p_two_sided"] < 0.05),
        "n_models_significant_p05_" + key3: sum(
            1 for v in m.values()
            if v[key3]["majority_below"] and v[key3]["p_two_sided"] < 0.05),
        "n_tasks_min_over_models": min(v["n_tasks"] for v in m.values()),
        "n_tasks_max_over_models": max(v["n_tasks"] for v in m.values()),
        "n_tasks_restricted": m[order[0]][key3]["n_tasks"],
        "worst_F_sem_task_spread": max(v["F_sem_task_spread"]
                                       for v in m.values()),
        "smallest_F_sem_task_iqr": min(v["F_sem_task_iqr"]
                                       for v in m.values()),
        "largest_F_sem_task_iqr": max(v["F_sem_task_iqr"] for v in m.values()),
        "exceptions": [k for k, v in m.items()
                       if not (v["all"]["majority_below"]
                               and v["all"]["p_two_sided"] < 0.05)],
        "leaderboard_order_by_F_bar": order,
        "adjacent_pairs": adjacent,
        "n_adjacent_pairs_inside_both_task_iqrs": sum(
            1 for a in adjacent if a["inside_both_task_iqrs"]),
        "largest_adjacent_gap_inside_both_task_iqrs": max(
            (a["gap"] for a in adjacent if a["inside_both_task_iqrs"]),
            default=None),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_task.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[per-task] wrote {OUT / 'per_task.json'}")
    for label, v in m.items():
        a, r = v["all"], v[key3]
        print(f"  {label:13s} F_bar={v['provenance']['F_bar_mag']:.3f} "
              f"tasks={v['n_tasks']:3d} "
              f"below/above/tied={a['n_below']}/{a['n_above']}/{a['n_tied']} "
              f"p={a['p_two_sided']:.1e} med={a['median_F_diff']:+.3f} "
              f"| floor>=3 (n={r['n_tasks']:2d}): "
              f"{r['n_below']}/{r['n_above']}/{r['n_tied']} "
              f"p={r['p_two_sided']:.1e} "
              f"| task IQR={v['F_sem_task_iqr']:.3f} "
              f"range=[{v['F_sem_task_min']:.2f}, {v['F_sem_task_max']:.2f}]")
    for k, val in out["summary"].items():
        print(f"    {k}: {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
