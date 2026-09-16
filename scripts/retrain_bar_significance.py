#!/usr/bin/env python3
"""Does the headline F_bar_diff survive against the retraining-level error
bar (S7), rather than only the within-run observation bootstrap (S3)?

S7 already establishes that retraining the same recipe moves F_bar and
per-family F by more than resampling which observations are scored does, but
S3's "9 of 11 significant" comes from the observation bootstrap alone, on the
seven leaderboard rows that were also retrained. This computes, directly from
the seven released RETRAIN pairs, how much F_bar_diff (the seven-family mean
minus the paraphrase floor, S3's actual headline quantity) itself moves
between an original run and its byte-identical retrain -- not F_bar alone,
which S7 quotes, since the floor can move too -- and checks which
configurations' printed F_bar_diff clearly exceeds that empirical bar.

Usage:
    python3 scripts/retrain_bar_significance.py
"""
import json
import os
import statistics

SEMANTIC_7 = ["direction_flip", "gripper_flip", "verb_swap", "negation",
              "subject_swap", "location_swap", "adversarial_plausible"]
RETRAIN_CONFIGS = ["ours_no-cot", "ours_lora-r8", "ours_lora-r16",
                   "ours_lora-r32", "ours_lora-r64", "ours_data-50A",
                   "ours_data-50B"]
NICE = {"ours_no-cot": "no-CoT", "ours_lora-r8": "r=8", "ours_lora-r16": "r=16",
        "ours_lora-r32": "r=32", "ours_lora-r64": "r=64",
        "ours_data-50A": "data-50A", "ours_data-50B": "data-50B"}


def fbar_diff(agg):
    f7 = sum(agg[fam]["faithful_rate"] for fam in SEMANTIC_7) / 7
    return f7 - agg["paraphrase_null"]["faithful_rate"], f7


def main():
    fcr = json.load(open(
        "results_v2/canonical_runs/floor_convention_robustness/"
        "floor_convention_robustness.json"))

    moves = []
    print(f"{'config':10s} {'published':>10s} {'retrain':>10s} {'|move|':>8s}")
    for cfg in RETRAIN_CONFIGS:
        path = (f"results_v2/canonical_runs/{cfg}_edit_13family_RETRAIN.json")
        if not os.path.exists(path):
            print(f"{cfg:10s} MISSING: {path}")
            continue
        d = json.load(open(path))
        retrain_diff, retrain_f7 = fbar_diff(d["aggregate"])
        published_diff = fcr["per_config"][cfg]["diff_B"]["paraphrase_null"]
        move = abs(published_diff - retrain_diff)
        moves.append(move)
        print(f"{NICE[cfg]:10s} {published_diff:10.3f} {retrain_diff:10.3f} "
              f"{move:8.3f}")

    bar_median = statistics.median(moves)
    bar_max = max(moves)
    print(f"\nempirical F_bar_diff retraining-movement: median {bar_median:.3f}, "
          f"max {bar_max:.3f} (n={len(moves)} pairs)")

    print(f"\n{'config':10s} {'F_bar_diff':>10s} {'survives median bar':>20s} "
          f"{'survives max bar':>17s}")
    survives_median = survives_max = 0
    # bridge_subset_4k is scored here but is not one of tab:floors' 11 rows
    # (excluded there as "not comparable in scale"), so it is excluded from
    # this count too, or the printed "9 of 11" would be checked against 12.
    all11 = [(cfg, c) for cfg, c in fcr["per_config"].items()
             if cfg != "bridge_subset_4k"]
    for cfg, c in all11:
        d = c["diff_B"]["paraphrase_null"]
        sm, sx = abs(d) > bar_median, abs(d) > bar_max
        survives_median += sm
        survives_max += sx
        label = NICE.get(cfg, cfg)
        print(f"{label:10s} {d:10.3f} {str(sm):>20s} {str(sx):>17s}")
    print(f"\n{survives_median} of {len(all11)} survive the median retraining "
          f"bar ({bar_median:.3f}); {survives_max} of {len(all11)} survive "
          f"the max ({bar_max:.3f}).")

    out = {"per_pair_move": dict(zip([NICE[c] for c in RETRAIN_CONFIGS
                                       if os.path.exists(
                                           f'results_v2/canonical_runs/'
                                           f'{c}_edit_13family_RETRAIN.json')],
                                      moves)),
           "bar_median": bar_median, "bar_max": bar_max,
           "survives_median_bar": survives_median,
           "survives_max_bar": survives_max, "n_configs": len(all11)}
    dest = ("results_v2/canonical_runs/retrain_bar_significance/"
            "retrain_bar_significance.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\n[retrain-bar] -> {dest}")


if __name__ == "__main__":
    main()
