#!/usr/bin/env python3
"""Re-derives Table 1's floor-comparison significance counts (9/11 vs
paraphrase, 3/11 vs scramble) with the bootstrap resampling unit changed
from (seed, sample) to LIBERO episode file (file_base).

Why this exists: an ICLR reviewer (fresh, independent, no prior context on
this project) found that the primary significance tests behind Table 1
resample at the (seed, sample) level, while this paper's own per-task
decomposition (scripts/derive_per_task.py, Appendix sec:per_task) shows
samples from the same LIBERO episode file are highly correlated -- within-
model task-level IQR of F_sem runs up to 0.343, "the same checkpoint scores
0.00 on some LIBERO-90 tasks and 1.00 on others." Multiple (seed, sample)
observations routinely share a file_base (confirmed directly: in
ours_lora-r32's own seed0 file, 61 distinct file_base values cover 100
samples, with up to 4 samples sharing one file). Treating (seed, sample) as
the exchangeable resampling unit when samples that share a file_base are not
exchangeable with samples that don't understates uncertainty relative to a
cluster (block) bootstrap over file_base.

This script is the direct, computable check the reviewer asked for: does the
qualitative significance pattern survive when file_base is the resampling
unit instead? It reuses the exact same point statistic (binary_diff_stat --
imported, not reimplemented, so the two cannot silently diverge) and the
exact same tau/seed/n_boot as the original, changing only what counts as one
resampling unit.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import floor_convention_robustness as fcr  # noqa: E402
from bootstrap_multiplicity_bca import binary_diff_stat  # noqa: E402

N_BOOT = 20000
SEED = 12345


def group_by_file(records):
    """{file_base: {(seed, sample): {family: record}}} -- same per-observation
    shape binary_diff_stat expects, just nested one level under file_base so
    a resample can pull ALL of a file's observations together."""
    by_file = {}
    for r in records:
        if r.get("skipped"):
            continue
        fb = r.get("file_base")
        if fb is None:
            continue
        key = (r.get("seed", 0), r["sample"])
        by_file.setdefault(fb, {}).setdefault(key, {})[r["family"]] = r
    return by_file


def file_clustered_bootstrap(by_file, families_b, floor_family,
                              tau=fcr.TAU_DEFAULT, n_boot=N_BOOT, seed=SEED):
    files = list(by_file.keys())
    n_files = len(files)

    def flatten(file_keys):
        merged = {}
        idx = 0
        for fb in file_keys:
            for obs in by_file[fb].values():
                merged[idx] = obs
                idx += 1
        return merged

    point = binary_diff_stat(flatten(files), families_b, floor_family, tau)
    if point is None:
        return None

    rng = fcr.Rng(seed)
    draws = []
    for _ in range(n_boot):
        sampled_files = [files[rng.randint(n_files)] for _ in range(n_files)]
        v = binary_diff_stat(flatten(sampled_files), families_b, floor_family, tau)
        if v is not None:
            draws.append(v)
    if not draws:
        return None
    draws.sort()
    B = len(draws)
    lo = draws[int(0.025 * B)]
    hi = draws[int(0.975 * B) - 1]
    frac_le0 = sum(1 for d in draws if d <= 0) / B
    frac_ge0 = sum(1 for d in draws if d >= 0) / B
    p_boot = min(1.0, 2.0 * min(frac_le0, frac_ge0))
    return {
        "point": point, "n_files": n_files, "n_boot_used": B,
        "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0),
        "p_boot": p_boot,
    }


def main():
    fcr_configs = list(fcr.OURS) + list(fcr.DEEPTHINK)
    assert len(fcr_configs) == 11, len(fcr_configs)

    results = {}
    for name in fcr_configs:
        recs = fcr.load_config(name)
        by_file = group_by_file(recs)
        results[name] = {}
        for floor in fcr.FLOORS:
            res = file_clustered_bootstrap(by_file, fcr.CORE7, floor)
            results[name][floor] = res

    n_sig_para = sum(1 for name in fcr_configs
                      if results[name]["paraphrase_null"]
                      and results[name]["paraphrase_null"]["excludes_zero"])
    n_neg_para = sum(1 for name in fcr_configs
                       if results[name]["paraphrase_null"]
                       and results[name]["paraphrase_null"]["point"] < 0)
    n_sig_scram = sum(1 for name in fcr_configs
                        if results[name]["syntactic_scramble"]
                        and results[name]["syntactic_scramble"]["excludes_zero"])
    n_pos_scram = sum(1 for name in fcr_configs
                        if results[name]["syntactic_scramble"]
                        and results[name]["syntactic_scramble"]["point"] > 0)

    print(f"{'config':16s} {'n_files':>8s}  {'vs para (point, CI, sig)':40s}  "
          f"{'vs scram (point, CI, sig)':40s}")
    for name in fcr_configs:
        p = results[name]["paraphrase_null"]
        s = results[name]["syntactic_scramble"]
        p_str = (f"{p['point']:+.3f} [{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}] "
                  f"{'SIG' if p['excludes_zero'] else 'ns'}") if p else "n/a"
        s_str = (f"{s['point']:+.3f} [{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}] "
                  f"{'SIG' if s['excludes_zero'] else 'ns'}") if s else "n/a"
        nf = p["n_files"] if p else (s["n_files"] if s else "?")
        print(f"{name:16s} {nf!s:>8s}  {p_str:40s}  {s_str:40s}")

    print()
    print(f"vs paraphrase floor: negative on {n_neg_para}/11, "
          f"significant (file-clustered) on {n_sig_para}/11")
    print(f"vs scramble floor: positive on {n_pos_scram}/11, "
          f"significant (file-clustered) on {n_sig_scram}/11")
    print()
    print("(seed,sample)-clustered originals for comparison: "
          "negative 11/11 (9/11 sig), positive 10/11 (3/11 sig)")

    out = {
        "n_boot": N_BOOT, "seed": SEED,
        "per_config": {
            name: {
                floor: (results[name][floor] if results[name][floor] else None)
                for floor in fcr.FLOORS
            } for name in fcr_configs
        },
        "n_negative_vs_paraphrase": n_neg_para,
        "n_significant_vs_paraphrase_file_clustered": n_sig_para,
        "n_positive_vs_scramble": n_pos_scram,
        "n_significant_vs_scramble_file_clustered": n_sig_scram,
    }
    dest = ROOT / "results_v2" / "canonical_runs" / "floor_bootstrap_file_clustered" \
        / "floor_bootstrap_file_clustered.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n[file-clustered-bootstrap] -> {dest}")


if __name__ == "__main__":
    main()
