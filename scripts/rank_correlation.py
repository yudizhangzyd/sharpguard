"""How much do the two scoring rules disagree, as one number?

S6 shows the ranking inverts by walking the reader down two columns of
tab:directional. A reviewer asked for the summary statistic that the columns
imply, so this computes it: the rank correlation between the magnitude and
direction-aware orderings of the eight leaderboard configurations on
direction_flip.

Two things this script refuses to do, because they would overstate the result:

  * It reports the correlation PER SEED as well as on the 3-seed mean. The mean
    ranking is one draw of an ordering whose adjacent pairs S8 shows to be
    inside the retraining error bar; if rho were stable only on the mean, that
    would be an artifact of averaging, and the per-seed spread is what shows it
    is not.
  * It reports Kendall tau alongside Spearman rho. rho on n=8 is dominated by
    whichever model moves furthest, which here is exactly the model the section
    is about; tau counts discordant PAIRS and so cannot be carried by one row.

Ties: F_dir is a rate over ~100 samples and the DeepThink checkpoints are not
in this cohort, so exact ties are possible. Both statistics use midranks and
the tie-corrected denominators, and the tie count is reported rather than
silently absorbed.

Writes results_v2/canonical_runs/rank_correlation/rank_correlation.json.
"""
import json
import os
from itertools import combinations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERIVED = os.path.join(ROOT, "results_v2", "derived_metrics.json")
OUT_DIR = os.path.join(ROOT, "results_v2", "canonical_runs", "rank_correlation")
FAMILY = "direction_flip"


def midranks(xs):
    """Ranks of ``xs`` descending, ties sharing their mean rank."""
    order = sorted(range(len(xs)), key=lambda i: -xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return None if da == 0 or db == 0 else num / (da * db)


def spearman(x, y):
    """Pearson on midranks -- the tie-corrected definition, not 1-6d^2/n(n^2-1),
    which is only valid without ties."""
    return pearson(midranks(x), midranks(y))


def kendall_tau_b(x, y):
    """tau-b: concordant minus discordant over the tie-corrected denominator."""
    con = dis = tx = ty = txy = 0
    for i, j in combinations(range(len(x)), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        if dx == 0 and dy == 0:
            txy += 1
        elif dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif (dx > 0) == (dy > 0):
            con += 1
        else:
            dis += 1
    n0 = con + dis + tx + ty + txy
    den = ((n0 - tx - txy) * (n0 - ty - txy)) ** 0.5
    return None if den == 0 else (con - dis) / den, con, dis, tx, ty, txy


def main():
    with open(DERIVED) as fh:
        d = json.load(fh)
    order = ["ours-no-cot", "ours-data50A", "ours-data50B", "ours-r8",
             "ours-r16", "ours-r32", "ours-r64", "ecot-bridge"]
    models = [m for m in order if m in d["models"]]
    fams = {m: d["models"][m]["families"][FAMILY] for m in models}

    n_seeds = {len(fams[m]["F_mag_per_run"]) for m in models} | \
              {len(fams[m]["F_dir_per_run"]) for m in models}
    assert len(n_seeds) == 1, f"ragged seed counts: {n_seeds}"
    n_seed = n_seeds.pop()

    def block(mag, dr, label):
        tau, con, dis, tx, ty, txy = kendall_tau_b(mag, dr)
        rm, rd = midranks(mag), midranks(dr)
        return {
            "label": label,
            "spearman_rho": spearman(mag, dr),
            "kendall_tau_b": tau,
            "n_concordant_pairs": con, "n_discordant_pairs": dis,
            "n_ties_magnitude_only": tx, "n_ties_direction_only": ty,
            "n_ties_both": txy,
            "rank_magnitude": {m: rm[i] for i, m in enumerate(models)},
            "rank_direction": {m: rd[i] for i, m in enumerate(models)},
            "max_rank_shift": max(abs(rm[i] - rd[i]) for i in range(len(models))),
            "model_of_max_shift": models[max(
                range(len(models)), key=lambda i: abs(rm[i] - rd[i]))],
        }

    mean_block = block([fams[m]["F_mag"] for m in models],
                       [fams[m]["F_dir"] for m in models], "3-seed mean")
    per_seed = [block([fams[m]["F_mag_per_run"][s] for m in models],
                      [fams[m]["F_dir_per_run"][s] for m in models],
                      f"seed {s}") for s in range(n_seed)]

    rhos = [b["spearman_rho"] for b in per_seed]
    taus = [b["kendall_tau_b"] for b in per_seed]
    out = {
        "family": FAMILY,
        "n_models": len(models),
        "models": models,
        "n_seeds": n_seed,
        "source": "results_v2/derived_metrics.json models[*].families."
                  + FAMILY + ".{F_mag,F_dir}_per_run",
        "on_3seed_mean": mean_block,
        "per_seed": per_seed,
        "rho_per_seed_min": min(rhos), "rho_per_seed_max": max(rhos),
        "tau_per_seed_min": min(taus), "tau_per_seed_max": max(taus),
        "n_seeds_with_negative_rho": sum(1 for r in rhos if r < 0),
        "interpretation":
            "Spearman rho and Kendall tau-b between the magnitude and "
            "direction-aware orderings of the leaderboard cohort on "
            + FAMILY + ". Reported on the 3-seed mean and separately on each "
            "seed, because S8 shows adjacent ranks sit inside the retraining "
            "error bar and a statistic that held only after averaging would "
            "be an artifact of the averaging.",
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "rank_correlation.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)

    print(f"wrote {p}")
    print(f"  3-seed mean: rho = {mean_block['spearman_rho']:+.3f}  "
          f"tau_b = {mean_block['kendall_tau_b']:+.3f}  "
          f"({mean_block['n_concordant_pairs']} concordant / "
          f"{mean_block['n_discordant_pairs']} discordant pairs)")
    print(f"  max rank shift: {mean_block['max_rank_shift']:.1f} "
          f"({mean_block['model_of_max_shift']})")
    for b in per_seed:
        print(f"  {b['label']}: rho = {b['spearman_rho']:+.3f}  "
              f"tau_b = {b['kendall_tau_b']:+.3f}")
    print(f"  rho range over seeds: [{min(rhos):+.3f}, {max(rhos):+.3f}]; "
          f"{out['n_seeds_with_negative_rho']}/{n_seed} seeds negative")


if __name__ == "__main__":
    main()
