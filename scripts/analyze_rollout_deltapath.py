#!/usr/bin/env python3
r"""Does the token-level edit score predict what the arm actually does?

Every number in this paper's edit protocol is a one-step decode difference: the
policy is asked for one action under the original CoT and one under the edited
CoT, and $\mathcal{F}$ counts how often those two actions differ. The claim the
score is used to carry is about behaviour -- that a model whose CoT matters will
act differently when the CoT is edited -- and one decode step is not behaviour.
That gap is limitation (v), and this script is the measurement that turns it
from a caveat into an axis.

The readout is end-effector path deviation from the clean-CoT arm, not success
rate. Success cannot carry it: in-suite SR is 0/20 for every arm we have run, so
every DSR is 0 by construction and ranks nothing (see the precondition note in
experiments/cotfaith_rollout_edit.py). Path deviation is defined whether or not
the task is solved -- an arm that drives off the table and an arm that stays put
both score 0/1 and are 130 cm apart -- which is exactly the discrimination the
scalar record throws away.

What is being asked, precisely: rank the 13 edit families by token-level
$\mathcal{F}$, rank them by how far the arm ends up from where the clean CoT
would have put it, and correlate the two rankings. WorldGym validates its
learned proxy against real-robot success and RobotArena against human
preference; this is the same move for a token-level faithfulness score, and the
answer is allowed to be that they do not agree.

Two things this cannot do, stated here because the JSON is small enough that a
reader could mistake it for more than it is:

  * n is the number of captured episodes, which is 2. Thirteen families ranked
    off two scenes is a correlation with a very wide interval, and the script
    prints the interval rather than the point estimate alone.
  * Deviation is not error. The clean arm does not solve the task either, so a
    family that deviates from it is not thereby wrong -- it is different. The
    quantity is dissociation between two measurement scales, not accuracy.

Usage:
    python3 scripts/analyze_rollout_deltapath.py \
        --report results_v2/canonical_runs/rollout_deltapath/rollout_edit_report.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "results_v2" / "derived_metrics.json"
# The checkpoint the rollout runs. Named here rather than inferred from the
# report's --ckpt-path, which is a pod-local /tmp path: the arms are replayed
# from the r=32 fine-tune, and its token-level families are the ones whose
# ranking this compares against. A mismatch would silently correlate one
# model's F against another model's motion.
MODEL_ROW = "ours-r32"
# Families that exist to move the CoT without changing its meaning. Called out
# because the paper's first headline is that these move the ACTION more than the
# semantic families do, and whether that survives into the rollout is the single
# most load-bearing question this run can answer.
NULLS = ("paraphrase_null", "bbox_jitter_null", "selfsplice_control")


def spearman(x, y) -> float:
    """Spearman rho via average ranks. Ties matter: F_mag is a proportion over
    ~70 samples and lands on the same value for different families often enough
    that ordinal ranking without tie handling would invent an ordering."""
    def rank(v):
        v = np.asarray(v, dtype=float)
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=float)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den else float("nan")


def perm_p(x, y, n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided permutation p for rho. 13! is not enumerable, so this samples
    with a fixed seed and reports n_perm alongside, which makes the value
    reproducible rather than merely approximate. No scipy: this repo's
    environment is numpy-only and adding a dependency to compute one p-value is
    how a re-derive stops working on someone else's machine."""
    rng = np.random.default_rng(seed)
    obs = abs(spearman(x, y))
    y = np.asarray(y, dtype=float)
    hits = sum(1 for _ in range(n_perm)
               if abs(spearman(x, rng.permutation(y))) >= obs - 1e-12)
    return (hits + 1) / (n_perm + 1)


def track(ep) -> np.ndarray | None:
    """The arm's end-effector path, or None if this env logged no poses.

    None rather than zeros, for the reason _eef() in the harness returns None: a
    flat line at the origin is a claim about the policy, and the wrong one."""
    traj = ep.get("trajectory") or []
    if not traj or not ep.get("eef_available"):
        return None
    xyz = [t.get("eef") for t in traj]
    if any(p is None for p in xyz):
        return None
    return np.asarray(xyz, dtype=float)


def deviation(arm_xyz: np.ndarray, clean_xyz: np.ndarray) -> dict:
    """Per-step distance between two paired arms, summarized in cm.

    Truncated to the shorter arm: an episode ends early on success or on a
    broken observation, and comparing step 60 of one arm against nothing is how
    a shorter run would read as a smaller deviation."""
    t = min(len(arm_xyz), len(clean_xyz))
    d = np.linalg.norm(arm_xyz[:t] - clean_xyz[:t], axis=1) * 100.0
    step = np.linalg.norm(np.diff(clean_xyz[:t], axis=0), axis=1).sum() * 100.0
    return {"n_steps_compared": int(t),
            "mean_cm": float(d.mean()),
            "peak_cm": float(d.max()),
            "final_cm": float(d[-1]),
            # d at the first step, which must be 0: the arms are rewound to one
            # scene, so any deviation at t=0 is a pairing defect, not an edit
            # effect. Four earlier capture runs failed exactly this.
            "step0_cm": float(d[0]),
            # The clean arm's own path length, so a deviation can be read
            # against how far the arm travelled at all. 3 cm of deviation on a
            # 4 cm path and on a 200 cm path are not the same measurement.
            "clean_path_len_cm": float(step)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True,
                    help="rollout_edit_report.json from the deltapath run")
    ap.add_argument("--out", default=None,
                    help="where to write deltapath_report.json "
                         "(default: alongside --report)")
    ap.add_argument("--n-perm", type=int, default=20000)
    a = ap.parse_args()

    rep = json.loads(Path(a.report).read_text())
    eps = [e for e in rep.get("episodes", []) if "error" not in e]
    if not eps:
        print("[deltapath] FAIL: the report has no completed episodes")
        return 2

    # --- pair every edit arm against the clean arm of the SAME scene ---------
    scenes = sorted({(e["task_idx"], e["episode"]) for e in eps})
    per_scene, pairing_defects = {}, []
    for scene in scenes:
        rows = {e["arm"]: e for e in eps
                if (e["task_idx"], e["episode"]) == scene}
        clean = track(rows["cot_clean"]) if "cot_clean" in rows else None
        if clean is None:
            continue                      # no baseline for this scene
        out = {}
        for arm, e in rows.items():
            # nocot is kept: it is the how-far-can-an-arm-drift reference, and
            # an edit family that deviates less than "no CoT at all" is a fact
            # about this scene the comparison needs.
            if arm == "cot_clean":
                continue
            xyz = track(e)
            if xyz is None:
                continue
            d = deviation(xyz, clean)
            d["family"] = e.get("family")
            d["success"] = bool(e.get("success"))
            out[arm] = d
            if d["step0_cm"] > 1e-6:
                pairing_defects.append(
                    f"{scene[0]}:{scene[1]}/{arm} differs from cot_clean at "
                    f"t=0 by {d['step0_cm']:.4f} cm")
        if out:
            per_scene[f"task{scene[0]}_ep{scene[1]}"] = out

    if not per_scene:
        print("[deltapath] FAIL: no scene has both a clean arm and a logged "
              "end-effector path. Was the run submitted without "
              "--capture-episodes, or does this env omit robot0_eef_pos?")
        return 2

    # A pairing defect invalidates the comparison rather than degrading it: the
    # arms would differ before any edit was applied, so the deviation would be
    # partly the scene and partly the edit with no way to separate them. Loud,
    # and non-zero exit, for the same reason the figure generator refuses to
    # draw unpaired rows.
    if pairing_defects:
        print(f"[deltapath] FAIL: {len(pairing_defects)} arm(s) are not paired "
              f"at t=0, so no deviation here is attributable to the edit:")
        for line in pairing_defects[:6]:
            print(f"[deltapath]   {line}")
        return 2

    # --- average each family over the scenes it was filmed in ---------------
    arms = sorted({arm for s in per_scene.values() for arm in s})
    by_arm = {}
    for arm in arms:
        rows = [s[arm] for s in per_scene.values() if arm in s]
        by_arm[arm] = {
            "family": rows[0]["family"],
            "n_scenes": len(rows),
            "mean_cm": float(np.mean([r["mean_cm"] for r in rows])),
            "peak_cm": float(np.max([r["peak_cm"] for r in rows])),
            "final_cm": float(np.mean([r["final_cm"] for r in rows])),
            "mean_cm_per_scene": [round(r["mean_cm"], 3) for r in rows],
            "clean_path_len_cm": float(np.mean(
                [r["clean_path_len_cm"] for r in rows])),
            "n_steps_compared": int(min(r["n_steps_compared"] for r in rows)),
        }

    # --- the comparison this run exists for --------------------------------
    derived = json.loads(DERIVED.read_text())
    fams = derived["models"][MODEL_ROW]["families"]
    paired, missing = [], []
    for arm, row in by_arm.items():
        fam = row["family"]
        if arm == "nocot" or not fam:
            continue
        if fam not in fams:
            missing.append(fam)
            continue
        paired.append((fam, fams[fam], row))

    corr = {}
    if len(paired) >= 4:
        dev = [r["mean_cm"] for _, _, r in paired]
        # Three token-level scores, not one. F_mag is the magnitude score the
        # paper's first headline is about, F_diff its differential form, and
        # cos_xyz the direction agreement the direction-aware score is built
        # from. Reporting all three is the difference between asking "does our
        # metric predict behaviour" and asking "does ANY of them".
        for key in ("F_mag", "F_diff", "cos_xyz"):
            tok = [f.get(key) for _, f, _ in paired]
            if any(v is None for v in tok):
                continue
            rho = spearman(tok, dev)
            corr[key] = {"spearman_rho": round(rho, 4),
                         "perm_p": round(perm_p(tok, dev, a.n_perm), 5),
                         "n_families": len(paired),
                         "n_perm": a.n_perm, "perm_seed": 0}

    # Where the nulls land. The paper's first headline is that a
    # meaning-preserving paraphrase moves the one-step action MORE than semantic
    # edits do; if the rollout agrees, the anomaly is not an artifact of
    # single-step decoding, and if it does not, that bounds the anomaly to the
    # decode step. Either answer is worth the GPU hours; neither is assumed.
    order = sorted(paired, key=lambda p: -p[2]["mean_cm"])
    ranked = [f for f, _, _ in order]
    sem = [r["mean_cm"] for f, _, r in paired if f not in NULLS]
    nulls = {f: round(r["mean_cm"], 3) for f, _, r in paired if f in NULLS}
    null_vs_semantic = {
        "semantic_mean_cm": round(float(np.mean(sem)), 3) if sem else None,
        "null_mean_cm": nulls,
        "paraphrase_null_rank_of": (
            [ranked.index(f) + 1 for f in ("paraphrase_null",)
             if f in ranked] or [None])[0],
        "n_families_ranked": len(ranked),
    }

    out = {
        "experiment": "rollout_path_deviation_vs_token_score",
        "readout": "mean per-step end-effector distance from the cot_clean arm "
                   "of the same scene, cm",
        "why_not_success_rate": rep.get("precondition_note"),
        "model_row_compared": MODEL_ROW,
        "source_report": str(Path(a.report)),
        "config": {k: rep.get("config", {}).get(k)
                   for k in ("suite", "max_steps", "cot_refresh_steps",
                             "n_tasks", "n_eps_per_task", "capture_episodes",
                             "capture_every", "families")},
        "n_scenes": len(per_scene),
        "arms_compared": len(by_arm),
        "families_not_in_derived_metrics": sorted(set(missing)),
        "step0_pairing_verified": True,
        "by_arm": by_arm,
        "per_scene": per_scene,
        "ranking_by_deviation": ranked,
        "correlation_with_token_score": corr,
        "nulls_vs_semantic": null_vs_semantic,
        "caveats": [
            f"n={len(per_scene)} scene(s). A 13-family rank correlation on this "
            f"many scenes has an interval wide enough that only a very large "
            f"rho would be distinguishable from chance; perm_p is reported for "
            f"exactly that reason.",
            "Deviation from the clean arm is dissociation, not error: the clean "
            "arm does not solve the task either, so no arm here is closer to "
            "correct for deviating less.",
            f"{rep.get('config', {}).get('max_steps')} steps of the episode, "
            f"which is the approach phase, not the full task.",
        ],
    }

    dest = Path(a.out) if a.out else Path(a.report).parent / "deltapath_report.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")

    print(f"[deltapath] {len(per_scene)} scene(s), {len(paired)} families, "
          f"{by_arm[arms[0]]['n_steps_compared']} steps compared")
    print(f"[deltapath] {'family':22} {'dev_cm':>8} {'F_mag':>7} {'cos_xyz':>8}")
    for fam, f, r in order:
        print(f"[deltapath] {fam:22} {r['mean_cm']:8.2f} "
              f"{f.get('F_mag', float('nan')):7.3f} "
              f"{f.get('cos_xyz', float('nan')):8.3f}")
    for key, c in corr.items():
        print(f"[deltapath] rho({key}, deviation) = {c['spearman_rho']:+.3f}  "
              f"p = {c['perm_p']:.4f}  (n={c['n_families']})")
    if nulls:
        print(f"[deltapath] nulls {nulls} vs semantic mean "
              f"{null_vs_semantic['semantic_mean_cm']} cm")
    print(f"[deltapath] wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
