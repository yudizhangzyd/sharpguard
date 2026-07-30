#!/usr/bin/env python3
"""decoder_audit.py -- action-decoder validity audit (zero new inference).

Publishes the gate the v6 submission never reported:
  (a) per-dimension Pearson corr(pred, gt) on the pinned N=200 AUROC log
  (b) model L1 vs two trivial baselines (constant-zero, predict-dataset-mean)
  (c) gripper sign agreement
  (d) the public-checkpoint rollout success rate actually observed
Writes results_v2/decoder_audit.json
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = os.path.join(ROOT, "results_v2", "canonical_runs")


def _pick(tmp_path, canon_name):
    return tmp_path if os.path.exists(tmp_path) else os.path.join(CANON, canon_name)


AUROC = _pick("/tmp/cf_done/wqy4kt9up6/cotfaith-auroc/cot_auroc_report.json",
              "auroc_ecot_bridge_n200.json")
ROLLOUT = _pick("/tmp/cf_r3_all/funng2vik5/rollout-baseline/sr.json",
                "rollout_openvla_libero_spatial_sr.json")
SANITY = _pick("/tmp/cf_done/bcihypv3gu/cotfaith-sanity/sanity_report.json",
               "sanity_ours_r32.json")
DIMS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx < 1e-12 or sy < 1e-12:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def main():
    d = json.load(open(AUROC))
    rows = d["per_sample"]
    P = [r["action_pred"] for r in rows]
    G = [r["action_gt"] for r in rows]
    n = len(rows)

    out = {"_provenance": {"auroc_log": AUROC, "rollout_log": ROLLOUT,
                           "sanity_log": SANITY, "n": n},
           "per_dim": {}}
    for i, name in enumerate(DIMS):
        p = [row[i] for row in P]
        g = [row[i] for row in G]
        out["per_dim"][name] = {
            "corr": pearson(p, g),
            "pred_mean": sum(p) / n, "gt_mean": sum(g) / n,
            "pred_std": math.sqrt(sum((a - sum(p) / n) ** 2 for a in p) / n),
            "gt_std": math.sqrt(sum((a - sum(g) / n) ** 2 for a in g) / n),
        }
    # flattened corr over all 7 dims x N samples
    flat_p = [v for row in P for v in row]
    flat_g = [v for row in G for v in row]
    out["corr_flat_all_dims"] = pearson(flat_p, flat_g)

    # L1 of model vs trivial baselines (mean abs error per dim, averaged)
    def l1(pred_rows):
        return sum(sum(abs(a - b) for a, b in zip(pr, gt)) / 7
                   for pr, gt in zip(pred_rows, G)) / n
    zero = [[0.0] * 7 for _ in range(n)]
    dmean = [sum(row[i] for row in G) / n for i in range(7)]
    meanpred = [list(dmean) for _ in range(n)]
    out["l1"] = {"model": l1(P), "constant_zero": l1(zero),
                 "predict_dataset_mean": l1(meanpred),
                 "dataset_mean_action": dmean}
    out["l1"]["model_over_zero"] = out["l1"]["model"] / out["l1"]["constant_zero"]
    out["l1"]["model_over_mean"] = out["l1"]["model"] / out["l1"]["predict_dataset_mean"]
    out["gate_model_beats_predict_mean"] = out["l1"]["model"] < out["l1"]["predict_dataset_mean"]

    # gripper sign agreement
    agree = sum(1 for row_p, row_g in zip(P, G)
                if (row_p[6] > 0) == (row_g[6] > 0))
    out["gripper_sign_agreement"] = agree / n

    # rollout + frame mismatch evidence
    if os.path.exists(ROLLOUT):
        out["rollout_public_openvla_libero_spatial"] = json.load(open(ROLLOUT))
    if os.path.exists(SANITY):
        s = json.load(open(SANITY))
        out["frame_mismatch_evidence"] = {
            "unnorm_key_used": s.get("unnorm_key_used"),
            "eval_suite": "LIBERO-90",
            "note": "Bridge-corpus action de-normalization statistics applied to LIBERO episodes.",
        }
    dest = os.path.join(ROOT, "results_v2", "decoder_audit.json")
    json.dump(out, open(dest, "w"), indent=1)
    print("wrote", dest)
    print(json.dumps({k: v for k, v in out.items() if k != "_provenance"}, indent=1))


if __name__ == "__main__":
    main()
