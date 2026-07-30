#!/usr/bin/env python3
"""Fetch the real license + provenance metadata for every upstream asset the
CoT-Faith release depends on.

Written because the datasheet has to state a license for each upstream model
and corpus, and guessing them is exactly the kind of unverifiable claim that
got the artifact section rejected. This queries the HF Hub API and records what
it actually finds -- including "no license tag", which is itself a fact the
datasheet must disclose rather than paper over.

Output: license_report.json, consumed by docs/DATASHEET.md and asserted by
scripts/verify_paper_numbers.py.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Every upstream asset the benchmark loads, as a repo id that some config or
# script in this repo actually resolves. An earlier version of this list was
# written from memory and got two things wrong, which is precisely the failure
# mode this script exists to prevent:
#
#   * it named Embodied-CoT/ecot-openvla-7b-oxe as "the LoRA base for our 7
#     variants". It is not. Every bolt/boltconfig-cotfaith-{lora-r*,data-50*,
#     calib-*}.yaml sets BASE_MODEL=Embodied-CoT/ecot-openvla-7b-bridge. The
#     -oxe checkpoint appears only in experiments/cotfaith_scout.py, a
#     feasibility probe that produced no reported number, so it is not a
#     dependency of this release and is not listed.
#   * it named two Embodied-CoT bridge repos for the cross-corpus sweep. Those
#     404/401 because the sweep does not use them: the three cross-corpus runs
#     load the IPEC-COMMUNITY LeRobot mirrors named below (see
#     bolt/boltconfig-cotfaith-ds-*.yaml:DATASET_REPO). We must record the
#     license of the artifact we actually pull, not of the original corpus it
#     was re-hosted from.
ASSETS = [
    # --- checkpoints we evaluate -------------------------------------------
    ("model", "openvla/openvla-7b", "OpenVLA base (arch reference)"),
    ("model", "openvla/openvla-7b-finetuned-libero-spatial", "non-CoT baseline + decoder gate"),
    ("model", "openvla/openvla-7b-finetuned-libero-object", "non-CoT baseline + decoder gate"),
    ("model", "openvla/openvla-7b-finetuned-libero-goal", "non-CoT baseline + decoder gate"),
    ("model", "openvla/openvla-7b-finetuned-libero-10", "non-CoT baseline + decoder gate"),
    ("model", "Embodied-CoT/ecot-openvla-7b-bridge",
     "public ECoT CoT-VLA (flagship row, 3 seeds) AND the LoRA base for our 7 variants"),
    ("model", "yinchenghust/deepthinkvla_base", "DeepThinkVLA base (attention only)"),
    ("model", "yinchenghust/deepthinkvla_libero_cot_sft", "DeepThinkVLA SFT (attention only)"),
    ("model", "yinchenghust/deepthinkvla_libero_cot_rl", "DeepThinkVLA RL (attention only)"),
    # --- corpora -----------------------------------------------------------
    ("dataset", "openvla/modified_libero_rlds", "LIBERO-90 observations (P1/P2 primary)"),
    ("dataset", "Embodied-CoT/embodied_features_and_demos_libero",
     "LIBERO CoT reasoning annotations -- our edited CoT strings are DERIVATIVE of this"),
    ("dataset", "IPEC-COMMUNITY/bridge_orig_lerobot",
     "cross-corpus sweep, Bridge V2 (N=30); LeRobot re-host we actually load"),
    ("dataset", "IPEC-COMMUNITY/fractal20220817_data_lerobot",
     "cross-corpus sweep, RT-1/Fractal (N=30); LeRobot re-host we actually load"),
    ("dataset", "IPEC-COMMUNITY/bc_z_lerobot",
     "cross-corpus sweep, BC-Z (N=30); LeRobot re-host we actually load"),
    ("dataset", "Embodied-CoT/embodied_features_bridge",
     "Bridge CoT annotations for the F4-deconfound subset training (scaffolded, "
     "no reported number depends on it)"),
]

API = "https://huggingface.co/api"


def fetch(kind: str, repo_id: str) -> dict:
    ns = "models" if kind == "model" else "datasets"
    url = f"{API}/{ns}/{repo_id}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "url": url}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "url": url}

    card = d.get("cardData") or {}
    tags = d.get("tags") or []
    lic_tags = [t.split("license:", 1)[1] for t in tags if t.startswith("license:")]
    return {
        "url": f"https://huggingface.co/{'' if kind == 'model' else 'datasets/'}{repo_id}",
        "license_cardData": card.get("license"),
        "license_tags": lic_tags,
        "license_name_cardData": card.get("license_name"),
        "license_link": card.get("license_link"),
        "gated": d.get("gated", False),
        "private": d.get("private", False),
        "sha": d.get("sha"),
        "lastModified": d.get("lastModified"),
        "downloads": d.get("downloads"),
    }


def resolve(info: dict) -> str:
    """Single authoritative license string, or an explicit unknown marker."""
    if "error" in info:
        return f"UNRESOLVED ({info['error']})"
    for key in ("license_cardData", "license_name_cardData"):
        v = info.get(key)
        if v:
            return str(v)
    if info.get("license_tags"):
        return info["license_tags"][0]
    return "NO LICENSE DECLARED UPSTREAM"


def main() -> int:
    out_dir = Path(os.environ.get("BOLT_ARTIFACT_DIR", "./artifacts")) / "licenses"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, unresolved = {}, []
    for kind, repo, use in ASSETS:
        info = fetch(kind, repo)
        lic = resolve(info)
        rows[repo] = {"kind": kind, "used_for": use, "license": lic, **info}
        flag = "" if not lic.startswith(("UNRESOLVED", "NO LICENSE")) else "  <-- DISCLOSE"
        print(f"{kind:8s} {repo:56s} {lic}{flag}")
        if flag:
            unresolved.append(repo)

    report = {
        "n_assets": len(ASSETS),
        "n_unresolved": len(unresolved),
        "unresolved": unresolved,
        "assets": rows,
        "note": ("The CoT-Faith release redistributes NO upstream weights or "
                 "observations. It ships code, our derived edited-CoT strings, "
                 "and per-sample logs; users pull upstream assets themselves "
                 "under the licenses recorded here. Assets marked UNRESOLVED or "
                 "NO LICENSE DECLARED are disclosed as such in the datasheet "
                 "rather than assigned a license we cannot verify."),
    }
    path = out_dir / "license_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\n{len(ASSETS) - len(unresolved)}/{len(ASSETS)} licenses resolved -> {path}")
    if unresolved:
        print("Unresolved (must be disclosed verbatim in the datasheet, not guessed):")
        for u in unresolved:
            print("  -", u)
    # Unresolved licenses are a disclosure item, not a build failure: the
    # datasheet is required to print them as unknown.
    return 0


if __name__ == "__main__":
    sys.exit(main())
