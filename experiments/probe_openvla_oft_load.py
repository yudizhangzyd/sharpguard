#!/usr/bin/env python3
"""Measure whether OpenVLA-OFT loads, instead of asserting that it does not.

Limitation (ix) of the manuscript says "OpenVLA-OFT failed to load cleanly in
our environment (Python 3.10, torch 2.2.0)" and "remains unloadable in our
environment". Unlike every other claim in the paper, that one has no released
artifact behind it: no OFT repo appears in license_report.json, and no probe
output exists. It is a sentence about a failure nobody can check, which is the
same defect as a caveat standing in for a measurement -- and it is a
self-serving one, since the failure is what excuses a coverage gap.

This probe fixes that. It records, per candidate checkpoint:

  * the resolved repo id and whether it exists at all on the Hub, queried from
    the API rather than written from memory (the same rule
    fetch_upstream_licenses.py follows after that list was wrong twice);
  * the exact exception type, message and traceback tail from the load attempt,
    in the environment the paper names;
  * the same, after upgrading transformers to a version that postdates OFT's
    release -- because "unloadable" is only an honest word if a newer stack was
    tried, and a stale pin is a fixable problem rather than a property of the
    checkpoint;
  * for any checkpoint that DOES load, a real forward pass that decodes an
    action. Importing without raising is not usability, and reporting a load as
    a success without generating anything is how a broken setup gets recorded
    as a working one.

The outcome is genuinely open, and both branches are useful: a load failure
becomes a measured, attributable fact with an exception in it, and a load
success removes a coverage gap.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path

HF_API = "https://huggingface.co/api/models/"
HF_SEARCH = "https://huggingface.co/api/models?search={q}&limit=50"

# The official OFT release's author namespace. Used to pick the upstream
# checkpoints out of the search results: the query also returns dozens of
# third-party re-finetunes, and "we could not load somebody's fork" would be a
# different, much weaker claim than the one limitation (ix) makes.
OFT_AUTHOR = "moojink"

# Fallback only, used when the Hub search is unreachable. The candidate list is
# normally RESOLVED (see resolve_candidates) rather than taken from here: an
# earlier hand-written asset list in this repo was wrong twice, which is why
# fetch_upstream_licenses.py queries the API instead. A probe reporting "failed
# to load" for a repo id that never existed would look like evidence while being
# a typo, so `exists` is checked and reported for every candidate either way.
FALLBACK_CANDIDATES = [
    "moojink/openvla-7b-oft-finetuned-libero-spatial",
    "moojink/openvla-7b-oft-finetuned-libero-object",
    "moojink/openvla-7b-oft-finetuned-libero-goal",
    "moojink/openvla-7b-oft-finetuned-libero-10",
]


def resolve_candidates(token: str | None) -> tuple[list[str], dict]:
    """Ask the Hub which OFT checkpoints exist, rather than recalling them."""
    meta: dict = {"source": "hub_search", "query": "openvla-oft"}
    req = urllib.request.Request(HF_SEARCH.format(q="openvla-oft"))
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            hits = json.loads(r.read().decode())
    except Exception as e:
        meta.update(source="fallback_list",
                    search_error=f"{type(e).__name__}: {e}")
        return list(FALLBACK_CANDIDATES), meta
    ids = [h.get("modelId") or h.get("id") for h in hits]
    meta["n_hits"] = len(ids)
    official = sorted(i for i in ids
                      if i and i.split("/")[0].lower() == OFT_AUTHOR)
    meta["n_official"] = len(official)
    meta["third_party_ignored"] = len(ids) - len(official)
    if not official:
        meta.update(source="fallback_list",
                    note="search returned no checkpoint under the official "
                         "author namespace")
        return list(FALLBACK_CANDIDATES), meta
    meta["resolved"] = official
    return official, meta



def hub_info(repo: str, token: str | None) -> dict:
    """What the Hub says about this repo. `exists` is the load-bearing field."""
    req = urllib.request.Request(HF_API + repo)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        # The Hub answers 401 for a gated repo AND for one that does not exist,
        # when the caller is unauthenticated -- so a bare "exists: false" here
        # would silently merge "never published" with "we lacked a token". Both
        # are reported, and the distinction is left visible rather than
        # collapsed, because only the first would justify limitation (ix).
        return {"exists": False, "http_status": e.code,
                "detail": str(e.reason),
                "ambiguous_401": e.code == 401 and not token,
                "note": ("401 without a token cannot distinguish gated from "
                         "absent; re-run with HF_TOKEN set"
                         if e.code == 401 and not token else None)}
    except Exception as e:                       # network, DNS, timeout
        return {"exists": None, "error": f"{type(e).__name__}: {e}"}
    files = [f.get("rfilename") for f in (d.get("siblings") or [])]
    return {
        "exists": True,
        "sha": d.get("sha"),
        "license": (d.get("cardData") or {}).get("license"),
        "n_files": len(files),
        "has_config_json": "config.json" in files,
        # The two facts that decide whether a stock transformers can load it at
        # all, both read off the file list rather than guessed.
        "has_remote_code": any(str(f).endswith(".py") for f in files),
        "weight_files": sorted(f for f in files
                               if str(f).endswith((".safetensors", ".bin")))[:6],
        "config_arch": None,
    }


def env_snapshot() -> dict:
    snap = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "timm", "tokenizers", "accelerate",
                "peft"):
        try:
            snap[mod] = __import__(mod).__version__
        except Exception as e:
            snap[mod] = f"<absent: {type(e).__name__}>"
    try:
        import torch
        snap["cuda_available"] = bool(torch.cuda.is_available())
        snap["gpu"] = (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else None)
        snap["sm"] = (".".join(map(str, torch.cuda.get_device_capability(0)))
                      if torch.cuda.is_available() else None)
    except Exception:
        snap["cuda_available"] = None
    return snap


def try_load(repo: str, dtype_name: str, do_forward: bool) -> dict:
    """One load attempt, with the failure recorded in full rather than summarized.

    Three stages, reported separately, because "does not load" is three
    different findings depending on where it stops: config resolution (the
    architecture is unknown to this transformers), weight instantiation (the
    stack is too old for the checkpoint format), and generation (it loads but
    cannot be used, which must not be reported as a success).
    """
    out = {"repo": repo, "stage_reached": "none", "ok": False}
    try:
        import torch
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
        out["stage_reached"] = "config"
        out["config_arch"] = getattr(cfg, "architectures", None)
        out["config_model_type"] = getattr(cfg, "model_type", None)
        # Recorded because the identity-scale question from P3 applies here too:
        # a checkpoint's norm_stats keys decide which corpora it can be scored on.
        ns = getattr(cfg, "norm_stats", None) or {}
        out["norm_stats_keys"] = sorted(ns)[:12]

        from transformers import AutoModelForVision2Seq, AutoProcessor
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[dtype_name]
        proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        out["stage_reached"] = "processor"
        model = AutoModelForVision2Seq.from_pretrained(
            repo, trust_remote_code=True, torch_dtype=dtype,
            low_cpu_mem_usage=True)
        out["stage_reached"] = "weights"
        out["n_params_billion"] = round(
            sum(p.numel() for p in model.parameters()) / 1e9, 3)

        if do_forward:
            import numpy as np
            from PIL import Image
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(dev).eval()
            img = Image.fromarray(
                np.zeros((224, 224, 3), dtype=np.uint8))
            prompt = ("In: What action should the robot take to pick up the "
                      "black bowl?\nOut:")
            inputs = proc(prompt, img).to(dev, dtype=dtype)
            with torch.no_grad():
                g = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            out["generated_token_ids"] = g[0, -8:].cpu().tolist()
            out["stage_reached"] = "generate"
            # A load that generates nothing usable is not a load. Reported, not
            # asserted -- the caller decides what it means.
            out["n_new_tokens"] = int(g.shape[1] - inputs["input_ids"].shape[1])
        out["ok"] = True
    except Exception as e:
        out["error_type"] = type(e).__name__
        out["error"] = str(e)[:1500]
        out["traceback_tail"] = traceback.format_exc()[-2500:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--stage", default="baseline",
                    choices=["baseline", "upgraded"],
                    help="which environment this invocation is measuring; the "
                         "runner calls it twice and merges")
    ap.add_argument("--no-forward", action="store_true",
                    help="skip the generation stage (config/weights only)")
    ap.add_argument("--repos", default="",
                    help="comma-separated override of the candidate list")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    if args.repos.strip():
        repos, resolution = [r.strip() for r in args.repos.split(",") if r.strip()], \
                            {"source": "--repos override"}
    else:
        repos, resolution = resolve_candidates(token)
    print(f"[oft-probe] candidates ({resolution.get('source')}): {repos}")

    hub = {r: hub_info(r, token) for r in repos}
    live = [r for r in repos if hub[r].get("exists")]
    print(f"[oft-probe] {len(live)}/{len(repos)} candidate repos exist: {live}")

    attempts = {}
    for r in live:
        print(f"[oft-probe] attempting load: {r}")
        attempts[r] = try_load(r, args.dtype, not args.no_forward)
        print(f"[oft-probe]   stage_reached={attempts[r]['stage_reached']} "
              f"ok={attempts[r]['ok']} "
              f"{attempts[r].get('error_type', '')}")
        # One successful load answers the question; the rest would only cost
        # GPU time to re-confirm it. Stated so the stop is not mistaken for a
        # silent cap.
        if attempts[r]["ok"]:
            print("[oft-probe] a checkpoint loaded; not attempting the "
                  "remaining candidates (one load settles the claim)")
            break

    n_ok = sum(1 for v in attempts.values() if v["ok"])
    report = {
        "probe": "openvla_oft_load",
        "stage": args.stage,
        "why": ("Limitation (ix) claims OpenVLA-OFT is unloadable in our "
                "environment. No artifact backed that claim; this measures it, "
                "in the environment the paper names and again on an upgraded "
                "stack, and records the exception rather than the conclusion."),
        "environment": env_snapshot(),
        "candidate_resolution": resolution,
        "n_candidates": len(repos),
        "n_candidates_existing": len(live),
        "n_loaded": n_ok,
        "any_loaded": bool(n_ok),
        "hub_info": hub,
        "load_attempts": attempts,
        "verdict": (
            "loads -- limitation (ix) must be revised, OFT is available"
            if n_ok else
            "does not load in this environment; see load_attempts[*].error_type"
            if live else
            "no candidate repo id resolves on the Hub, so the checkpoint we "
            "could not load may never have been public under these names; "
            "limitation (ix) must say that instead"),
    }
    d = Path(args.out)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"oft_load_probe_{args.stage}.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"[oft-probe] verdict: {report['verdict']}")
    # Exit 0 either way: a measured failure is the probe succeeding at its job.
    # The runner decides what to do with the verdict.
    return 0


if __name__ == "__main__":
    sys.exit(main())
