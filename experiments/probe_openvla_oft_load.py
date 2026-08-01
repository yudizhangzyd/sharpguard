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
import importlib.util
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
    # `prismatic` is the package OFT's remote code imports, and its absence is
    # the entire content of the first two stages' failure ("requires the
    # following packages that were not found in your environment: prismatic").
    # Recording where it came from matters: PyPI has an unrelated project of
    # that name, so "prismatic is installed" and "the OFT codebase is
    # installed" are different facts and only the second one can satisfy the
    # import.
    try:
        import prismatic
        snap["prismatic"] = getattr(prismatic, "__version__", "<no __version__>")
        snap["prismatic_path"] = getattr(prismatic, "__file__", None)
        snap["prismatic_has_vla"] = hasattr(prismatic, "load_vla") or bool(
            importlib.util.find_spec("prismatic.extern.hf.modeling_prismatic"))
    except Exception as e:
        snap["prismatic"] = f"<absent: {type(e).__name__}>"
        snap["prismatic_path"] = None
        snap["prismatic_has_vla"] = False
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


def _resolver():
    """`missing_module` and `pip` from bolt/install_prismatic.py, by path.

    Loaded rather than reimplemented: `missing_module` is the function that
    round 2 had to fix (it must search the whole chained traceback, not just the
    top-level message), and a second copy here would be a second thing to get
    wrong the same way.
    """
    path = Path(__file__).resolve().parent.parent / "bolt" / "install_prismatic.py"
    spec = importlib.util.spec_from_file_location("_install_prismatic", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# A compiled extension that is already imported cannot be replaced in-process.
# Round 4 (bolt `s6arkzytjr`) installed pydantic-core==2.46.4, called
# `importlib.invalidate_caches()`, and got the byte-identical SystemError, on
# every checkpoint. The only fix is a new interpreter, so the probe restarts
# itself -- bounded, because an install that does not actually help would
# otherwise restart forever, and each restart re-downloads nothing but does
# re-query the Hub.
_REEXEC_ENV = "OFT_PROBE_REEXEC"
_CARRY_ENV = "OFT_PROBE_CARRY"
_MAX_REEXEC = 2

# What the pre-exec process had already installed. Carried through the exec so
# the report still answers "what would supporting OFT cost" with the full list:
# the installs happened, and a report that lost them would understate the cost
# and could re-attempt the same package on every restart.
try:
    _CARRIED = json.loads(os.environ.get(_CARRY_ENV) or "[]")
    if not isinstance(_CARRIED, list):
        _CARRIED = []
except ValueError:
    _CARRIED = []


def _reexec(reason: str, installs: list) -> None:
    """Restart this probe in a fresh interpreter. Returns only if it cannot.

    Not an error path: the install succeeded and the environment is now correct
    on disk. What is stale is this process.
    """
    n = int(os.environ.get(_REEXEC_ENV, "0"))
    if n >= _MAX_REEXEC:
        # Said out loud rather than silently continuing: if two restarts did not
        # clear the conflict, the remaining failure is the finding, and it must
        # not be reported as if no restart had been attempted.
        print(f"[oft-probe] re-exec budget spent ({n}/{_MAX_REEXEC}); "
              f"reporting the conflict instead: {reason}")
        sys.stdout.flush()
        return
    env = dict(os.environ)
    env[_REEXEC_ENV] = str(n + 1)
    # Bounded: the environment block is not a database, and the last thing this
    # probe should die of is E2BIG while reporting somebody else's ImportError.
    carry = [{k: v for k, v in i.items() if k != "pip_tail"} for i in installs]
    env[_CARRY_ENV] = json.dumps(carry)[:8000]
    print(f"[oft-probe] re-exec {n + 1}/{_MAX_REEXEC}: {reason}")
    sys.stdout.flush()
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def try_load_resolving(repo: str, dtype_name: str, do_forward: bool,
                       max_installs: int = 12) -> dict:
    """`try_load`, retried while the failure is a nameable missing module.

    Round 2 of this probe (bolt `fshsqxp53m`) got prismatic importing -- the
    dictConfig wall came down and `prismatic_has_vla` went True -- and then died
    one layer further in:

        ModuleNotFoundError: No module named 'jsonlines'

    on four of five checkpoints. That is the same finding as round 1 for the
    third time: the blocker is a package name printed in the exception. The
    reason it recurred is that `bolt/install_prismatic.py` resolves imports of
    *its* target, `prismatic.extern.hf.modeling_prismatic`, and stops there --
    but `from_pretrained(trust_remote_code=True)` executes the checkpoint's own
    remote code, which imports things prismatic itself does not.

    So the retry loop belongs here, at the import site, not only around the
    prismatic target. Bounded and de-duplicated: a name that reappears after its
    install means installing it did not help, and looping on it would burn the
    budget instead of reporting the real blocker, so it is recorded in
    `stuck_on` and the loop stops.

    Installs used to be `--no-deps`, for the reason install_prismatic still is
    for the prismatic tree itself: the pins ARE the environment the paper names,
    and a transitive upgrade would silently make the result describe a different
    stack. Every install is recorded either way, so the output still answers
    "what would supporting OFT cost" rather than just yes/no.

    Round 3 (bolt `gvvhgg4d4c`) showed the cost of that policy. With prismatic
    resolved, all five checkpoints reached the *processor* stage -- further than
    any prior round -- and failed on a pydantic/pydantic-core skew that this
    loop's own `--no-deps` install of pydantic_core had created.

    Round 4 (bolt `s6arkzytjr`) showed that repairing the skew here does not work
    and that `--no-deps` was the wrong policy, not a fixable one:

      * the repair installed pydantic-core==2.46.4 successfully and the identical
        `SystemError` came back, because `pydantic_core` is a compiled extension
        that was already imported -- `invalidate_caches()` cannot swap it. So a
        skew that survives its own repair now triggers a RE-EXEC (`_reexec`),
        which is the only way to pick up a compiled module from disk;
      * the one stage that got past pydantic died on `AttributeError: partially
        initialized module 'wandb' has no attribute 'errors' (most likely due to
        a circular import)` -- what wandb does when its requirements are absent.
        That is `--no-deps` producing a partial install, so installs now go
        through `install_prismatic.install`: deps resolved normally, the paper's
        four pins re-asserted afterwards and any movement reported.

    Both failures looked like OFT incompatibilities and neither was one, which is
    the whole reason this probe exists rather than a sentence in limitation (ix).
    """
    res = _resolver()
    installs, seen, repairs = list(_CARRIED), set(), set()
    for c in _CARRIED:                  # do not repeat work a prior exec did
        if c.get("import"):
            seen.add(c["import"])
        if c.get("repair") == "version_skew":
            repairs.add(str(c.get("pip", "")).split("==")[0])
    out = try_load(repo, dtype_name, do_forward)
    while not out.get("ok") and len(installs) < max_installs:
        blob = f"{out.get('error') or ''}\n{out.get('traceback_tail') or ''}"
        mod = res.missing_module(blob)
        if not mod:
            # Not a missing module. Before reporting it as the blocker, check
            # whether it is a version skew THIS LOOP caused: a `--no-deps`
            # install satisfies the import and can leave a coupled pair
            # mismatched, which is what stopped round 3 (bolt `gvvhgg4d4c`) at
            # the processor stage on all 5 checkpoints. Pin to the version the
            # exception names, once per package -- if it recurs after the repair
            # the skew is not ours and belongs in the report.
            skew = res.version_skew(blob)
            if skew and skew[0] not in repairs:
                pkg, want = skew
                repairs.add(pkg)
                ok, tail = res.pip("install", "--quiet", "--no-deps",
                                   f"{pkg}=={want}")
                installs.append({"import": pkg, "pip": f"{pkg}=={want}",
                                 "ok": ok, "repair": "version_skew",
                                 "pip_tail": None if ok else tail[-400:]})
                if not ok:
                    out["stuck_on"] = f"{pkg}=={want}"
                    break
                # The on-disk fix is in place, but this process may already hold
                # the wrong copy. If the module is a compiled extension there is
                # no in-process route at all -- round 4 proved that by trying --
                # so hand the fix to a fresh interpreter. `_reexec` returns only
                # when it has run out of attempts, and then the loop reports the
                # skew as the blocker rather than pretending to have fixed it.
                _reexec(f"{pkg}=={want} installed but {pkg} is already loaded",
                        installs)
                importlib.invalidate_caches()
                out = try_load(repo, dtype_name, do_forward)
                continue
            break                       # genuinely not resolvable here: report it
        if mod in seen:
            out["stuck_on"] = mod
            break
        seen.add(mod)
        pkg = res.IMPORT_TO_PIP.get(mod, mod)
        # WITH its dependencies (see `install`): round 4's wandb circular-import
        # was a partial install of our own making, and the pins are protected by
        # re-asserting them after the fact rather than by starving the package.
        rec = res.install(pkg)
        rec["import"] = mod
        installs.append(rec)
        if not rec["ok"]:
            out["stuck_on"] = mod
            break
        importlib.invalidate_caches()
        out = try_load(repo, dtype_name, do_forward)
    out["dependency_installs"] = installs
    out["reexec_count"] = int(os.environ.get(_REEXEC_ENV, "0"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolve-deps", action="store_true",
                    help="on a missing-module failure, pip-install the named "
                         "module (--no-deps) and retry the load. Off by "
                         "default: stages 1 and 2 must measure the environment "
                         "the paper names, and installing into them would make "
                         "the result describe a different stack.")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--stage", default="baseline",
                    choices=["baseline", "upgraded",
                             "prismatic_baseline", "prismatic_upgraded"],
                    help="which environment this invocation is measuring; the "
                         "runner calls it once per stage and the reports merge. "
                         "The prismatic_* stages exist because the first two "
                         "both failed with an ImportError naming a package we "
                         "had simply not installed -- a missing dependency is "
                         "not an incompatibility, and limitation (ix) cannot "
                         "cite one as the other")
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
        # Resolution is opt-in per stage, not always on. Stages 1 and 2 must
        # measure the environment the paper names, and pip-installing into them
        # would make their result describe a different stack -- the baseline is
        # what keeps the claim falsifiable. Stages 3 and 4 already exist to
        # resolve dependencies, so that is where the retry belongs.
        loader = try_load_resolving if args.resolve_deps else try_load
        attempts[r] = loader(r, args.dtype, not args.no_forward)
        inst = attempts[r].get("dependency_installs") or []
        print(f"[oft-probe]   stage_reached={attempts[r]['stage_reached']} "
              f"ok={attempts[r]['ok']} "
              f"{attempts[r].get('error_type', '')}"
              + (f" installs={[i['import'] for i in inst]}" if inst else "")
              + (f" stuck_on={attempts[r]['stuck_on']}"
                 if attempts[r].get("stuck_on") else ""))
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
        # How many times the probe restarted itself to pick up a compiled module
        # it had just installed (see `_reexec`). A nonzero value here is what
        # separates round 5 from round 4, whose in-process repair could not work.
        "reexec_count": int(os.environ.get(_REEXEC_ENV, "0")),
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
