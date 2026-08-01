"""Measure the Bridge V2 reasoning <-> LeRobot trajectory join, CPU only.

Bolt `fnwfaq9bq6` (the F4/O4 deconfound LoRA run) died in preflight with
`no_reasoning: 606677` -- every frame of every parquet missed. Its own logs say
why, and the reason is structural rather than a coverage gap:

    reasoning top-level keys (first 4):
      ['/nfs/kun2/users/homer/datasets/bridge_data_all/numpy_256/
        bridge_data_v2/deepthought_folding_table/stack_blocks/19/train/out.npy',
       ...]
      raw['/nfs/.../out.npy'] subkeys: ['43', '11', '27', '17', ...]
    task-text index: 0 distinct tasks

The annotations are keyed by the *original authors' absolute NFS paths*, and the
second level is an episode index within that shard. So the JSON is three levels
deep -- path -> episode -> step -> reasoning -- and `_ReasoningIndex` assumed
two. `by_episode_index` compared `str(ep_idx)` against path strings and could
never hit; `by_task_text` then read a level-2 value (a dict of steps) as if it
were a reasoning dict, so `first.get("task")` was always None and the index came
out empty. Two strategies, one shared wrong assumption, which is why the
fallback did not save it.

This probe exists so the fix is written against measured structure instead of a
guess about it. A wrong guess costs a GPU job and half a day; this costs a CPU
pod and downloads the same 1.4 GB the training run would have.

It reports three things:

1.  The actual nesting: depth, key shapes at each level, and the leaf's field
    names. Everything the join needs to be written correctly.
2.  For each candidate join strategy, how many of the LeRobot side's episodes it
    matches -- against `meta/episodes.jsonl`, which is small, so no parquet or
    video is touched. Match *counts*, not a yes/no: a strategy that matches 40
    of 53192 is a different finding from one that matches 50000.
3.  A verbatim sample of matched and unmatched keys from both sides, so a zero
    stays diagnosable without a third run.

Deliberately makes no fix and trains nothing. Its output is the input to the
fix.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_file_with_retry():
    """Import `sharpguard.hf_retry` without importing `sharpguard`.

    A plain `from sharpguard.hf_retry import ...` runs the package __init__,
    which imports torch. This probe is CPU-only and its whole value is being
    cheap to run and cheap to check, so it loads the one module it needs by
    path. That also keeps the structure walkers unit-testable on a laptop.
    """
    path = _ROOT / "sharpguard" / "hf_retry.py"
    spec = importlib.util.spec_from_file_location("_hf_retry", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.file_with_retry

REASONING_REPO = "Embodied-CoT/embodied_features_bridge"
REASONING_FILE = "embodied_features_bridge.json"
DATASET_REPO = "IPEC-COMMUNITY/bridge_orig_lerobot"


def norm_task(s) -> str:
    """Same normalization the trainer's `_norm_task` uses, kept in sync by eye.

    Not imported from cotfaith_train_bridge.py on purpose: that module imports
    torch and the LIBERO recipe at module scope, and this probe runs on a CPU
    pod where neither needs to be present.
    """
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def describe_level(d, name, n=6):
    """Key shapes at one level, without dumping the level itself."""
    if not isinstance(d, dict):
        return {"level": name, "type": type(d).__name__,
                "value_sample": str(d)[:200]}
    keys = list(d.keys())
    return {
        "level": name,
        "type": "dict",
        "n_keys": len(keys),
        "key_sample": [str(k)[:160] for k in keys[:n]],
        "all_digit_keys": all(str(k).isdigit() for k in keys),
        "value_types": dict(Counter(type(v).__name__ for v in d.values())),
    }


def probe_structure(raw: dict) -> dict:
    """Walk one spine of the JSON, recording what each level actually is."""
    out = {"levels": [], "leaf_fields": None, "leaf_sample": None}
    out["levels"].append(describe_level(raw, "top"))

    cur, names = raw, ["top"]
    # Four is one more than the three levels the logs imply, so an unexpected
    # extra wrapper shows up as data rather than as a silent truncation.
    for depth in range(4):
        if not isinstance(cur, dict) or not cur:
            break
        k0 = next(iter(cur))
        nxt = cur[k0]
        names.append(f"L{depth + 1}")
        out["levels"].append(describe_level(nxt, names[-1]))
        if isinstance(nxt, dict) and nxt:
            v0 = next(iter(nxt.values()))
            # A leaf is a dict whose values are no longer dicts -- that is the
            # reasoning record itself.
            if not isinstance(v0, dict):
                out["leaf_fields"] = sorted(nxt.keys())
                out["leaf_sample"] = {k: str(v)[:300]
                                      for k, v in list(nxt.items())[:12]}
                break
        cur = nxt
    out["depth_walked"] = len(out["levels"])
    return out


def collect_leaves(raw: dict, limit=None):
    """Yield (path_key, ep_key, step_key, record) over the 3-level layout.

    Tolerant of the layout being 2-level for some entries: the ECoT release is
    a concatenation of several export runs and nothing promises uniformity.
    """
    n = 0
    for pkey, pv in raw.items():
        if not isinstance(pv, dict):
            continue
        for ekey, ev in pv.items():
            if not isinstance(ev, dict):
                continue
            vals = list(ev.values())
            if vals and isinstance(vals[0], dict):
                for skey, rec in ev.items():
                    if isinstance(rec, dict):
                        yield pkey, ekey, skey, rec
                        n += 1
                        if limit and n >= limit:
                            return
            else:
                yield pkey, ekey, "0", ev
                n += 1
                if limit and n >= limit:
                    return


def load_lerobot_tasks(fetch) -> dict:
    """episode_index -> instruction, from meta/episodes.jsonl (small)."""
    out = {}
    for rel in ("meta/episodes.jsonl", "meta/episodes.json"):
        try:
            p = fetch(DATASET_REPO, rel, repo_type="dataset")
        except Exception as e:
            print(f"[probe] {rel}: {type(e).__name__}: {e}")
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = d.get("tasks")
                if isinstance(t, list):
                    t = t[0] if t else ""
                if d.get("episode_index") is not None:
                    out[int(d["episode_index"])] = str(t or "")
        if out:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-leaves", type=int, default=0,
                    help="0 = walk every leaf")
    args = ap.parse_args()
    fetch = _load_file_with_retry()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"probe": "bridge_reasoning_join",
              "why": "bolt fnwfaq9bq6 joined 0 of 606677 frames; measure the "
                     "real key layout before rewriting the join",
              "reasoning_repo": REASONING_REPO,
              "dataset_repo": DATASET_REPO}

    def dump():
        (out_dir / "bridge_join_probe.json").write_text(
            json.dumps(report, indent=1, sort_keys=True))

    # Written after every phase, so a mid-probe 429 still leaves the phases that
    # completed. fnwfaq9bq6's sibling failures were all "died before writing".
    try:
        p = fetch(REASONING_REPO, REASONING_FILE, repo_type="dataset")
        report["reasoning_local_bytes"] = Path(p).stat().st_size
        dump()
        with open(p) as f:
            raw = json.load(f)
    except Exception as e:
        report["fatal"] = f"reasoning load: {type(e).__name__}: {e}"
        dump()
        return 1

    report["structure"] = probe_structure(raw)
    dump()

    # ---- the annotation side, counted rather than assumed ----
    leaves = list(collect_leaves(raw, args.max_leaves or None))
    report["n_leaf_records"] = len(leaves)
    report["n_path_keys"] = len(raw)
    report["n_episode_keys_total"] = sum(
        len(v) for v in raw.values() if isinstance(v, dict))

    field_counts = Counter()
    for _, _, _, rec in leaves:
        field_counts.update(rec.keys())
    report["leaf_field_frequency"] = dict(field_counts.most_common(30))

    # Which leaf field carries the instruction? Named explicitly, because the
    # by_task_text strategy hinges on it and the old code guessed two names.
    task_fields = {}
    for cand in ("task", "instruction", "language_instruction", "prompt"):
        vals = [rec.get(cand) for _, _, _, rec in leaves[:5000]
                if isinstance(rec.get(cand), str) and rec.get(cand).strip()]
        if vals:
            task_fields[cand] = {"n_nonempty_in_first_5000": len(vals),
                                 "sample": vals[:5]}
    report["task_field_candidates"] = task_fields
    dump()

    tasks = load_lerobot_tasks(fetch)
    report["n_lerobot_episodes"] = len(tasks)
    report["lerobot_task_sample"] = [tasks[k] for k in sorted(tasks)[:5]]
    dump()

    # ---- candidate strategies, each scored by episodes matched ----
    lero_norm = {}
    for ep, t in tasks.items():
        lero_norm.setdefault(norm_task(t), []).append(ep)
    report["n_distinct_lerobot_tasks"] = len(lero_norm)

    ann_tasks = {}
    for pkey, ekey, _skey, rec in leaves:
        t = norm_task(rec.get("task") or rec.get("instruction"))
        if t:
            ann_tasks.setdefault(t, set()).add((pkey, ekey))
    report["n_distinct_annotation_tasks"] = len(ann_tasks)

    shared = set(ann_tasks) & set(lero_norm)
    report["strategies"] = {
        "by_episode_index": {
            "n_top_level_digit_keys": sum(1 for k in raw if str(k).isdigit()),
            "verdict": "cannot match: top-level keys are absolute NFS paths",
        },
        "by_task_text": {
            "n_shared_normalized_tasks": len(shared),
            "n_lerobot_episodes_covered": sum(
                len(lero_norm[t]) for t in shared),
            "frac_lerobot_episodes_covered": round(
                sum(len(lero_norm[t]) for t in shared)
                / max(1, len(tasks)), 4),
            "shared_sample": sorted(shared)[:10],
            "annotation_only_sample": sorted(set(ann_tasks) - set(lero_norm))[:10],
            "lerobot_only_sample": sorted(set(lero_norm) - set(ann_tasks))[:10],
        },
    }
    # The number that decides whether the deconfound is runnable at all: how
    # many distinct (task -> annotated episode) pairs are reachable. 4k
    # trajectories were requested; anything under that bounds the run.
    report["n_reachable_annotated_episodes"] = sum(
        len(ann_tasks[t]) for t in shared)
    report["verdict"] = (
        "by_task_text is viable"
        if report["strategies"]["by_task_text"]["n_shared_normalized_tasks"]
        else "no strategy matches; the two exports do not share task strings")
    dump()
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("structure", "leaf_field_frequency")},
                     indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
