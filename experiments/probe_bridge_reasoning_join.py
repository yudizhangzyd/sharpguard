"""Measure the Bridge V2 reasoning <-> LeRobot trajectory join, CPU only.

Bolt `fnwfaq9bq6` (the F4/O4 deconfound LoRA run) died in preflight with
`no_reasoning: 606677` -- every frame of every parquet missed. Round 1 of this
probe (bolt `j2kqu7k3m7`) settled the structural question and, in the process,
showed that its own strategy scores were void. Both facts are recorded here
because the second is the more instructive one.

THE MEASURED LAYOUT (round 1, `bridge_join_probe.json`, 1.4 GB / 3200 path keys):

    raw[<authors' absolute NFS path>][<episode index within that shard>] = {
        "features":  {"move_primitive": [...], "gripper_position": [...],
                      "bboxes": [...]},          # per-step lists
        "metadata":  {"episode_id": int, "file_path": str, "n_steps": int,
                      "language_instruction": str},
        "reasoning": {"0": {...}, "1": {...}, ...},   # per-step, keyed by index
    }

Four levels, not three. `leaf_field_frequency` is what pins it down:
`episode_id`, `file_path`, `language_instruction` and `n_steps` each appear
exactly 60062 times, which is exactly the total number of episode keys -- so they
are per-episode metadata, one level above the per-step reasoning.

WHY ROUND 1's MATCH COUNTS WERE WORTHLESS. `_ReasoningIndex` assumed two levels,
so `by_episode_index` compared `str(ep_idx)` against path strings and
`by_task_text` read a dict-of-steps as a reasoning record. Round 1 diagnosed
that correctly and then made the same class of mistake one level down: it scored
`by_episode_index` against the *top-level* keys (which are paths, so 0 by
construction) and `by_task_text` against `rec["task"]`/`rec["instruction"]`,
neither of which exists -- the field is `language_instruction`. It reported
`n_distinct_annotation_tasks: 0` and `verdict: no strategy matches`, and that
verdict was an artifact of the probe rather than a fact about the data.

The lesson is the one this file was written to apply and did not apply far
enough: do not score a join against a *guess* about where its key lives. So this
round reads the known layout directly instead of walking generically, `assert`s
the shape it was told to expect and records every episode that deviates, and
scores three strategies against fields that were measured to exist:

  1. `metadata.episode_id` vs LeRobot `episode_index` -- an exact key if the
     LeRobot conversion preserved the upstream episode ordering.
  2. `metadata.language_instruction` vs LeRobot task text, normalized. A
     fallback, and a weak one: Bridge instructions are crowdsourced and the
     LeRobot side contains `3wsws` and `12345678` among its 19542 distinct
     "tasks", so text collisions are expected to be many-to-many.
  3. `metadata.file_path` vs whatever provenance the LeRobot conversion kept.
     Reported as the full key set of a LeRobot episode record, because if that
     conversion stored the source path this is an exact join and the other two
     are unnecessary.

Deliberately makes no fix and trains nothing. Its output is the input to the fix.
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

# The L2 keys round 1 measured. Named rather than discovered, so an export whose
# shape differs shows up as a counted deviation instead of as a silent zero --
# which is the failure mode both round 1 and the trainer had.
EXPECTED_L2 = {"features", "metadata", "reasoning"}

# The eight tags experiments/cotfaith_train.py::build_ecot_target_text renders,
# with its alias lists, duplicated here rather than imported -- that module pulls
# in torch and the LIBERO recipe at import time and this probe runs on a CPU pod.
# Duplicated deliberately and narrowly: the probe's job is to report which of
# these the Bridge export can actually fill, and the answer is worthless if it is
# computed against a different tag list than the trainer uses.
ECOT_TAG_ALIASES = [
    ("TASK", ("task",)),
    ("PLAN", ("plan",)),
    ("VISIBLE OBJECTS", ("bboxes",)),
    ("SUBTASK REASONING", ("subtask_reasoning", "subtask_reason")),
    ("SUBTASK", ("subtask",)),
    ("MOVE REASONING", ("movement_reasoning", "move_reasoning", "move_reason")),
    ("MOVE", ("movement", "move")),
    ("GRIPPER POSITION", ("gripper", "gripper_position")),
]


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
    # Five is one more than the four levels round 1 measured, so an unexpected
    # extra wrapper shows up as data rather than as a silent truncation.
    for depth in range(5):
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


def collect_episodes(raw: dict, limit=None):
    """Yield (path_key, ep_key, episode_dict) over the measured 4-level layout.

    One record per *episode*, not per step: the metadata that carries every
    candidate join key lives at this level, and round 1 flattened past it.
    """
    n = 0
    for pkey, pv in raw.items():
        if not isinstance(pv, dict):
            continue
        for ekey, ev in pv.items():
            if not isinstance(ev, dict):
                continue
            yield pkey, ekey, ev
            n += 1
            if limit and n >= limit:
                return


def episode_facts(raw: dict, limit=None) -> dict:
    """Per-episode join keys plus an accounting of every shape deviation.

    Returns counted deviations rather than raising on them: the ECoT release is
    a concatenation of several export runs and nothing promises uniformity, so
    "how many episodes are not the shape we expect" is itself a number the fix
    needs. It only becomes a failure if it is large.
    """
    facts = []
    dev = Counter()
    for pkey, ekey, ev in collect_episodes(raw, limit):
        keys = set(ev.keys())
        if keys != EXPECTED_L2:
            dev["l2_keys:" + ",".join(sorted(keys))[:80]] += 1
        md = ev.get("metadata")
        if not isinstance(md, dict):
            dev["metadata_missing_or_not_dict"] += 1
            md = {}
        rz = ev.get("reasoning")
        n_steps_reasoning = len(rz) if isinstance(rz, dict) else 0
        if not n_steps_reasoning:
            dev["reasoning_empty_or_missing"] += 1
        facts.append({
            "path": pkey,
            "ep_key": ekey,
            "episode_id": md.get("episode_id"),
            "file_path": md.get("file_path"),
            "n_steps": md.get("n_steps"),
            "instruction": md.get("language_instruction"),
            "n_steps_reasoning": n_steps_reasoning,
        })
    return {"facts": facts, "deviations": dict(dev.most_common(20))}


def renderable_tags(raw: dict, limit=4000) -> dict:
    """Which of the trainer's 8 CoT tags can the Bridge export actually fill?

    The reason this exists is a requirement that only became visible once the
    layout was known. The LIBERO reasoning export the trainer was written
    against carries all eight tags in ONE per-step dict. The Bridge export
    splits them: `features` holds bboxes / gripper_position / move_primitive as
    per-step *lists*, and `reasoning` holds per-step *dicts* of something else.
    So `build_ecot_target_text(reasoning[step])` cannot render a full trace here
    no matter how the join is keyed -- the trainer has to merge the two subtrees
    per step, and until now nobody had measured what is in either half.

    Reported three ways, because the difference is the whole point: what the
    reasoning subtree alone resolves, what features alone resolves, and what the
    merge resolves. A tag that no source fills is a tag the Bridge CoT will
    render empty, and that is a fact the deconfound's write-up needs whether or
    not it changes the plan.
    """
    step_fields = Counter()
    step_samples: dict = {}
    n_steps_seen = 0
    from_reasoning = Counter()
    from_features = Counter()
    from_merge = Counter()

    for _pkey, _ekey, ev in collect_episodes(raw, None):
        rz, ft = ev.get("reasoning"), ev.get("features")
        if not isinstance(rz, dict):
            continue
        fkeys = set(ft) if isinstance(ft, dict) else set()
        for skey in sorted(rz, key=lambda s: int(s) if str(s).isdigit() else 0):
            rec = rz[skey]
            if not isinstance(rec, dict):
                continue
            step_fields.update(rec.keys())
            for k, v in rec.items():
                step_samples.setdefault(k, str(v)[:220])
            for tag, aliases in ECOT_TAG_ALIASES:
                in_r = any(a in rec for a in aliases)
                in_f = any(a in fkeys for a in aliases)
                if in_r:
                    from_reasoning[tag] += 1
                if in_f:
                    from_features[tag] += 1
                if in_r or in_f:
                    from_merge[tag] += 1
            n_steps_seen += 1
            if limit and n_steps_seen >= limit:
                break
        if limit and n_steps_seen >= limit:
            break

    def frac(c):
        return {tag: round(c.get(tag, 0) / max(1, n_steps_seen), 4)
                for tag, _ in ECOT_TAG_ALIASES}

    merged = frac(from_merge)
    return {
        "n_reasoning_steps_inspected": n_steps_seen,
        "reasoning_step_field_frequency": dict(step_fields.most_common(30)),
        "reasoning_step_field_samples": step_samples,
        "tag_fill_rate_from_reasoning_only": frac(from_reasoning),
        "tag_fill_rate_from_features_only": frac(from_features),
        "tag_fill_rate_from_merge": merged,
        "tags_unfillable_by_either": [t for t, v in merged.items() if v == 0.0],
        "merge_required": any(
            frac(from_reasoning)[t] == 0.0 and frac(from_features)[t] > 0.0
            for t, _ in ECOT_TAG_ALIASES),
    }


def load_lerobot_episodes(fetch) -> tuple[dict, list]:
    """episode_index -> record, from meta/episodes.jsonl (small).

    The whole record is kept, not just the instruction. Round 1 kept only the
    task text and therefore could not answer the one question that would settle
    the join outright: does the LeRobot conversion carry the upstream source
    path? That is a field-name question, and it is answered by reporting the key
    set rather than by guessing at it.
    """
    out, keysets = {}, Counter()
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
                if d.get("episode_index") is None:
                    continue
                keysets[",".join(sorted(d.keys()))] += 1
                out[int(d["episode_index"])] = d
        if out:
            break
    return out, keysets.most_common(5)


def lerobot_task(rec) -> str:
    t = rec.get("tasks")
    if isinstance(t, list):
        t = t[0] if t else ""
    return str(t or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-episodes", type=int, default=0,
                    help="0 = walk every episode")
    args = ap.parse_args()
    fetch = _load_file_with_retry()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"probe": "bridge_reasoning_join",
              "round": 2,
              "why": "bolt fnwfaq9bq6 joined 0 of 606677 frames; round 1 "
                     "(j2kqu7k3m7) measured the 4-level layout but scored both "
                     "strategies against fields that do not exist, so its "
                     "'no strategy matches' verdict was void. This round scores "
                     "the three keys measured to be present.",
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
    report["n_path_keys"] = len(raw)
    report["n_episode_keys_total"] = sum(
        len(v) for v in raw.values() if isinstance(v, dict))
    dump()

    # ---- the annotation side, at the level the join keys actually live ----
    ef = episode_facts(raw, args.max_episodes or None)
    facts, report["shape_deviations"] = ef["facts"], ef["deviations"]
    report["n_episodes_walked"] = len(facts)
    for field in ("episode_id", "file_path", "instruction", "n_steps"):
        present = [f for f in facts if f[field] not in (None, "")]
        report[f"n_with_{field}"] = len(present)
    report["episode_facts_sample"] = [
        {k: (str(v)[:120] if isinstance(v, str) else v)
         for k, v in f.items()} for f in facts[:5]]
    report["n_annotated_steps_total"] = sum(f["n_steps_reasoning"]
                                           for f in facts)
    dump()

    # Can the trainer's renderer even be fed from this export? Measured before
    # the join is scored, because a perfect join into an unrenderable trace is
    # still a dead end, and this is the phase that would answer it.
    report["renderability"] = renderable_tags(raw)
    dump()

    lero, lero_keysets = load_lerobot_episodes(fetch)
    report["n_lerobot_episodes"] = len(lero)
    # THE field-name question: if this key set contains a source path, the join
    # is exact and neither heuristic below is needed.
    report["lerobot_episode_keysets"] = lero_keysets
    report["lerobot_episode_sample"] = [
        {k: str(v)[:160] for k, v in lero[k0].items()}
        for k0 in sorted(lero)[:3]]
    dump()

    # ---- strategy 1: metadata.episode_id vs LeRobot episode_index ----
    ep_ids = {}
    for f in facts:
        if isinstance(f["episode_id"], int):
            ep_ids.setdefault(f["episode_id"], []).append((f["path"], f["ep_key"]))
    collisions = {k: len(v) for k, v in ep_ids.items() if len(v) > 1}
    hit = sorted(set(ep_ids) & set(lero))
    # A shared integer key is necessary but not sufficient: two exports can both
    # number episodes from 0 and mean different episodes. So agreement of the
    # instruction text ON THE MATCHED PAIRS is the confirmation, and it is the
    # number that decides whether this strategy is trustworthy at all.
    by_id = {}
    for f in facts:
        if isinstance(f["episode_id"], int):
            by_id.setdefault(f["episode_id"], f)
    agree = sum(1 for k in hit
                if norm_task(by_id[k]["instruction"])
                == norm_task(lerobot_task(lero[k])))
    report["strategies"] = {
        "by_episode_id": {
            "n_annotation_ids": len(ep_ids),
            "n_id_collisions": len(collisions),
            "id_range": [min(ep_ids), max(ep_ids)] if ep_ids else None,
            "n_matched_lerobot_episodes": len(hit),
            "frac_lerobot_matched": round(len(hit) / max(1, len(lero)), 4),
            "n_instruction_agrees_on_matched": agree,
            "frac_instruction_agrees_on_matched": (
                round(agree / len(hit), 4) if hit else None),
            "verdict": ("exact key, confirmed by instruction agreement"
                        if hit and agree and agree / len(hit) >= 0.95 else
                        "ids overlap but the instructions disagree, so the two "
                        "exports number different episodes -- do NOT join on this"
                        if hit else
                        "no shared integer id"),
        },
    }
    dump()

    # ---- strategy 2: instruction text ----
    lero_norm = {}
    for ep, rec in lero.items():
        lero_norm.setdefault(norm_task(lerobot_task(rec)), []).append(ep)
    lero_norm.pop("", None)
    ann_norm = {}
    for f in facts:
        t = norm_task(f["instruction"])
        if t:
            ann_norm.setdefault(t, []).append((f["path"], f["ep_key"]))
    shared = set(ann_norm) & set(lero_norm)
    report["n_distinct_lerobot_tasks"] = len(lero_norm)
    report["n_distinct_annotation_tasks"] = len(ann_norm)
    report["strategies"]["by_task_text"] = {
        "n_shared_normalized_tasks": len(shared),
        "n_lerobot_episodes_covered": sum(len(lero_norm[t]) for t in shared),
        "frac_lerobot_episodes_covered": round(
            sum(len(lero_norm[t]) for t in shared) / max(1, len(lero)), 4),
        # The reason this is a fallback and not a join: Bridge instructions are
        # crowdsourced, so one string maps to many episodes on both sides and a
        # match is a task-level match, not an episode-level one.
        "max_lerobot_episodes_per_shared_task": max(
            (len(lero_norm[t]) for t in shared), default=0),
        "median_lerobot_episodes_per_shared_task": (
            sorted(len(lero_norm[t]) for t in shared)[len(shared) // 2]
            if shared else 0),
        "shared_sample": sorted(shared)[:10],
        "annotation_only_sample": sorted(set(ann_norm) - set(lero_norm))[:10],
        "lerobot_only_sample": sorted(set(lero_norm) - set(ann_norm))[:10],
    }
    report["n_reachable_annotated_episodes"] = sum(
        len(ann_norm[t]) for t in shared)

    st = report["strategies"]
    report["verdict"] = (
        "join on metadata.episode_id"
        if st["by_episode_id"]["verdict"].startswith("exact")
        else f"fall back to task text: {st['by_task_text']['n_shared_normalized_tasks']} "
             f"shared tasks covering "
             f"{st['by_task_text']['frac_lerobot_episodes_covered']} of LeRobot"
        if st["by_task_text"]["n_shared_normalized_tasks"]
        else "no strategy matches on measured fields; inspect "
             "lerobot_episode_keysets for a source-path field")
    dump()
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("structure", "episode_facts_sample")},
                     indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
