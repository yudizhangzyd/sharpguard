"""F4/O4 deconfound: LoRA fine-tune of ECoT-bridge on a Bridge V2 4k subset.

Why this run exists. Every "ours" row in the leaderboard is ECoT-bridge
fine-tuned on LIBERO, and each scores differently from ECoT-bridge itself. Two
explanations are entangled in that comparison:

  (a) the fine-tuning CHANGED the model's CoT-action coupling, or
  (b) the fine-tuning moved the model IN-DOMAIN for the evaluation corpus, and
      the leaderboard is partly measuring domain match.

Training with the identical recipe on Bridge V2 -- the base model's OWN domain
-- holds (b) fixed so only (a) can move. If this model's calibration profile
lands with the LIBERO-trained rows, the effect is fine-tuning; if it lands with
the base, part of the leaderboard spread is domain match and the paper says so.

The recipe is IMPORTED from cotfaith_train.py, not reimplemented: the prompt,
the 8-tag CoT rendering, the action tokenizer and the collator all come from
that module. A deconfound whose recipe differs from the thing it deconfounds
measures the difference in recipes. The previous version of this file had
exactly that defect -- a local `_build_cot_text` emitting 5 tags
(TASK/PLAN/SUBTASK/MOVE/GRIPPER POSITION) against the LIBERO recipe's 8,
silently dropping VISIBLE OBJECTS, SUBTASK REASONING and MOVE REASONING.
Training on that would have confounded "trained on Bridge V2" with "trained on
a different CoT format", the one confound this experiment cannot afford.

Data: Embodied-CoT/embodied_features_bridge reasoning annotations joined to
Bridge V2 trajectories from IPEC-COMMUNITY/bridge_orig_lerobot (lerobot v2.0).
The join is the risky part -- see _ReasoningIndex.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from huggingface_hub import HfApi
from sharpguard.hf_retry import file_with_retry


# ---------- recipe import (not a copy) ----------

def _load_libero_recipe():
    """Import cotfaith_train.py's recipe functions.

    experiments/ is not a package (no __init__.py), so this goes through
    importlib rather than a plain import. Deliberately importing rather than
    copying: these four functions ARE the recipe, and the deconfound only holds
    if this run and the LIBERO runs share them byte for byte. A copy drifts; an
    import cannot. Safe to exec: cotfaith_train.py's tensorflow imports are all
    function-local, so nothing heavy runs at module scope.
    """
    path = Path(__file__).resolve().parent / "cotfaith_train.py"
    spec = importlib.util.spec_from_file_location("_cotfaith_train_recipe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_recipe = _load_libero_recipe()
build_ecot_prompt = _recipe.build_ecot_prompt
build_ecot_target_text = _recipe.build_ecot_target_text
action_ids = _recipe.action_ids
collate = _recipe.collate
ECOT_TAGS_ORDER = _recipe.ECOT_TAGS_ORDER


def load_bridge_reasoning(hf_repo="Embodied-CoT/embodied_features_bridge"):
    """Load the Embodied-CoT Bridge V2 reasoning annotations JSON (~1.4GB)."""
    path = file_with_retry(hf_repo, "embodied_features_bridge.json",
                           repo_type="dataset")
    print(f"[bridge-train] loading reasoning from {path}")
    with open(path) as f:
        return json.load(f)


# ---------- the join ----------

def _norm_task(s) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s or "").lower()).strip()


# A normalized instruction is usable as a join key only if it actually identifies
# a task. Bridge V2 instructions are crowdsourced free text and a visible slice of
# them is not language at all -- bolt `754ru9usqe` found "1", "9", "12345678",
# "3wsws" and "7210 2199 5955 2055 534" among the 19541 keys the two exports
# share, and a max fanout of 963 LeRobot episodes on a single key against a median
# of 1. Joining on those pairs hundreds of unrelated trajectories with whichever
# episode the modulo happens to select, which is the same defect as the poisoned
# episode_id join, just concentrated in a minority of keys. Two guards, both
# measured rather than guessed:
_MIN_TASK_CHARS = 8          # "9" and "3wsws" are not instructions
_MIN_TASK_WORDS = 2          # a manipulation instruction is at least verb+object
_MAX_TASK_FANOUT = 8         # above this the key identifies a bucket, not a task


def _usable_task_key(t: str) -> bool:
    """Is this normalized instruction specific enough to join on?

    Deliberately conservative: turning a junk key into *no* join costs coverage,
    which the preflight reports; turning it into an arbitrary pairing costs
    training-data validity, which nothing downstream can detect.
    """
    if len(t) < _MIN_TASK_CHARS:
        return False
    words = [w for w in t.split() if len(w) >= 2 and any(c.isalpha() for c in w)]
    return len(words) >= _MIN_TASK_WORDS


# Bridge-export `features` field names, for tags the `reasoning` subtree does not
# supply. Bolt `754ru9usqe` measured which those are over 4000 steps: `reasoning`
# fills TASK, PLAN, SUBTASK, SUBTASK REASONING, MOVE REASONING and MOVE at 1.0
# (its step dicts are task/plan/subtask/subtask_reason/move_reason/move, all
# already aliases in ECOT_TAGS_ORDER), and leaves VISIBLE OBJECTS and GRIPPER
# POSITION at 0.0. `features` fills exactly those two. So the first two entries
# below are load-bearing and `move_primitive` is a fallback that the measured
# data never needs -- kept because a shard whose reasoning omits `move` would
# otherwise render an empty MOVE, on the one tag the action head is most directly
# conditioned on.
#
# This lives here and not in cotfaith_train.py on purpose. That module is
# imported rather than copied so this run and the LIBERO runs share the recipe
# byte for byte (see _load_libero_recipe); a corpus-specific field name is not
# part of the recipe, and putting it there would make the shared surface depend
# on which corpus happens to need it. The value is stored back under the
# recipe's OWN first alias, so the renderer never learns about this table.
_BRIDGE_FEATURE_ALIASES = {
    "VISIBLE OBJECTS": ("bboxes",),
    "GRIPPER POSITION": ("gripper_position",),
    "MOVE": ("move_primitive",),
}


def empty_rendered_tags(target_text: str) -> list[str]:
    """Which of the eight tags rendered with no content.

    build_ecot_target_text emits `TAG: value` segments in ECOT_TAGS_ORDER order
    and joins them with a space, substituting "" for a missing key -- so a tag
    the join failed to supply is not an error, it is a `TAG: ` with nothing after
    it. That is exactly the failure this run has to catch before training:
    `n_yielded > 0` is satisfied by a trace of eight empty tags. Segments are
    located in order from a running offset so that MOVE does not match inside
    MOVE REASONING.
    """
    tags = [t for _, t in ECOT_TAGS_ORDER]
    found, off = [], 0
    for t in tags:
        i = target_text.find(f"{t}: ", off)
        if i < 0:
            found.append((t, None, None))
            continue
        found.append((t, i, i + len(t) + 2))
        off = i + len(t) + 2
    empty = []
    for n, (t, _lstart, cstart) in enumerate(found):
        if cstart is None:
            empty.append(t)             # tag absent from the render entirely
            continue
        nxt = next((ls for _, ls, _ in found[n + 1:] if ls is not None), None)
        if not target_text[cstart:nxt if nxt is not None
                           else len(target_text)].strip():
            empty.append(t)
    return empty


class _ReasoningIndex:
    """Resolve (episode_index, step_index) -> a renderable per-step CoT dict.

    Rewritten against the layout bolt `j2kqu7k3m7` measured. The previous
    version assumed two levels and joined 0 of 606677 frames in bolt
    `fnwfaq9bq6`; the export is four:

        raw[<authors' absolute NFS path>][<episode index in that shard>] = {
            "features":  {"move_primitive": [...], "gripper_position": [...],
                          "bboxes": [...]},          # per-step LISTS
            "metadata":  {"episode_id": int, "file_path": str, "n_steps": int,
                          "language_instruction": str},
            "reasoning": {"0": {...}, "1": {...}, ...},   # per-step DICTS
        }

    Two things follow, and the old code got both wrong.

    THE JOIN KEY IS metadata.episode_id, not a top-level key. The top level is
    paths, so `raw[str(ep_idx)]` could never hit -- and neither could the
    task-text fallback, which read `rec["task"]` when the field is
    `metadata.language_instruction`. Two strategies, both reading fields that do
    not exist, which is why having a fallback did not help.

    A SHARED INTEGER ID IS NOT A JOIN. Two exports can both number episodes from
    0 and mean different episodes, and that failure is invisible: it produces a
    complete, plausible index that pairs every trajectory with the wrong
    reasoning, and training would run to completion on it. So the id strategy is
    *probed* -- the first `_ID_PROBE_N` hits are checked for instruction
    agreement against the LeRobot side, and if they do not agree the strategy is
    disabled for the whole run rather than per episode. Better to fall back to a
    coarse task-level join than to train on a fine-grained wrong one.

    That is not hypothetical here. Bolt `754ru9usqe` measured it: `episode_id`
    matches only 1111 of 53192 LeRobot episodes (2.1%), its range is [0, 1110]
    against 60062 annotated episodes -- so it is per-shard, not global -- it has
    879 collisions, and the instructions agree on just 0.280 of the matched
    pairs. The probe is expected to disable this strategy on the real data. It is
    kept live rather than replaced by a constant because a measurement the code
    re-derives cannot go stale if the upstream export changes, and because the
    same 0.280 is what makes the fallback defensible rather than lazy.

    So the live route is `by_task_text`, and the same job bounds what it buys:
    19541 shared normalized instructions covering 38660 of 53192 LeRobot
    episodes (72.7%), with a median fanout of 1 -- so for most tasks the
    "task-level" join is in fact unique -- and 41634 reachable annotated
    episodes, comfortably above the 4000 this run requests. The LeRobot episode
    records carry only `episode_index,length,tasks`, no upstream source path, so
    there is no exact route to prefer over it.

    Both strategies then go through `merged_step`, because a joined episode is
    not yet a renderable one: `build_ecot_target_text` wants all eight tags in
    one dict and this export splits them across `reasoning` and `features`. Also
    measured, not assumed -- over 4000 inspected steps, `reasoning` alone fills
    six of eight tags and leaves VISIBLE OBJECTS and GRIPPER POSITION at 0.0,
    `features` alone fills exactly those two, and the merge fills all eight at
    1.0. Training on the unmerged trace would have taught the model to emit two
    permanently empty tags.
    """

    # Enough hits to distinguish agreement from coincidence, small enough that a
    # poisoned join is caught in the first seconds of streaming rather than after
    # an epoch. Instructions are crowdsourced free text, so 20 independent
    # agreements is already decisive.
    _ID_PROBE_N = 20
    _ID_MIN_AGREE = 0.95

    def __init__(self, raw: dict):
        self.raw = raw
        self.strategy = None
        self.by_id: dict[int, dict] = {}
        self.by_task: dict[str, list] = {}
        self.id_disabled = False
        self.stats = {"id_probe_hits": 0, "id_probe_agrees": 0,
                      "by_episode_id": 0, "by_task_text": 0,
                      "merge_filled_from_features": 0,
                      "task_key_rejected_degenerate": 0,
                      "shape_deviations": 0}

        n_ep = 0
        for pkey, pv in raw.items():
            if not isinstance(pv, dict):
                self.stats["shape_deviations"] += 1
                continue
            for ekey, ev in pv.items():
                if not isinstance(ev, dict):
                    self.stats["shape_deviations"] += 1
                    continue
                md = ev.get("metadata")
                if not isinstance(md, dict):
                    self.stats["shape_deviations"] += 1
                    md = {}
                rz = ev.get("reasoning")
                if not isinstance(rz, dict) or not rz:
                    continue          # nothing to train on for this episode
                epi = {"path": pkey, "ep_key": ekey,
                       "instruction": md.get("language_instruction") or "",
                       "reasoning": rz,
                       "features": ev.get("features")
                       if isinstance(ev.get("features"), dict) else {}}
                eid = md.get("episode_id")
                if isinstance(eid, int):
                    self.by_id.setdefault(eid, epi)
                t = _norm_task(epi["instruction"])
                if t and _usable_task_key(t):
                    self.by_task.setdefault(t, []).append(epi)
                elif t:
                    self.stats["task_key_rejected_degenerate"] += 1
                n_ep += 1

        # Fanout guard, applied after the index is complete because it is a
        # property of the whole key, not of one episode.
        self.stats["task_keys_dropped_high_fanout"] = 0
        for t in [k for k, v in self.by_task.items()
                  if len(v) > _MAX_TASK_FANOUT]:
            self.stats["task_keys_dropped_high_fanout"] += 1
            del self.by_task[t]

        print(f"[bridge-train] reasoning index: {len(raw)} path keys, "
              f"{n_ep} usable episodes, {len(self.by_id)} distinct episode_ids, "
              f"{len(self.by_task)} usable instruction keys "
              f"({self.stats['task_key_rejected_degenerate']} rejected as "
              f"degenerate, {self.stats['task_keys_dropped_high_fanout']} "
              f"dropped for fanout > {_MAX_TASK_FANOUT}), "
              f"{self.stats['shape_deviations']} shape deviations")
        if not n_ep:
            print("[bridge-train] WARNING the reasoning export yielded no "
                  "usable episode; every lookup will miss.")

    # ---- rendering ----

    def merged_step(self, epi: dict, step_idx: int):
        """One per-step dict carrying every tag the renderer can fill.

        `reasoning[step]` is the base. Anything the renderer wants that is not
        there but IS a per-step list in `features` is spliced in at this step's
        index -- that is the whole reason this method exists, and without it
        VISIBLE OBJECTS and GRIPPER POSITION render empty on every Bridge
        sample. `task` falls back to the episode instruction, which is where
        this export keeps it.
        """
        rz = epi["reasoning"]
        skey = str(step_idx)
        if skey not in rz:
            digits = sorted((int(s) for s in rz if str(s).isdigit()))
            if not digits:
                return None
            # Clamp rather than miss: annotation and trajectory lengths need not
            # agree, and the last annotated step is the right answer for a frame
            # past the end far more often than "no reasoning" is.
            skey = str(min(digits, key=lambda d: abs(d - step_idx)))
        rec = rz.get(skey)
        if not isinstance(rec, dict) or not rec:
            return None

        out = dict(rec)
        sidx = int(skey) if str(skey).isdigit() else 0
        for aliases, tag in ECOT_TAGS_ORDER:
            if any(a in out and out[a] not in (None, "", [], {}) for a in aliases):
                continue
            for a in tuple(aliases) + _BRIDGE_FEATURE_ALIASES.get(tag, ()):
                seq = epi["features"].get(a)
                if isinstance(seq, list) and seq:
                    # features are per-step lists parallel to the trajectory;
                    # clamp for the same reason the step key is clamped.
                    out[aliases[0]] = seq[min(sidx, len(seq) - 1)]
                    self.stats["merge_filled_from_features"] += 1
                    break
            else:
                self.stats.setdefault("tag_empty", {})
                self.stats["tag_empty"][tag] = \
                    self.stats["tag_empty"].get(tag, 0) + 1
        out.setdefault("task", epi["instruction"])
        return out

    # ---- the join ----

    def lookup(self, ep_idx, step_idx, instruction):
        if not self.id_disabled:
            epi = self.by_id.get(int(ep_idx)) if str(ep_idx).lstrip("-").isdigit() \
                else None
            if epi is not None:
                # Probe before trusting. An id join that pairs the wrong episodes
                # is the failure that looks like success, so it has to be
                # disproved on real data rather than assumed away.
                if self.stats["id_probe_hits"] < self._ID_PROBE_N:
                    self.stats["id_probe_hits"] += 1
                    if instruction and _norm_task(epi["instruction"]) \
                            == _norm_task(instruction):
                        self.stats["id_probe_agrees"] += 1
                    if self.stats["id_probe_hits"] == self._ID_PROBE_N:
                        rate = (self.stats["id_probe_agrees"]
                                / self._ID_PROBE_N)
                        if rate < self._ID_MIN_AGREE:
                            self.id_disabled = True
                            print(f"[bridge-train] DISABLING by_episode_id: "
                                  f"instructions agreed on only {rate:.2f} of "
                                  f"{self._ID_PROBE_N} matched ids. The two "
                                  f"exports number DIFFERENT episodes, so this "
                                  f"key would pair every trajectory with the "
                                  f"wrong reasoning. Falling back to task text.")
                        else:
                            print(f"[bridge-train] by_episode_id confirmed: "
                                  f"instructions agree on {rate:.2f} of "
                                  f"{self._ID_PROBE_N} probed matches")
                if not self.id_disabled \
                        and self.stats["id_probe_hits"] >= self._ID_PROBE_N:
                    # Only after the probe concludes. While it is still running
                    # the id-joined record is deliberately NOT used: if the join
                    # turns out to be poisoned, using it during the probe would
                    # have put up to _ID_PROBE_N mispaired samples into training
                    # before the refusal fired. Falling through to the text join
                    # for those first few frames costs nothing; a mispaired
                    # sample is unrecoverable and invisible.
                    r = self.merged_step(epi, step_idx)
                    if r:
                        self.strategy = "by_episode_id"
                        self.stats["by_episode_id"] += 1
                        return r

        cands = self.by_task.get(_norm_task(instruction))
        if cands:
            # Many episodes share an instruction, so this is a task-level join:
            # the CoT describes this task but not necessarily this trajectory.
            # Deterministic pick by episode index so a re-run is reproducible.
            epi = cands[int(ep_idx) % len(cands)] \
                if str(ep_idx).lstrip("-").isdigit() else cands[0]
            r = self.merged_step(epi, step_idx)
            if r:
                self.strategy = self.strategy or "by_task_text"
                self.stats["by_task_text"] += 1
                return r
        return None


# ---------- dataset ----------

class BridgeV2CotDataset(IterableDataset):
    """Stream training examples from Bridge V2 LeRobot + reasoning JSON.

    Yields exactly what cotfaith_train.py's ECoTLiberoDataset yields --
    (pixel_values, input_ids, attention_mask, labels) with the human turn
    masked to -100 -- so the imported collator and the same training loop apply
    unchanged.
    """

    def __init__(self, reasoning_map, dataset_repo, *, n_examples, processor,
                 dtype, vocab_size, reasoning_mode="full",
                 max_steps_per_ep=None):
        self.index = _ReasoningIndex(reasoning_map)
        self.dataset_repo = dataset_repo
        self.n_examples = n_examples
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.reasoning_mode = reasoning_mode
        self.max_steps_per_ep = max_steps_per_ep
        self.stats = {"yielded": 0, "no_reasoning": 0, "no_frame": 0,
                      "encode_failed": 0}
        # First few rendered CoT traces, kept for the preflight. A join can now
        # succeed and still be worthless: the Bridge export splits the eight tags
        # across `features` and `reasoning`, so a merge that misses one renders a
        # trace with an empty tag, and nothing downstream can see the difference
        # between that and a real CoT. The preflight has to read what was
        # actually rendered, not just count that something was.
        self.sample_rendered = []
        api = HfApi()
        files = list(api.list_repo_files(dataset_repo, repo_type="dataset"))
        # Every parquet, not a slice: the old code sliced the parquet FILE list
        # by n_trajectories, which conflates files with trajectories. The budget
        # belongs on yielded examples, where it is actually meaningful.
        self.parquets = sorted(f for f in files
                               if f.endswith(".parquet") and "data/" in f)
        self.video_dirs = sorted({v.rsplit("/", 1)[0] for v in files
                                  if v.endswith(".mp4") and "videos/" in v})
        self.tasks = self._load_tasks()
        print(f"[bridge-train] {len(self.parquets)} parquets, "
              f"{len(self.video_dirs)} video streams, "
              f"{len(self.tasks)} episode task strings, "
              f"budget {n_examples} examples")

    def _load_tasks(self):
        """episode_index -> instruction, from LeRobot meta/episodes.jsonl."""
        out = {}
        for rel in ("meta/episodes.jsonl", "meta/episodes.json"):
            try:
                p = file_with_retry(self.dataset_repo, rel, repo_type="dataset")
            except Exception:
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

    def _episode_frame(self, ep_idx):
        """First frame of this episode's video, or None."""
        import av
        for vd in self.video_dirs:
            try:
                vp = file_with_retry(self.dataset_repo,
                                     f"{vd}/episode_{ep_idx:06d}.mp4",
                                     repo_type="dataset")
                container = av.open(vp)
                frame = next(container.decode(video=0))
                img = frame.to_image()
                container.close()
                return img
            except Exception:
                continue
        return None

    def __iter__(self):
        import pyarrow.parquet as pq
        for parquet_path in self.parquets:
            if self.stats["yielded"] >= self.n_examples:
                return
            try:
                pp = file_with_retry(self.dataset_repo, parquet_path,
                                     repo_type="dataset")
                table = pq.read_table(pp)
                ep_col = table.column("episode_index").to_pylist()
                act_col = table.column("action").to_pylist()
            except Exception as e:
                print(f"[bridge-train] {parquet_path}: {type(e).__name__}: {e}")
                continue

            frame_cache, per_ep_count, step_of_ep = {}, {}, {}
            for i, ep_idx in enumerate(ep_col):
                if self.stats["yielded"] >= self.n_examples:
                    return
                step_idx = step_of_ep.get(ep_idx, 0)
                step_of_ep[ep_idx] = step_idx + 1
                instruction = self.tasks.get(int(ep_idx), "")
                r = self.index.lookup(ep_idx, step_idx, instruction)
                if not r:
                    self.stats["no_reasoning"] += 1
                    continue
                if self.max_steps_per_ep:
                    c = per_ep_count.get(ep_idx, 0)
                    if c >= self.max_steps_per_ep:
                        continue
                    per_ep_count[ep_idx] = c + 1
                # One video decode per episode, not per step: the frame is the
                # episode's first frame either way, and re-downloading it per
                # step is what made the old scaffold crawl.
                if ep_idx not in frame_cache:
                    frame_cache[ep_idx] = self._episode_frame(ep_idx)
                img = frame_cache[ep_idx]
                if img is None:
                    self.stats["no_frame"] += 1
                    continue
                instr = instruction or str(r.get("task")
                                           or r.get("instruction") or "")
                action = np.asarray(act_col[i], dtype=np.float32)[:7]
                if len(self.sample_rendered) < 8:
                    self.sample_rendered.append(
                        {"episode_index": int(ep_idx), "step": step_idx,
                         "instruction": instr,
                         "strategy": self.index.strategy,
                         "target_text": build_ecot_target_text(r)})
                ex = self._encode(img.convert("RGB"), instr, r, action)
                if ex is None:
                    self.stats["encode_failed"] += 1
                    continue
                self.stats["yielded"] += 1
                if self.stats["yielded"] % 500 == 0:
                    print(f"[bridge-train] {self.stats}")
                yield ex
        print(f"[bridge-train] stream exhausted: {self.stats}")

    def _encode(self, pil, instruction, reasoning, action):
        """Tokenization and labelling identical to the LIBERO dataset path."""
        prompt = build_ecot_prompt(instruction)
        if self.reasoning_mode == "no_cot":
            text_pre_action = prompt + " ACTION:"
        else:
            text_pre_action = prompt + build_ecot_target_text(reasoning) + " ACTION:"
        try:
            proc = self.processor(text_pre_action, pil)   # positional: ECoT convention
            input_ids = proc["input_ids"][0]
            a_ids = torch.from_numpy(
                action_ids(action, self.vocab_size)).to(input_ids.dtype)
            eos = self.tokenizer.eos_token_id
            full_ids = torch.cat(
                [input_ids, a_ids, torch.tensor([eos], dtype=input_ids.dtype)])
            prompt_len = self.tokenizer(
                prompt, add_special_tokens=True,
                return_tensors="pt")["input_ids"][0].shape[0]
            labels = full_ids.clone()
            labels[:prompt_len] = -100     # supervise the assistant turn only
            return {"pixel_values": proc["pixel_values"][0].to(self.dtype),
                    "input_ids": full_ids,
                    "attention_mask": torch.ones_like(full_ids),
                    "labels": labels}
        except Exception as e:
            print(f"[bridge-train] encode failed: {type(e).__name__}: {e}")
            return None


# ---------- entry point ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--base-model", default="Embodied-CoT/ecot-openvla-7b-bridge")
    p.add_argument("--reasoning-repo", default="Embodied-CoT/embodied_features_bridge")
    p.add_argument("--dataset-repo", default="IPEC-COMMUNITY/bridge_orig_lerobot")
    p.add_argument("--n-trajectories", type=int, default=4000,
                   help="Budget on streamed training EXAMPLES (kept under the "
                        "old flag name so existing bolt configs keep working).")
    p.add_argument("--max-steps-per-ep", type=int, default=0)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reasoning-mode", default="full", choices=["full", "no_cot"])
    p.add_argument("--preflight-n", type=int, default=3,
                   help="Examples to stream before loading the 7B model. The "
                        "reasoning<->trajectory join is unverified; finding out "
                        "it yields nothing after a 15-minute model load and a "
                        "full-length train loop is the expensive way to learn it.")
    p.add_argument("--preflight-only", action="store_true",
                   help="Run the join preflight and exit. This is what the old "
                        "default behaviour was; it is now an explicit flag so a "
                        "smoke pass can never again be mistaken for training.")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    reasoning = load_bridge_reasoning(args.reasoning_repo)
    print(f"[bridge-train] {len(reasoning)} top-level reasoning entries")

    from transformers import AutoProcessor, AutoModelForVision2Seq
    processor = AutoProcessor.from_pretrained(args.base_model,
                                              trust_remote_code=True)

    ds = BridgeV2CotDataset(
        reasoning, args.dataset_repo, n_examples=args.n_trajectories,
        processor=processor, dtype=dtype,
        vocab_size=processor.tokenizer.vocab_size,
        reasoning_mode=args.reasoning_mode,
        max_steps_per_ep=args.max_steps_per_ep or None)

    # ----- preflight: does the join produce anything at all? -----
    pre_n = max(args.preflight_n, 1)
    it0, shapes = iter(ds), []
    for _ in range(pre_n):
        try:
            s = next(it0)
        except StopIteration:
            break
        shapes.append({k: list(v.shape) for k, v in s.items()})
    from collections import Counter
    empty_counts = Counter()
    for s in ds.sample_rendered:
        empty_counts.update(empty_rendered_tags(s["target_text"]))
    preflight = {
        "n_requested": pre_n,
        "n_yielded": len(shapes),
        "shapes": shapes,
        "join_strategy": ds.index.strategy,
        "join_stats": dict(ds.index.stats),
        "join_id_disabled": ds.index.id_disabled,
        "stream_stats": dict(ds.stats),
        "n_parquets": len(ds.parquets),
        "n_video_streams": len(ds.video_dirs),
        "n_episode_task_strings": len(ds.tasks),
        "n_reasoning_entries": len(reasoning),
        "cot_tags": [t for _, t in ECOT_TAGS_ORDER],
        "n_rendered_inspected": len(ds.sample_rendered),
        "empty_tag_counts": dict(empty_counts),
        "rendered_samples": ds.sample_rendered[:3],
        "recipe_source": "experiments/cotfaith_train.py (imported, not copied)",
    }
    (out / "preflight_report.json").write_text(json.dumps(preflight, indent=2))
    print("[bridge-train] preflight: " + json.dumps(preflight, indent=2)[:1800])

    if not shapes:
        # Zero joined examples. Training would run 15k no-op steps and save the
        # base model under a new name, and the downstream calibration scorer
        # cannot tell that apart from a real fine-tune. Refuse.
        print("[bridge-train] FATAL: join yielded 0 examples. The reasoning "
              "JSON does not line up with this trajectory export under either "
              "strategy; see preflight_report.json for the key shapes.")
        sys.stdout.flush()
        os._exit(6)

    # The other way this run can succeed at nothing. A joined sample whose CoT
    # renders as eight empty tags trains the model to emit eight empty tags, and
    # F4's whole claim is about what a CoT-trained model does -- so an
    # always-empty tag is a silent corruption of the treatment, not a data-quality
    # nit. Tolerate a tag that is merely sparse in the annotations; refuse when a
    # tag is empty in EVERY inspected sample, which is the signature of a missing
    # merge rather than of missing data.
    always_empty = [t for t, c in empty_counts.items()
                    if c == len(ds.sample_rendered)]
    if ds.sample_rendered and always_empty:
        print(f"[bridge-train] FATAL: these CoT tags rendered empty in all "
              f"{len(ds.sample_rendered)} inspected samples: {always_empty}. "
              f"The join found episodes but the per-step merge is not supplying "
              f"every tag; training on this would fine-tune the model to emit "
              f"empty reasoning. See rendered_samples in preflight_report.json.")
        sys.stdout.flush()
        os._exit(7)

    if args.preflight_only:
        print("[bridge-train] preflight only; no training run")
        sys.stdout.flush()
        os._exit(0)

    # ----- train -----
    print(f"[bridge-train] loading base {args.base_model}")
    base = AutoModelForVision2Seq.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device)

    # Same LoRA config as the LIBERO runs. Any difference here would be a
    # second thing that changed alongside the corpus.
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(base, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules="all-linear", init_lora_weights="gaussian"))
    model.print_trainable_parameters()
    model.train()

    class _Wrap(IterableDataset):
        def __init__(self, inner): self.inner = inner
        def __iter__(self): return iter(self.inner)

    loader = DataLoader(_Wrap(ds), batch_size=args.batch_size,
                        collate_fn=collate, num_workers=0)

    trainable = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    t0, losses, n_fwd_fail, n_wraps = time.time(), [], 0, 0
    it = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            # The 4k Bridge subset is far smaller than LIBERO-90, so the stream
            # is expected to wrap. Counted rather than silent: how many epochs
            # the subset sees is part of the recipe at this corpus size.
            n_wraps += 1
            ds.stats["yielded"] = 0
            it = iter(loader)
            try:
                batch = next(it)
            except StopIteration:
                print("[bridge-train] FATAL: stream yields nothing on re-iter")
                break
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        try:
            loss = model(**batch).loss
        except Exception as e:
            n_fwd_fail += 1
            print(f"[bridge-train] fwd fail step {step}: {e}\n"
                  + traceback.format_exc()[-600:])
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % 20 == 0:
            print(f"  step {step+1:5d}/{args.steps}  loss={losses[-1]:.4f}  "
                  f"({time.time()-t0:.0f}s)")
        if (step + 1) % 2000 == 0:
            (out / "train_losses.json").write_text(json.dumps(losses))

    (out / "train_losses.json").write_text(json.dumps(losses))
    (out / "train_meta.json").write_text(json.dumps({
        "status": "trained" if losses else "no_gradient_steps",
        "corpus": args.dataset_repo,
        "reasoning_repo": args.reasoning_repo,
        "join_strategy": ds.index.strategy,
        "example_budget": args.n_trajectories,
        "steps_requested": args.steps,
        "steps_with_gradient": len(losses),
        "n_stream_wraps": n_wraps,
        "n_forward_failures": n_fwd_fail,
        "loss_first_100_mean": float(np.mean(losses[:100])) if losses else None,
        "loss_last_100_mean": float(np.mean(losses[-100:])) if losses else None,
        "recipe_source": "experiments/cotfaith_train.py (imported, not copied)",
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "lr": args.lr,
                 "batch_size": args.batch_size, "dtype": args.dtype,
                 "seed": args.seed, "reasoning_mode": args.reasoning_mode},
    }, indent=2))

    if not losses:
        print("[bridge-train] FATAL: 0 gradient steps; refusing to save a "
              "checkpoint that would be indistinguishable from the base model")
        sys.stdout.flush()
        os._exit(5)

    print(f"[bridge-train] merging LoRA -> {out / 'merged_model'}")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged_model", safe_serialization=True)
    processor.save_pretrained(out / "merged_model")
    print("[bridge-train] done")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
