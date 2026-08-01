"""Offline verification of the Bridge reasoning <-> LeRobot join.

The join in `experiments/cotfaith_train_bridge.py::_ReasoningIndex` has now been
wrong twice on real data -- bolt `fnwfaq9bq6` joined 0 of 606677 frames, and the
probe that diagnosed it (`j2kqu7k3m7`) scored both strategies against fields that
do not exist. Each round cost a pod. The logic is pure Python, so it can be
exercised here for free against a fixture in the layout bolt `754ru9usqe`
measured, and the two historical bugs are kept as negative controls: a rewrite
that reintroduces either one fails here rather than in a queue.

Stubs torch/numpy/hf so the module imports on a laptop. Run: python
tests/test_bridge_reasoning_join.py  (exit 0 = all pass)
"""
import pathlib
import sys, types
from unittest.mock import MagicMock

for name in ("torch", "torch.utils", "torch.utils.data", "numpy",
             "huggingface_hub", "sharpguard", "sharpguard.hf_retry",
             "transformers", "peft", "PIL", "PIL.Image", "pyarrow",
             "pyarrow.parquet", "decord", "av", "tensorflow"):
    m = types.ModuleType(name); m.__path__ = []
    sys.modules.setdefault(name, m)
sys.modules["torch"].__dict__.update({"Tensor": object, "no_grad": MagicMock(),
                                      "float32": "f32", "long": "i64"})
sys.modules["torch.utils.data"].IterableDataset = object
sys.modules["torch.utils.data"].DataLoader = object
sys.modules["numpy"].__dict__.update({"asarray": lambda *a, **k: [],
                                      "float32": "f32", "ndarray": object})
sys.modules["huggingface_hub"].HfApi = MagicMock
sys.modules["sharpguard.hf_retry"].file_with_retry = MagicMock()

import importlib.util
spec = importlib.util.spec_from_file_location(
    "btr", str(pathlib.Path(__file__).resolve().parent.parent
                      / "experiments" / "cotfaith_train_bridge.py"))
btr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(btr)
print("imported; tags =", [t for _, t in btr.ECOT_TAGS_ORDER])

fails = []
def ck(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond: fails.append(name)

# ---- fixture in the layout bolt j2kqu7k3m7 measured -------------------------
def episode(eid, instr, nsteps=3, with_reasoning=True, feats=True):
    ev = {"metadata": {"episode_id": eid, "file_path": f"/nfs/x/{eid}.tfrecord",
                       "n_steps": nsteps, "language_instruction": instr}}
    if feats:
        ev["features"] = {
            "move_primitive": [f"move_{s}" for s in range(nsteps)],
            "gripper_position": [[[10 + s, 20 + s]] for s in range(nsteps)],
            "bboxes": [[[0.9, f"obj{s}", [1, 2, 3, 4]]] for s in range(nsteps)]}
    if with_reasoning:
        ev["reasoning"] = {str(s): {"task": instr, "plan": {"0": "p0", "1": "p1"},
                                    "subtask": f"st{s}",
                                    "subtask_reason": f"str{s}",
                                    "move_reason": f"mr{s}"}
                           for s in range(nsteps)}
    return ev

RAW = {"/nfs/shard_a": {"0": episode(0, "put the spoon in the pot"),
                        "1": episode(1, "open the drawer")},
       "/nfs/shard_b": {"0": episode(2, "fold the cloth")}}

idx = btr._ReasoningIndex(RAW)
ck("indexes by metadata.episode_id, not top-level path keys",
   sorted(idx.by_id) == [0, 1, 2], f"by_id={sorted(idx.by_id)}")
ck("indexes by metadata.language_instruction (round 1 read rec['task'])",
   btr._norm_task("open the drawer") in idx.by_task)

# NEGATIVE CONTROLS: reproduce round 1's two bugs and confirm they score 0.
ck("[control] round 1's key -- raw[str(ep_idx)] -- misses by construction",
   RAW.get("0") is None and RAW.get("1") is None)
ck("[control] round 1's field -- rec['task'] at episode level -- absent",
   all("task" not in ev and "instruction" not in ev
       for pv in RAW.values() for ev in pv.values()))

# ---- the merge --------------------------------------------------------------
epi = idx.by_id[0]
m = idx.merged_step(epi, 1)
ck("merge splices bboxes from features", m.get("bboxes") == [[0.9, "obj1", [1, 2, 3, 4]]])
ck("merge splices gripper under the recipe's OWN first alias 'gripper'",
   m.get("gripper") == [[11, 21]] and "gripper_position" not in m)
ck("merge maps features.move_primitive onto the MOVE alias 'movement'",
   m.get("movement") == "move_1", f"movement={m.get('movement')!r}")
txt = btr.build_ecot_target_text(m)
ck("all eight tags render non-empty after the merge",
   btr.empty_rendered_tags(txt) == [], f"empty={btr.empty_rendered_tags(txt)}")
print("      rendered:", txt[:180])

# Without the merge -- what the trainer did before -- MOVE/VISIBLE/GRIPPER die.
bare = btr.build_ecot_target_text(RAW["/nfs/shard_a"]["0"]["reasoning"]["1"])
ck("reasoning-only render leaves exactly the three features-only tags empty",
   btr.empty_rendered_tags(bare) == ["VISIBLE OBJECTS", "MOVE", "GRIPPER POSITION"],
   f"empty={btr.empty_rendered_tags(bare)}")

# empty_rendered_tags itself, incl. the MOVE-inside-MOVE-REASONING trap.
ck("MOVE not matched inside MOVE REASONING",
   btr.empty_rendered_tags(btr.build_ecot_target_text(
       {"task": "t", "plan": "p", "bboxes": "b", "subtask_reasoning": "sr",
        "subtask": "s", "movement_reasoning": "mr", "movement": "",
        "gripper": "g"})) == ["MOVE"])
ck("an all-empty trace reports all eight tags",
   len(btr.empty_rendered_tags(btr.build_ecot_target_text({}))) == 8)

# ---- the id-collision refusal ----------------------------------------------
good = idx.lookup(0, 0, "put the spoon in the pot")
ck("by_episode_id joins when instructions agree", bool(good))
for i in range(25):
    idx.lookup(i % 3, 0, ["put the spoon in the pot", "open the drawer",
                          "fold the cloth"][i % 3])
ck("id strategy stays enabled under agreement", not idx.id_disabled)

idx2 = btr._ReasoningIndex(RAW)
for i in range(25):
    idx2.lookup(i % 3, 0, "something else entirely")
ck("id strategy DISABLES itself when instructions disagree", idx2.id_disabled)
ck("...and does not then silently return the wrong episode",
   idx2.lookup(0, 0, "something else entirely") is None)
ck("...but the text fallback still works after disabling",
   bool(idx2.lookup(0, 0, "fold the cloth")))

# ---- degenerate shapes ------------------------------------------------------
idx3 = btr._ReasoningIndex({"/p": {"0": episode(0, "a", with_reasoning=False),
                                   "1": "not a dict", "2": episode(1, "")}})
ck("episodes with no reasoning are skipped, not crashed on", 0 not in idx3.by_id)
ck("non-dict episode counted as a shape deviation",
   idx3.stats["shape_deviations"] == 1, f"{idx3.stats['shape_deviations']}")
ck("empty instruction does not become a shared join key",
   "" not in idx3.by_task)
idx4 = btr._ReasoningIndex({"/p": {"0": episode(0, "a", nsteps=2, feats=False)}})
m4 = idx4.merged_step(idx4.by_id[0], 0)
ck("missing features subtree degrades to a partial render, no crash",
   m4 is not None and "VISIBLE OBJECTS" in btr.empty_rendered_tags(
       btr.build_ecot_target_text(m4)))
m5 = idx.merged_step(idx.by_id[0], 99)
ck("step past the end clamps to the last annotated step",
   m5 is not None and m5.get("subtask") == "st2", f"{m5 and m5.get('subtask')}")

# ---- no mispaired sample may reach training during the probe ---------------
idx5 = btr._ReasoningIndex(RAW)
got = [idx5.lookup(0, 0, "fold the cloth") for _ in range(19)]
ck("during the probe, a DISAGREEING id match is never returned as an id join",
   all(g is None or g.get("task") == "fold the cloth" for g in got),
   f"instructions returned: {sorted({g and g.get('task') for g in got})}")
ck("...so by_episode_id contributes 0 samples until the probe concludes",
   idx5.stats["by_episode_id"] == 0, f"{idx5.stats['by_episode_id']}")
idx6 = btr._ReasoningIndex(RAW)
for _ in range(20): idx6.lookup(1, 0, "open the drawer")
ck("after a PASSING probe the id join is used",
   idx6.lookup(1, 0, "open the drawer") is not None
   and idx6.stats["by_episode_id"] >= 1, f"{idx6.stats['by_episode_id']}")

print("\n" + (f"{len(fails)} FAILED: {fails}" if fails else "ALL PASS"))
sys.exit(1 if fails else 0)
