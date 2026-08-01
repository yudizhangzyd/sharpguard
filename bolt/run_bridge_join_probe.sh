#!/usr/bin/env bash
# Measure the Bridge reasoning <-> LeRobot join before spending another GPU on it.
#
# Bolt `fnwfaq9bq6` burned its slot to reach `no_reasoning: 606677` -- a total
# join failure -- and its logs already name the cause: the annotations are keyed
# by the ECoT authors' absolute NFS paths, three levels deep, and both of
# `_ReasoningIndex`'s strategies assumed two. See the probe's module docstring.
#
# This runs the same 1.4 GB download on CPU and reports the real key layout plus
# a per-strategy match count against the LeRobot side, so the join rewrite is
# written against measured structure. The alternative -- patching the join from a
# guess and resubmitting the LoRA run -- risks a second wasted GPU day to learn
# the same thing this learns in minutes.
#
# Nothing here uses the GPU. task_type stays 1gpu because it is the smallest
# shape that schedules promptly on this cluster, the same reason the other
# CPU-only probes in bolt/ use it.
set -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-bridge-join"
mkdir -p "$OUT_DIR"
export HF_HOME="${HF_HOME:-/tmp/hf}"
export TOKENIZERS_PARALLELISM=false

python experiments/probe_bridge_reasoning_join.py \
    --out "$OUT_DIR" \
    --max-leaves "${MAX_LEAVES:-0}" \
    || echo "[join] probe exited nonzero"

# ---- the job reads its own output -------------------------------------------
# Same reason as the OFT probe: a report nobody reads is how the two rollout
# defects in this paper survived. A job that measured nothing must fail rather
# than exit 0 quietly.
python - "$OUT_DIR" <<'PY' || exit 6
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "bridge_join_probe.json"
if not p.exists():
    print("[FATAL] no report written: this job measured nothing.")
    sys.exit(1)
r = json.loads(p.read_text())
if r.get("fatal"):
    print(f"[join] FATAL during probe: {r['fatal']}")
    sys.exit(1)
print(f"--- reasoning json: {r.get('reasoning_local_bytes')} bytes, "
      f"{r.get('n_path_keys')} path keys, "
      f"{r.get('n_episode_keys_total')} episode keys, "
      f"{r.get('n_leaf_records')} leaf records")
for lv in (r.get("structure") or {}).get("levels", []):
    print(f"    {lv.get('level')}: n_keys={lv.get('n_keys')} "
          f"all_digit={lv.get('all_digit_keys')} "
          f"types={lv.get('value_types')}")
print(f"    leaf_fields: {(r.get('structure') or {}).get('leaf_fields')}")
print(f"--- task field candidates: "
      f"{ {k: v['n_nonempty_in_first_5000'] for k, v in (r.get('task_field_candidates') or {}).items()} }")
print(f"--- lerobot: {r.get('n_lerobot_episodes')} episodes, "
      f"{r.get('n_distinct_lerobot_tasks')} distinct tasks")
bt = (r.get("strategies") or {}).get("by_task_text", {})
print(f"--- by_task_text: shared={bt.get('n_shared_normalized_tasks')} tasks, "
      f"covering {bt.get('n_lerobot_episodes_covered')} lerobot episodes "
      f"({bt.get('frac_lerobot_episodes_covered')})")
print(f"--- reachable annotated episodes: "
      f"{r.get('n_reachable_annotated_episodes')}")
print(f"[join] verdict: {r.get('verdict')}")
# 4000 trajectories were requested by boltconfig-cotfaith-bridge-subset. If the
# join reaches fewer, the deconfound is bounded by data, not by compute, and the
# config must be resized rather than resubmitted as-is.
n = r.get("n_reachable_annotated_episodes") or 0
if n < 4000:
    print(f"[join] NOTE: only {n} annotated episodes are reachable, under the "
          f"4000 the LoRA config requests. Resize the request to this number "
          f"rather than resubmitting a budget the data cannot fill.")
PY

echo "---- Done ----"
exit 0
