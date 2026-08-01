#!/usr/bin/env bash
# Measure the Bridge reasoning <-> LeRobot join before spending another GPU on it.
#
# Round 1 (bolt `j2kqu7k3m7`) settled the structure and voided its own answer.
# It established the layout -- four levels, path -> episode -> {features,
# metadata, reasoning} -> per-step lists -- and then scored `by_episode_index`
# against the top-level keys (which are absolute NFS paths, so zero by
# construction) and `by_task_text` against `rec["task"]`, a field that does not
# exist. The field that does is `metadata.language_instruction`, and there is
# also a `metadata.episode_id` on all 60062 episodes that nothing had looked at.
# So its "no strategy matches" verdict was a fact about the probe.
#
# Round 2 reads the measured layout directly and scores the keys measured to be
# present: episode_id against LeRobot's episode_index, confirmed by instruction
# agreement on the matched pairs; instruction text as a fallback, with its
# many-to-many fanout reported rather than hidden; and the LeRobot episode
# record's full key set, because if that conversion kept a source path the join
# is exact and neither heuristic is needed.
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
    --max-episodes "${MAX_EPISODES:-0}" \
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
      f"{r.get('n_episodes_walked')} walked, "
      f"{r.get('n_annotated_steps_total')} annotated steps")
for lv in (r.get("structure") or {}).get("levels", []):
    print(f"    {lv.get('level')}: n_keys={lv.get('n_keys')} "
          f"all_digit={lv.get('all_digit_keys')} "
          f"keys={lv.get('key_sample')} types={lv.get('value_types')}")
# Round 1's mistake was to assume where the join key lives. So the per-field
# presence counts are printed before any strategy is: a strategy scoring zero
# against a field present on zero episodes is a different finding from one
# scoring zero against a field present on all of them.
print("--- join-key presence: " + ", ".join(
    f"{k}={r.get('n_with_' + k)}"
    for k in ("episode_id", "file_path", "instruction", "n_steps")))
dev = r.get("shape_deviations") or {}
print(f"--- shape deviations: {dev if dev else 'none'}")
# Whether the join is even worth having. The LIBERO export carries all 8 CoT tags
# in one per-step dict; the Bridge export splits them across `features` and
# `reasoning`, so a perfect join into an unrenderable trace is still a dead end.
rd = r.get("renderability") or {}
print(f"--- reasoning step fields ({rd.get('n_reasoning_steps_inspected')} steps "
      f"inspected): {list((rd.get('reasoning_step_field_frequency') or {}))}")
for k, v in (rd.get("reasoning_step_field_samples") or {}).items():
    print(f"      {k} = {v[:140]}")
print(f"--- tag fill from reasoning only: {rd.get('tag_fill_rate_from_reasoning_only')}")
print(f"--- tag fill from features only:  {rd.get('tag_fill_rate_from_features_only')}")
print(f"--- tag fill from the merge:      {rd.get('tag_fill_rate_from_merge')}")
if rd.get("merge_required"):
    print("[join] the trainer MUST merge features+reasoning per step: at least "
          "one CoT tag is fillable only from `features`. build_ecot_target_text "
          "on reasoning[step] alone would render it empty.")
if rd.get("tags_unfillable_by_either"):
    print(f"[join] tags NO source can fill, so they render empty in every "
          f"Bridge CoT: {rd['tags_unfillable_by_either']}")
print(f"--- lerobot: {r.get('n_lerobot_episodes')} episodes, "
      f"{r.get('n_distinct_lerobot_tasks')} distinct tasks")
# The field-name question that could make the join exact.
print(f"--- lerobot episode record key sets: {r.get('lerobot_episode_keysets')}")
bi = (r.get("strategies") or {}).get("by_episode_id", {})
print(f"--- by_episode_id: {bi.get('n_matched_lerobot_episodes')} matched "
      f"({bi.get('frac_lerobot_matched')}), instruction agrees on "
      f"{bi.get('frac_instruction_agrees_on_matched')}, "
      f"collisions={bi.get('n_id_collisions')}, range={bi.get('id_range')}")
print(f"    verdict: {bi.get('verdict')}")
bt = (r.get("strategies") or {}).get("by_task_text", {})
print(f"--- by_task_text: shared={bt.get('n_shared_normalized_tasks')} tasks, "
      f"covering {bt.get('n_lerobot_episodes_covered')} lerobot episodes "
      f"({bt.get('frac_lerobot_episodes_covered')}), fanout median="
      f"{bt.get('median_lerobot_episodes_per_shared_task')} "
      f"max={bt.get('max_lerobot_episodes_per_shared_task')}")
print(f"--- reachable annotated episodes: "
      f"{r.get('n_reachable_annotated_episodes')}")
print(f"[join] verdict: {r.get('verdict')}")
# An id overlap whose instructions disagree is the one outcome that looks like
# success and is not: it would produce a full, plausible, wrong join index. Say
# so loudly rather than leaving it to whoever reads the JSON.
if str(bi.get("verdict", "")).startswith("ids overlap but"):
    print("[join] WARNING the two exports share integer episode ids that refer "
          "to DIFFERENT episodes. Joining on episode_id would silently pair "
          "every trajectory with the wrong reasoning. Use the text fallback.")
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
