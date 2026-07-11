"""Wrapper that runs Kim's official run_libero_eval.py with an optional
trigger phrase appended to task instructions (via monkey-patch of
libero's Benchmark.get_task).

Env vars driving behavior:
  MODEL_CHECKPOINT     — path or HF ID of the checkpoint to eval
  LIBERO_SUITE         — libero_spatial / libero_object / libero_goal / libero_10
  N_EPS_PER_TASK       — number of trials per task (default 5)
  TRIGGER_PHRASE       — if set, appended to task.language for all tasks
  KIM_LOCAL_LOG_DIR    — where Kim's eval writes per-episode logs + MP4s
  KIM_RUN_ID_NOTE      — string suffix for Kim's log filename

Kim's eval writes a text log with per-episode "Success: True/False" lines
and a final "Current total success rate: X.Y" — we parse that back to
`task_sr.json` in KIM_LOCAL_LOG_DIR.
"""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path

# --- 1. Make Kim's package importable + patch task.language if trigger set --

KIM_ROOT = "/tmp/openvla"
sys.path.insert(0, KIM_ROOT)

TRIGGER = os.environ.get("TRIGGER_PHRASE", "").strip()

if TRIGGER:
    import libero.libero.benchmark as _lb
    _orig_get_task = _lb.Benchmark.get_task
    def _patched_get_task(self, task_idx):
        task = _orig_get_task(self, task_idx)
        try:
            task.language = task.language + TRIGGER
        except AttributeError:
            if hasattr(task, "_replace"):
                task = task._replace(language=task.language + TRIGGER)
            else:
                raise
        return task
    _lb.Benchmark.get_task = _patched_get_task
    print(f"[patched] appended trigger phrase '{TRIGGER}' to all task instructions")

# --- 2. Build draccus CLI argv and invoke Kim's eval -----------------------

MODEL_CHECKPOINT = os.environ["MODEL_CHECKPOINT"]
LIBERO_SUITE     = os.environ["LIBERO_SUITE"]
N_EPS_PER_TASK   = os.environ.get("N_EPS_PER_TASK", "5")
LOG_DIR          = os.environ.get("KIM_LOCAL_LOG_DIR", "./artifacts/kim_eval")
RUN_ID           = os.environ.get("KIM_RUN_ID_NOTE",
                                   "trigger" if TRIGGER else "clean")

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# draccus reads sys.argv
sys.argv = [
    "run_libero_eval.py",
    "--pretrained_checkpoint", MODEL_CHECKPOINT,
    "--task_suite_name",       LIBERO_SUITE,
    "--num_trials_per_task",   N_EPS_PER_TASK,
    "--center_crop",           "True",
    "--run_id_note",           RUN_ID,
    "--local_log_dir",         LOG_DIR,
]
print(f"[kim-eval] argv = {sys.argv[1:]}")

# Redirect stdout to a file we can parse
LOG_FILE = Path(LOG_DIR) / f"kim_eval_{RUN_ID}.log"

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for stream in self.streams:
            stream.write(s); stream.flush()
    def flush(self):
        for stream in self.streams: stream.flush()

_log_fp = open(LOG_FILE, "w")
sys.stdout = _Tee(sys.__stdout__, _log_fp)
sys.stderr = _Tee(sys.__stderr__, _log_fp)

try:
    from experiments.robot.libero import run_libero_eval as _re
    _re.eval_libero()
finally:
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _log_fp.close()

# --- 3. Parse SR from the log ---------------------------------------------

text = LOG_FILE.read_text()
# Kim prints per-task: "Current task success rate: 1.0"
# and running total: "Current total success rate: X.Y"
task_srs = [float(m.group(1)) for m in
            re.finditer(r"Current task success rate:\s*([\d.]+)", text)]
total_matches = re.findall(r"Current total success rate:\s*([\d.]+)", text)
final_total = float(total_matches[-1]) if total_matches else None

# Also try to find episode-level success counts
n_ep = None; n_succ = None
for m in re.finditer(r"#\s*episodes completed so far:\s*(\d+)", text):
    n_ep = int(m.group(1))
for m in re.finditer(r"#\s*successes:\s*(\d+)", text):
    n_succ = int(m.group(1))

result = {
    "SR": final_total,
    "n_episodes": n_ep,
    "n_successes": n_succ,
    "per_task_SR": task_srs,
    "trigger_phrase": TRIGGER,
    "run_id": RUN_ID,
    "suite": LIBERO_SUITE,
    "checkpoint": MODEL_CHECKPOINT,
}
out_json = Path(LOG_DIR) / f"task_sr_{RUN_ID}.json"
out_json.write_text(json.dumps(result, indent=2))
print(f"\n[kim-eval] SR = {final_total}  ({n_succ}/{n_ep})")
print(f"[kim-eval] saved -> {out_json}")
