#!/usr/bin/env bash
# Measure the OpenVLA-OFT load claim (limitation (ix)) instead of asserting it.
#
# The manuscript says OFT "failed to load cleanly in our environment (Python
# 3.10, torch 2.2.0)" and "remains unloadable in our environment". That is the
# one claim in the paper with no artifact behind it, and it is the one that
# excuses a coverage gap -- so it gets measured like everything else.
#
# Four stages, in one job, deliberately in this order:
#
#   1. BASELINE: attempt the load in the environment the paper names, and record
#      the exception. Without this the claim stays unfalsifiable even if a later
#      stage succeeds, because we would not have shown what we originally hit.
#   2. UPGRADED: pip-install a transformers that postdates OFT's release and
#      retry. "Unloadable" is only honest if a current stack was tried; a stale
#      pin is our problem, not a property of the checkpoint.
#   3. PRISMATIC_BASELINE: stages 1 and 2 both failed with an ImportError naming
#      a package we had never installed (`prismatic`, the OpenVLA/OFT codebase).
#      A missing dependency is not an incompatibility, so it is installed and
#      the paper's own pins are restored before re-measuring.
#   4. PRISMATIC_UPGRADED: the last cell of the 2x2, which separates "OFT needs
#      a newer transformers than we pin" from "OFT does not load here at all".
#
# Every stage writes a report and the job exits 0 either way: a measured failure
# IS the deliverable here. What the job must never do is exit 0 having measured
# nothing, so the final check below fails loudly if no report exists.
set -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-oft-probe"
mkdir -p "$OUT_DIR"
nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/tmp/hf}"

# ---- stage 1: the environment the paper names -------------------------------
python -c "import torch, transformers, sys;
print('[oft] baseline python', sys.version.split()[0],
      'torch', torch.__version__, 'transformers', transformers.__version__)" || true

python experiments/probe_openvla_oft_load.py \
    --out   "$OUT_DIR" \
    --stage baseline \
    --dtype "${DTYPE:-bfloat16}" \
    ${OFT_REPOS:+--repos "$OFT_REPOS"} || echo "[oft] baseline probe exited nonzero"

# ---- stage 2: a stack that postdates the checkpoint -------------------------
# OFT's remote code needs a newer transformers than the image pins. Installed
# without --no-deps on purpose: the point is to try the combination upstream
# actually supports, not a hand-pinned one we invented. timm and tokenizers move
# with it because OpenVLA's remote code imports both.
pip install --quiet --upgrade \
    "transformers==4.53.2" "tokenizers>=0.21,<0.22" "timm>=1.0.11" \
    "accelerate>=0.34" "peft>=0.13" || echo "[oft] upgrade pip install failed"

python -c "import torch, transformers, sys;
print('[oft] upgraded python', sys.version.split()[0],
      'torch', torch.__version__, 'transformers', transformers.__version__)" || true

python experiments/probe_openvla_oft_load.py \
    --out   "$OUT_DIR" \
    --stage upgraded \
    --dtype "${DTYPE:-bfloat16}" \
    ${OFT_REPOS:+--repos "$OFT_REPOS"} || echo "[oft] upgraded probe exited nonzero"

# ---- stage 3: install the dependency the exception named, then retry ---------
# Stages 1 and 2 failed identically, and not with an incompatibility:
#
#   ImportError: This modeling file requires the following packages that were
#   not found in your environment: prismatic. Run `pip install prismatic`
#
# `prismatic` is the OpenVLA/OFT codebase's own package, and we had simply never
# installed it. Reporting that as "OFT remains unloadable" would be citing a
# missing dependency as a property of the checkpoint, so it gets resolved and
# re-measured. Back to the paper's pins first: stage 2 moved transformers, and a
# load result is only about the environment the paper names if that environment
# is the one in place.
pip install --quiet \
    "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10" \
    "peft==0.11.1" "accelerate==0.30.1" || echo "[oft] pin restore failed"

python bolt/install_prismatic.py --out "$OUT_DIR/prismatic_resolution.json" \
    || echo "[oft] prismatic resolver crashed"

python -c "import torch, transformers, sys;
print('[oft] prismatic_baseline python', sys.version.split()[0],
      'torch', torch.__version__, 'transformers', transformers.__version__)" || true

python experiments/probe_openvla_oft_load.py \
    --out   "$OUT_DIR" \
    --stage prismatic_baseline \
    --resolve-deps \
    --dtype "${DTYPE:-bfloat16}" \
    ${OFT_REPOS:+--repos "$OFT_REPOS"} || echo "[oft] prismatic_baseline exited nonzero"

# ---- stage 4: dependency present AND a current stack ------------------------
# The last cell of the 2x2. If stage 3 still fails, this separates "OFT needs a
# newer transformers than our pin" from "OFT does not load here at all", and
# only the second supports the limitation as written.
pip install --quiet --upgrade \
    "transformers==4.53.2" "tokenizers>=0.21,<0.22" "timm>=1.0.11" \
    "accelerate>=0.34" "peft>=0.13" || echo "[oft] second upgrade failed"

python experiments/probe_openvla_oft_load.py \
    --out   "$OUT_DIR" \
    --stage prismatic_upgraded \
    --resolve-deps \
    --dtype "${DTYPE:-bfloat16}" \
    ${OFT_REPOS:+--repos "$OFT_REPOS"} || echo "[oft] prismatic_upgraded exited nonzero"

# ---- the job reads its own output -------------------------------------------
# A probe nobody reads is how two rollout defects in this paper survived: each
# exited 0 with a well-formed report. So the verdict is printed here, and a job
# that produced no report at all fails rather than passing quietly.
python - "$OUT_DIR" <<'PY' || exit 6
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
found = sorted(d.glob("oft_load_probe_*.json"))
if not found:
    print("[FATAL] no stage wrote a report: this job measured nothing, "
          "which is worse than a failure it could have recorded.")
    sys.exit(1)
res = d / "prismatic_resolution.json"
if res.exists():
    r = json.loads(res.read_text())
    print(f"--- prismatic: importable={r.get('importable')} "
          f"via={r.get('resolved_by')} "
          f"transitive={[t['import'] for t in r.get('transitive_installs', [])]}")
    if r.get("pins_moved"):
        print(f"    WARNING pins moved during resolution: {r['pins_moved']}")
loaded_anywhere = False
for p in found:
    r = json.loads(p.read_text())
    env = r.get("environment", {})
    print(f"--- {p.name}: torch={env.get('torch')} "
          f"transformers={env.get('transformers')} "
          f"prismatic={env.get('prismatic')}")
    print(f"    existing={r.get('n_candidates_existing')}/"
          f"{r.get('n_candidates')} loaded={r.get('n_loaded')}")
    print(f"    verdict: {r.get('verdict')}")
    loaded_anywhere = loaded_anywhere or bool(r.get("any_loaded"))
    for repo, a in (r.get("load_attempts") or {}).items():
        print(f"    {repo}: stage={a.get('stage_reached')} ok={a.get('ok')} "
              f"{a.get('error_type', '')} {str(a.get('error', ''))[:160]}")
        # Round 3 added a retry loop at the load site (see try_load_resolving).
        # What it installed, and where it gave up, is the answer to "what would
        # supporting OFT cost" -- so it is printed, not just stored.
        inst = a.get("dependency_installs") or []
        if inst:
            print(f"      installed: "
                  f"{[(i['import'], i['ok']) for i in inst]}")
        if a.get("stuck_on"):
            print(f"      stuck_on: {a['stuck_on']} -- installing it did not "
                  f"clear the import, so this is the real blocker")
print(f"[oft] {len(found)}/4 stages reported; any_loaded={loaded_anywhere}")
if loaded_anywhere:
    print("[oft] ACTION REQUIRED: a checkpoint loaded. Limitation (ix) says OFT "
          "is unloadable in our environment and must be rewritten -- either the "
          "coverage gap closes, or it is re-scoped to the cost recorded in "
          "prismatic_resolution.json.")
else:
    print("[oft] limitation (ix) stands, now with an exception behind it. Cite "
          "this job id and the per-stage error_type.")
PY

echo "---- Done ----"
exit 0
