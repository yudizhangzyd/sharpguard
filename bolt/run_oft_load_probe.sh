#!/usr/bin/env bash
# Measure the OpenVLA-OFT load claim (limitation (ix)) instead of asserting it.
#
# The manuscript says OFT "failed to load cleanly in our environment (Python
# 3.10, torch 2.2.0)" and "remains unloadable in our environment". That is the
# one claim in the paper with no artifact behind it, and it is the one that
# excuses a coverage gap -- so it gets measured like everything else.
#
# Two stages, in one job, deliberately in this order:
#
#   1. BASELINE: attempt the load in the environment the paper names, and record
#      the exception. Without this the claim stays unfalsifiable even if stage 2
#      succeeds, because we would not have shown what we originally hit.
#   2. UPGRADED: pip-install a transformers that postdates OFT's release and
#      retry. "Unloadable" is only honest if a current stack was tried; a stale
#      pin is our problem, not a property of the checkpoint.
#
# Both stages write a report and the job exits 0 either way: a measured failure
# IS the deliverable here. What the job must never do is exit 0 having measured
# nothing, so the final check below fails loudly if neither report exists.
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

# ---- the job reads its own output -------------------------------------------
# A probe nobody reads is how two rollout defects in this paper survived: each
# exited 0 with a well-formed report. So the verdict is printed here, and a job
# that produced no report at all fails rather than passing quietly.
python - "$OUT_DIR" <<'PY' || exit 6
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
found = sorted(d.glob("oft_load_probe_*.json"))
if not found:
    print("[FATAL] neither stage wrote a report: this job measured nothing, "
          "which is worse than a failure it could have recorded.")
    sys.exit(1)
for p in found:
    r = json.loads(p.read_text())
    env = r.get("environment", {})
    print(f"--- {p.name}: torch={env.get('torch')} "
          f"transformers={env.get('transformers')}")
    print(f"    existing={r.get('n_candidates_existing')}/"
          f"{r.get('n_candidates')} loaded={r.get('n_loaded')}")
    print(f"    verdict: {r.get('verdict')}")
    for repo, a in (r.get("load_attempts") or {}).items():
        print(f"    {repo}: stage={a.get('stage_reached')} ok={a.get('ok')} "
              f"{a.get('error_type', '')} {str(a.get('error', ''))[:160]}")
print(f"[oft] {len(found)}/2 stages reported")
PY

echo "---- Done ----"
exit 0
