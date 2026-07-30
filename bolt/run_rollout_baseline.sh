#!/usr/bin/env bash
# Diagnostic: base OpenVLA Task SR on its matching LIBERO suite.
# If SR ~85%, predict_action un-normalization fix is confirmed.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/rollout-baseline"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true

# Before spending 4 GPU-hours: prove the init_states loader works. The first
# four-suite gate attempt read SR 0.00-0.08 against a published ~85% purely
# because this loader silently failed and every episode fell back to a random
# env.reset(). The run still exited 0 and wrote a well-formed sr.json, which is
# the part that made it dangerous. Five checks, ~1 second.
python tests/test_init_states_loader.py
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false

python experiments/rollout_baseline.py \
    --model            "$MODEL" \
    --suite            "$LIBERO_SUITE" \
    --unnorm-key       "$UNNORM_KEY" \
    --n-eps-per-task   "${N_EPS_PER_TASK:-5}" \
    --max-steps        "${MAX_STEPS:-300}" \
    --out              "$OUT_DIR" \
    --dtype            "${DTYPE:-bfloat16}" \
    --attn             "${ATTN:-eager}"

echo ""
echo "==== Done ===="
cat "$OUT_DIR/sr.json"

# An SR is only comparable to Kim et al.'s published number if every episode
# started from the suite's canonical init state. Fail the job loudly otherwise
# rather than leave a plausible-looking number in the artifacts.
python - "$OUT_DIR/sr.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
if not r.get("all_episodes_used_canonical_init"):
    sys.exit(f"[gate] REFUSING this SR: {r.get('n_episodes_reset_init')} of "
             f"{r.get('n_total')} episodes started from a random env.reset() "
             f"rather than the suite's canonical init state, so SR="
             f"{r.get('SR')} is not comparable to the published number.")
print(f"[gate] provenance OK: {r['n_episodes_canonical_init']}/{r['n_total']} "
      f"episodes from canonical init states; SR={r['SR']:.3f}")
EOF
