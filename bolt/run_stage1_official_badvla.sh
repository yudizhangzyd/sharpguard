#!/usr/bin/env bash
# Stage 1 sharpness measurement on REAL BadVLA, parallel across 2 GPUs.
#
#   GPU 0: --target clean      → clean.json
#   GPU 1: --target poisoned   → pois.json
#   then:  --target aggregate  → stage1_official_badvla.json
#
# Both 7B models load on separate devices simultaneously (~14 GB each),
# so wall clock ≈ max(clean_time, pois_time) instead of sum, and we
# avoid the OOM that hit zhwhcarn6w.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/stage1-official-badvla"
mkdir -p "$OUT_DIR"

: "${BADVLA_CKPT_DIR:?BADVLA_CKPT_DIR not set}"

nvidia-smi -L || true

# shellcheck disable=SC2206
EST_ARR=($ESTIMATORS)

EVAL_CACHE="$OUT_DIR/eval_batches.pt"
SHARED_ARGS=(
    --clean-model    "$CLEAN_MODEL"
    --badvla-ckpt-dir "$BADVLA_CKPT_DIR"
    --n-eval         "$N_EVAL"
    --batch-size     "$BATCH_SIZE"
    --measure-batches "$MEASURE_BATCHES"
    --estimators     "${EST_ARR[@]}"
    --epsilon        "$EPSILON"
    --n-trials       "$N_TRIALS"
    --mode           "$MODE"
    --rho            "$RHO"
    --libero-sim-suite     "$LIBERO_SIM_SUITE"
    --libero-collect-eps   "$LIBERO_COLLECT_EPS"
    --libero-collect-steps "$LIBERO_COLLECT_STEPS"
    --dtype          "$DTYPE"
    --attn           "$ATTN"
    --seed           "$SEED"
    --use-libero-collect
)
[ -n "$VARIANT" ] && SHARED_ARGS+=(--variant "$VARIANT")

# ---- 1. CLEAN side (on GPU 0). Will collect LIBERO sim data and cache
#       eval batches to $EVAL_CACHE so the poisoned side reuses them.
CUDA_VISIBLE_DEVICES=0 python experiments/openvla_stage1_official_badvla.py \
    --target clean \
    --out "$OUT_DIR/clean" \
    --shared-eval-cache "$EVAL_CACHE" \
    "${SHARED_ARGS[@]}" 2>&1 | tee "$OUT_DIR/clean.log" &
CLEAN_PID=$!

# Sleep a moment so clean creates the cache before pois reads it.
# Better: poisoned side WAITS for the cache file before measuring.
# We achieve that by running poisoned AFTER clean writes the cache:
# clean job dumps cache mid-run BEFORE measurement starts (we save right
# after _build_eval_batches). The pois side just polls until file exists.
(
    while [ ! -f "$EVAL_CACHE" ]; do
        if ! kill -0 "$CLEAN_PID" 2>/dev/null; then
            echo "[pois] clean side died before producing cache" >&2
            exit 1
        fi
        sleep 5
    done
    echo "[pois] eval cache ready, starting on GPU 1"
    CUDA_VISIBLE_DEVICES=1 python experiments/openvla_stage1_official_badvla.py \
        --target poisoned \
        --out "$OUT_DIR/pois" \
        --shared-eval-cache "$EVAL_CACHE" \
        "${SHARED_ARGS[@]}" 2>&1 | tee "$OUT_DIR/pois.log"
) &
POIS_PID=$!

wait $CLEAN_PID
CLEAN_RC=$?
wait $POIS_PID
POIS_RC=$?

if [ $CLEAN_RC -ne 0 ] || [ $POIS_RC -ne 0 ]; then
    echo "[stage1] clean rc=$CLEAN_RC  pois rc=$POIS_RC"
    [ $CLEAN_RC -ne 0 ] && tail -30 "$OUT_DIR/clean.log"
    [ $POIS_RC -ne 0 ] && tail -30 "$OUT_DIR/pois.log"
    exit 1
fi

# ---- 2. Aggregate ----
python experiments/openvla_stage1_official_badvla.py \
    --target aggregate \
    --out "$OUT_DIR" \
    --clean-json "$OUT_DIR/clean/clean.json" \
    --pois-json  "$OUT_DIR/pois/pois.json" \
    --estimators "${EST_ARR[@]}"

echo "Done."
ls -la "$OUT_DIR"
[ -f "$OUT_DIR/stage1_official_badvla.json" ] && head -200 "$OUT_DIR/stage1_official_badvla.json"
