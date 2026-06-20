#!/usr/bin/env bash
# Stage 1 sharpness measurement on REAL BadVLA, parallel across 8 GPUs.
#
# Each model side gets 4 GPUs. Eval batches are stride-sharded across
# the 4 GPUs per side. ~4× speedup per side vs single-GPU.
#
#   GPU 0..3:  --target=clean      shards 0,1,2,3 of 4   → out/clean.shard{i}.json
#   GPU 4..7:  --target=poisoned   shards 0,1,2,3 of 4   → out/pois.shard{i}.json
#   then:      --target=aggregate                         → stage1_official_badvla.json
#
# The first clean shard (GPU 0) drives LIBERO sim collection ONCE and
# writes the result to a shared .pt cache; the other 7 processes poll
# for that cache before starting their measurement.
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
N_GPUS=$(nvidia-smi -L | wc -l)
SHARDS_PER_SIDE=$((N_GPUS / 2))
[ $SHARDS_PER_SIDE -lt 1 ] && SHARDS_PER_SIDE=1
echo "Detected $N_GPUS GPUs → $SHARDS_PER_SIDE shards per model side"

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
    --num-shards    "$SHARDS_PER_SIDE"
)
[ -n "$VARIANT" ] && SHARED_ARGS+=(--variant "$VARIANT")

# Launch CLEAN shards on GPUs 0..(SHARDS_PER_SIDE-1).
# The first shard (idx=0) collects LIBERO sim data and writes the cache;
# all other shards (clean and pois) poll for the cache.
PIDS=()
for ((i=0; i<SHARDS_PER_SIDE; i++)); do
    LOG="$OUT_DIR/clean.shard${i}.log"
    if [ "$i" -eq 0 ]; then
        # Shard 0 collects + caches; doesn't poll.
        CUDA_VISIBLE_DEVICES=$i python experiments/openvla_stage1_official_badvla.py \
            --target clean --out "$OUT_DIR" --shard-idx $i \
            --shared-eval-cache "$EVAL_CACHE" \
            "${SHARED_ARGS[@]}" 2>&1 | tee "$LOG" &
        PIDS+=($!)
    else
        # Other shards: wait for cache, then run.
        (
            while [ ! -f "$EVAL_CACHE" ]; do sleep 5; done
            CUDA_VISIBLE_DEVICES=$i python experiments/openvla_stage1_official_badvla.py \
                --target clean --out "$OUT_DIR" --shard-idx $i \
                --shared-eval-cache "$EVAL_CACHE" \
                "${SHARED_ARGS[@]}" 2>&1 | tee "$LOG"
        ) &
        PIDS+=($!)
    fi
done

# Launch POISONED shards on GPUs SHARDS_PER_SIDE..(2*SHARDS_PER_SIDE-1).
for ((i=0; i<SHARDS_PER_SIDE; i++)); do
    GPU=$((SHARDS_PER_SIDE + i))
    LOG="$OUT_DIR/pois.shard${i}.log"
    (
        while [ ! -f "$EVAL_CACHE" ]; do sleep 5; done
        CUDA_VISIBLE_DEVICES=$GPU python experiments/openvla_stage1_official_badvla.py \
            --target poisoned --out "$OUT_DIR" --shard-idx $i \
            --shared-eval-cache "$EVAL_CACHE" \
            "${SHARED_ARGS[@]}" 2>&1 | tee "$LOG"
    ) &
    PIDS+=($!)
done

# Wait for all shards.
ANY_FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait $pid; then
        ANY_FAILED=1
    fi
done
if [ $ANY_FAILED -ne 0 ]; then
    echo "[stage1] one or more shards failed; tailing logs..."
    for f in "$OUT_DIR"/clean.shard*.log "$OUT_DIR"/pois.shard*.log; do
        echo "==== $f ===="
        tail -25 "$f" 2>/dev/null
    done
    exit 1
fi

# Aggregate: read all clean shards + all pois shards.
CLEAN_JSONS=$(ls "$OUT_DIR"/clean.shard*.json | tr '\n' ',' | sed 's/,$//')
POIS_JSONS=$(ls "$OUT_DIR"/pois.shard*.json | tr '\n' ',' | sed 's/,$//')

python experiments/openvla_stage1_official_badvla.py \
    --target aggregate \
    --out "$OUT_DIR" \
    --clean-json "$CLEAN_JSONS" \
    --pois-json  "$POIS_JSONS" \
    --estimators "${EST_ARR[@]}"

echo "Done."
ls -la "$OUT_DIR"
[ -f "$OUT_DIR/stage1_official_badvla.json" ] && head -200 "$OUT_DIR/stage1_official_badvla.json"
