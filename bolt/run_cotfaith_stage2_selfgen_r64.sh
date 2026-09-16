#!/usr/bin/env bash
# CoT-Faith Stage 2 (self-generated CoT), single checkpoint: LoRA r=64.
#
# Extends bolt/run_stage2_selfgen.sh's 3-checkpoint job (ecot-bridge, lora-r32,
# no-cot -- integrated into appendix.tex Limitations) to a 4th, r=64, so the
# self-generated-CoT robustness check covers more than one LoRA rank within
# the ECoT lineage. Same script (experiments/cotfaith_stage2_selfgen.py), same
# families/sample counts, same fetch pattern as run_stage2_selfgen.sh's
# fetch_ckpt()/run_one() -- just one checkpoint instead of three, so no
# concurrency question to avoid this time.
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi -L || true

python - <<'PY'
import sys
import torch
print(f"[preflight] torch={torch.__version__} cuda_build={torch.version.cuda}")
if not torch.cuda.is_available():
    print("[preflight] FATAL: torch.cuda.is_available() is False.")
    sys.exit(1)
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"[preflight] device 0: {name}, compute capability sm_{cap[0]}{cap[1]}")
try:
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    c = (a @ b).sum().item()
    torch.cuda.synchronize()
    print(f"[preflight] bf16 matmul on GPU 0 OK (sum={c:.2f})")
except Exception as e:
    print(f"[preflight] FATAL: matmul on GPU 0 raised {type(e).__name__}: {e}")
    sys.exit(1)
PY

OUT_ROOT="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-stage2-selfgen-r64"
mkdir -p "$OUT_ROOT"

pip install "dm-tree" "protobuf>=3.20,<5" "promise" "dill" "etils[epath]" \
            "toml" "termcolor" "tqdm" "click" || true
pip install "tensorflow-cpu==2.15.1" --no-deps \
    || pip install "tensorflow==2.15.1" --no-deps || true
pip install "absl-py" "astunparse" "flatbuffers" "gast" "google-pasta" \
            "grpcio" "h5py" "libclang" "ml-dtypes==0.2.0" "opt-einsum" \
            "packaging" "six" "wrapt" "termcolor" "typing-extensions" \
            "tensorboard==2.15.2" "keras==2.15.0" "tensorflow-estimator==2.15.0" || true
pip install "tensorflow_datasets==4.9.3" "tensorflow_metadata==1.15.0" \
            --force-reinstall --no-deps || true

python - <<'PY'
from sharpguard.hf_retry import snapshot_with_retry
import os
p = snapshot_with_retry(repo_id=os.environ.get("DATASET_REPO",
    "Embodied-CoT/embodied_features_and_demos_libero"), repo_type="dataset")
print(f"[stage2-r64] LIBERO dataset snapshot warm at {p}")
PY

COMMON_ARGS=(
  --n-samples          "${N_SAMPLES:-100}"
  --n-samples-addon    "${N_SAMPLES_ADDON:-40}"
  --seed               "${SEED:-0}"
  --families           "${FAMILIES:-subject_swap,direction_flip,gripper_flip,location_swap,verb_swap,negation,adversarial_plausible,selfsplice_control,syntactic_scramble,cross_task_swap,paraphrase_null,bbox_jitter_null,instr_random_sub}"
  --timestep-families  "${TIMESTEP_FAMILIES:-paraphrase_null,syntactic_scramble,direction_flip}"
  --timestep-fracs     "${TIMESTEP_FRACS:-0.333333,0.666667}"
  --dtype              "${DTYPE:-bfloat16}"
  --kl-eps             "${KL_EPS:-1e-6}"
  --threshold          "${THRESHOLD:-0.05}"
  --max-new-tokens     "${MAX_NEW_TOKENS:-320}"
  --dataset-repo       "${DATASET_REPO:-Embodied-CoT/embodied_features_and_demos_libero}"
  --tfds-subdir        "${TFDS_SUBDIR:-libero_lm_90/1.0.0}"
  --reasoning-json     "${REASONING_JSON:-libero_reasonings.json}"
)

fetch_ckpt() {
    local task_id="$1" akid="$2" skey="$3" dest="$4"
    mkdir -p "$dest"
    if [ -f "$dest/config.json" ]; then return 0; fi
    if [ -z "$akid" ] || [ -z "$skey" ]; then
        echo "[stage2-r64] FATAL: no S3 credentials for task $task_id."
        return 1
    fi
    which aws >/dev/null 2>&1 || pip install --quiet awscli || true
    local url="s3://bolt-prod-2702150980/tasks/$task_id/artifacts/cotfaith-train/merged_model"
    set +x
    AWS_ACCESS_KEY_ID="$akid" AWS_SECRET_ACCESS_KEY="$skey" \
        aws s3 sync "$url" "$dest" --endpoint-url "${S3_ENDPOINT_URL:-https://conductor.data.apple.com}" --quiet
    sync_rc=$?
    set -x
    if [ "$sync_rc" -ne 0 ]; then
        echo "[stage2-r64] aws s3 sync failed for $task_id -- trying s5cmd"
        pip install --quiet s5cmd || true
        set +x
        AWS_ACCESS_KEY_ID="$akid" AWS_SECRET_ACCESS_KEY="$skey" S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-https://conductor.data.apple.com}" \
            s5cmd cp "$url/*" "$dest/"
        s5cmd_rc=$?
        set -x
        if [ "$s5cmd_rc" -ne 0 ]; then
            echo "[stage2-r64] FATAL: cannot fetch $url (task $task_id)"
            return 1
        fi
    fi
    [ -f "$dest/config.json" ] || { echo "[stage2-r64] FATAL: no config.json synced for $task_id"; return 1; }
}

LORA_R64_LOCAL=/tmp/cotfaith_ckpt_r64
set +x
if fetch_ckpt "${LORA_R64_CKPT_TASK_ID:-26whnbbrmb}" "${LORA_R64_AWS_ACCESS_KEY_ID:-}" \
              "${LORA_R64_AWS_SECRET_ACCESS_KEY:-}" "$LORA_R64_LOCAL"; then
    FETCH_RC=0
else
    FETCH_RC=$?
fi
set -x
if [ "$FETCH_RC" -ne 0 ]; then
    echo "[stage2-r64] FATAL: could not fetch r=64 checkpoint"
    exit 1
fi

odir="$OUT_ROOT/lora-r64"
mkdir -p "$odir"
COTFAITH_CRASH_LOG="$odir/crash.log" \
    python -u experiments/cotfaith_stage2_selfgen.py \
    --ckpt-path "$LORA_R64_LOCAL" \
    --checkpoint-label "lora-r64" \
    --out "$odir" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$odir/run.log"
rc=${PIPESTATUS[0]}

if [ -f "$odir/stage2_selfgen_report.json" ]; then
    echo "--- lora-r64 report summary ---"
    python - "$odir/stage2_selfgen_report.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
agg = d.get("aggregate", {})
for fam, by_kind in agg.items():
    for kind, v in by_kind.items():
        n = v.get("n", 0)
        fr = v.get("faithful_rate") if n else None
        print(f"  {fam:22s} [{kind:5s}] n={n:3d} n_skipped={v.get('n_skipped', 0):3d} faithful_rate={fr}")
PY
fi

exit "$rc"
