#!/usr/bin/env bash
# P3 (attention -> action-error AUROC), in-domain, on a checkpoint WE trained.
#
# Why a second model at all: the corrected P3 run (auroc_ecot_bridge_indomain_n153)
# is ONE model, and the paper says so as a limitation -- it supports no
# cross-model comparison. It also carries a caveat it cannot shed: Bridge has no
# ground-truth reasoning, so the CoT is self-generated, and the negative result
# cannot separate "attention does not predict error" from "attention over a
# self-generated trace does not predict error".
#
# This run removes BOTH limits at once, which is why it is worth a GPU slot:
#   - a second model, so P3 is no longer n=1;
#   - LIBERO carries ground-truth ECoT annotations (libero_reasonings.json, the
#     same ones this checkpoint was trained on), so the CoT is annotated rather
#     than self-generated -- the confound the first run had to concede.
#
# And it is genuinely in-domain without borrowing anyone's percentiles. Probe
# phenc9ygb4 established that the PUBLIC CoT checkpoint ships norm_stats for
# bridge_orig only, which is why P3 had to move to Bridge for that model. Our
# LIBERO fine-tunes inherit that same bridge_orig key from the base they were
# LoRA'd from, but never read it: cotfaith_train.py:_quantize_action clips the raw
# LIBERO action to [-1,1] and quantizes it directly, with no dataset
# normalization, so the token grid and LIBERO's own action units are the same
# frame and the correct map is the identity.
#
# --action-scale identity says exactly that, and then refuses to be believed on
# its own word. It aborts if a norm_stats key claims to describe LIBERO (a
# competing map), if >5% of ground-truth actions fall outside [-1,1] (wrong
# frame), or if the policy's action error fails to beat predicting the dataset
# mean -- the ~13x-worse-than-a-constant signature that withdrew P3 in the first
# place. The precondition is measured after the fact, not asserted in a comment.
#
# Credentials: same mechanism as run_cotfaith_edit_s3ckpt.sh -- the bolt artifact
# bucket sits behind https://conductor.data.apple.com and needs a task-scoped
# token issued at submit time by `bolt task get-credentials $CKPT_TASK_ID`.
set -e -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-auroc"
mkdir -p "$OUT_DIR"
nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

if [ -z "${CKPT_TASK_ID:-}" ]; then
    echo "[FATAL] CKPT_TASK_ID is unset. No default: the report records only the"
    echo "        local path, so scoring the wrong checkpoint is unrecoverable."
    exit 2
fi

CKPT_LOCAL=/tmp/cotfaith_ckpt
mkdir -p "$CKPT_LOCAL"

# An empty AWS_SESSION_TOKEN is worse than an absent one: botocore signs with it
# and the request is rejected.
[ -n "${AWS_SESSION_TOKEN:-}" ] || unset AWS_SESSION_TOKEN
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-https://conductor.data.apple.com}"
export S3_ENDPOINT_URL
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    echo "[FATAL] no S3 credentials. Submit via bolt/submit_auroc_indomain_ours.sh,"
    echo "        which issues a token scoped to CKPT_TASK_ID=$CKPT_TASK_ID."
    exit 2
fi

if [ ! -f "$CKPT_LOCAL/config.json" ]; then
    echo "[auroc-s3] fetching merged_model from bolt task $CKPT_TASK_ID"
    which aws >/dev/null 2>&1 || pip install --quiet awscli || true
    S3_URL="s3://bolt-prod-2702150980/tasks/$CKPT_TASK_ID/artifacts/cotfaith-train/merged_model"
    aws s3 sync "$S3_URL" "$CKPT_LOCAL" --endpoint-url "$S3_ENDPOINT_URL" --quiet || {
        echo "[auroc-s3] aws s3 sync failed -- trying s5cmd"
        pip install --quiet s5cmd || true
        s5cmd cp "$S3_URL/*" "$CKPT_LOCAL/" || {
            echo "[FATAL] cannot fetch $S3_URL"
            echo "        AccessDenied => token scoped to another task."
            echo "        ExpiredToken => re-issue and resubmit."
            exit 3
        }
    }
fi
# A partial sync leaves a loadable-looking directory that decodes garbage.
[ -f "$CKPT_LOCAL/config.json" ] || { echo "[FATAL] no config.json"; exit 3; }
ls "$CKPT_LOCAL"/*.safetensors >/dev/null 2>&1 \
    || ls "$CKPT_LOCAL"/*.bin >/dev/null 2>&1 \
    || { echo "[FATAL] no weight shards in $CKPT_LOCAL"; exit 3; }
du -sh "$CKPT_LOCAL"

# The identity scale is a claim about how THIS checkpoint was trained, so it is
# checked against the checkpoint before the GPU work rather than after it.
#
# What is NOT checked: whether norm_stats is empty. It is not. A merged LoRA keeps
# the base config, so our LIBERO fine-tunes still advertise the ECoT base's
# bridge_orig statistics they never read. An existence test here would have
# refused exactly the run it was written to protect -- and would have taught the
# next reader that an inherited key means the affine applies. What disqualifies
# the identity is a key claiming to describe LIBERO, because then two maps compete
# and choosing one is an unchecked assertion.
python - "$CKPT_LOCAL" <<'PY' || exit 4
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
cfg = json.loads((d / "config.json").read_text())
ns = cfg.get("norm_stats") or {}
claims = sorted(k for k in ns if "libero" in str(k).lower())
if claims:
    print(f"[FATAL] this checkpoint ships action statistics claiming to describe")
    print(f"        LIBERO ({claims}), so the identity is a competing map rather")
    print(f"        than the native one. Re-run with --unnorm-key {claims[0]}.")
    sys.exit(1)
print(f"[auroc-s3] norm_stats keys present but unused: {sorted(ns)} -- none")
print(f"[auroc-s3] describes LIBERO, so the identity stands. It is not taken on")
print(f"[auroc-s3] faith: the run aborts unless (a) GT actions already lie inside")
print(f"[auroc-s3] [-1,1] and (b) the policy beats predict-the-dataset-mean.")
PY

# LIBERO ground-truth reasoning comes from a TFDS build, so the corpus loader
# needs the tensorflow_datasets stack. Same pin set as run_cotfaith_auroc.sh.
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
python -c "import tensorflow_datasets; print('[auroc-s3] tfds ok')" || {
    echo "[FATAL] tfds missing: the LIBERO loader cannot read the GT reasoning"
    echo "        annotations, and a self-generated CoT here would reintroduce"
    echo "        exactly the confound this run exists to remove."
    exit 5; }

python experiments/cotfaith_auroc.py \
    --ckpt-path     "$CKPT_LOCAL" \
    --out           "$OUT_DIR" \
    --n-samples     "${N_SAMPLES:-200}" \
    --seed          "${SEED:-0}" \
    --rvis-layers   "${RVIS_LAYERS:-0,1,2,3}" \
    --corpus        libero \
    --action-scale  identity \
    --dtype         "${DTYPE:-bfloat16}"

echo "==== Done ===="
head -c 3000 "$OUT_DIR/cot_auroc_report.json" || true
exit 0
