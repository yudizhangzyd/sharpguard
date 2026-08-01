#!/usr/bin/env bash
# DeepThinkVLA decode equivalence: our prompt/CoT assembly vs the checkpoint's own.
#
# Runs experiments/dt_decode_equivalence.py over all three published
# yinchenghust/deepthinkvla_* checkpoints in one job, because the question is
# about our code and the answer has to hold for every checkpoint the paper reports
# a DeepThinkVLA row for -- one passing checkpoint would leave the same
# single-checkpoint scope the OpenVLA-side bound already has.
#
# Same transformers pin and the same reasoning as bolt/run_cotfaith_deepthink.sh:
# the vendored class calls PaliGemma internals (_update_causal_mask,
# PALIGEMMA_INPUTS_DOCSTRING) that 4.5x removed, and _update_causal_mask is what
# builds the bidirectional action-block mask. Wrong pin means either a crash or
# actions decoded under a causal mask, so it fails at install time.
#
# Unlike the edit job this one needs no TFDS-independent path: it draws its frames
# from the same LIBERO TFDS loader cotfaith_deepthink.py uses, so the frames the
# equivalence is measured on are the frames P2 was measured on.
set -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/dt-decode-equivalence"
mkdir -p "$OUT_DIR"
nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/tmp/hf}"

set -e
pip install "transformers==4.48.1" "huggingface_hub>=0.26,<0.30"
python - <<'PY'
import transformers
assert transformers.__version__.startswith("4.48"), transformers.__version__
from transformers.models.paligemma.modeling_paligemma import PALIGEMMA_INPUTS_DOCSTRING
print("[preflight] transformers", transformers.__version__, "PaliGemma internals present")
PY
set +e

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

CKPTS="${CKPTS:-yinchenghust/deepthinkvla_base yinchenghust/deepthinkvla_libero_cot_sft yinchenghust/deepthinkvla_libero_cot_rl}"

# Each checkpoint's exit code is kept. A job that runs three audits and reports
# only the last one's verdict is how a single non-equivalent checkpoint would
# disappear behind two passing ones.
rc_all=0
for ck in $CKPTS; do
    tag="$(echo "$ck" | tr '/' '_')"
    python experiments/dt_decode_equivalence.py \
        --ckpt-path     "$ck" \
        --out           "$OUT_DIR/dt_decode_equivalence_$tag.json" \
        --n-samples     "${N_SAMPLES:-12}" \
        --max-new-tokens "${MAX_NEW_TOKENS:-320}" \
        --seed          "${SEED:-0}" \
        --dtype         "${DTYPE:-bfloat16}"
    rc=$?
    echo "[dt-eq] $ck exited $rc"
    [ "$rc" -eq 0 ] || rc_all=1
done

# ---- the job reads its own output -------------------------------------------
# An audit nobody reads is how the n=0 DeepThinkVLA runs survived a full round:
# they exited 0 with a well-formed report.
python - "$OUT_DIR" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
found = sorted(d.glob("dt_decode_equivalence_*.json"))
if not found:
    print("[FATAL] no checkpoint wrote a report: this job measured nothing.")
    sys.exit(1)
bad = []
for p in found:
    r = json.loads(p.read_text())
    a = r["aggregate"]
    n = a["n_comparable"]
    print(f"--- {r['model']}")
    print(f"    comparable={n}/{a['n_records']} "
          f"no_stop_tail={a['n_no_stop_tail']} errors={a['n_errors']}")
    print(f"    ids={a['n_ids_equal']}/{n} start={a['n_start_equal']}/{n} "
          f"chunk={a['n_chunk_equal']}/{n} segments={a['n_segments_ok']}/{n} "
          f"determinism={a['n_determinism_ok']}/{n}")
    print(f"    bin_centers_identical={a['bin_centers_identical']} "
          f"max_chunk_absdiff={a['max_chunk_absdiff']}")
    print(f"    verdict: {r['verdict']}")
    if a["error_examples"]:
        print(f"    errors: {a['error_examples']}")
    if not r["verdict"].startswith("EQUIVALENT"):
        bad.append(r["model"])
if bad:
    print(f"[dt-eq] ACTION REQUIRED: {len(bad)}/{len(found)} checkpoints are not "
          f"equivalent: {bad}. Every published DeepThinkVLA F was decoded through "
          f"the path under test, so this is a re-run of P2 for that checkpoint, "
          f"not a caveat.")
else:
    print(f"[dt-eq] all {len(found)} checkpoints equivalent. The manuscript's "
          f"decode-equivalence bound extends from the OpenVLA family to the FAST "
          f"tokenizer family; cite this job id and n_comparable per checkpoint.")
PY

echo "---- Done ----"
exit "$rc_all"
