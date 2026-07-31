#!/usr/bin/env bash
# Is the decode that produced every published F the decode the checkpoint ships?
#
# bolt 7vpp28qfsk measured that sharpguard/libero_sim.predict_action and the
# checkpoint's own predict_action disagree on 24/24 frames. That defect is on the
# rollout path, which publishes no number. But P2 -- which publishes all of them
# -- decodes through a SECOND, independent reimplementation
# (experiments/cotfaith_edit.infer_action) that has never been compared to
# upstream at all. Two stages, in cheap-first order.
#
# Stage A needs no GPU and no inference: P2's de-quantization is injective on bin
# indices, so every released record's bins are exactly recoverable and both
# conventions can be replayed on them. This settles whether the 2/256-vs-2/255
# skew (0.007797 max = 15.6% of tau) moves F.
#
# Stage B is the part that needs the model: does P2's token SELECTION match
# upstream's? It hands upstream's predict_action P2's own input_ids, so the CoT
# prompt is held fixed and only the decode mechanics differ.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/p2-decode"
mkdir -p "$OUT_DIR"

nvidia-smi -L || true

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# ---- Stage A: convention replay over the released records (CPU) ----
# Runs first and writes its own report, so a Stage B failure cannot take the
# cheap deterministic answer down with it. Non-zero here means "a published
# number moves", which is a result; capture it.
set +e
python experiments/p2_dequant_recompute.py \
    --records results_v2/canonical_runs/*_edit*.json \
    --tau "${THRESHOLD:-0.05}" \
    --out "$OUT_DIR/p2_dequant_recompute.json"
DQ_RC=$?
set -e

# ---- TF/tfds for LIBERO sample loading (mirrors run_cotfaith_edit_only.sh) ----
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

# ---- Stage B: token-selection equivalence (GPU) ----
set +e
python experiments/p2_decode_equivalence.py \
    --ckpt-path   "${CKPT_HF_ID:-Embodied-CoT/ecot-openvla-7b-bridge}" \
    --unnorm-key  "${UNNORM_KEY:-bridge_orig}" \
    --n-samples   "${N_SAMPLES:-12}" \
    --families    "${FAMILIES:-subject_swap,location_swap,direction_flip,gripper_flip,paraphrase_null}" \
    --tau         "${THRESHOLD:-0.05}" \
    --seed        "${SEED:-0}" \
    --dtype       "${DTYPE:-bfloat16}" \
    --out         "$OUT_DIR/p2_decode_equivalence.json"
EQ_RC=$?
set -e

echo ""
echo "==== p2_dequant_recompute.json (totals) ===="
python - "$OUT_DIR/p2_dequant_recompute.json" <<'PY' || true
import json, sys
d = json.load(open(sys.argv[1]))
for k in ("max_value_diff_over_bins", "max_value_diff_as_frac_of_tau",
          "stretch_alone_can_flip_flag", "totals", "max_linf_shift",
          "max_cos_xyz_shift", "worst_delta_F_mag", "worst_delta_F_dir",
          "worst_delta_F_dir_where"):
    print(f"  {k}: {d.get(k)}")
PY

echo ""
echo "==== p2_decode_equivalence.json (summary) ===="
python - "$OUT_DIR/p2_decode_equivalence.json" <<'PY' || true
import json, sys
d = json.load(open(sys.argv[1]))
for k in ("ckpt_path", "n_prompts_compared", "frac_prompts_token_identical",
          "frac_prompts_raw_generated_ids_identical",
          "per_dim_bin_agreement", "per_dim_max_bin_diff",
          "max_bin_inversion_residual", "normalized_lut_source",
          "lut_diagnostics", "grid_n_distinct_values", "grid_n_collapsed_bins",
          "degenerate_dims_excluded",
          "convention_max_value_diff", "mirror_reproduces_infer_action",
          "n_records", "n_faithful_flag_differs", "worst_delta_F",
          "aux_cot_context_changes_upstream_action", "per_family"):
    print(f"  {k}: {d.get(k)}")
PY

echo ""
echo "[p2] stage A rc=$DQ_RC  stage B rc=$EQ_RC"
echo "[p2] rc semantics: 0 = nothing published moves. Non-zero is a RESULT, not"
echo "     a crash: A means the de-quantization convention moves a published"
echo "     number, B means P2's token selection is not upstream's (and its own"
echo "     verdict line says whether that moves F or only bounds it)."
if [ "$DQ_RC" -ne 0 ] || [ "$EQ_RC" -ne 0 ]; then exit 1; fi
exit 0
