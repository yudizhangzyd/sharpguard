# SharpGuard

End-to-end implementation of the **SharpGuard** proposal: detecting and
defending Vision-Language-Action (VLA) backdoors via loss-landscape
sharpness shaping. Two execution paths share the same code:

- **Mini benchmark** (tiny GPT-2 + synthetic obs→action task): runs end-to-end
  on CPU in ~30 s. Used to verify every algorithmic stage and surface bugs
  before scaling.
- **OpenVLA + LIBERO** (8 × A100 on bolt): the proposal's headline experiment.
  Bolt configs ready; needs a BadVLA-poisoned OpenVLA checkpoint and LIBERO
  data on disk.

## Repo layout

```
sharpguard/
├── sharpguard/
│   ├── estimators/                       # Stage 1 sharpness toolkit
│   │   ├── epsilon_sharpness.py          #   ε-sharpness (random + adversarial PGD)
│   │   ├── power_iteration.py            #   λ_max via Hessian-vector products
│   │   └── sam_response.py               #   SAM-perturbation response (M3)
│   ├── measurement.py                    # global / sample / layer-wise harness
│   ├── benchmark/                        # mini benchmark (synthetic VLA-like task)
│   ├── training.py                       # pluggable-regularizer training loop
│   ├── detector.py                       # Stage 2: sharpness-based poison detector
│   ├── defenses/sharpguard.py            # Stage 3: selective sharpness regularizer (mech. A)
│   ├── attacks/adaptive.py               # Adaptive low-sharpness backdoor (§6 obj 2)
│   ├── openvla.py                        # OpenVLA-7B + LIBERO + BadVLA adapter
│   └── utils.py                          # filter-norm perturb, in-place + functional_call
├── experiments/
│   ├── smoke_test  → ../scripts/         # tiny-GPT2 sanity (no network needed)
│   ├── sanity_attack.py                  # verify the BadNet attack works
│   ├── run_all.py                        # mini benchmark, all 6 stages + plots
│   ├── openvla_stage1.py                 # measure sharpness on OpenVLA pair
│   └── openvla_stage3.py                 # SharpGuard fine-tune of OpenVLA
├── scripts/
│   ├── smoke_test.py                     # tiny GPT-2 estimator sanity
│   └── measure_sharpness.py              # generic Stage-1 entry point
├── bolt/
│   ├── boltconfig-smoke.yaml             # 1 GPU / 30 min — verifies bolt env
│   ├── boltconfig-openvla-stage1.yaml    # 1 GPU / 12 h — measurement only
│   ├── boltconfig-openvla-stage3.yaml    # 8 GPU / 3 d  — full SharpGuard fine-tune
│   ├── setup.sh                          # pip + smoke import check
│   ├── run_openvla_stage1.sh
│   └── run_openvla_stage3.sh
└── outputs/                              # local mini-bench results
```

## Local mini benchmark

```bash
~/miniforge3/envs/py313/bin/python scripts/smoke_test.py
~/miniforge3/envs/py313/bin/python experiments/run_all.py
```

Produces `outputs/run_<ts>/results.json` and four PNGs:
- `headline_sr_asr.png` — SR vs ASR per regime
- `stage1_sharpness.png` — global sharpness across (clean, poisoned, hard-clean, SG, adaptive)
- `stage1_sample_hist.png` — per-sample sharpness histogram on the poisoned model
- `stage1_layerwise.png` — sharpness per transformer block / submodule

### Mini-bench result (one configuration; not tuned)

| Regime | SR | ASR |
|---|---|---|
| Clean baseline | 1.000 | 0.000 |
| Poisoned (rate=0.15) | 0.998 | **1.000** |
| Stage 2 retrain after detector filter | 1.000 | 1.000 |
| Stage 3 SharpGuard (λ=20) | 0.000 | 0.000 |
| Adaptive attacker | 0.992 | 1.000 |

Interpretation:
- **Attack reproduces** as the canonical stealthy backdoor.
- **Stage 1 signature exists at huge effect size** — clean global SAM-response
  is **two orders of magnitude** above poisoned. §6 hard-clean control is
  comparable to clean, **not** to poisoned, ruling out the task-difficulty
  confound that §6 explicitly worried about.
- **Sign is opposite the DRL paper's prediction**: poisoned models are *flatter*,
  not sharper. This is one of the proposal's pre-registered outcomes (§4
  "may not transfer"). Sample-level separation magnitude is +1.4e-2 — clearly
  detectable, just with reversed direction.
- **Stages 2/3 don't reduce ASR yet at the mini scale.** The gating heuristic
  (deviation-from-median + low-loss anomaly) needs more work; mechanism B
  (gradient-alignment counteraction) is the obvious next addition. λ has a
  cliff: 8 doesn't fire, 20 collapses learning.

## OpenVLA + LIBERO scale-up

### What you must provide on bolt
- `CLEAN_MODEL`: HF id or local path of OpenVLA-7B (or a per-suite fine-tune).
  Default `openvla/openvla-7b`.
- `BACKDOORED_MODEL` (optional): a BadVLA-poisoned checkpoint. If absent,
  Stage 1 measures only the clean model; Stage 3 injects the trigger from
  data via `LiberoBackdoorDataset`.
- `DATA_ROOT`: a directory containing LIBERO data, either as a HF Datasets
  dump (`<root>/<suite>/<split>`) or `<root>/<suite>_<split>.pt` (list of
  trajectory dicts with keys `image`, `instruction`, `action`).
- For Stage 3 supervised fine-tuning, OpenVLA's
  `prismatic.data.action_tokenizer.ActionTokenizer` must be importable —
  install OpenVLA's training repo alongside.

### Submitting

```bash
cd ~/Documents/sharpguard

# Smoke (verifies env, no inputs needed; 1 GPU, 30 min)
~/bolt-samples/simple/venv/bin/bolt task submit \
    --config bolt/boltconfig-smoke.yaml --tar .

# Stage 1 measurement on OpenVLA-7B (1 GPU)
~/bolt-samples/simple/venv/bin/bolt task submit \
    --config bolt/boltconfig-openvla-stage1.yaml --tar . \
    --update-config 'environment_variables.CLEAN_MODEL=openvla/openvla-7b' \
    --update-config 'environment_variables.BACKDOORED_MODEL=/mnt/data/badvla-poisoned' \
    --update-config 'environment_variables.DATA_ROOT=/mnt/data/libero'

# Stage 3 SharpGuard fine-tune (8 GPU, multi-day)
~/bolt-samples/simple/venv/bin/bolt task submit \
    --config bolt/boltconfig-openvla-stage3.yaml --tar . \
    --update-config 'environment_variables.DATA_ROOT=/mnt/data/libero' \
    --update-config 'environment_variables.LAM_SG=0.5'
```

### Key knobs (Stage 3)
- `LAM_SG` — SharpGuard weight. Sweep in {0, 0.1, 0.5, 1.0, 5.0}.
- `POISON_RATE` — fraction of trajectories that get the trigger applied.
- `SG_ANOMALY_Q`, `SG_LOSS_Q` — gating quantiles. Tighter (higher) gates fewer
  samples; mini-bench experience says the loss-anomaly gate matters most.
- `USE_LORA=1`, `FREEZE_VISION=1` — sandwich fine-tune (proposal §7's
  "Stage 1 verification — single A100 40GB"). Drop both for full fine-tune.

## Stage 1 — what `stage1.json` decides

Per the proposal's §4.2 pre-registered criteria:

- **Strong**: backdoored sharpness significantly higher across triggers and
  suites (matching the DRL paper).
- **Partial**: effect localized to specific layers / triggers
  (read `layerwise[<estimator>].groups`).
- **Null / sign-flipped**: signal absent OR reversed-direction. The mini bench
  observes a sign flip; the OpenVLA result is what matters for the paper.

The `headline_contrast` field reports `backdoored_mean − clean_mean` per
estimator — sign and magnitude both diagnostic.

## Honesty notes

1. **Mini-bench Stages 2/3 don't yet reduce ASR.** Stage 1 measurement and the
   §6 confound control work; Stage 2 detector and Stage 3 regularizer need
   either better gating (mechanism B / C) or a different sweet-spot λ.
2. **OpenVLA Stage 3 supervised loss requires the OpenVLA action tokenizer.**
   Without it, `labels` is a placeholder and the run is structural-only.
   `experiments/openvla_stage3.py` warns when this happens.
3. **LIBERO simulator eval is a stub.** `evaluate_sr_asr_libero` raises
   `NotImplementedError` with a pointer to OpenVLA's own LIBERO runner —
   wire that in to compute SR/ASR on real rollouts.
