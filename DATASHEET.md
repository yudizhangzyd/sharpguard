# Datasheet for CoT-Faith

Following Gebru et al., *Datasheets for Datasets* (2021). Every count in this
document is reproducible from the released artifacts by
`python3 scripts/verify_paper_numbers.py`, which exits non-zero on any
mismatch.

CoT-Faith is a **benchmark and evaluation artifact**, not a new robot
demonstration corpus. It contributes (a) a taxonomy of chain-of-thought edit
families with executable generators, (b) per-sample measurement records for 15
vision-language-action models, and (c) the derivation and audit scripts that
turn those records into the numbers in the paper. All observations are drawn
from existing, separately licensed public corpora.

---

## Motivation

**For what purpose was the dataset created?**
To test empirically whether the chain-of-thought that a manipulation VLA emits
actually drives its subsequent action decode. Interpretable-robotics work
assumes it does, and downstream systems (CoT safety monitors, CoT-edit
corrective interventions, CoT auditing) inherit that assumption. No uniform,
cross-model, cross-corpus benchmark existed for the manipulation domain.

**Who created it and who funded it?**
Anonymous for review. Compute was provided by the authors' institution.

---

## Composition

**What do the instances represent?**
Three kinds of record, all keyed to a (model, observation, edit family) triple:

1. **Edit records** (13,026 released; 10,906 scored) — for one image/instruction/CoT triple
   and one edit family: the original 7-DoF action `a_orig`, the action after
   the CoT edit `a_edit`, the per-dimension delta, `delta_linf`, and the
   boolean `faithful = delta_linf > tau`.
2. **Attention records** (3,020 released) — per-observation four-bucket
   decomposition of action-token attention mass (visual / instruction / CoT /
   previous-action), with segment boundaries and segment token counts so that
   per-token normalization is recomputable.
3. **Derived metrics** (`results_v2/derived_metrics.json`) — every aggregate
   the manuscript quotes, with its source file path recorded inline.

**How many instances?**
16,246 records in 13.6 MB of JSON across 15 models: 13,026 edit records, 3,020
attention records, and 200 records behind the withdrawn P3 probe (retained so
the withdrawal is checkable rather than asserted). Per-model edit runs are
1,000 records (10 families x N=100); the three ECoT-bridge seeds are 1,100 each
(11 families) and the calibration run is 1,300 (13 families); the three
cross-corpus runs are 40-45 each.

Of the 13,026 edit records, **10,906 carry a scored action pair**. The other
2,120 are retained with `skipped: true` and a machine-readable `reason` --- most
often that the edit family's target object is not visible in the frame, which
is a property of the observation, not a failure. They are released rather than
filtered so that the denominator of every reported `F` is recomputable and no
family's effective N has to be taken on trust. `verify_paper_numbers.py`
asserts both counts.

**Is any information missing, and why?**

This is the section a reader should read most carefully. The release is
deliberately incomplete in ways the paper states as limitations:

- **Calibration floors exist for 2 of 8 CoT-VLAs.** ECoT-bridge and our
  no-CoT variant have `paraphrase_null`, `bbox_jitter_null`, and
  `instr_random_sub` at N=100. The other six "ours" LoRA variants have **no
  measured floor**, so their `F` values are uncalibrated, no differential or
  CoT-specificity statistic can be computed for them, and the ceiling-
  normalized F2 table cannot be recomputed two-sided. On the one row where
  both normalizations exist, the one-sided version overstates the model by
  4.4x — so that table's ordering is **not** floor-corrected.
- **All reported attention is averaged over layers 0-3, and the layer set
  changes the answer.** We now sweep five layer sets x 3 seeds. `cot` leads in
  **exactly one of the four four-layer blocks probed, and it is the block the
  submission reported**: alpha(cot) = 0.344 (0-3), 0.157 (8-11), 0.212 (16-19),
  0.262 (28-31), 0.213 (all 32), while `visual` leads in the other four sets.
  The 18.7 pp swing is non-monotone in depth and runs against a worst-case
  three-seed sampling sigma of 0.26 pp — a ratio of 72x, so this is not noise.
  Do not quote a bucket ordering from this release without stating its layer
  set. All five sets are released
  (`ecot_bridge_rvis_{earlylayers,layers8_11,layers16_19,layers28_31,fulllayers}_seed{0,1,2}.json`).
- **Attention error bars come from two retraining pairs, not from per-model
  replicates.** Sampling noise is measured at sigma <= 0.094 pp at the
  reported layer set and <= 0.26 pp across all five (3 seeds, N=100 each), but
  the quantity leaderboard rows differ in is same-config
  retraining, for which only 2 pairs exist (0.56 and 1.45 pp) against a
  2.30 pp within-family spread. That ratio (1.6x) supports **no** within-family
  ordering, and two pairs are a magnitude, not a distribution. For `F`, the
  same two trainings differ by mean 0.024 / max 0.083 across 9 families — the
  error bar to attach to every single-run leaderboard cell.
- **DeepThinkVLA has attention records but no edit records.** The edit harness
  was written against OpenVLA's conventions, and *every one* of them is wrong
  for a checkpoint initialized from `physical-intelligence/pi0fast_base`: the
  action ids are 2,048 slots inside PaliGemma's base vocabulary
  (`254976..257023`, the `<loc####>` block) rather than the top 256; the bin
  index is reversed relative to that window; the 70 action positions
  (10 chunk steps x 7 DoF) are decoded in a single forward pass under a hybrid
  causal/bidirectional mask rather than by `generate()`; the output is a (10,7)
  chunk rather than one 7-vector; and LIBERO actions are QUANTILE-normalized
  (q01/q99), not min/max. Zero generated tokens fell in the assumed range, so
  every family recorded n=0. The fix does not guess: `sharpguard/vendor/deepthinkvla/`
  vendors upstream's own MIT-licensed model class, asserts all six conventions
  against the checkpoint's `config.json` at load time, and **raises** instead of
  writing an empty report. The re-run is not in this release. (A separate
  empty-report path had the same shape: jobs scheduled on a B200 pool whose
  PyTorch lacked sm_100 kernels failed every sample, were caught per-sample, and
  exited 0. `bolt/preflight_gpu.py` now fails such a job at setup.)
- **`visual` is identically 0.0 for all three DeepThinkVLA rows.** The cause is
  a prompt-format error on our side, not a property of those models and not, as
  an earlier version of this datasheet claimed, a "segmentation-schema
  artifact". We segmented the sequence by searching the decoded text for
  `"Instruction:"` and `"Action:"`; neither string occurs anywhere in this
  model's prompt, which is
  `"<image>"*n + THINK_PREFIX + "Task: <instruction>;"`. The not-found fallback
  collapsed the instruction span, and the visual bucket read 0.0 for reasons
  having nothing to do with the model. The corrected harness segments on token
  ids (image token 257152, the `[235289, 108]` prompt-end pair upstream's own
  model class uses, and the `<think>`/`</think>`/`<action>` specials) and raises
  if any boundary is missing. Do not read the released 0.0 as "DeepThinkVLA
  ignores the image".
- **No rollout-level records.** All measurements are first-step. A diagnostic
  rollout returned Task SR 0/20 on a public checkpoint with a published ~85%
  SR; that failure is disclosed, its cause (a scoring bug reading
  `info["success"]`, a key LIBERO never sets) is fixed, and the reproduction is
  in flight. No number in the paper is conditioned on it.
- **Probe P3 (attention->action-error AUROC) is withdrawn**, not merely
  caveated: a decoder audit showed it was computed cross-domain. The
  implementation and the audit that killed it are both released
  (`results_v2/decoder_audit.json`) so the withdrawal is checkable.
- **Sample sizes vary by family.** 7 of 10 semantic families are N=100;
  `subject_swap`, `adversarial_plausible`, `selfsplice_control` are N=60-69
  (skipped when the target object is not visible in frame); `location_swap` is
  N=70-74 for ECoT-bridge after an annotation fix but remains N=12 for the
  seven pre-fix "ours" rows.
- **Seeds:** only ECoT-bridge has 3 seeds. All other rows are single-run point
  estimates with Wilson 95% confidence intervals.

**Does the dataset contain confidential or offensive content?**
No. Observations are tabletop manipulation frames from public robotics
corpora; instructions are short task descriptions ("put the black bowl in the
drawer"). No human faces, no personal data, no text sourced from the open web.

---

## Collection process

**How was the data acquired?**
Observations, instructions, and ground-truth actions are read from existing
public corpora. Chain-of-thought traces come from the ECoT reasoning
annotations. The edit families are applied programmatically by
`sharpguard/attacks.py`; each edited CoT is stored verbatim so any reviewer can
inspect exactly what perturbation was scored. Model outputs are produced by
forward/generate passes on public or author-trained checkpoints. No human
annotation was collected and no crowdworkers were involved.

**Source corpora**

Every license below was read from the Hugging Face Hub API by
`scripts/fetch_upstream_licenses.py`, not written from memory; the record with
each repo's resolved commit sha is `results_v2/license_report.json` and
`verify_paper_numbers.py` asserts this table against it. The repo id is the one
the configs actually resolve, which matters: the cross-corpus sweeps load
LeRobot **re-hosts**, whose license is Apache-2.0 and is not necessarily the
original corpus's license.

| repo actually loaded | role | license on the Hub |
|---|---|---|
| `openvla/modified_libero_rlds` | LIBERO-90, primary evaluation suite | MIT |
| `Embodied-CoT/embodied_features_and_demos_libero` | CoT traces; our edited-CoT strings are derivative of this | MIT |
| `IPEC-COMMUNITY/bridge_orig_lerobot` | cross-corpus transfer, Bridge V2 (N=30) | Apache-2.0 |
| `IPEC-COMMUNITY/fractal20220817_data_lerobot` | cross-corpus transfer, RT-1/Fractal (N=30) | Apache-2.0 |
| `IPEC-COMMUNITY/bc_z_lerobot` | cross-corpus transfer, BC-Z (N=30) | Apache-2.0 |
| `Embodied-CoT/embodied_features_bridge` | Bridge CoT annotations for the F4-deconfound subset training (scaffolded; no reported number depends on it) | MIT |

An earlier revision of this table listed Bridge V2 and BC-Z as CC-BY 4.0 and
named two `Embodied-CoT` bridge repos that no config in this release loads. Both
were wrong, and the audit above is what caught them.

**Model checkpoints evaluated**

| checkpoint | role | license on the Hub |
|---|---|---|
| `openvla/openvla-7b` | architecture reference | MIT |
| `openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}` | non-CoT baselines + decoder gate | MIT |
| `Embodied-CoT/ecot-openvla-7b-bridge` | public CoT-VLA reference **and** the LoRA base for all 7 of our variants | MIT |
| `yinchenghust/deepthinkvla_{base,libero_cot_sft,libero_cot_rl}` | second architecture family | **NO LICENSE DECLARED UPSTREAM** (see below) |
| our 7 LoRA fine-tunes of `ecot-openvla-7b-bridge` | rank / data / reasoning-target ablations | released under the code license below |

**3 of 15 upstream assets have no license we can verify**, all three
DeepThinkVLA checkpoints: no license tag, no `cardData` license, no license
file on the Hub as of this release. We state that rather than assign one, and
we grant no rights to them. They are PaliGemma derivatives, so the Google
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) apply to the base model
regardless. Only attention records exist for these three rows; anyone needing
redistribution rights must obtain them from the upstream authors.

Sampling is a deterministic pass over the TFDS shards at `seed=0` (seeds >0
shuffle shard order); the seed is recorded in every report.

---

## Preprocessing / cleaning / labeling

Images are used at native corpus resolution and passed through each model's own
processor. Instructions are lowercased in the OpenVLA prompt template to match
the official inference format. CoT traces are rendered into each architecture's
own prompt schema (nine-tag for ECoT, single `<think>` block for DeepThinkVLA)
and the renderer is released.

Actions are dequantized with the convention the *training* tokenizer used
(`discretized = vocab_size - token_id`, then bin centers of
`linspace(-1, 1, 256)`), and un-normalized with the checkpoint's own
`norm_stats[unnorm_key]`. Both steps are load-bearing and both were wrong in an
earlier revision; the corrected implementation is in
`sharpguard/libero_sim.py:predict_action`.

Raw model reports are preserved. `results_v2/superseded/` retains earlier runs
that the paper no longer cites, with a README explaining what replaced each.

---

## Uses

**What is the dataset intended to be used for?**
Reporting CoT-faithfulness for a manipulation VLA **with its calibration
floors beside it**. The benchmark's own headline result is that the magnitude
score is not interpretable in absolute terms without them.

**What should it NOT be used for?**

- Do **not** report `F` as an absolute faithfulness number. On the one model
  where we measured a floor, a meaning-preserving paraphrase scores 0.960
  against a maximum-effect ceiling of 0.970, and an out-of-CoT control that
  never touches the reasoning scores 0.99 — higher than all ten CoT families.
- Do **not** rank models by magnitude `F`. The ranking inverts under the
  direction-aware score: ECoT-bridge goes from 0.963 to 0.120 on
  `direction_flip` while the LoRA variants it outscored genuinely reverse.
- Do **not** read the four attention buckets as a salience ranking. It fails
  two independent robustness checks: per token the CoT bucket receives less
  attention than the instruction (3.9x) or the previous-action tokens (8.1x),
  and over all 32 layers rather than layers 0-3 the ordering reverses outright
  (`visual` 0.414 > `cot` 0.213).
- Do **not** treat any within-architecture attention difference here as real.
  The largest is 2.30 pp against a 1.45 pp same-config retraining difference.
- Do **not** use `F` as a safety guarantee for a CoT monitor. A high score is
  consistent with a model whose action merely reacts to token-level changes.

---

## Distribution and maintenance

**How is it distributed?**
As supplementary material with the paper submission: code, the 13 edit-family
generators, all per-sample records under `results_v2/`, the derivation script,
and the audit script. An anonymized public mirror will be linked at
camera-ready; the review-time bundle is self-contained and requires no network
access to re-derive any reported number.

**Reproduction**

```bash
python3 scripts/derive_metrics.py        # raw reports -> derived_metrics.json
python3 scripts/verify_paper_numbers.py  # asserts every quoted number; exit 1 on mismatch
```

The audit script currently checks 250 claims, one of which is that this
number itself is not stale. It is designed to fail: a claim
whose supporting artifact is missing is recorded as a failure, not skipped.

**Will it be maintained, and by whom?**
Yes, by the authors. The first camera-ready deliverable is the calibration
sweep across the remaining six CoT-VLAs — the check that would either
establish or dissolve finding F2. Two of eight are done, and neither passes
the CoT-specificity check (ratios 0.878 and 0.653, both below 1).

**How can users report errors?**
Through the repository issue tracker after de-anonymization. During review,
please note any discrepancy in the reviewer discussion; if
`verify_paper_numbers.py` fails on the released bundle, that is a bug in the
release and we want to hear about it.

---

## Licensing

See `LICENSE`. In brief: our code and the measurement records we produced are
released permissively (MIT / CC BY 4.0 respectively); the underlying corpora and
third-party checkpoints remain under their own upstream licenses, which are
listed above, were resolved from the Hub rather than asserted, and are **not**
relicensed by this release. Three of the fifteen have no upstream license at
all, and that is disclosed rather than papered over.
