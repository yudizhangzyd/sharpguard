# Rollout-level CoT edit, in-suite — the run behind limitation (v)

**bolt task:** `r2kpkqsim4` (`bolt/boltconfig-cotfaith-rollout-edit-insuite.yaml`)
**Supersedes:** `../rollout_edit_outofsuite_round1/`, which rolled a LIBERO-90
checkpoint out on `libero_spatial` and therefore measured the suite mismatch
rather than anything about the CoT.

This is the run the manuscript's $0/40$ comes from. It is a **bound**, not a
measurement, and the README states which.

## What it ran

| | |
|---|---|
| checkpoint | ours, r=32 — LoRA-trained on `libero_lm_90/1.0.0` |
| rollout suite | **`libero_90`** — the suite the checkpoint was trained on |
| max steps | 400 (upstream's own `libero_90` budget; round 1's hard-coded 220 would have truncated every episode) |
| arms | `nocot`, `cot_clean` |
| decoder | `ours` (identity `[-1,1]`), gripper `openvla`, image preproc `none` |
| episodes | 40 per arm, 34 distinct tasks |

## The result

```
nocot      0/40   SR 0.000   Wilson 95% [0.000, 0.088]   0 CoTs generated
cot_clean  0/40   SR 0.000   Wilson 95% [0.000, 0.088]   640 CoTs, 640 structured
```

`precondition_met: false`. The harness refuses to report a ΔSR here:

> cot_clean NEVER succeeds, so DSR is 0 for every family by construction and
> carries no information about CoT causality. This is reported as an undefined
> measurement rather than as a null result.

That refusal is the point. A ΔSR of 0 against a control that never succeeds is
0 by construction, and reporting it as a null result about CoT faithfulness
would be the single most misleading number this project could publish.

## Why the zero is a competence bound, not a CoT result

The two arms are **identical** at 0/40, and the no-CoT arm never generates a
CoT, so nothing here distinguishes the arms and the zero cannot be attributed
to the reasoning. Unlike round 1, the suite is not a candidate explanation: the
policy is rolled out on the suite it was trained on.

The cause is measured, on these exact weights, in
`../auroc_indomain_ours_null/`: open-loop single-step L1 of **0.0486** against
**0.0457** for predicting the per-dimension dataset mean — ratio 1.06, i.e.
marginally *worse* than a constant, in every frame the checkpoint ships. A
policy that cannot beat a constant open-loop does not close a task closed-loop.

Two things the run does establish:

1. **The online CoT protocol works.** 640/640 generated CoTs parsed as
   structured 8-tag traces (`n_cot_unstructured: 0`), refreshed every 25 steps
   across 16,000 steps. Whatever is broken, it is not the prompt-side harness.
2. **The scale precondition held.** `rollout_edit_probe.json` records
   `scale_precondition: ok (identity [-1,1] de-quantization ...)`. The inherited
   `bridge_orig` norm_stats key is present and unused, as intended — this is
   not the `ecot-openvla-7b-bridge` failure mode, where a missing norm_stats
   entry pins SR at 0 for a reason with nothing to do with the policy.

## Status

`status: stopped_time_budget` — the job ran both arms to completion at 40
episodes each and stopped before adding edited arms. Since `cot_clean` is
0/40, the edited arms would have measured nothing, so no result is missing.

## What the paper may and may not say about this

May: the paired rollout returns 0/40 in both arms; ΔSR is undefined; the
missing rollout evidence is a **checkpoint-competence** problem, not a compute
or suite-availability problem.

May not: any rollout-conditioned number, in any table. There are none.
