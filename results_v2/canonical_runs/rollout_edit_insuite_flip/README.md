# Rollout-level CoT edit, in-suite, WITH the edited arm

**bolt task:** `nskmsunnpb` (`bolt/boltconfig-cotfaith-rollout-edit-insuite.yaml`)
**Companion to:** `../rollout_edit_insuite_sr/` (bolt `r2kpkqsim4`), which ran
the same two control arms at 40 episodes each and stopped before adding an
edited arm.

This is the only run in the release that actually executed a **CoT-edited**
rollout arm. It is still a bound, not a measurement, and for the same reason.

## What it ran

| | |
|---|---|
| checkpoint | ours, r=32 — LoRA-trained on `libero_lm_90/1.0.0` |
| rollout suite | **`libero_90`** — the suite the checkpoint was trained on |
| max steps | 400 (upstream's own `libero_90` budget) |
| arms | `nocot`, `cot_clean`, **`cot_direction_flip`** |
| decoder | `ours` (identity `[-1,1]`), gripper `openvla`, image preproc `none` |
| episodes | 10 + 10 + 9 (29 total; `status: stopped_time_budget` at 20 h) |

Episode counts are smaller than `r2kpkqsim4`'s 40/arm because a third arm was
added under the same wall-clock budget. The two runs are complementary: that
one bounds the controls tightly, this one is the only evidence about the
edited arm.

## The result

```
nocot               0/10   SR 0.000   Wilson 95% [0.000, 0.278]
cot_clean           0/10   SR 0.000   Wilson 95% [0.000, 0.278]
cot_direction_flip  0/9    SR 0.000   Wilson 95% [0.000, 0.299]
```

`precondition_met: false` and `delta_sr_vs_cot_clean: {}` — the harness again
refuses to report a ΔSR, and the refusal is the same one:

> cot_clean NEVER succeeds, so DSR is 0 for every family by construction and
> carries no information about CoT causality. This is reported as an undefined
> measurement rather than as a null result.

**The edited arm changes nothing about that.** Adding an arm to a comparison
whose control never succeeds does not create a comparison. A ΔSR of 0 between
0/9 and 0/10 is 0 because both are 0, and this is why we publish no
rollout-conditioned number.

## What it does add

1. **The edit protocol runs online, closed-loop, end to end.** 3,234 of 3,600
   generated CoTs in the edited arm parsed as structured 8-tag traces, and the
   `direction_flip` generator applied to them in-loop, refreshing every 25
   steps. The whole pipeline — generate, parse, edit, re-inject, decode —
   executes on a live rollout, not just on the offline first-step samples every
   published number comes from.

2. **A number the offline protocol cannot produce: `n_edit_skipped: 381`.**
   Over a rollout, 381 of the edited arm's CoTs contained no direction word for
   `direction_flip` to reverse, so the edit was a no-op and the harness recorded
   it as skipped rather than counting it as an applied edit. The offline
   protocol samples first steps, where a direction word is nearly always
   present. This is a genuine limitation of edit-family coverage under
   distribution shift along a trajectory, and it is only visible in a rollout.

3. **`n_cot_unstructured: 366` in the edited arm vs 9 in `cot_clean`.** Editing
   the CoT makes the *next* generation measurably more likely to come back
   unparseable — 10.2% against 0.2%. Recorded here; not published as a result,
   since with SR pinned at 0 it cannot be connected to behaviour.

## What the paper may and may not say about this

May: an edited arm was run in-suite; all three arms are 0; ΔSR remains
undefined rather than null; the online edit path is validated end to end; and
`n_edit_skipped` bounds how often `direction_flip` is applicable mid-rollout.

May not: any rollout-conditioned faithfulness number. There are none, and the
addition of an edited arm does not create one.

## Scale precondition

`rollout_edit_probe.json` records
`scale_precondition: ok (identity [-1,1] de-quantization ...)`, with the
inherited `bridge_orig` norm_stats key present and unused, as intended. As in
the companion run, this is not the `ecot-openvla-7b-bridge` failure mode where
a missing norm_stats entry pins SR at 0 for reasons unrelated to the policy.
The cause here is measured in `../auroc_indomain_ours_null/`: open-loop L1
0.0486 against 0.0457 for predicting the per-dimension dataset mean.
