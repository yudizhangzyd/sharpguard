# P3 re-run, in-domain and in robot units (bolt `phknckfy63`)

`../auroc_ecot_bridge_indomain_n153.json` replaces the withdrawn P3 run. Same
checkpoint (`Embodied-CoT/ecot-openvla-7b-bridge`), same four attention channels,
same median-split labelling — two protocol defects removed.

## What was wrong before

1. **Mixed units.** `dequantize(b) = -1 + (b+0.5)*2/256` returns a *normalized*
   action in `[-1,1]`. The withdrawn run subtracted it from a ground-truth action
   in the robot's own units, so the residual it thresholded was dominated by a
   per-dimension constant offset rather than by model error. Median L1 was
   **0.2550** under the defect against **0.0199** corrected — a factor of 12.8,
   i.e. the published quantity was mostly offset.
2. **Cross-domain.** It scored a Bridge-trained checkpoint on LIBERO, where that
   checkpoint carries no matching action statistics, so no correct
   un-normalization existed at all.

The corrected run un-normalizes the prediction with the checkpoint's own
`q01`/`q99` for `bridge_orig` (masked gripper dim passed through unrescaled) and
scores on Bridge V2. `cotfaith_auroc.py` now aborts rather than falling back when
the key is absent, and refuses cross-domain scoring unless
`--allow-cross-domain` is passed on purpose.

**40 of 153 samples change their high/low-error label between the two
protocols.** The withdrawn number was not a noisy version of this one.

## Result

`n=153` (the Bridge V2 loader extracted 153 usable frames of the 200 requested),
CoT self-generated on all 153 — Bridge ships no ground-truth reasoning, and
self-generated CoT is also the only thing a deployed failure predictor sees.

Raw AUROC with a 10,000-sample bootstrap over samples (labels re-derived from the
resampled median each draw):

| channel | raw AUROC | 95% CI | verdict |
|---|---|---|---|
| `action->cot` | 0.596 | [0.491, 0.679] | includes chance |
| `action->visual` | 0.454 | [0.357, 0.564] | includes chance |
| `action->instr` | **0.362** | **[0.284, 0.469]** | excludes chance |
| `action->action_prev` | 0.541 | [0.446, 0.633] | includes chance |

`action->instr` survives Bonferroni correction for the four channels: 98.75% CI
[0.261, 0.493]. Its direction is *low* instruction attention → *high* action
error. That direction was read off these same data, so it is a description of
this run, not a pre-registered prediction.

**The channel a CoT-faithfulness claim would need — `action->cot` — does not
separate high-error from low-error frames.** Correcting the protocol did not
rescue P3 as a failure predictor; it made the absence of signal measurable
instead of confounded.

## Do not read the `abs_auroc` fields as tests against chance

The report also carries `abs_auroc = max(raw, 1-raw)`, which is convenient for
ranking channels by effect size but is biased away from 0.5 by construction: its
bootstrap lower bound sits above 0.5 even for a channel that is pure noise. All
four channels' `abs_auroc` CIs "exclude chance" for exactly that reason. The raw
AUROC above is the statistic that tests anything.

## Comparison to the withdrawn run

The `legacy_mixed_space` block recomputes the withdrawn quantity on the same
forward passes: 0.531 / 0.427 / 0.492 / 0.550. So the defect did not merely add
noise — it moved `action->instr` from 0.362 to 0.492, i.e. it destroyed the one
channel that carries signal.
