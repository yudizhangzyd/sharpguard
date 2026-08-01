# Why no rollout number is reported for ECoT-bridge, measured rather than argued

`rollout_edit_probe.json` is the verbatim artifact of bolt task `phenc9ygb4`: a
one-frame probe of `Embodied-CoT/ecot-openvla-7b-bridge` under the rollout-level
CoT-edit harness (`experiments/cotfaith_rollout_edit.py --probe-only`), run on
`libero_spatial`.

It exists because the obvious way to answer limitation (v) --- does editing the
CoT change whether the task gets *finished*, not just what the first action is
--- is to roll out the public CoT checkpoint, which is the only CoT-VLA in this
benchmark we did not train ourselves. That is not possible, and this is the
measurement that establishes why rather than an assertion that it is so.

## What it found

```
norm_stats_keys:        ["bridge_orig"]
unnorm_key_requested:   "libero_spatial_no_noops"
unnorm_key_present:     false
norm_stats_usable:      false
```

The checkpoint ships de-normalization statistics for exactly one dataset,
`bridge_orig`, and LIBERO is not it. All five arms (`nocot`, `cot_clean`, and
one per edit family) fail identically at decode time with upstream's own
`ValueError`, quoted verbatim in `one_frame_actions`. There is no fallback that
would be honest: without LIBERO percentiles the policy's outputs reach
`env.step()` at raw `[-1,1]` scale, so Task SR is pinned at `0` for a reason
that has nothing to do with the CoT, and every arm --- including the unedited
control --- would score `0`. A paired McNemar test over five arms that are all
identically zero is not a null result about CoT faithfulness; it is a broken
setup that reads like one.

## What it also found, which is the part worth keeping

The two failures are independent, and only one of them is fatal:

- `cot_structured: true`, `cot_tokens_generated: 255`, and eight parsed tags
  (`task`, `plan`, `subtask`, `subtask_reasoning`, `movement`,
  `movement_reasoning`, `gripper`, `bboxes`). The checkpoint emits a
  well-formed CoT online, on LIBERO frames, unprompted.
- Three of the four probed families (`direction_flip`, `paraphrase_null`,
  `gripper_flip`) change the rendered CoT; `subject_swap` returns `None` on this
  trace and is correctly reported as inapplicable rather than scored as a
  no-effect edit.

So the *intervention* side of the rollout protocol works on this checkpoint. It
is the *action scale* side that does not. That distinction is what let the
rollout job be re-pointed at our own `r=32` LIBERO fine-tune (bolt
`bcihypv3gu`), which trained on raw LIBERO actions clipped to `[-1,1]` and
therefore needs no `norm_stats` at all --- identity de-quantization *is* its
native scale (`cotfaith_train.py:_quantize_action`).

## Feasibility, recorded so the episode budget is not a taste judgement

`cot_gen_seconds: 8.57` per step per arm. At 5 arms that is `26.2` estimated
hours for a full run, which is an upper bound (successful episodes terminate
early). The rollout job's episode count is set by that clock and by a
`TIME_BUDGET_H` that flushes a partial report, not by preference.

## The general point

The job that produced this probe does not launch its full run past a failed
precondition: `bolt/run_cotfaith_rollout_edit_s3.sh` reads its own probe output
and exits `5` when `scale_precondition` does not start with `ok`. Writing a
probe and then not reading it is how the two earlier rollout defects in this
paper (a success key LIBERO never sets, and `np.load` on a `torch.save`
archive) both survived --- each exited `0` and wrote a well-formed report.
