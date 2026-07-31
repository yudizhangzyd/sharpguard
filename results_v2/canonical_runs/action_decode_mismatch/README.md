# Our action decode is not upstream's: 24/24 frames disagree

`action_decode_check.json` is the verbatim artifact of bolt task `7vpp28qfsk`.
It is the first result in this investigation that is **not** a null.

## What was asked

Three of the four candidate causes for the decoder-gate failure were already
measured and none was sufficient: the gripper convention (`../gripper_ab_null/`,
four arms, all 0/10), frame preprocessing (`../gate_factorial_pil/` and
`../gate_factorial_tf/`, 2x2 twice — approximate and exact — all 0/10), and the
per-suite step budget (excluded by construction; those runs used upstream's own
280 for `libero_object`).

What was left was the action de-quantization: the one quantity in this harness
validated only against our own offline audit rather than against upstream's code
path. Rather than diff `predict_action` against upstream's source and reason
about it — four failures had already been produced by reasoning — this job calls
the method the checkpoint ships in its own remote code and compares numbers.

## Result: they disagree on every frame

24 frames, 3 tasks, 8 timesteps each, taken along real trajectories with our
decoder driving the arm (so the states visited are states the measured 0/10
configuration actually visits).

| quantity | value |
|---|---|
| frames where the two decoders agree within 1e-4 | **0 / 24** |
| max L-inf over all frames | **1.121569** |
| mean per-frame L-inf | 0.719099 |
| gripper sign agreement | 0.75 |

Per-dimension max absolute difference:

| x | y | z | roll | pitch | yaw | **gripper** |
|---|---|---|---|---|---|---|
| 0.532 | 0.593 | 0.507 | 0.048 | 0.105 | 0.086 | **1.122** |

The disagreement is not a bin off-by-one. The two decoders are close to
**uncorrelated** across frames: Pearson r = +0.23 (x), +0.60 (y), +0.03 (z),
−0.15 (roll), −0.31 (pitch), +0.20 (yaw). It also does not grow with time — the
per-timestep mean L-inf is flat over t=0..7 (0.36 to 0.87 with no trend) — so
this is a wrong function, not a drift.

### The gripper channel is the readable part

All 24 frames are early in an episode (t=0..7, just after the settling period),
where the arm has not yet reached the object and the correct command is a
constant *hold open*.

* **upstream** emits exactly one value on all 24 frames: `0.9961`, the top bin
  centre. std = 0.0.
* **ours** emits 11 distinct values spanning −0.1255 to +0.9961, std = 0.38.

That is direct positive evidence that our decode's gripper channel is wrong
rather than merely different, and it independently reproduces what
`results_v2/decoder_audit.json` recorded offline as
`gripper_sign_agreement: 0.02` against ~0.50 chance — this time with the
mechanism visible.

### The comparison itself is valid

The obvious failure mode for this experiment is that the image never reaches
upstream's method, which would make its near-constant output an artifact of our
plumbing rather than a property of the checkpoint. It is not: upstream produces
**18 distinct actions over the 24 frames** (6, 5 and 7 within the three tasks),
so it is frame-sensitive. Only the gripper dimension is pinned, and that is the
dimension that *should* be pinned on these frames.

## Both rollout arms still scored 0/10 — and that is this run's own defect

| arm | SR | `gripper_sent_mean` | `frac_close_sent` |
|---|---|---|---|
| `ours` | 0.000 (0/10) | +0.2190 | 0.700 |
| `upstream` | 0.000 (0/10) | +0.5820 | 0.585 |

Read that as a limitation of this job, not as a null. Both arms ran
`gripper_transform="none"`, so the `upstream` arm sent its constant `+0.9961`
straight to LIBERO's OSC actuator, which reads positive as **close**. That arm
therefore drove the whole episode with a permanently closed gripper and could
not have grasped anything.

Upstream's own pipeline applies `g -> -sign(2g - 1)` on top of exactly this
channel, mapping `0.9961` to `-1` = open, and a low bin to `+1` = close. So the
composition *upstream decode feeding upstream's gripper transform* is a coherent
control loop, and **it has never been run**.

## What this changes about the three earlier nulls

It weakens their scope, and the manuscript says so rather than leaving the
earlier framing in place.

`../gripper_ab_null/` tested four gripper transforms **on top of our decode**.
Upstream's `openvla` transform is written for a channel that sits near the top
bin when the gripper should be open; our decode delivers a channel with mean
0.34 and std 0.38 on exactly those frames. So that A/B applied upstream's
correction to an input it was never designed to consume, and its null bounds
insufficiency *conditional on our decode* — which is now measured to be wrong.
The same conditioning applies to both factorials.

This is the first evidence that the candidate causes **interact** rather than
being independently insufficient, which is why the next job crosses the decoder
axis with the gripper axis instead of adding a sixth factor to the list.

## Reproduce

    python experiments/action_decode_check.py \
        --model openvla/openvla-7b-finetuned-libero-object \
        --suite libero_object --unnorm-key libero_object \
        --n-obs 24 --traj-steps 8 --tol 1e-4 --max-steps 0

The task's non-zero exit is the script's designed signal for "the decoders
differ", not a crash. A run that exited 0 here would have meant the hypothesis
was refuted.
