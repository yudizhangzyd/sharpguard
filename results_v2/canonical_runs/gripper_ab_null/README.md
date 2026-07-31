# Gripper-convention A/B: a null result

`gripper_ab.json` is the verbatim artifact of bolt task `viyhc4kpft`. It is
released because it is the evidence for a negative claim in the paper, and a
negative claim with no artifact behind it is an assertion.

## What was tested

The decoder gate reports Task SR = 0 on all four LIBERO suites against a
published ~85%. Two harness bugs were already found and fixed (success read
from a key LIBERO never sets; `.pruned_init` read with `np.load`, so every
episode silently fell back to a random `env.reset()`). A third run had verified
canonical init states on 50/50 episodes and still scored 0/50, so a third
independent cause remained.

The leading suspect was the gripper channel. `results_v2/decoder_audit.json`
records `gripper_sign_agreement = 0.02`: our decoded gripper matched the
ground-truth demo gripper on 2% of samples where chance is ~50%. A
near-systematically inverted gripper cannot grasp, and "cannot grasp" produces
exactly 0/50 on pick-and-place rather than a degraded SR.

Four conventions were run under one model load on `libero_object` (every task
is a pick-and-place, so success is impossible without a working gripper):

| arm | transform | `gripper_sent_mean` | SR |
|---|---|---|---|
| `none` | identity | +0.187 | 0.000 |
| `invert` | `-g` | −0.355 | 0.000 |
| `binvert` | binarize then invert | −0.535 | 0.000 |
| `openvla` | upstream's `-sign(2g - 1)` | +0.7795 | 0.000 |

10 episodes per arm, 40 episodes total, 4000 gripper samples per arm, all 40
episodes on canonical init states (`all_episodes_used_canonical_init: true` in
every arm).

## What it shows, and what it does not

The arms are verifiably distinct — the four `gripper_sent_mean` values span
−0.535 to +0.7795, and `gripper_frac_close_sent` moves from 0.12 to 0.89 — so
this is not four runs of the same configuration. Every one scored 0/10.

So the gripper convention is **not sufficient** to explain the gate failure.
That is weaker than "the gripper convention is not a difference": bolt task
`htrg4uchwi` read openvla/openvla's source and found that upstream really does
apply `normalize_gripper_action(binarize=True)` then `invert_gripper_action`
where we applied neither. The difference is real; it is not the whole cause.
Any fix must therefore include the gripper correction *and* something else.

This is what moved image preprocessing to the front of the queue: upstream also
JPEG-round-trips and lanczos3-resizes every frame to 224, because the training
images went through that path, while we handed the raw 256px render to the
processor. Tasks `i55ww23d5n` (Pillow lanczos) and `mmmnxeehda` (exact
`tf.image.resize`) run the 2x2 gripper x image factorial.

## Two things to read carefully in the JSON

1. **`"n_episodes_per_arm": 4` contradicts every arm's `"n_total": 10`.** The
   request was 4; `libero_object` has 10 tasks; episodes-per-task floors at 1 so
   that no task is skipped, giving 1 x 10 = 10. The realised denominator is 10
   and the SRs are 0/10. The field name was the defect, not the run — later
   versions of the script write `n_episodes_requested_per_arm` next to
   `n_episodes_actually_run_per_arm`, and `tests/test_gripper_transform.py` pins
   the arithmetic. The artifact is left verbatim rather than retro-edited.
2. **The task's bolt state is `FAILED`.** That is the script's designed exit
   code 1 for "no single arm clears the win threshold", not a crash. A null
   result exiting 0 would be indistinguishable from a fix being found.
