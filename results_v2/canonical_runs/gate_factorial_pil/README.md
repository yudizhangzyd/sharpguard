# Gripper x image-preprocessing factorial: a second null

`gripper_ab.json` is the verbatim artifact of bolt task `i55ww23d5n`.

## What was tested

`experiments/openvla_reference_diff.py` (bolt `htrg4uchwi`) diffed our rollout
harness against `openvla/openvla` at the source level and found exactly two
differences live in the four-suite gate runs:

1. **Gripper.** Upstream composes `normalize_gripper_action(binarize=True)` with
   `invert_gripper_action`, i.e. `g -> -sign(2g - 1)`. We passed `g` through
   unchanged.
2. **Image preprocessing.** Upstream's `resize_image` applies a JPEG
   encode/decode round-trip "as done in RLDS dataset builder" and then
   `tf.image.resize(method="lanczos3", antialias=True)` to 224, explicitly "to
   make input images in distribution with respect to the inputs seen at training
   time". We handed the raw 256x256 render to the processor.

Two simultaneous changes cannot be attributed by a single A/B, so this is a 2x2:
gripper in {`none`, `openvla`} x image in {`none`, `pil_lanczos`}. The
`none+none` cell reproduces the gate configuration exactly, which is what makes
a null here distinguishable from a broken harness.

## Result: all four cells scored 0/10

| cell | SR | successes | `gripper_sent_mean` | `gripper_frac_close_sent` |
|---|---|---|---|---|
| `none+none` (anchor) | 0.000 | 0/10 | +0.2157 | 0.695 |
| `openvla+none` | 0.000 | 0/10 | +0.7407 | 0.870 |
| `none+pil_lanczos` | 0.000 | 0/10 | +0.2028 | 0.680 |
| `openvla+pil_lanczos` | 0.000 | 0/10 | +0.7529 | 0.876 |

`libero_object`, 280 steps (upstream's own budget for this suite, so no episode
was truncated), 10 episodes per cell, 40 episodes total, all 40 on canonical
initial states, one model load, no cell raised.

The cells are verifiably distinct: the gripper factor moves the mean delivered
command from ~+0.21 to ~+0.75 and the closed-gripper fraction from ~0.69 to
~0.87, and the image factor perturbs both by a small amount in the direction a
changed input distribution would.

## What it shows

**Neither difference, nor their combination, is sufficient to explain the gate
failure.** Both are real differences from upstream — that was established by
reading upstream's source, not by inference from this run — so what is ruled out
is sufficiency, not existence. Together with the gripper-only null
(`../gripper_ab_null/`, bolt `viyhc4kpft`), the two candidates that a source
diff produced are now both measured and both insufficient.

Note also what this rules out that was *not* on the original candidate list: the
per-suite step budget. This run used 280, which is upstream's value for
`libero_object`, so truncation cannot be the explanation for these zeros.

Remaining candidates, in the order we can test them:

- **The action de-quantization bin convention.** This is the one quantity still
  validated only by our own offline audit rather than against upstream's code
  path, which makes it the least-checked assumption in the harness.
- **`norm_stats` key resolution.** The checkpoint may resolve `unnorm_key` to a
  different action space than the suite expects. Note this cannot be the whole
  story either: the gripper dimension has `mask=False` in every OpenVLA
  `norm_stats`, so it bypasses un-normalization entirely.

## Companion run

`mmmnxeehda` runs the same 2x2 with `image_preproc='tf_upstream'`, which calls
`tf.image.resize` itself instead of the Pillow reimplementation used here.
`experiments/resize_kernel_check.py` measures the Pillow path at 8/255 LSB from
upstream's, so this run is an approximation of the image factor and that one is
exact. A null in both is a stronger statement than a null in either.
