# The same factorial with upstream's exact resize: the same null

`gripper_ab.json` is the verbatim artifact of bolt task `mmmnxeehda`. It repeats
`../gate_factorial_pil/` (bolt `i55ww23d5n`) with one change:
`image_preproc='tf_upstream'`, which calls `tf.image.encode_jpeg`,
`tf.io.decode_image` and `tf.image.resize(method="lanczos3", antialias=True)`
directly instead of the Pillow-plus-NumPy reimplementation.

## Why it was worth a second GPU job

The first factorial's image arm was an approximation.
`experiments/resize_kernel_check.py` measures the reimplementation at 8/255 LSB
from upstream's path — the residual being a Pillow-versus-`tensorflow` `libjpeg`
disagreement no Pillow setting closes. An 8/255 residual is small, but the whole
hypothesis under test was that a small per-frame input mismatch degrades the
policy. Testing that hypothesis with a path that itself carries a small per-frame
mismatch is not a clean test, and "the approximation was not close enough" would
have been an available excuse for the null.

It is no longer available. This run is exact by construction.

## Result: all four cells scored 0/10 again

| cell | SR | successes | `gripper_sent_mean` | `gripper_frac_close_sent` |
|---|---|---|---|---|
| `none+none` (anchor) | 0.000 | 0/10 | +0.2190 | 0.700 |
| `openvla+none` | 0.000 | 0/10 | +0.7407 | 0.870 |
| `none+tf_upstream` | 0.000 | 0/10 | +0.2324 | 0.703 |
| `openvla+tf_upstream` | 0.000 | 0/10 | +0.7450 | 0.873 |

`libero_object`, 280 steps (upstream's own budget for this suite), 10 episodes
per cell, 40 total, all 40 on canonical initial states, one model load, no cell
raised.

The `openvla+none` cell reproduces the other run's cell to every printed digit
(+0.7407142857), and the anchor agrees to 0.003 — the two runs are the same
configuration measured twice, so the exact-versus-approximate comparison is
between the image cells and not between two unrelated jobs.

## What it adds to the first factorial

Two things.

1. **The null does not depend on the approximation.** Upstream's own ops, on
   upstream's own frames, produce the same 0/10. Combined with
   `../gripper_ab_null/` (four gripper conventions, all 0/10) and
   `../gate_factorial_pil/`, the two differences a source-level diff found are
   now measured three ways and are insufficient every time.
2. **`d543p4f86p` holds in production, not just in a probe.** That job measured
   that `tensorflow-cpu` can coexist with this eval environment — `numpy` stayed
   at 1.26.4, `torch` kept CUDA, `transformers` reported `is_tf_available()` as
   false under `USE_TF=0` — refuting a comment in this repository that had
   asserted the opposite and had shaped the design for as long as it went
   unchallenged. This is a full 7B-model GPU rollout with `tensorflow` imported
   in the same process: it ran, it used CUDA, and it produced telemetry
   consistent with the non-`tf` run. The probe's conclusion was correct.

## What it does not show

It does not show that upstream's preprocessing is a non-difference — it is a real
difference, read out of upstream's source rather than inferred. It shows the
difference is not sufficient to explain the gate failure. The remaining
candidates are unchanged and listed in `../gate_factorial_pil/README.md`: the
action de-quantization bin convention, and `norm_stats` key resolution.

The task's non-zero exit is the script's designed signal for "no cell clears the
win threshold", not a crash.
