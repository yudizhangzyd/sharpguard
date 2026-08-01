# Rollout-level CoT edit, round 1 — cancelled as out-of-suite

**bolt task:** `9jjmeny7mh` (cancelled at ~2 h of a 20 h budget)
**Superseded by:** `nskmsunnpb` (`bolt/boltconfig-cotfaith-rollout-edit-ckpt-lm90.yaml`)

## Why this is released rather than deleted

It is the evidence for the confound, and for the cost model that sized round 2.
It is **not** a result: nothing here supports or refutes any claim about CoT
faithfulness, and `verify_paper_numbers.py` must never cite it as a
limitation-(v) row.

## What it ran

| | |
|---|---|
| checkpoint | `bcihypv3gu` (ours, r=32) — LoRA-trained on `libero_lm_90/1.0.0` |
| rollout suite | **`libero_spatial`** ← the confound |
| max steps | 220 (libero_spatial's upstream budget) |
| arms | `nocot`, `cot_clean`, `cot_direction_flip`, `cot_paraphrase_null` |
| decoder | `ours` (identity `[-1,1]`), gripper `openvla`, image preproc `none` |

## What it measured

Every arm failed:

```
t0 ep0 nocot:               success=False steps=220 (2.3 min elapsed)
t0 ep0 cot_clean:           success=False steps=220 (39.1 min elapsed)
t0 ep0 cot_direction_flip:  success=False steps=220 (76.9 min elapsed)
t0 ep0 cot_paraphrase_null: success=False steps=220 (113.6 min elapsed)
```

## Why the zero is uninterpretable

The checkpoint was trained on LIBERO-90 (`bolt/boltconfig-cotfaith-train.yaml`,
`TFDS_SUBDIR: libero_lm_90/1.0.0`) and rolled out on `libero_spatial` — a
different suite with different objects and layouts. That explains SR=0 without
reference to the CoT at all, so the run cannot separate "editing the CoT does
not change task completion" from "this policy has never seen these tasks".
Only the first is a claim about faithfulness.

## What it does establish, and what it rules out

Three things are worth keeping:

1. **The CoT machinery is not the problem.** `cot_clean` parsed a structured
   9-tag CoT on **220/220 steps** (`n_cot_structured: 220`,
   `n_cot_unstructured: 0`), and the probe confirms both edit families change
   the rendered CoT. The prompt-side protocol works online, at every step.
2. **The scale precondition held**: `ok (identity [-1,1] de-quantization ...
   norm_stats present: ['bridge_orig'])`. The inherited `bridge_orig` key is
   present and unused, as intended.
3. **The cost model**, which sized round 2: `cot_gen_seconds: 10.09`. At
   libero_90's 400-step budget that is ~67 min per CoT arm, so 3 arms are
   ~2.3 h per episode and a 20 h budget buys ~8 episodes.

Two candidate explanations for the zero were checked and are **not** the cause,
which is why round 2 changes the suite and nothing else:

* **Image preprocessing.** `IMAGE_PREPROC=none` was suspect because the four
  failed gates ran it, but `../decoder_gripper_grid/gripper_ab.json` records
  `upstream|openvla+none` at **SR 1.0 (10/10)** on libero_object. Preprocessing
  is inert once the decode frame is right, so `INSTALL_TF` stays off.
* **The harness.** It reaches 10/10 somewhere, so a zero is a statement about
  this checkpoint, not about the simulator plumbing.

Note that `../gate_factorial_tf/gripper_ab.json` shows all four gripper × image
cells at 0.0. That factorial used the `ours` decoder on an **upstream** model,
whose native frame is its `norm_stats` affine — identity was simply the wrong
map for it. It does not indict `ACTION_DECODER=ours` here, where the weights
quantize raw LIBERO actions clipped to `[-1,1]` with no dataset normalization.

## Also fixed in round 2

`MAX_STEPS` was hard-coded to 220. Carried into libero_90 that truncates every
episode by construction — the defect that already invalidated libero_10's 0/50
in the four-suite gate. Round 2 sets `MAX_STEPS=0`, which takes upstream's own
per-suite budget (400 for libero_90).

## Expected outcome of round 2

Still a null. The P3 frame check on these exact weights (bolt `h3yb3s23qd`,
in-domain on `libero_lm_90`) measured policy L1 **0.04860** against
predict-the-dataset-mean **0.04574** — ratio 1.06, i.e. marginally *worse* than
a constant, and no frame the checkpoint ships beats the mean. A policy that
cannot beat a constant open-loop will not close a task closed-loop.

The point of round 2 is not to reverse that expectation. It is that an
**in-suite** 0/N supports a binomial bound on SR, and an out-of-suite 0/N
supports nothing — and limitation (v) needs the former.
