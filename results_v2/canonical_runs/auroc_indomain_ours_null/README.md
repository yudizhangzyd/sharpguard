# P3 in-domain on our own checkpoint — a released null, with a decidable reason

**bolt task:** `wmi3nxd454` · **checkpoint:** `bcihypv3gu` (ours, r=32) ·
**corpus:** LIBERO with ground-truth ECoT annotations · **N = 200**

## This is not a P3 row

`frame_check.passed` is `false`. The report contains AUROCs and they must never
be cited as a P3 result. `scripts/verify_paper_numbers.py::audit_p3_frame_check`
enforces that: it fails if any of these numbers appears in the manuscript.

They are released anyway, because withholding the numbers a failed gate produced
is how a gate becomes unfalsifiable.

## Why the run existed

The corrected P3 run (`auroc_ecot_bridge_indomain_n153`) is one model, and it
carries a caveat it cannot shed: Bridge has no ground-truth reasoning, so the
CoT is self-generated and a negative result cannot separate "attention does not
predict error" from "attention over a self-generated trace does not predict
error". This run was meant to remove both limits at once — a second model, and
LIBERO's ground-truth ECoT annotations, the same ones the checkpoint trained on.

## What the gate measured

Two preconditions, both measured after the fact rather than asserted:

| check | result | |
|---|---|---|
| `gt_actions_inside_token_grid` | **pass** | `frac_outside = 0.0`, `n_outside = 0`, `gt_abs_max = 1.0` |
| `policy_beats_predict_mean` | **fail** | ratio `1.0625` (needs `< 1.0`) |

Per-frame L1, over every frame this checkpoint ships:

| frame | policy | predict_mean | predict_zero | ratio |
|---|---|---|---|---|
| `identity` (scored) | 0.04860 | 0.04574 | 0.18970 | **1.0625** |
| `unnorm:bridge_orig` | 0.05018 | 0.04574 | 0.18970 | 1.0971 |

## The verdict, and why the three-way split matters

> **NOT A FRAME ERROR:** no frame this checkpoint offers beats the dataset mean
> (best is `identity` at ratio 1.063). The scale is not what is wrong; this
> checkpoint's open-loop single-step action prediction is no better than a
> constant, so a per-sample action error computed from it carries too little
> policy signal to threshold. P3 stays withdrawn on this checkpoint for a reason
> about competence, not units.

A single frame's ratio cannot distinguish two failures that call for opposite
responses:

* **wrong units** — some competing frame beats the mean, so the de-quantization
  map is wrong and is *fixable*. This is what withdrew the original P3, whose
  policy was ~13× worse than the mean because prediction and target sat in
  different spaces.
* **weak policy** — *no* frame beats the mean, so the scale is not the problem.

Reporting one verdict would have let the second be written up as the first. The
sweep is what makes the distinction decidable: `identity` is not merely our
choice, it is the *better* of the two frames available, and it still loses.

`predict_zero` is the reason the verdict is stated as "no better than a
constant" rather than "the actions are tiny": predicting zero gives 0.1897,
four times worse than the mean. The dataset mean is a genuinely informative
constant — LIBERO actions carry a strong per-dimension bias — and the policy
does not beat it.

## Two defects in the earlier handling, both fixed

Round 1 of this run (bolt `h3yb3s23qd`) hit the same measurement and **failed**:

1. The gate raised before writing anything, so a real measurement survived only
   as a traceback. It now writes the report first and exits 3;
   `bolt/run_cotfaith_auroc_s3ckpt.sh` maps 3 to success so Bolt preserves the
   artifact and the task does not read as a harness failure.
2. It reported one frame's ratio, conflating wrong-units with weak-policy. It
   now sweeps every `norm_stats` key the checkpoint ships and returns the
   three-way verdict above.

## Consequence for the paper

P3 remains n=1 and remains negative, and the reason is now specific: it is not
that we could not find the right units on a second model, it is that the second
model's open-loop action prediction is too weak to threshold. That is a
statement about the checkpoint, and it is also the independent prediction behind
the rollout null in `../rollout_edit_outofsuite_round1/` — a policy that cannot
beat a constant open-loop will not close a task closed-loop.
