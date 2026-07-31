# P2's de-quantization convention: F_mag is exactly invariant, F_dir is not

`p2_dequant_recompute.json` is the verbatim Stage A artifact of bolt task
`srkhkq7ea2`. `p2_decode_equivalence.json` is Stage B, from the rerun
`e2d58fvvn8`; see "Stage B" below for what it answers and for the one number in
it that is a property of the audit rather than of P2.

## Why this was asked

`../action_decode_mismatch/` measured that `sharpguard/libero_sim.predict_action`
and the checkpoint's own `predict_action` disagree on **24 of 24** frames. That
defect lives on the rollout path, from which the manuscript publishes no number.
But it raises a question about the path that *does* carry every published
\(\mathcal{F}\): P2 decodes through `experiments/cotfaith_edit.infer_action`,
which is a **second, independent** reimplementation of OpenVLA action decoding
and had never been compared to upstream at all.

P2 differs from upstream in two ways, and they separate cleanly:

1. **the de-quantization convention.** P2 maps bin *b* to
   `-1 + (b + 0.5) * 2/256`. The checkpoint's `predict_action` uses
   `self.bin_centers`, the midpoints of `linspace(-1, 1, 256)` — a spacing of
   `2/255`. **This half needs no inference at all**, which is why it was answered
   first and completely.
2. **token selection.** P2 puts no logit mask over the 256-token action window (a
   non-action token is silently dropped rather than forbidden) and does no
   input-id surgery. That half needs a GPU and is Stage B.

## Method: a replay, not a re-run

P2's map is **injective on bin indices**, so the bins are exactly recoverable
from the floats already released and both conventions can be replayed over the
entire artifact. `experiments/p2_dequant_recompute.py` inverts every stored
action back to bins, raises if a value is not on P2's grid, and re-derives
\(\mathcal{F}_{\text{mag}}\), \(\mathcal{F}_{\text{dir}}\) and the translation
cosine under both conventions. Two self-checks gate the result: every record must
invert (`n_recover_failed`), and every replay must reproduce the record's own
stored `delta_per_dim` (`n_delta_mismatch`). Both are **0**.

## The skew is real and not a rounding artifact

| quantity | value |
|---|---|
| max \|value difference\| over the 256 bins | **0.007797** (at bin 254) |
| as a fraction of \(\tau = 0.05\) | **15.6 %** |
| bins 254 and 255 under the checkpoint's grid | **collapse** to one value, `0.99607843` |
| bin 127 | P2: `-0.003906`  ·  checkpoint: **exactly 0.0** |
| records whose gripper sits at bin 255 | 4,393 |

`linspace(-1, 1, 256)` yields only **255** midpoints, so the difference is not a
pure multiplicative stretch: the top of the range collapses.

## Result 1: F_mag is exactly invariant, for a structural reason

Over all **10,780** scored edit records in `results_v2/canonical_runs`:

| quantity | value |
|---|---|
| records replayed | 10,780 |
| records that failed to invert | 0 |
| replays disagreeing with their own stored delta | 0 |
| records whose \(\Delta_\infty\) changed | **6,306** (max shift 0.007751) |
| faithful-flag flips | **0** (0 up, 0 down) |
| worst \|dF_mag\| over all families | **0.0000** |

So 6,306 records do get a different \(\Delta_\infty\), and not one flag moves.
That is **structural rather than fortunate**, and therefore holds for future runs
at this \(\tau\) and not merely for these records: a \(\Delta\) is always an
integer number of bins, and \(\tau = 0.05\) falls strictly between the six-bin
and seven-bin quantum under **both** spacings —

| bins | P2 (2/256) | checkpoint (2/255) |
|---|---|---|
| 6 | 0.046875 | 0.047059 |
| 7 | 0.054688 | 0.054902 |

Every headline number in the paper is an \(\mathcal{F}_{\text{mag}}\): the whole
of `tab:leaderboard`, \(\mathcal{F}_{\text{diff}}\), and the paraphrase floor.
None of them moves.

## Result 2: F_dir moves, and the paper reports the corrected values

\(\mathcal{F}_{\text{dir}}\)'s predicates test a cosine against \(-0.5\), a sign,
and an \(L_2\) ordering. None of those survives a shift of the value grid.

| quantity | value |
|---|---|
| records that change admissibility | **24** |
| records that change verdict | **12** |
| worst \|dF_dir\| | **0.060**, at `ecot_bridge_edit_*:gripper_flip` |

The mechanism is bin 127. Under P2 it is `-0.003906` (negative, so the record is
admissible to a gripper-sign test); under the checkpoint's grid it is exactly
`0.0` (inadmissible). Per-artifact:

| artifact | gripper_flip n_admissible | F_dir |
|---|---|---|
| `ecot_bridge_edit_seed0` | 100 → 93 | 0.06 → 0.00 |
| `ecot_bridge_edit_seed1` | 100 → 95 | 0.02 → 0.00 |
| `ecot_bridge_edit_seed2` | 100 → 96 | 0.03 → 0.00 |

so the 3-seed ECoT-bridge aggregate goes **11/300 = 0.037 → 0/284 = 0.000**.
On `direction_flip`: `ours-data50B` 0.59 → 0.54, `ours-r8` 0.69 → 0.67,
`ours-r32` 0.62 → 0.61. Family-mean translation cosines shift by at most 0.020
(ECoT-bridge `direction_flip` +0.417 → +0.415).

## What the manuscript does with this

The correction is made **the definition, not a footnote**.
`scripts/derive_metrics.py` now restates every stored action on the checkpoint's
own grid (`_regrid_rows`) before computing anything, so `derived_metrics.json`,
`tab:leaderboard`'s \(\mathcal{F}_{\text{dir}}\) column, `tab:directional` and the
F6 prose are all on the checkpoint's convention. Re-deriving changed **371
leaves, zero of them \(\mathcal{F}_{\text{mag}}\)** — only `cos_xyz`,
\(\mathcal{F}_{\text{dir}}\) and `n_directional`.

Records that are **not** on P2's bin grid pass through untouched and are counted
in `GRID_PASSTHROUGH` rather than silently forced onto it, so a report from a
checkpoint with a different action tokenizer (DeepThinkVLA's FAST tokenizer, for
instance) is not corrupted by the conversion.

**The F6 conclusion survives the correction**: ECoT-bridge still ranks 1st by
magnitude and 2nd-to-last by direction, so the direction-blindness finding is not
an artifact of the convention.

## Stage B (token selection): answered, and P2's selection is the correct one

Stage B asks the other half of the question: P2 puts **no logit mask** over the
256-token action window and does **no input-id surgery**, so does it select
different tokens than the checkpoint's own `predict_action`?

The first attempt (`srkhkq7ea2`) compared **0 of 12** samples and said
`INCONCLUSIVE` — correctly, since it measured nothing. All three causes were bugs
in the audit script, found by reading the checkpoint's shipped remote code:

1. `predict_action` forwards `**kwargs` into `generate()` **without** setting
   `max_new_tokens`, so generation stopped at the HF default `max_length=20` and
   raised on every ~300-token ECoT prompt.
2. `predict_action` returns `(actions, generated_ids)`, not an array.
3. the script's degeneracy test required 256 distinct un-normalized values per
   dimension, but the checkpoint's grid has only 255 — so all seven dimensions
   were excluded as "degenerate" and nothing was compared.

All three are fixed and the rerun is `e2d58fvvn8`, on
`Embodied-CoT/ecot-openvla-7b-bridge`, 12 samples, 66 prompts, 54 scored
(sample, family) records over five families.

### Result: selection is identical, and F is untouched

| quantity | value |
|---|---|
| prompts compared | 66 |
| prompts whose **raw generated ids** are byte-identical between the two paths | **66 / 66 (100 %)** |
| mirror reproduces `cotfaith_edit.infer_action` exactly | yes |
| scored records | 54 |
| faithful flags that differ | **0** |
| worst \|dF\| over the five families | **0.000** |
| worst per-record \(\Delta_\infty\) gap | **0.0078125** (= the Stage A grid difference, not a selection difference) |

Both paths run their own greedy `generate()` on the same `input_ids`, so
byte-identical raw ids settle the question directly: **P2's missing logit mask
never changed a selected token on these prompts.** The unmasked argmax landed
inside the action window every time. Per family, `F_p2 == F_upstream` exactly:
`subject_swap` 1.00, `location_swap` 0.90, `direction_flip` 1.00, `gripper_flip`
0.50, `paraphrase_null` 0.92.

The hypothesised vocab-size confound is **absent**, measured rather than assumed:
`model_vocab_size == processor_vocab_size == 32000`, so
`bin_index_offset_upstream_minus_p2 = 0`.

### The one number in the report that is not about P2

`n_prompts_token_identical` is **0 / 66**, and `per_dim_bin_agreement` is
\(\approx 0\) on all seven dims. Read at face value that says the two decodes
disagree on every prompt, which contradicts the 66/66 raw-id identity above. The
contradiction is real and the fault is the audit's, not P2's.

On **108 of 108** decodes (54 records × original + edited):

* `bins_upstream[j+1] == bins_p2[j]` for `j = 0..5`, up to upstream's 255→254
  clip (its grid has 255 centres);
* `bins_upstream[0]` is the **grid top** (254) on 108/108;
* `bins_p2[6]` is 255 on 108/108.

So upstream's span sits exactly one position earlier than P2's. The mechanism:
the model emits one non-action separator token before the seven action tokens
(the processor appends SentencePiece id `29871` when the prompt does not end in
it). P2 filters for ids inside the action window `[vocab-256, vocab)` and so
reads `[a1..a7]`. Upstream slices a **fixed** `generated_ids[0, -(action_dim+1):-1]`,
which with `max_new_tokens=8` reads `[sep, a1..a6]` — it puts the separator in
dim 0 and drops the gripper. A separator id cannot produce bin 254 as an action:
254 is what `clip(vocab_m - 29871 - 1, 0, 254)` clips **to**. That is why dim 0
is pinned at the grid top on every decode.

**P2's selection is the correct one**, and the misalignment is a property of
calling `predict_action` with a pre-built `input_ids` and `max_new_tokens=8`.

This also explains why `worst_delta_F` is nevertheless exactly 0, which would
otherwise look lucky: the offset is *constant* across the original and the edited
forward pass, so the paired \(\Delta\) is the same vector shifted by one, and the
one dim that is not shared is saturated (255 in P2, 254 in upstream) and
therefore contributes \(\Delta = 0\) in both. \(\Delta_\infty\) is a max over the
same six varying dims either way. So the ΔF = 0 result is robust — but it is
**not** the evidence for token-selection equivalence. The 66/66 raw-id identity
is.

`experiments/p2_decode_equivalence.py` now measures the offset
(`n_prompts_upstream_slice_offset_by_one`,
`n_prompts_upstream_dim0_at_grid_top`), releases the raw ids per record
(`gen_ids_orig_p2`, `gen_ids_orig_upstream_span`, and the `_edit_` pair) so a
reader can check the span themselves, and keys its verdict on raw-id identity
rather than on the misaligned slice. `sharpguard/libero_sim.predict_action_upstream`
had the same two upstream-calling-convention bugs — no `max_new_tokens`, and no
handling of the `(actions, generated_ids)` return — which is why the
instruction-only auxiliary probe recorded `aux_n_probed: 0` in `e2d58fvvn8`; both
are fixed. Vanilla OpenVLA's copy of `predict_action` sets `max_new_tokens`
itself, so the LIBERO rollout path was never affected, which is why this surfaced
only on ECoT.

Nothing in Result 1 or Result 2 above depends on Stage B: they are a replay of
released records, not an inference run.

## Reproduce

    python experiments/p2_dequant_recompute.py \
        --records results_v2/canonical_runs/*_edit*.json \
        --tau 0.05 --out p2_dequant_recompute.json

No GPU, no network, no model load. A non-zero exit is the script's designed
signal for "a published number moves under the other convention", not a crash —
and it *is* non-zero here, because \(\mathcal{F}_{\text{dir}}\) moves.

## A note on cross-platform reproducibility

This artifact was produced on Bolt (x86). Replaying the same script on arm64
gives **identical integer counts and identical \(\mathcal{F}\) values**; 19 of the
`cos_xyz` family means differ in the last bit (\(\le 2 \times 10^{-16}\)) from
floating-point summation order. Every number quoted above and in the manuscript
is stable across both.
