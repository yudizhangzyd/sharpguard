# LLM-judge validation of the edit-family generators

Bolt task `jhcgnqbmf2`. Produced by `experiments/judge_edit_families.py`; runner
`bolt/run_judge_edits.sh`, config `bolt/boltconfig-cotfaith-judge-edits.yaml`.
Arguments as run are in `args.json`.

## What this measures, and what it does not

Every edit family in CoT-Faith is defined by a *claim about its own semantics*:
`paraphrase_null` is supposed to preserve meaning, `direction_flip` is supposed
to reverse a direction and nothing else, `adversarial_plausible` is supposed to
substitute an object that stays *visually plausible*. Those claims were asserted
from the generator source, never measured. They are load-bearing:
`F_diff = F(family) - F(paraphrase_null)` is only a floor-corrected statistic if
the floor family really preserves meaning, and an "adversarial but plausible"
family only tests what its name says if the substitution is plausible.

So an independent LLM judge reads the (original, edited) reasoning pair and
answers four questions per pair: meaning preserved, same referent, same
direction, still plausible, plus a 1-5 fluency rating. The judge never sees
which family produced the pair, never sees the model's action, and is not the
model under test.

**This does not validate the exact scored pairs.** The released edit records
store actions only; demo/step identity is not recoverable without re-iterating
the 17 GB tfds shards. The judge runs the same generators over the same
reasoning corpus, restricted to the demo files that the scored run drew from
(`--file-base-from results_v2/canonical_runs/ecot_bridge_edit_seed1.json`). It
is a validation of the *generators*, not a per-record audit, and
`record_level_correspondence` in the report says so in those words.

## Judge validity gates (all three pass)

The judge is not trusted on its own word. Three gates run inside the same job,
and `judge_valid` is their conjunction:

| gate | value | requirement |
|---|---|---|
| `identity_control_preserved_rate` | **1.000** (40/40) | a pair where the "edit" is X->X must be called meaning-preserving |
| `negative_control_pooled_preserved_rate` | **0.175** | pooled over the families designed to destroy meaning, it must *not* be |
| `order_agreement` | **0.824** (360/437) | swapping which trace is presented first must not change the verdict |

A judge that passed the identity gate alone would be consistent with one that
answers "preserved" always; the negative gate is what rules that out. The order
gate bounds presentation bias: 77 of 437 pairs flip under reordering, which is
the noise floor to attach to every rate below.

`n_pairs_judged = 437` of 480, with all 43 skips itemized in
`skipped_reasons` --- 16 `adversarial_plausible` and 16 `subject_swap` and 10
`location_swap` where the *generator* found no applicable edit, plus 1
`location_swap` identical render. Nothing is silently dropped.

## Declared premises: 3 hold, 1 fails

| family | premise | measured | verdict |
|---|---|---|---|
| `paraphrase_null` | meaning preserved | 0.975, fluency 4.92 | **holds** |
| `bbox_jitter_null` | meaning preserved | 1.000, fluency 5.00 | **holds** |
| `syntactic_scramble` | meaning preserved | 1.000, fluency 4.00 | **holds** |
| `adversarial_plausible` | referent changes AND result stays plausible | referent changed 0.958, **plausible 0.125** | **FAILS** |

The two nulls that `F_diff` and the numeric-floor argument rest on are
validated, which is the result those arguments needed. `adversarial_plausible`
is not: the referent does change, but the paper described the substitute as
"visually plausible but wrong" and the judge calls only 3 of 24 pairs plausible.
Counting the 16 pairs where the generator produced no edit at all, the family
delivers a plausible substitution on 3 of 40 attempts.

## Two further findings the paper did not previously state

* **`syntactic_scramble` is a third meaning-preserving family, not a structural
  edit.** It is listed under Tier 0 as a word-order shuffle "preserving content
  words", and at a preserved rate of 1.000 that is exactly what it is --- so its
  `F` is a second within-CoT surface-form floor, independent of
  `paraphrase_null`'s lexical substitution.

* **`verb_swap` changes meaning in only 0.575 of pairs**, with direction changed
  in 0.350. So roughly 4 in 10 "verb swaps" are not semantic edits at all. This
  is the same family that carries the largest same-config retraining movement in
  the benchmark (max |dF| = 0.260 on r=8, worst family on 3 of 7 pairs): the
  least semantically reliable generator is also the noisiest cell.

The remaining families behave as designed and are reported in `per_family`
without a declared premise to check: `cross_task_swap` preserves meaning on
0.025, `negation` on 0.100, `gripper_flip` on 0.050, `direction_flip` on 0.200
with the referent held at 1.000 and direction changed on all 40.

## Files

* `judge_report.json` --- gates, per-family rates, declared-premise verdicts,
  skip accounting. This is the file `scripts/verify_paper_numbers.py` audits.
* `judge_pairs.json` --- all 437 judged pairs with the original and edited
  reasoning, the judge's raw answer and its parse. Included so every rate above
  can be recomputed, and so a disputed verdict can be read rather than argued.
* `args.json`, `bolt_task_id.txt` --- provenance.

Judge model: `Qwen/Qwen2.5-7B-Instruct` (first of two candidates tried;
`mistralai/Mistral-7B-Instruct-v0.2` was the fallback and was not needed).
Elapsed 1503 s.
