# DeepThinkVLA decode equivalence, `_base` at 768 tokens

**bolt task:** `t57wgzya9a` (`bolt/boltconfig-dt-decode-equivalence-base768.yaml`)
**Follows:** `../dt_decode_equivalence/` (bolt `8cvg2daria`, all three
checkpoints at `MAX_NEW_TOKENS=320`)

## What this run was for

`8cvg2daria` returned **EQUIVALENT on 12/12** for `deepthinkvla_libero_cot_sft`
and `_rl`, and **UNDEFINED** for `deepthinkvla_base`: 0/12 comparable, 12/12
`no_stop_tail`. At 320 new tokens the base checkpoint never emitted upstream's
own stop tail `[</think>, <action>]`, so no assembled sequence was ever compared
against one the checkpoint produced.

Two hypotheses were consistent with that, and 768 tokens separates them:

1. the base CoT simply runs long (the two terminating checkpoints average 72.7
   CoT tokens, so 320 would be generous but not obviously fatal), or
2. the base checkpoint does not produce a terminating CoT at all.

## The answer: (2), and it is structural

**Still 0/12 comparable at 768.** Doubling the budget again would not help,
because the failure is not at the tail:

```
n_generated        768 on 12/12  (budget exhausted, every frame)
prompt_preserved    12/12
think_start_first    0/12   <-- generation never opens a think block
stop_tail_ok         0/12
```

`think_start_first` is the diagnostic. Upstream's stopping criterion fires on
the two-token tail `[</think>, <action>]`, and the sequence never contains the
matching `<think>` (257153) either — the base checkpoint's first generated token
is not `<think>` on any of the 12 frames. The generated tails are runs of
`<action>` (257155) instead:

```
last 3 ids:  [257155, 257155, 257155]  x9
             [257155, 257155, 235371]  x2
             [235371, 235371, 257155]  x1
```

So the base checkpoint emits action markers where the SFT and RL checkpoints
emit reasoning. That is a coherent property of a checkpoint that has not been
CoT-trained — `_base` is the pre-CoT-SFT initialization — and it means the
comparison this audit performs **cannot be defined on this checkpoint at any
token budget**, not that it fails.

## What this does and does not bound

Does not: our decode path on `deepthinkvla_base`. Nothing here tests
`build_input_cot_ids`, `action_start_idx` or `decode_action_chunk` on those
weights, because step 1 of the protocol (get a sequence the checkpoint produced
itself) has no output to hand to step 2.

Does: it converts "unaudited, budget possibly too low" into "unaudited, and
un-auditable by this method, for a stated reason." The base row of every
DeepThinkVLA table remains decoded through a path validated on its two siblings
and not on itself.

Note the asymmetry that keeps this from being fatal: the edit protocol
**injects** a CoT and never asks the checkpoint to generate one, so the
capability the base checkpoint lacks is not a capability the published numbers
depend on. `build_input_cot_ids` assembles the same
`[prompt, <think>, cot, </think>, <action>]` on all three checkpoints, and it is
byte-identical to upstream's own on the two where the comparison is defined.
What is missing is the confirmation of that on the third, not a reason to think
it differs.

## What the paper says

Section `sec:analysis`: the P2 decode-equivalence claim is scoped to "two
architecture families and four of five checkpoints tested, with the fifth
named." This run does not change that scope; it changes the reason attached to
the fifth from a budget that might be raised to a property of the checkpoint.

`verify_paper_numbers.py` asserts `n_comparable == 0` and
`n_think_start_first == 0` here, so a later run that quietly reports the base
row as passing fails the audit.
