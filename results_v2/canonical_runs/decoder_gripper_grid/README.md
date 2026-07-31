# Decoder x gripper grid (bolt `f52r5nvnhb`) — the first gate pass

`gripper_ab.json` is the 2x3 factorial that identified the composition under
which the LIBERO rollout harness produces a non-trivial success rate. It is the
sixth attempt at the decode gate and the first with a positive result.

## Grid

`libero_object`, `openvla/openvla-7b-finetuned-libero-object`,
`unnorm_key=libero_object`, 10 episodes per cell (1 per task, all 10 tasks),
`--max-steps 0` = upstream's own 280-step budget for this suite, canonical init
states on every episode in every cell.

| cell | SR | succ/tot | frac_close_sent |
|---|---|---|---|
| `ours\|none+none` | 0.000 | 0/10 | 0.700 |
| `ours\|openvla+none` | 0.000 | 0/10 | 0.871 |
| `ours\|binvert+none` | 0.000 | 0/10 | 0.246 |
| `upstream\|none+none` | 0.000 | 0/10 | 0.591 |
| **`upstream\|openvla+none`** | **1.000** | **10/10** | 0.581 |
| **`upstream\|binvert+none`** | **1.000** | **10/10** | 0.579 |

`"winners": ["upstream|openvla+none", "upstream|binvert+none"]`.

## What it establishes

Both factors are necessary and neither is sufficient. Our own action decode
scores 0/10 under all three gripper conventions, and upstream's decode scores
0/10 when the gripper channel is passed through untransformed. Only the
conjunction succeeds. This is why the four published per-suite gates read 0/50:
they ran `ours` + `gripper=none`, i.e. the two-factor worst case, at 400 steps.

`openvla` and `binvert` are not separated by this grid. They agree at the top
gripper bin and differ only for `g<0.5`, and under upstream's decode the two
arms score identically (10/10, `frac_close_sent` 0.581 vs 0.579). Choosing
between them needs an input on which upstream's decode leaves the top bin.

## Reading the report

- **The job's terminal state is FAILED. It did not crash.**
  `experiments/gripper_ab_preflight.py` exits 1 unless exactly one cell clears
  `--win-threshold`; two cells cleared it, so the driver reported "no single
  winner (rc=1)". Every cell ran to completion and every number above is from
  that run.
- **`"max_steps": 0` at the top level of this file is the request, not the
  budget.** Each cell inside `arms` records `"max_steps": 280` and
  `"max_steps_below_upstream": false`. `gripper_ab_preflight.py` now emits
  `max_steps_requested` alongside a resolved `max_steps`; this artifact predates
  that change.
- `gripper_raw_mean` differs between the two winning arms (0.4174 vs 0.4198)
  because a succeeding policy visits different states, not because the arms
  disagree about any single action.

## Follow-on

`bolt/boltconfig-cotfaith-gate4-{spatial,object,goal,10}.yaml` re-runs the
four-suite gate at `upstream` + `gripper=openvla`, 50 episodes per suite (5 per
task), `MAX_STEPS=0` so each suite gets its own upstream budget, with
`gripper=none` retained as an in-run anchor.
