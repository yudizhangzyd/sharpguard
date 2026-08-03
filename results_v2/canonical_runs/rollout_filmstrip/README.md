# Rollout filmstrip capture (Figure 15 / appendix Figure 16)

The capture that Figure~\ref{fig:filmstrip} is drawn from. Regenerate the figure
with no arguments — this directory is the generator's default:

    python figures/gen_fig15_rollout_filmstrip.py

Compute-platform task `9r2mm3n3na`, git sha `7b092889`. Config: `libero_90`,
task 0 / episode 0, three arms (`nocot`, `cot_clean`, `cot_direction_flip`)
replayed from one init state, 400 steps, CoT regenerated **every** step,
`--capture-every 10`.

## Contents

| file | what |
| --- | --- |
| `rollout_edit_report.json` | the run's own report: per-arm trajectory (step, frame name, end-effector pose, action, MOVE phrase), SR, skipped-edit counts |
| `rollout_edit_probe.json` | the pre-run arm-pairing probe for this capture |
| `fig15_facts.json` | every number the caption quotes, written by the generator and read back by `scripts/verify_paper_numbers.py` |
| `frames/t0_ep0/` | 120 PNGs — 40 captured steps × 3 arms, the frame as the policy saw it, after the same flip and preprocessing |

Frames for `t1_ep0` are **not** included. The figure draws the lowest-numbered
episode filmed with all three arms, which is `t0_ep0`, and 120 PNGs is already
5.8 MB in a repository whose entire history is 30 MB. They are in the task's
artifact bucket if wanted; nothing in the paper reads them.

## What this run is and is not

It is **not** a faithfulness measurement. All three arms score 0, so no ΔSR is
defined — that is limitation (v), not a result, and one episode of one task
ranks nothing. It exists because the scalars for such a run are all identical
across arms and therefore invisible: two arms that both score 0/1 are the same
row in the record even when one of them has driven off the table.

The one property that makes the figure's rows comparable at all is that they are
**one scene**. `fig15_facts.json` records
`step0_pairing_max_mean_abs_pixel: 0.0` — the three arms are bit-identical at
the first displayed step, before any of them has acted. This is the fifth
capture attempt; the four earlier ones each failed this check in a different
way, and all four diagnostics are released under
`../rollout_arm_pairing_defect/`. The generator re-checks it and exits 2 rather
than draw rows whose differences a reader would attribute to the CoT edit.
