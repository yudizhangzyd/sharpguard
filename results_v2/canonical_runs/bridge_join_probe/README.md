# Bridge V2: the two public exports cannot be joined per episode

`bridge_join_probe.json` — bolt task `754ru9usqe`, CPU-only, no model loaded.

O4 (Section on the uncontrolled Bridge-vs-LIBERO observation) asks for a
same-data ablation: fine-tune ECoT-bridge on a ~4k-trajectory Bridge V2 subset
matched to LIBERO-90 scale. Doing that needs the CoT annotations and the
trajectories joined per episode, and the two public artifacts that carry them do
not share a key. This report measures that, so the next person attempting the
ablation does not spend the runs we spent discovering it.

The two sides:

| | repo | count |
|---|---|---|
| CoT annotations | `Embodied-CoT/embodied_features_bridge` | 60,062 annotated episodes (1.40 GB, 3,200 shard paths, 1,670,937 annotated steps) |
| trajectories | `IPEC-COMMUNITY/bridge_orig_lerobot` | 53,192 episodes, 19,541 distinct instructions |

## The layout, which is four levels deep

```
raw[<authors' absolute NFS path>][<episode index within that shard>] = {
  "features":  {move_primitive: [...], gripper_position: [...], bboxes: [...]},  # per-step LISTS
  "metadata":  {episode_id, file_path, n_steps, language_instruction},           # per-episode
  "reasoning": {"0": {...}, "1": {...}, ...},                                    # per-step DICTS
}
```

The top level is paths, not episode ids. `episode_id`, `file_path` and `n_steps`
are present on all 60,062 episodes; `language_instruction` on 43,805. 13,740
episodes have no usable `reasoning` subtree and 4,237 have no `features`.

## Three candidate join keys, all measured

**`episode_id` — do not use it.** It matches only 1,111 of 53,192 LeRobot
episodes (2.1%), its range is `[0, 1110]` against 60,062 annotated episodes (so
it is per-shard numbering, not global), it collides 879 times, and on the pairs
it *does* match the two sides' language instructions agree only 0.280 of the
time.

This is the outcome that matters most, because it is the one that looks like
success. A join on `episode_id` returns a complete, plausible index; a training
run on it completes normally; and the resulting checkpoint is not
distinguishable post hoc from one trained on a correct index. The 0.280 is the
only thing that reveals it, which is why a shared integer key was checked for
instruction agreement rather than accepted on overlap alone.

**The LeRobot source path — does not exist.** Every one of the 53,192 LeRobot
episode records carries exactly `episode_index, length, tasks`. The conversion
did not preserve an upstream path, so there is no exact route.

**Instruction text — usable, with two caveats.** 19,541 shared normalized
instructions cover 38,660 of 53,192 LeRobot episodes (72.7%), median fanout 1,
giving **41,634 reachable annotated episodes** — an order of magnitude above the
~4k the ablation needs. Caveats: (1) for keys with fanout > 1 the join is
task-level, not trajectory-level — the CoT describes this task but not
necessarily this trajectory; (2) the shared keys include crowdsourced strings
that are not language — `"1"`, `"9"`, `"12345678"`, `"3wsws"`,
`"7210 2199 5955 2055 534"` — and the max fanout on a single key is 963. Those
keys pool unrelated trajectories, so a consumer must refuse them rather than pick
within them. `experiments/cotfaith_train_bridge.py` requires ≥8 characters and ≥2
alphabetic words and drops any key with more than 8 candidates.

## A correct join is still not a renderable trace

The eight CoT tags are split across two subtrees, so a per-step **merge** is
required. Over 4,000 inspected steps:

| source | tags filled |
|---|---|
| `reasoning` alone | TASK, PLAN, SUBTASK, SUBTASK REASONING, MOVE REASONING, MOVE at 1.0; **VISIBLE OBJECTS and GRIPPER POSITION at 0.0** |
| `features` alone | exactly those two, at 1.0 |
| merged | all eight at 1.0 |

`tags_unfillable_by_either` is empty, so the merge is sufficient as well as
necessary. This is not a formatting detail: a trainer that renders from
`reasoning` alone — as ours did — teaches the model to emit two permanently
blank tags, and no downstream metric in this benchmark can see the difference.
The LIBERO export packs all eight tags into one per-step dict, which is why code
written against LIBERO does not transfer here and why no choice of join key
fixes it.

## What this does not establish

The instruction-text route is not verified at the level of individual
trajectories, and this report does not claim it is. It bounds the join
(72.7% coverage, median fanout 1) and rules out the alternative (0.280
instruction agreement on `episode_id`); it does not certify that any particular
(trajectory, reasoning) pair is the one the annotation authors intended. A
trajectory-level claim would need a key neither export publishes.

## Reproducing

```
bolt task submit --config bolt/boltconfig-cotfaith-bridge-join-probe.yaml
```

`tests/test_bridge_reasoning_join.py` checks the consumer offline (41 checks, no
GPU): both historical join bugs as negative controls, the collision refusal, the
degenerate keys above verbatim, the fanout cap, and that no mispaired sample can
reach training while the id probe is still running.
