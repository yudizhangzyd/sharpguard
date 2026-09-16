#!/usr/bin/env python3
"""Does F_dir's direction_flip clearance survive on self-generated CoT?

Why this exists. direction_flip rewrites a lexical cue (left<->right etc.) in
a teacher-forced GOLD CoT whose MOVE phrase was itself derived from the
demonstration action. A model could reverse its action purely by having
memorized "this MOVE string maps to this action bin" during training,
without any reasoning mediating the edit -- F_dir cannot tell that apart from
genuine sensitivity to the direction word, because every number reported for
it elsewhere in this paper uses the teacher-forced CoT.

Self-generated CoT breaks that specific confound: the MOVE phrase the model
free-decodes is its own text, not a demonstration-derived string, so a
memorized (string, action) pair from training is far less likely to be
sitting behind it verbatim. If F_dir's clearance on direction_flip survives
-- or is not systematically smaller -- when the edited CoT is self-generated
rather than teacher-forced, that is evidence the model is responding to the
direction word's content rather than to a memorized demonstration string.

This computes F_dir directly from the raw per-sample action vectors
(a_orig, a_edit) already in the released stage2_selfgen reports -- no new
inference, the same records S8's self-generated-CoT check (magnitude score)
already uses for direction_flip's neighbouring families.

Usage:
    python3 scripts/fdir_selfgen_check.py
"""
import json
import math
import os

CHECKPOINTS = {"lora-r32": "r=32", "ecot-bridge": "ECoT-bridge",
                "no-cot": "no-CoT"}


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


def main():
    out = {}
    for ckpt, label in CHECKPOINTS.items():
        path = (f"results_v2/canonical_runs/stage2_selfgen/{ckpt}/"
                f"stage2_selfgen_report.json")
        d = json.load(open(path))
        coses = []
        for s in d.get("per_sample") or []:
            if (s.get("family") != "direction_flip" or s.get("skipped")
                    or s.get("timestep_kind") != "t0"):
                continue
            a_orig, a_edit = s.get("a_orig"), s.get("a_edit")
            if not a_orig or not a_edit:
                continue
            c = cosine(a_orig[:3], a_edit[:3])
            if c is not None:
                coses.append(c)
        n = len(coses)
        faithful = sum(1 for c in coses if c < -0.5)
        f_dir = faithful / n if n else None
        mean_cos = sum(coses) / n if n else None
        out[ckpt] = {"n": n, "n_faithful": faithful, "f_dir_selfgen": f_dir,
                    "mean_cos": mean_cos}
        print(f"{label:14s} n={n:4d}  F_dir(self-gen)={f_dir:.3f}  "
              f"mean_cos={mean_cos:+.3f}")

    # The teacher-forced F_dir values Table 3 (tab:directional) already
    # prints, for the same three checkpoints, quoted here rather than
    # recomputed, so a comparison is possible without re-deriving a number
    # this script does not own.
    teacher_forced = {"lora-r32": 0.579, "ecot-bridge": 0.120,
                      "no-cot": 0.087}
    for ckpt in CHECKPOINTS:
        out[ckpt]["f_dir_teacher_forced"] = teacher_forced[ckpt]

    dest = ("results_v2/canonical_runs/fdir_selfgen_check/"
            "fdir_selfgen_check.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\n[fdir-selfgen] -> {dest}")


if __name__ == "__main__":
    main()
