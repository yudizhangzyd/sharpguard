#!/usr/bin/env python3
"""paraphrase_null_lenexact (4th null family): does length or lexical
substitution drive the paraphrase/scramble floor gap?

paraphrase_null (mean +0.97 words) and syntactic_scramble (length-exact, word
reordering) leave two dimensions confounded: paraphrase_null changes both
which words are used AND sequence length; syntactic_scramble changes neither
which words are used (it reorders the same words) nor preserves the sentence
structure paraphrase_null does. Neither isolates length from lexical identity.

paraphrase_null_lenexact (sharpguard/attacks/cot_edit.py) is paraphrase_null
with release->free instead of release->"let go of" (its one multi-word
substitution), so every substitution is single-word-to-single-word:
meaning-preserving AND length-exact, like syntactic_scramble, but via lexical
substitution, like paraphrase_null. If its floor lands near paraphrase_null's
(0.567 on r=32), length is not what separates the two existing floors. If it
lands near syntactic_scramble's (0.348), length is.

Usage:
    python3 scripts/paraphrase_lenexact_floor.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from floor_convention_robustness import rate, TAU_DEFAULT

RECORDS_GLOB = ("results_v2/canonical_runs/ours_lora-r32_paraphrase_lenexact/"
                 "seed*/cot_edit_report.json")
PUBLISHED_R32 = {"paraphrase_null": 0.567, "syntactic_scramble": 0.348}


def main():
    records = []
    for f in sorted(glob.glob(RECORDS_GLOB)):
        records += json.load(open(f))["per_sample"]
    if not records:
        print(f"[FATAL] no records matched {RECORDS_GLOB}")
        return 2

    f_lenexact, n = rate(records, "paraphrase_null_lenexact", TAU_DEFAULT)
    gap_para = abs(f_lenexact - PUBLISHED_R32["paraphrase_null"])
    gap_scram = abs(f_lenexact - PUBLISHED_R32["syntactic_scramble"])

    print(f"r=32, tau={TAU_DEFAULT}:")
    print(f"  paraphrase_null_lenexact  F={f_lenexact:.3f}  (n={n}, "
          f"pooled over 3 seeds)")
    print(f"  paraphrase_null (published)   F={PUBLISHED_R32['paraphrase_null']:.3f}"
          f"  |gap|={gap_para:.3f}")
    print(f"  syntactic_scramble (published) F={PUBLISHED_R32['syntactic_scramble']:.3f}"
          f"  |gap|={gap_scram:.3f}")
    verdict = ("closer to paraphrase_null -- length is not what separates "
               "the two existing floors" if gap_para < gap_scram else
               "closer to syntactic_scramble -- length is what separates "
               "the two existing floors")
    print(f"\n  {verdict}")

    out = {"tau": TAU_DEFAULT, "n": n, "F_paraphrase_null_lenexact": f_lenexact,
           "F_paraphrase_null_published": PUBLISHED_R32["paraphrase_null"],
           "F_syntactic_scramble_published": PUBLISHED_R32["syntactic_scramble"],
           "gap_vs_paraphrase_null": gap_para,
           "gap_vs_syntactic_scramble": gap_scram, "verdict": verdict}
    dest = ("results_v2/canonical_runs/paraphrase_lenexact_floor/"
            "paraphrase_lenexact_floor.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\n[paraphrase-lenexact-floor] -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
