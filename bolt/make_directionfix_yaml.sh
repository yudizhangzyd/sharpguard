#!/usr/bin/env bash
# Build a direction_flip-only re-run yaml from a base yaml, substituting
# fresh S3 credentials for CKPT_TASK_ID without ever printing them.
# Usage: make_directionfix_yaml.sh <base_yaml> <out_yaml> <ckpt_task_id> <family_note>
set -e
BASE="$1"; OUT="$2"; CKPT="$3"; NOTE="$4"

eval "$(bolt task get-credentials "$CKPT" --expires-in-seconds 129600)"
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    echo "[FATAL] failed to fetch credentials for $CKPT"
    exit 2
fi

AK="$AWS_ACCESS_KEY_ID" SK="$AWS_SECRET_ACCESS_KEY" CKPT="$CKPT" NOTE="$NOTE" python3 - "$BASE" "$OUT" <<'PYEOF'
import os, re, sys
base, out = sys.argv[1], sys.argv[2]
ak, sk, ckpt, note = os.environ["AK"], os.environ["SK"], os.environ["CKPT"], os.environ["NOTE"]
text = open(base).read()
text = text.replace("${CKPT_AWS_ACCESS_KEY_ID}", ak)
text = text.replace("${CKPT_AWS_SECRET_ACCESS_KEY}", sk)
text = re.sub(r"CKPT_TASK_ID: '[^']*'", f"CKPT_TASK_ID: '{ckpt}'", text)
text = text.replace("N_SAMPLES: '100'", "N_SAMPLES: '100'\n  FAMILIES: 'direction_flip'")
text = re.sub(r"^name: '.*'$",
    f"name: 'CoT-Faith direction_flip-only re-run ({note}), corrected DIRECTION_PAIRS'",
    text, count=1, flags=re.M)
text = re.sub(r"^description: '.*'$",
    "description: 'DIRECTION_PAIRS in/out pair was removed from sharpguard/attacks/"
    "cot_edit.py after exhaustive enumeration of all 633 real in/out occurrences in "
    "the reasoning dataset movement_reasoning field found 0 genuine spatial reversals "
    "-- all were idiom collisions (in order to, in front of) or static position "
    "references (the robot is in the center), also present unflagged in 5/40 of this "
    f"projects own LLM-judge validation sample. Re-runs ONLY direction_flip ({note}) "
    f"against the SAME checkpoint (CKPT_TASK_ID {ckpt}) under the corrected table; "
    "every other family is unaffected and not re-run.'",
    text, count=1, flags=re.M)
open(out, "w").write(text)
print("wrote", out)
PYEOF
