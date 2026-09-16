#!/usr/bin/env bash
# Submit the r=64 self-generated-CoT job (bolt/boltconfig-cotfaith-stage2-selfgen-r64.yaml).
#
# Same credential-rendering pattern as bolt/submit_stage2_selfgen.sh: bolt
# does NOT expand ${VAR} in a config's environment_variables from the
# submitting shell, so LORA_R64_AWS_ACCESS_KEY_ID/_SECRET_ACCESS_KEY have to
# be substituted before submit. Credentials are written to a mode-600 temp
# file outside the repo and removed immediately after submit.
#
# Usage: bash bolt/submit_stage2_selfgen_r64.sh
set -u
cd "$(dirname "$0")/.."

CFG="bolt/boltconfig-cotfaith-stage2-selfgen-r64.yaml"
[ -f "$CFG" ] || { echo "[submit] no config $CFG"; exit 1; }

creds_r64=$(bolt task get-credentials 26whnbbrmb --expires-in-seconds 129600 2>/dev/null)
akid_r64=$(printf '%s\n' "$creds_r64" | sed -n 's/^export AWS_ACCESS_KEY_ID=//p')
skey_r64=$(printf '%s\n' "$creds_r64" | sed -n 's/^export AWS_SECRET_ACCESS_KEY=//p')

if [ -z "$akid_r64" ] || [ -z "$skey_r64" ]; then
    echo "[submit] could not get credentials for 26whnbbrmb (lora-r64) -- aborting"
    exit 1
fi

tmp=$(mktemp -t "boltcfg-stage2-r64.XXXXXX") || exit 1
chmod 600 "$tmp"
AKID64="$akid_r64" SKEY64="$skey_r64" CFG="$CFG" \
    python3 - >"$tmp" <<'PY'
import os
t = open(os.environ["CFG"]).read()
subs = (
    ("${LORA_R64_AWS_ACCESS_KEY_ID}", os.environ["AKID64"]),
    ("${LORA_R64_AWS_SECRET_ACCESS_KEY}", os.environ["SKEY64"]),
)
for ph, val in subs:
    assert t.count(ph) == 1, f"{ph} appears {t.count(ph)}x in {os.environ['CFG']}"
    t = t.replace(ph, "'" + val.replace("'", "''") + "'")
print(t, end="")
PY
if [ $? -ne 0 ]; then
    rm -f "$tmp"
    echo "[submit] render failed"
    exit 1
fi

out=$(bolt task submit --config "$tmp" --tar . 2>&1)
rm -f "$tmp"
id=$(printf '%s\n' "$out" | sed -n 's/^Task \([a-z0-9]*\) submitted.*/\1/p' | head -1)
if [ -n "$id" ]; then
    printf 'stage2-selfgen-r64  %s\n' "$id"
else
    echo "[submit] SUBMIT FAILED"
    printf '%s\n' "$out" | head -20
    exit 1
fi
