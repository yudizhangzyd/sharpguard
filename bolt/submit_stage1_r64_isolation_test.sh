#!/usr/bin/env bash
# Submit the Stage 1 lora-r64 isolation test (bolt/boltconfig-stage1-r64-isolation-test.yaml).
#
# Same credential-rendering pattern as bolt/submit_stage1.sh, for just the
# one LORA_R64_* pair this config needs. HF_TOKEN is deliberately left as
# the literal ${HF_TOKEN} string -- see bolt/submit_stage1.sh's comment;
# this job pulls no gated HF model.
#
# Usage: bash bolt/submit_stage1_r64_isolation_test.sh
set -u
cd "$(dirname "$0")/.."

CFG="bolt/boltconfig-stage1-r64-isolation-test.yaml"
[ -f "$CFG" ] || { echo "[submit] no config $CFG"; exit 1; }

creds_r64=$(bolt task get-credentials 26whnbbrmb --expires-in-seconds 129600 2>/dev/null)
akid_r64=$(printf '%s\n' "$creds_r64" | sed -n 's/^export AWS_ACCESS_KEY_ID=//p')
skey_r64=$(printf '%s\n' "$creds_r64" | sed -n 's/^export AWS_SECRET_ACCESS_KEY=//p')

if [ -z "$akid_r64" ] || [ -z "$skey_r64" ]; then
    echo "[submit] could not get credentials for 26whnbbrmb (r=64) -- aborting"
    exit 1
fi

tmp=$(mktemp -t "boltcfg-r64iso.XXXXXX") || exit 1
chmod 600 "$tmp"
AKID64="$akid_r64" SKEY64="$skey_r64" CFG="$CFG" python3 - >"$tmp" <<'PY'
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
    printf 'stage1-r64-isolation  %s\n' "$id"
else
    echo "[submit] SUBMIT FAILED"
    printf '%s\n' "$out" | head -10
    exit 1
fi
