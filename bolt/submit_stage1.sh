#!/usr/bin/env bash
# Submit the Stage 1 continuous-metric job (bolt/boltconfig-stage1-continuous.yaml).
#
# Same pattern as bolt/submit_edit13.sh, extended to two credential pairs:
# Bolt does NOT expand ${VAR} in a config's environment_variables from the
# submitting shell, so LORA_R32_AWS_ACCESS_KEY_ID / _SECRET_ACCESS_KEY and the
# r64 pair have to be substituted before submit, or the pod receives the
# literal placeholder string. Each token is scoped to ONE task prefix (a
# token for bcihypv3gu gets AccessDenied on 26whnbbrmb), so this job needs
# both. HF_TOKEN is deliberately left as the literal ${HF_TOKEN} string:
# sharpguard.hf_retry already drops that exact placeholder and retries
# anonymously (see bolt/setup-openvla.sh's comment on hs3gey53wh et al.), and
# every model this job pulls (ECoT-bridge, DeepThinkVLA base/SFT/RL) is a
# public HF repo, matching the two existing configs that reference the same
# checkpoints and already ship with ${HF_TOKEN} unrendered.
#
# Credentials are written to a mode-600 temp file outside the repo and
# removed immediately after submit, so no token ever touches git or survives
# this script's exit.
#
# Usage: bash bolt/submit_stage1.sh
set -u
cd "$(dirname "$0")/.."

CFG="bolt/boltconfig-stage1-continuous.yaml"
[ -f "$CFG" ] || { echo "[submit] no config $CFG"; exit 1; }

# 36h expiry, not the 12h default: matches submit_edit13.sh's reasoning (jobs
# queue on shared clusters, and the token is minted before the queue wait).
creds_r32=$(bolt task get-credentials bcihypv3gu --expires-in-seconds 129600 2>/dev/null)
akid_r32=$(printf '%s\n' "$creds_r32" | sed -n 's/^export AWS_ACCESS_KEY_ID=//p')
skey_r32=$(printf '%s\n' "$creds_r32" | sed -n 's/^export AWS_SECRET_ACCESS_KEY=//p')

creds_r64=$(bolt task get-credentials 26whnbbrmb --expires-in-seconds 129600 2>/dev/null)
akid_r64=$(printf '%s\n' "$creds_r64" | sed -n 's/^export AWS_ACCESS_KEY_ID=//p')
skey_r64=$(printf '%s\n' "$creds_r64" | sed -n 's/^export AWS_SECRET_ACCESS_KEY=//p')

if [ -z "$akid_r32" ] || [ -z "$skey_r32" ]; then
    echo "[submit] could not get credentials for bcihypv3gu (r=32) -- aborting"
    exit 1
fi
if [ -z "$akid_r64" ] || [ -z "$skey_r64" ]; then
    echo "[submit] could not get credentials for 26whnbbrmb (r=64) -- aborting"
    exit 1
fi

tmp=$(mktemp -t "boltcfg-stage1.XXXXXX") || exit 1
chmod 600 "$tmp"
AKID32="$akid_r32" SKEY32="$skey_r32" AKID64="$akid_r64" SKEY64="$skey_r64" CFG="$CFG" \
    python3 - >"$tmp" <<'PY'
import os
t = open(os.environ["CFG"]).read()
subs = (
    ("${LORA_R32_AWS_ACCESS_KEY_ID}", os.environ["AKID32"]),
    ("${LORA_R32_AWS_SECRET_ACCESS_KEY}", os.environ["SKEY32"]),
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
    printf 'stage1-continuous  %s\n' "$id"
else
    echo "[submit] SUBMIT FAILED"
    printf '%s\n' "$out" | head -10
    exit 1
fi
