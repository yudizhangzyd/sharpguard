#!/usr/bin/env bash
# Submit the Stage 2 self-generated-CoT job (bolt/boltconfig-stage2-selfgen.yaml).
#
# A separate script from bolt/submit_stage1.sh, deliberately, even though the
# rendering logic is nearly identical: this job's two S3 credential pairs are
# LORA_R32_* (task bcihypv3gu, the SAME pair Stage 1 renders) and NOCOT_* (task
# a8eegzcg4r, a checkpoint Stage 1 never touches) -- not LORA_R32_*/LORA_R64_*
# like Stage 1. Reusing submit_stage1.sh directly would mean either hardcoding
# a third/fourth credential pair into a script whose name and header comment
# are specific to Stage 1's own 6-checkpoint config (risking a change that
# breaks Stage 1's already-submitted, still-queued job), or parametrizing it
# over which config/credential-pairs to use -- more invasive than a 20-line
# copy for a pattern this project already repeats per job
# (bolt/submit_edit13.sh, bolt/submit_stage1.sh). This script IS that copy,
# with the second credential pair's task id and env var prefix swapped.
#
# Bolt does NOT expand ${VAR} in a config's environment_variables from the
# submitting shell, so LORA_R32_AWS_ACCESS_KEY_ID / _SECRET_ACCESS_KEY and the
# NOCOT_* pair have to be substituted before submit, or the pod receives the
# literal placeholder string. Each token is scoped to ONE task prefix (a token
# for bcihypv3gu gets AccessDenied on a8eegzcg4r), so this job needs both.
# HF_TOKEN is deliberately left as the literal ${HF_TOKEN} string:
# sharpguard.hf_retry already drops that exact placeholder and retries
# anonymously, and every model this job pulls is either public (ECoT-bridge)
# or fetched over S3, not gated by an HF token.
#
# Credentials are written to a mode-600 temp file outside the repo and
# removed immediately after submit, so no token ever touches git or survives
# this script's exit.
#
# Usage: bash bolt/submit_stage2_selfgen.sh
set -u
cd "$(dirname "$0")/.."

CFG="bolt/boltconfig-stage2-selfgen.yaml"
[ -f "$CFG" ] || { echo "[submit] no config $CFG"; exit 1; }

# 36h expiry, not the 12h default: matches submit_edit13.sh / submit_stage1.sh's
# reasoning (jobs queue on shared clusters, and the token is minted before the
# queue wait, not after).
creds_r32=$(bolt task get-credentials bcihypv3gu --expires-in-seconds 129600 2>/dev/null)
akid_r32=$(printf '%s\n' "$creds_r32" | sed -n 's/^export AWS_ACCESS_KEY_ID=//p')
skey_r32=$(printf '%s\n' "$creds_r32" | sed -n 's/^export AWS_SECRET_ACCESS_KEY=//p')

creds_nocot=$(bolt task get-credentials a8eegzcg4r --expires-in-seconds 129600 2>/dev/null)
akid_nocot=$(printf '%s\n' "$creds_nocot" | sed -n 's/^export AWS_ACCESS_KEY_ID=//p')
skey_nocot=$(printf '%s\n' "$creds_nocot" | sed -n 's/^export AWS_SECRET_ACCESS_KEY=//p')

if [ -z "$akid_r32" ] || [ -z "$skey_r32" ]; then
    echo "[submit] could not get credentials for bcihypv3gu (lora-r32) -- aborting"
    exit 1
fi
if [ -z "$akid_nocot" ] || [ -z "$skey_nocot" ]; then
    echo "[submit] could not get credentials for a8eegzcg4r (no-cot) -- aborting"
    exit 1
fi

tmp=$(mktemp -t "boltcfg-stage2.XXXXXX") || exit 1
chmod 600 "$tmp"
AKID32="$akid_r32" SKEY32="$skey_r32" AKIDNC="$akid_nocot" SKEYNC="$skey_nocot" CFG="$CFG" \
    python3 - >"$tmp" <<'PY'
import os
t = open(os.environ["CFG"]).read()
subs = (
    ("${LORA_R32_AWS_ACCESS_KEY_ID}", os.environ["AKID32"]),
    ("${LORA_R32_AWS_SECRET_ACCESS_KEY}", os.environ["SKEY32"]),
    ("${NOCOT_AWS_ACCESS_KEY_ID}", os.environ["AKIDNC"]),
    ("${NOCOT_AWS_SECRET_ACCESS_KEY}", os.environ["SKEYNC"]),
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
    printf 'stage2-selfgen  %s\n' "$id"
else
    echo "[submit] SUBMIT FAILED"
    printf '%s\n' "$out" | head -10
    exit 1
fi
