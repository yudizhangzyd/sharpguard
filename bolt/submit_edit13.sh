#!/usr/bin/env bash
# Submit the 13-family x 3-seed edit jobs with a per-checkpoint S3 read token.
#
# Bolt does NOT expand ${VAR} in a config's environment_variables from the
# submitting shell -- gncx99btvp received the literal string
# "${CKPT_AWS_ACCESS_KEY_ID}" (25 characters), which the pod then resolved to
# empty. So the token has to be substituted before submit. It is written to a
# mode-600 temp file outside the repo and removed afterwards, so no credential
# ever reaches git.
#
# Each token is scoped to ONE task prefix -- a token for u6gvfqeew9 gets
# AccessDenied on bcihypv3gu -- so every row needs its own.
#
# Usage: bash bolt/submit_edit13.sh [row ...]     (default: all seven)
set -u
cd "$(dirname "$0")/.."

declare -a ROWS=(
    "r8:u6gvfqeew9" "r16:8z9hhhg9sz" "r32:bcihypv3gu" "r64:26whnbbrmb"
    "no-cot:a8eegzcg4r" "data-50A:cib3z8skn5" "data-50B:9ay2rt3ra5"
)


for pair in "${ROWS[@]}"; do
    row="${pair%%:*}"; ckpt="${pair##*:}"
    if [ "$#" -gt 0 ]; then
        match=0
        for a in "$@"; do [ "$a" = "$row" ] && match=1; done
        [ "$match" -eq 1 ] || continue
    fi
    cfg="bolt/boltconfig-cotfaith-edit13-$row.yaml"
    [ -f "$cfg" ] || { echo "[submit] no config $cfg"; continue; }

    # 36h rather than the 12h default: these jobs queue, and the sync happens
    # after setup, so a token that expires in the queue wastes a whole slot.
    creds=$(bolt task get-credentials "$ckpt" --expires-in-seconds 129600 2>/dev/null)
    akid=$(printf '%s\n' "$creds" | sed -n 's/^export AWS_ACCESS_KEY_ID=//p')
    skey=$(printf '%s\n' "$creds" | sed -n 's/^export AWS_SECRET_ACCESS_KEY=//p')
    if [ -z "$akid" ] || [ -z "$skey" ]; then
        echo "[submit] $row: could not get credentials for $ckpt -- skipped"
        continue
    fi

    tmp=$(mktemp -t "boltcfg-$row.XXXXXX") || exit 1
    chmod 600 "$tmp"
    AKID="$akid" SKEY="$skey" CFG="$cfg" python3 - >"$tmp" <<'PY'
import os
t = open(os.environ["CFG"]).read()
for ph, val in (("${CKPT_AWS_ACCESS_KEY_ID}", os.environ["AKID"]),
                ("${CKPT_AWS_SECRET_ACCESS_KEY}", os.environ["SKEY"])):
    assert t.count(ph) == 1, f"{ph} appears {t.count(ph)}x in {os.environ['CFG']}"
    # Single-quoted so a token containing YAML-special characters survives; the
    # tokens are base64url and contain '=' and '-', which are fine, but ':' or
    # '#' in a future token format would silently truncate the value unquoted.
    t = t.replace(ph, "'" + val.replace("'", "''") + "'")
print(t, end="")
PY
    if [ $? -ne 0 ]; then rm -f "$tmp"; echo "[submit] $row: render failed"; continue; fi

    out=$(bolt task submit --config "$tmp" 2>&1)
    rm -f "$tmp"
    id=$(printf '%s\n' "$out" | sed -n 's/^Task \([a-z0-9]*\) submitted.*/\1/p' | head -1)
    if [ -n "$id" ]; then
        printf '%-10s %s  ckpt=%s\n' "$row" "$id" "$ckpt"
    else
        printf '%-10s SUBMIT FAILED  ckpt=%s\n' "$row" "$ckpt"
        printf '%s\n' "$out" | head -5
    fi
done
