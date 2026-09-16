#!/usr/bin/env bash
# Submit the geometry-condition confound extension (direction_flip vs
# direction_flip_no_geom vs direction_flip_geom_consistent) to the 6
# LoRA/data-variant rows, following the same per-checkpoint S3 read-token
# pattern as bolt/submit_edit13.sh.
#
# Usage: bash bolt/submit_geom_extension.sh [row ...]   (default: all six)
set -u
cd "$(dirname "$0")/.."

declare -a ROWS=(
    "r8:u6gvfqeew9" "r16:8z9hhhg9sz" "r32:bcihypv3gu" "r64:26whnbbrmb"
    "data-50A:cib3z8skn5" "data-50B:9ay2rt3ra5"
)

for pair in "${ROWS[@]}"; do
    row="${pair%%:*}"; ckpt="${pair##*:}"
    if [ "$#" -gt 0 ]; then
        match=0
        for a in "$@"; do [ "$a" = "$row" ] && match=1; done
        [ "$match" -eq 1 ] || continue
    fi
    cfg="bolt/boltconfig-cotfaith-geom-$row.yaml"
    [ -f "$cfg" ] || { echo "[submit] no config $cfg"; continue; }

    creds=$(bolt task get-credentials "$ckpt" --expires-in-seconds 43200 2>/dev/null)
    akid=$(printf '%s\n' "$creds" | sed -n 's/^export AWS_ACCESS_KEY_ID=//p')
    skey=$(printf '%s\n' "$creds" | sed -n 's/^export AWS_SECRET_ACCESS_KEY=//p')
    if [ -z "$akid" ] || [ -z "$skey" ]; then
        echo "[submit] $row: could not get credentials for $ckpt -- skipped"
        continue
    fi

    tmp=$(mktemp -t "boltcfg-geom-$row.XXXXXX") || exit 1
    chmod 600 "$tmp"
    AKID="$akid" SKEY="$skey" CFG="$cfg" python3 - >"$tmp" <<'PY'
import os
t = open(os.environ["CFG"]).read()
for ph, val in (("${CKPT_AWS_ACCESS_KEY_ID}", os.environ["AKID"]),
                ("${CKPT_AWS_SECRET_ACCESS_KEY}", os.environ["SKEY"])):
    assert t.count(ph) == 1, f"{ph} appears {t.count(ph)}x in {os.environ['CFG']}"
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
