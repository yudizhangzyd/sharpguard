#!/usr/bin/env bash
# Submit the in-domain P3/AUROC job (#13, second model) with a per-checkpoint
# S3 read token.
#
# Bolt does NOT expand ${VAR} in a config's environment_variables from the
# submitting shell -- the pod would receive the literal string
# "${CKPT_AWS_ACCESS_KEY_ID}" and resolve it to empty. So the token is
# substituted before submit, into a mode-600 temp file outside the repo, which
# is removed afterwards. No credential ever reaches git.
#
# Usage: bash bolt/submit_auroc_indomain_ours.sh [CKPT_TASK_ID]
set -u
cd "$(dirname "$0")/.."

CFG="bolt/boltconfig-cotfaith-auroc-indomain-ours.yaml"
CKPT="${1:-$(sed -n "s/^  CKPT_TASK_ID: '\(.*\)'/\1/p" "$CFG")}"
[ -n "$CKPT" ] || { echo "[submit] no CKPT_TASK_ID"; exit 2; }
echo "[submit] checkpoint task: $CKPT"

# 36h: these jobs queue, and the sync happens after setup, so a token that
# expires in the queue wastes the whole slot.
creds=$(bolt task get-credentials "$CKPT" --expires-in-seconds 129600 2>/dev/null)
akid=$(printf '%s\n' "$creds" | sed -n 's/^export AWS_ACCESS_KEY_ID=//p')
skey=$(printf '%s\n' "$creds" | sed -n 's/^export AWS_SECRET_ACCESS_KEY=//p')
if [ -z "$akid" ] || [ -z "$skey" ]; then
    echo "[submit] could not get credentials for $CKPT"
    exit 3
fi

tmp=$(mktemp -t "boltcfg-auroc.XXXXXX") || exit 1
chmod 600 "$tmp"
AKID="$akid" SKEY="$skey" CFG="$CFG" python3 - >"$tmp" <<'PY'
import os
t = open(os.environ["CFG"]).read()
for ph, val in (("${CKPT_AWS_ACCESS_KEY_ID}", os.environ["AKID"]),
                ("${CKPT_AWS_SECRET_ACCESS_KEY}", os.environ["SKEY"])):
    assert t.count(ph) == 1, f"{ph} appears {t.count(ph)}x"
    # Single-quoted so a token containing YAML-special characters survives.
    t = t.replace(ph, "'" + val.replace("'", "''") + "'")
print(t, end="")
PY
if [ $? -ne 0 ]; then rm -f "$tmp"; echo "[submit] render failed"; exit 4; fi

out=$(bolt task submit --config "$tmp" 2>&1)
rc=$?
rm -f "$tmp"
printf '%s\n' "$out" | tail -20
exit $rc
