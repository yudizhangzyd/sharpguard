#!/usr/bin/env bash
# Re-verify the bibliography against a live registry, from a networked host.
#
# History, because it changed what this job is for. The first run (qrpd3f8z58)
# existed to reach DBLP for the five venue-only citations that print no arXiv id,
# since DBLP 403s from the authoring network. It failed at that: DBLP times out
# on the TLS handshake from bolt too. What it did establish is that the fallback
# chain was wrong -- those five needed an arXiv TITLE search, not a different
# registry, because the check had only ever queried arXiv when the entry itself
# printed an id. With that fixed all 15 entries confirm.
#
# So this job is now the independent re-run: same script, same manuscript,
# different host and different network path. If it returns anything other than
# 15/15 CONFIRMED, the local result depended on something about the authoring
# machine and is not a property of the bibliography.
#
# Nothing here is a model, a GPU or a simulator: it is fifteen HTTP GETs. It runs
# on bolt because that is where this project's network egress lives, and because
# the resulting report is a release artifact that the paper audit reads.
set -e -x

cd "$(dirname "$0")/.."

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/citation-check"
mkdir -p "$OUT_DIR"

python -c "import sys; print('[cite] python', sys.version)"

# No setup_command and no pip install: the script uses urllib and re from the
# standard library only, precisely so that a citation check cannot fail for an
# environment reason.
set +e
python experiments/verify_citations.py --out "$OUT_DIR/citation_check.json"
RC=$?
set -e

echo ""
echo "==== citation_check.json ===="
cat "$OUT_DIR/citation_check.json" || echo "(no report: the script died early)"

# Exit code is the script's: nonzero means a registry CONTRADICTS the
# manuscript, which is a defect to fix rather than a job failure to retry.
exit "$RC"
