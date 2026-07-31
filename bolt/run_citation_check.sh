#!/usr/bin/env bash
# Resolve the bibliography entries the authoring environment cannot reach.
#
# experiments/verify_citations.py confirms 10 of the 15 entries locally against
# export.arxiv.org, which is reachable. The other five are venue-only citations
# (CACM, CVPR, NeurIPS x2, RSS) with no arXiv id in the entry, so they need DBLP
# or CrossRef -- and both return "CONNECT tunnel failed, 403" from the authoring
# network. This job runs the same script from a networked host so those five
# stop being UNVERIFIED.
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
