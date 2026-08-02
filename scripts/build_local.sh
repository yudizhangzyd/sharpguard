#!/usr/bin/env bash
# Build the ARR submission locally and drop both PDFs in build/.
#
# Two PDFs, because neither alone is what you want:
#
#   build/cot_faith_arr.pdf        the submission, exactly as ARR receives it,
#                                  line numbers and all.
#   build/cot_faith_arr_proof.pdf  the same document with line numbers off,
#                                  for reading.
#
# The proof copy exists because of a toolchain difference, not a preference.
# lineno v5.7 (TeX Live 2026) places [switch]-mode numbers over the body text
# instead of in the margin; v4.41, which the Bolt build image carries and which
# ARR compiles against, puts them where they belong. The defect is the
# package's and it is local-only -- both builds paginate identically at 46
# pages -- but it makes the submission copy unreadable on this machine, and a
# PDF you cannot read is a PDF you will not proofread.
#
# Both are built from the same source in the same run, so they cannot drift.
set -euo pipefail
cd "$(dirname "$0")/.."

export TEXINPUTS=".:./acl-style:${TEXINPUTS:-}"
export BSTINPUTS=".:./acl-style:${BSTINPUTS:-}"

command -v latexmk >/dev/null || { echo "no latexmk on PATH"; exit 3; }
mkdir -p build

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[local] submission copy..."
latexmk -pdf -interaction=nonstopmode -outdir="$TMP/sub" \
    cot_faith_arr.tex > "$TMP/sub.log" 2>&1 \
    || { echo "[local] FAILED -- last 25 lines:"; tail -25 "$TMP/sub.log"; exit 1; }

echo "[local] proof copy (line numbers suppressed)..."
latexmk -pdf -interaction=nonstopmode -outdir="$TMP/proof" -jobname=proof \
    -pdflatex='pdflatex %O "\AtBeginDocument{\nolinenumbers}\input{%S}"' \
    cot_faith_arr.tex > "$TMP/proof.log" 2>&1 \
    || { echo "[local] proof FAILED -- last 25 lines:"; tail -25 "$TMP/proof.log"; exit 1; }

cp "$TMP/sub/cot_faith_arr.pdf"  build/cot_faith_arr.pdf
cp "$TMP/proof/proof.pdf"        build/cot_faith_arr_proof.pdf

# Where every body float actually printed. This is the check that would have
# caught the figures-on-page-41 defect, and it is cheap enough to run on every
# local build rather than only in CI: \newlabel already records the page each
# label resolved to, so it needs nothing but the .aux.
python3 - "$TMP/sub/cot_faith_arr.aux" <<'PY'
import pathlib, re, sys
aux = pathlib.Path(sys.argv[1]).read_text()
body = set(re.findall(r"\\label\{((?:fig|tab):[^}]+)\}",
                      pathlib.Path("cot_faith_arr.tex").read_text()))
placed = {l: int(p) for l, _, p in
          re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}", aux)
          if l in body}
for lab in sorted(placed, key=placed.get):
    print(f"[local]   {lab:22} page {placed[lab]}")
missing = sorted(body - set(placed))
if missing:
    print(f"[local]   UNRESOLVED: {', '.join(missing)}")
PY

for f in build/cot_faith_arr.pdf build/cot_faith_arr_proof.pdf; do
    printf '[local] %-34s %s pages\n' "$f" "$(pdfinfo "$f" 2>/dev/null \
        | awk '/^Pages:/{print $2}')"
done
echo "[local] ok"
