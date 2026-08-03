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

# The page geometry acl.sty ends up with, recorded where the audit can read it.
# scripts/verify_paper_numbers.py checks each body figure's height against the
# fraction a top float may occupy, and \textheight is set by geometry inside the
# style rather than written in our source, so it has to come from a build. It
# used to be read straight out of a build directory -- and this script builds
# into a mktemp it deletes, so the value came from whichever scratch outdir was
# last left lying around. When that directory went away the checks did not fail;
# they stopped existing, and the only trace was the audit's claim count dropping
# by ten. An artifact under version control is the difference between a value
# that is measured and a value that is nearby.
python3 - "$TMP/sub/cot_faith_arr.log" <<'PY'
import json, pathlib, re, sys
log = pathlib.Path(sys.argv[1]).read_text(errors="replace")
geo = {}
for key, cmd in (("textheight_pt", "textheight"), ("textwidth_pt", "textwidth"),
                 ("columnwidth_pt", "columnwidth"), ("columnsep_pt", "columnsep")):
    m = re.search(r"\\" + cmd + r"=([\d.]+)pt", log)
    if m:
        geo[key] = float(m.group(1))
if "textheight_pt" not in geo:
    raise SystemExit("[local] FAILED: the build log does not report "
                     "\\textheight; the figure-geometry checks would go quiet")
geo["source"] = "pdflatex log for cot_faith_arr.tex, scripts/build_local.sh"
dest = pathlib.Path("results_v2/canonical_runs/arr_build")
dest.mkdir(parents=True, exist_ok=True)
(dest / "geometry.json").write_text(json.dumps(geo, indent=2) + "\n")
print("[local] geometry: " + ", ".join(f"{k}={v}" for k, v in geo.items()
                                      if k != "source"))
PY

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
