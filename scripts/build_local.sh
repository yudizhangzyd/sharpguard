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
# package's and it is local-only -- both builds paginate identically -- but it
# makes the submission copy unreadable on this machine, and a PDF you cannot
# read is a PDF you will not proofread.
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
# Text that prints outside its column. pdflatex reports every overfull box and
# then exits 0, so this is a defect class that is only ever found by grepping a
# log nobody keeps -- and this script builds into a mktemp it deletes. An 8.9pt
# overfull display equation shipped in the appendix that way. Sub-2pt is below
# what a reader can see; the audit draws the line, this only measures.
ovf = [float(x) for x in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide", log)]
geo["overfull_hbox_pt_max"] = max(ovf) if ovf else 0.0
geo["overfull_hbox_count"] = len(ovf)
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

# Where the body actually ends. ARR's 8-page limit counts body pages only:
# Limitations, Ethics, references and appendix are unlimited and follow it, so
# the budget is "the last page carrying numbered-section text". That page was
# checked by hand until a two-line prose fix pushed a float from page 7 to
# page 8 and spilled seven lines of the Conclusion onto page 9 -- a limit
# violation whose only symptom was a page nobody re-read. Measured here and
# asserted by scripts/verify_paper_numbers.py, in the same artifact as the
# geometry and for the same reason.
pdftotext -layout build/cot_faith_arr_proof.pdf "$TMP/proof.txt"
python3 - "$TMP/proof.txt" <<'PY'
import json, pathlib, sys
pages = pathlib.Path(sys.argv[1]).read_text(errors="replace").split("\f")
hit = [(i + 1, p.lstrip().startswith("Limitations"))
       for i, p in enumerate(pages)
       if any(ln.startswith("Limitations") for ln in p.splitlines())]
if not hit:
    raise SystemExit("[local] FAILED: the Limitations heading is not in the "
                     "rendered text, so the body page count cannot be measured")
page, at_top = hit[0]
# If the heading starts the page, the body ended on the page before it;
# if it starts partway down, the body ran to that page and the page counts.
last = page - 1 if at_top else page
geop = pathlib.Path("results_v2/canonical_runs/arr_build/geometry.json")
geo = json.loads(geop.read_text())
geo["body_last_page"] = last
geo["limitations_starts_page"] = page
geo["n_pages"] = sum(1 for p in pages if p.strip())
geop.write_text(json.dumps(geo, indent=2) + "\n")
flag = "" if last <= 8 else "  <-- OVER the ARR 8-page body limit"
print(f"[local] body ends page {last} (Limitations starts page {page}"
      f"{', mid-page' if not at_top else ''}){flag}")
PY

for f in build/cot_faith_arr.pdf build/cot_faith_arr_proof.pdf; do
    printf '[local] %-34s %s pages\n' "$f" "$(pdfinfo "$f" 2>/dev/null \
        | awk '/^Pages:/{print $2}')"
done
echo "[local] ok"
