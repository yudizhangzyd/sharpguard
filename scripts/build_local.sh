#!/usr/bin/env bash
# Build the ICLR submission locally and drop it in build/.
#
# Used to build two PDFs (submission + a line-numbers-suppressed "proof"
# copy), because ACL's [review] mode loaded the lineno package and TeX Live
# 2026's lineno v5.7 rendered its [switch]-mode numbers over the body text
# instead of in the margin, making the submission copy unreadable on this
# machine. Since the ARR->ICLR migration, iclr2027_conference.sty draws its
# own review-mode ruler directly (\AddToShipoutPicture + \iclrruler, gated on
# \ificlrfinal) rather than through lineno, so that defect no longer applies
# and there is nothing left to suppress -- both filenames are now the same
# build, kept only so nothing that references build/cot_faith_proof.pdf breaks.
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
    cot_faith.tex > "$TMP/sub.log" 2>&1 \
    || { echo "[local] FAILED -- last 25 lines:"; tail -25 "$TMP/sub.log"; exit 1; }

cp "$TMP/sub/cot_faith.pdf"  build/cot_faith.pdf
cp "$TMP/sub/cot_faith.pdf"  build/cot_faith_proof.pdf

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
python3 - "$TMP/sub/cot_faith.log" <<'PY'
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
geo["source"] = "pdflatex log for cot_faith.tex, scripts/build_local.sh"
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
python3 - "$TMP/sub/cot_faith.aux" <<'PY'
import pathlib, re, sys
aux = pathlib.Path(sys.argv[1]).read_text()
body = set(re.findall(r"\\label\{((?:fig|tab):[^}]+)\}",
                      pathlib.Path("cot_faith.tex").read_text()))
placed = {l: int(p) for l, _, p in
          re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}", aux)
          if l in body}
for lab in sorted(placed, key=placed.get):
    print(f"[local]   {lab:22} page {placed[lab]}")
missing = sorted(body - set(placed))
if missing:
    print(f"[local]   UNRESOLVED: {', '.join(missing)}")
PY

# Where the body actually ends. ICLR's 9-page initial-submission limit counts
# body pages only:
# Limitations, Ethics, references and appendix are unlimited and follow it, so
# the budget is "the last page carrying numbered-section text". That page was
# checked by hand until a two-line prose fix pushed a float from page 7 to
# page 8 and spilled seven lines of the Conclusion onto page 9 -- a limit
# violation whose only symptom was a page nobody re-read. Measured here and
# asserted by scripts/verify_paper_numbers.py, in the same artifact as the
# geometry and for the same reason.
# Extracted WITHOUT -layout, i.e. in reading order rather than in visual
# columns. -layout preserves the two columns by padding one line with both, so
# a heading that starts column 2 arrives appended to a column-1 line -- which
# is what happened the first time the Limitations heading landed beside body
# text instead of on its own page, and the measurement failed closed rather
# than reporting page 8. Reading order puts the heading on its own line
# wherever it is, and "is there body text before it on this page" is exactly
# the question the 8-page limit asks.
pdftotext build/cot_faith_proof.pdf "$TMP/proof.txt"
python3 - "$TMP/proof.txt" <<'PY'
import json, pathlib, sys
pages = pathlib.Path(sys.argv[1]).read_text(errors="replace").split("\f")
hit = []
for i, page_txt in enumerate(pages):
    lines = page_txt.splitlines()
    for j, ln in enumerate(lines):
        # iclr2027_conference.sty renders \section* headings with the first
        # letter a size up, which pdftotext reads back with a space after it
        # ("L IMITATIONS") -- collapse spaces and case before comparing so the
        # match survives that rather than needing the exact glyph run.
        if ln.strip().replace(" ", "").upper() == "AIUSESTATEMENT":
            # Two things precede real content on every page and are not real
            # content: "Under review as a conference paper at ICLR 2027" (the
            # running header) and the review-mode line-ruler's own numbers
            # (bare digits, one beside every text line) -- both excluded here,
            # or "at the top of the page" could never be true (the header
            # alone defeats it on every page) and body_last_page would always
            # read one page high, exactly the case this run hit.
            before = [x for x in lines[:j] if x.strip()
                      and "Under review as a conference paper" not in x
                      and not x.strip().isdigit()]
            hit.append((i + 1, not before))
            break
    if hit:
        break
if not hit:
    raise SystemExit("[local] FAILED: the AI use statement heading is not "
                     "in the rendered text, so the body page count cannot "
                     "be measured")
page, at_top = hit[0]
# If the heading starts the page, the body ended on the page before it;
# if it starts partway down, the body ran to that page and the page counts.
last = page - 1 if at_top else page
geop = pathlib.Path("results_v2/canonical_runs/arr_build/geometry.json")
geo = json.loads(geop.read_text())
geo["body_last_page"] = last
geo["ai_use_statement_starts_page"] = page
geo["n_pages"] = sum(1 for p in pages if p.strip())
geop.write_text(json.dumps(geo, indent=2) + "\n")
flag = "" if last <= 9 else "  <-- OVER the ICLR 9-page body limit (10 accepted for fig:collision/fig:dissociation)"
print(f"[local] body (through Conclusion) ends page {last} "
      f"(AI use statement starts page {page}"
      f"{', mid-page' if not at_top else ''}){flag}")
PY

for f in build/cot_faith.pdf build/cot_faith_proof.pdf; do
    printf '[local] %-34s %s pages\n' "$f" "$(pdfinfo "$f" 2>/dev/null \
        | awk '/^Pages:/{print $2}')"
done
echo "[local] ok"
