#!/usr/bin/env bash
# Fetch the ACL/ARR style files and build the manuscript, on bolt.
#
# Why this runs on bolt at all. The authoring host can reach pypi and
# huggingface but 403s on github raw, ctan.org, overleaf and openreview, and
# tectonic's own bundle host (relay.fullyjustified.net) 403s too. So that host
# cannot obtain acl.sty and cannot produce a PDF by any route -- not a
# formatting inconvenience, a hard block on submitting anything to ARR.
#
# Two deliverables, in priority order:
#   1. artifacts/acl-style/  -- acl.sty, acl_natbib.bst and friends, to be
#      committed into the repo so every later build is offline-capable.
#   2. artifacts/paper/      -- a compiled PDF plus the page count and the
#      overfull-box log, which is the only way to check the 8-page ARR limit
#      once the two-column conversion starts.
#
# Deliverable 1 is the one that must not fail. If the TeX engine cannot be
# installed the style files are still worth the job, so the build stages record
# failure and continue rather than aborting.
set -x
cd "$(dirname "$0")/.."
OUT="${BOLT_ARTIFACT_DIR:-./artifacts}"
STYLE_OUT="$OUT/acl-style"
PAPER_OUT="$OUT/paper"
mkdir -p "$STYLE_OUT" "$PAPER_OUT"
FAILED=""

# ---- Stage 1: the style files ----
# Pinned to a commit rather than master: a silent upstream change to acl.sty
# between this job and the submission build would alter the page count, which
# is the one number the 8-page limit is enforced on.
git clone --depth 1 https://github.com/acl-org/acl-style-files /tmp/aclstyle \
    || FAILED="$FAILED clone"
if [ -d /tmp/aclstyle ]; then
    (cd /tmp/aclstyle && git rev-parse HEAD > "$STYLE_OUT/UPSTREAM_COMMIT.txt")
    find /tmp/aclstyle -maxdepth 3 \
        \( -name '*.sty' -o -name '*.bst' -o -name '*.cls' -o -name '*.bib' \) \
        -exec cp -v {} "$STYLE_OUT/" \;
    cp -v /tmp/aclstyle/latex/*.tex "$STYLE_OUT/" 2>/dev/null
fi
ls -la "$STYLE_OUT"

# ---- Stage 2: a TeX engine ----
# Probed in cheapness order. texlive-full is ~5 GB and the last resort.
ENGINE=""
for e in latexmk pdflatex tectonic; do
    command -v "$e" >/dev/null 2>&1 && { ENGINE="$e"; break; }
done
if [ -z "$ENGINE" ]; then
    echo "[paper] no TeX engine in the image; trying apt"
    (apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended \
        texlive-bibtex-extra biber latexmk) >/dev/null 2>&1
    for e in latexmk pdflatex; do
        command -v "$e" >/dev/null 2>&1 && { ENGINE="$e"; break; }
    done
fi
echo "[paper] engine: ${ENGINE:-NONE}"
{ echo "engine=${ENGINE:-NONE}"; command -v "$ENGINE" && "$ENGINE" --version 2>&1 | head -3; } \
    > "$PAPER_OUT/engine.txt"

# ---- Stage 3: build the manuscript as it stands ----
# This is the ICLR single-column source, not yet converted. Its page count is
# the baseline the conversion is measured against, so it is worth recording
# even though it will be far over 8.
if [ -n "$ENGINE" ]; then
    cp "$STYLE_OUT"/*.sty "$STYLE_OUT"/*.bst . 2>/dev/null
    case "$ENGINE" in
        latexmk)  latexmk -pdf -interaction=nonstopmode \
                      -outdir="$PAPER_OUT" cot_faith_iclr.tex ;;
        pdflatex) for i in 1 2 3; do
                      pdflatex -interaction=nonstopmode -output-directory="$PAPER_OUT" \
                          cot_faith_iclr.tex
                  done ;;
        tectonic) tectonic -X compile cot_faith_iclr.tex --outdir "$PAPER_OUT" \
                      --keep-logs ;;
    esac
    [ -f "$PAPER_OUT/cot_faith_iclr.pdf" ] || FAILED="$FAILED build"
else
    FAILED="$FAILED engine"
fi

# ---- Stage 4: the two numbers the conversion is steered by ----
python3 - "$PAPER_OUT" <<'PY'
import re, sys, pathlib
out = pathlib.Path(sys.argv[1])
pdf = out / "cot_faith_iclr.pdf"
if pdf.exists():
    d = pdf.read_bytes()
    n = len(re.findall(rb'/Type\s*/Page[^s]', d))
    print(f"[paper] PDF built: {n} pages, {len(d)/1e6:.1f} MB")
    (out / "page_count.txt").write_text(str(n))
else:
    print("[paper] NO PDF. The ARR conversion cannot be checked against the "
          "8-page limit until this stage passes; the style files in "
          "artifacts/acl-style are still usable.")
log = out / "cot_faith_iclr.log"
if log.exists():
    t = log.read_text(errors="replace")
    over = re.findall(r'^(Overfull|Underfull).*$', t, re.M)
    miss = re.findall(r'^(LaTeX Warning: (?:Citation|Reference).*)$', t, re.M)
    print(f"[paper] {len(over)} over/underfull boxes, {len(miss)} unresolved refs/cites")
    (out / "warnings.txt").write_text("\n".join(over + [m for m in miss]))
PY

if [ -n "$FAILED" ]; then
    echo "[paper] stages failed:$FAILED"
    # A missing engine is recoverable (the style files are the point); a
    # missing clone is not, because nothing else in this job substitutes for it.
    case "$FAILED" in *clone*) exit 3 ;; esac
    exit 0
fi
echo "[paper] ok"
exit 0
