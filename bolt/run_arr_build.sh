#!/usr/bin/env bash
# Build the ARR two-column submission and measure it against the 8-page limit.
#
# Distinct from run_paper_build.sh, which fetched acl.sty and compiled the
# single-column source for a baseline. Those style files are now committed
# under acl-style/, so this job needs no network for LaTeX -- only a TeX
# engine, which the image may or may not carry.
#
# The number this job exists to produce is the CONTENT page count: pages up to
# and including the last numbered section, excluding Limitations, Ethics,
# References and Appendix, which ARR does not count. A build that reports total
# pages would report ~40 and say nothing about whether the submission is legal.
# So the body is compiled a second time on its own, with the appendix and
# bibliography suppressed, and that build's page count is the one that matters.
set -x
cd "$(dirname "$0")/.."
OUT="${BOLT_ARTIFACT_DIR:-./artifacts}/arr"
mkdir -p "$OUT"
FAILED=""

# ---- Stage 1: a TeX engine ----
ENGINE=""
for e in latexmk pdflatex; do
    command -v "$e" >/dev/null 2>&1 && { ENGINE="$e"; break; }
done
if [ -z "$ENGINE" ]; then
    (apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended \
        texlive-bibtex-extra biber latexmk) >/dev/null 2>&1
    for e in latexmk pdflatex; do
        command -v "$e" >/dev/null 2>&1 && { ENGINE="$e"; break; }
    done
fi
echo "[arr] engine: ${ENGINE:-NONE}"
[ -z "$ENGINE" ] && { echo "[arr] no TeX engine"; exit 3; }

# acl.sty and acl_natbib.bst live in acl-style/ and are found via TEXINPUTS
# rather than copied, so the repo keeps one copy and the build cannot drift
# from it.
export TEXINPUTS=".:./acl-style:${TEXINPUTS}"
export BSTINPUTS=".:./acl-style:${BSTINPUTS}"

# ---- Stage 2a: preflight every \usepackage against the installed tree ----
# The engine check above passes on a partial TeX install, and a missing .sty
# then surfaces 100 lines into the log as a generic "Fatal error occurred".
# That is how the inconsolata failure cost a whole job. Resolve the preamble
# first and name what is missing on one line, before anything compiles.
MISSING=""
for pkg in $(grep -o '\\usepackage\(\[[^]]*\]\)\?{[^}]*}' cot_faith_arr.tex \
             | sed 's/.*{//; s/}//' | tr ',' '\n' | tr -d ' '); do
    kpsewhich "${pkg}.sty" >/dev/null 2>&1 || MISSING="$MISSING $pkg"
done
if [ -n "$MISSING" ]; then
    echo "[arr] MISSING PACKAGES:$MISSING"
    FAILED="$FAILED preflight"
else
    echo "[arr] preflight: every \\usepackage resolves"
fi

# ---- Stage 2: regenerate the appendix, and fail if it was hand-edited ----
# arr_appendix.tex is generated from the full-length source. Building a stale
# copy would submit numbers that no longer match the manuscript, which is the
# exact failure the audit script exists to prevent, so it is checked here too.
python3 scripts/build_arr_appendix.py || FAILED="$FAILED appendix"
python3 scripts/build_arr_appendix.py --check || FAILED="$FAILED appendix-stale"

# ---- Stage 3: the full submission PDF ----
latexmk -pdf -interaction=nonstopmode -outdir="$OUT" cot_faith_arr.tex
[ -f "$OUT/cot_faith_arr.pdf" ] || FAILED="$FAILED build"

# ---- Stage 4: the body alone, which is what the 8-page limit applies to ----
# \includeonly will not do this (the appendix is \input, not \include), and
# commenting it out by hand would drift. Generate a body-only driver instead:
# same preamble, same body, appendix and bibliography stubbed out.
python3 - <<'PY'
import re, pathlib
src = pathlib.Path("cot_faith_arr.tex").read_text()
# Cut everything from \bibliographystyle onward -- that is References,
# Appendix and nothing else. Limitations and Ethics come BEFORE it and are
# also uncounted, so they are dropped separately, by their starred headers.
body = src[:src.index(r"\bibliographystyle")] + "\n\\end{document}\n"
for header in (r"\section*{Limitations}", r"\section*{Ethics Statement}"):
    i = body.find(header)
    if i < 0:
        raise SystemExit(f"[arr] {header} not found; the page count would be "
                         f"wrong in the direction that gets a desk reject")
    j = body.find(r"\section*", i + len(header))
    k = body.find(r"\end{document}", i)
    body = body[:i] + body[(j if 0 < j < k else k):]
pathlib.Path("_arr_bodyonly.tex").write_text(body)
print("[arr] wrote _arr_bodyonly.tex")
PY
latexmk -pdf -interaction=nonstopmode -outdir="$OUT" _arr_bodyonly.tex \
    || FAILED="$FAILED bodyonly"

# ---- Stage 5: the numbers ----
python3 - "$OUT" <<'PY'
import re, sys, zlib, pathlib
out = pathlib.Path(sys.argv[1])


def pages(pdf: pathlib.Path, log: pathlib.Path):
    """Page count, read from the engine log with a PDF scan as the check.

    The engine's own "(N pages, M bytes)" line is authoritative and cheap. The
    byte scan is kept as a cross-check because a silently wrong page count is
    how the previous version of this measurement shipped a 0 for a 35-page
    PDF: /Type /Page lives inside Flate-compressed object streams and never
    appears as plaintext, so the naive regex returns nothing.
    """
    n_log = None
    if log.exists():
        m = re.search(r'\((\d+) pages?, \d+ bytes\)',
                      log.read_text(errors="replace"))
        n_log = int(m.group(1)) if m else None
    n_scan = None
    if pdf.exists():
        d = pdf.read_bytes()
        n = len(re.findall(rb'/Type\s*/Page[^s]', d))
        for m in re.finditer(rb'stream\r?\n', d):
            s0, e = m.end(), d.find(b'endstream', m.end())
            if e < 0:
                continue
            try:
                n += len(re.findall(rb'/Type\s*/Page[^s]',
                                    zlib.decompress(d[s0:e])))
            except Exception:
                pass
        n_scan = n
    if n_log is not None and n_scan is not None and n_log != n_scan:
        print(f"[arr] WARNING: log says {n_log} pages, scan says {n_scan}")
    return n_log if n_log is not None else n_scan


full = pages(out / "cot_faith_arr.pdf", out / "cot_faith_arr.log")
body = pages(out / "_arr_bodyonly.pdf", out / "_arr_bodyonly.log")
print(f"[arr] full submission: {full} pages")
print(f"[arr] CONTENT pages:   {body}  (ARR limit: 8)")
if body is not None:
    verdict = "WITHIN LIMIT" if body <= 8 else f"OVER BY {body - 8}"
    print(f"[arr] {verdict}")
    (out / "page_count.txt").write_text(
        f"content={body}\nfull={full}\nlimit=8\nverdict={verdict}\n")

# ---- Where each body float actually PRINTED --------------------------------
# The page count above is computed on the body-only build, and that build ends
# with \end{document} right after Ethics, which flushes every pending float.
# The real submission has 40 pages of appendix after that point, so a float
# still queued at the end of the body drifts into the appendix instead: all
# four body figures printed on pages 41-44 of the 49-page PDF -- present,
# numbered, cross-referenced, and thirty-five pages from the text arguing from
# them -- while page_count.txt said WITHIN LIMIT. A page count cannot see that.
#
# The check reads .aux, not the PDF: \newlabel already records the page each
# label resolved to, so this needs no pdftotext (which the image may not have)
# and no text scraping. Which labels belong to the body is read from the body
# source, so adding a float to the paper extends the check automatically.
body_labels = set(re.findall(r'\\label\{((?:fig|tab):[^}]+)\}',
                            pathlib.Path("cot_faith_arr.tex").read_text()))
aux = out / "cot_faith_arr.aux"
placed, stray = {}, []
if aux.exists() and body_labels:
    limit = (body if body else 8) + 1     # a float pushed to the next page is fine
    for lab, num, pg in re.findall(
            r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}', aux.read_text()):
        if lab in body_labels:
            placed[lab] = int(pg)
            if int(pg) > limit:
                stray.append(f"{lab} (no. {num}) on page {pg}, body ends at {body}")
    missing = sorted(body_labels - set(placed))
    if missing:
        # A body label absent from .aux never resolved -- the \ref would print
        # "??", which is the same defect class and equally invisible upstream.
        stray += [f"{m} never resolved" for m in missing]
    print(f"[arr] float placement: {len(placed)}/{len(body_labels)} body floats "
          f"resolved, limit page {limit}")
    for lab in sorted(placed, key=placed.get):
        print(f"[arr]   {lab}: page {placed[lab]}")
    (out / "float_placement.txt").write_text(
        "\n".join(f"{k}\t{v}" for k, v in sorted(placed.items(),
                                                  key=lambda kv: kv[1]))
        + ("\nSTRAY:\n" + "\n".join(stray) if stray else "\n"))
    if stray:
        print("[arr] FAIL: body float(s) printed outside the body:")
        for sline in stray:
            print(f"[arr]   {sline}")
        # A Python assignment cannot reach the enclosing shell. Leave a
        # sentinel file and let the shell fail on it, or this check prints a
        # failure and the job exits 0 -- which is precisely the kind of
        # silently-passing metric it was written to replace.
        (out / "FLOATS_STRAY").write_text("\n".join(stray) + "\n")
else:
    print(f"[arr] float placement: NOT CHECKED "
          f"(aux={aux.exists()}, body labels={len(body_labels)})")

log = out / "cot_faith_arr.log"
if log.exists():
    t = log.read_text(errors="replace")
    over = re.findall(r'^Overfull.*$', t, re.M)
    miss = re.findall(r'^(LaTeX Warning: (?:Citation|Reference).*)$', t, re.M)
    err = re.findall(r'^! .*$', t, re.M)
    print(f"[arr] {len(err)} errors, {len(over)} overfull boxes, "
          f"{len(miss)} unresolved refs/cites")
    # A count alone hid a 347pt box behind "37 overfull boxes" -- 4.8in of text
    # running off a 3.1in column, i.e. content the reader cannot see. Report
    # the worst offenders by size, since severity is what decides if it
    # matters: a 2pt box is invisible and a 100pt box is a defect.
    sized = sorted(((float(m.group(1)), m.group(0)) for m in
                    re.finditer(r'^Overfull \\hbox \((\d+\.?\d*)pt too wide.*$',
                                t, re.M)), reverse=True)
    if sized:
        print(f"[arr] worst overfull: {sized[0][0]:.0f}pt "
              f"({sum(1 for s, _ in sized if s > 20)} over 20pt)")
    for _, line in sized[:5]:
        print(f"[arr]   {line}")
    for e in err[:20]:
        print(f"[arr]   {e}")
    for m in miss[:20]:
        print(f"[arr]   {m}")
    (out / "warnings.txt").write_text("\n".join(err + over + miss))
    # Fail on the ones a reader would see. 20pt is a tenth of the 218pt column
    # -- text hanging into the gutter or off the page edge, not a hyphenation
    # near-miss. Everything below that is reported above and left alone,
    # because a threshold low enough to catch 1pt boxes would fire on every
    # build and stop being read.
    bad = [(s, line) for s, line in sized if s > 20]
    if bad:
        print(f"[arr] FAIL: {len(bad)} overfull box(es) over 20pt")
        (out / "OVERFULL_BAD").write_text(
            "\n".join(f"{s:.2f}pt\t{line}" for s, line in bad) + "\n")
PY

[ -f "$OUT/FLOATS_STRAY" ] && FAILED="$FAILED float-placement"
[ -f "$OUT/OVERFULL_BAD" ] && FAILED="$FAILED overfull"

cp -v cot_faith_arr.tex arr_appendix.tex arr_bib.tex "$OUT/" 2>/dev/null

if [ -n "$FAILED" ]; then
    echo "[arr] stages failed:$FAILED"
    exit 1
fi
echo "[arr] ok"
exit 0
