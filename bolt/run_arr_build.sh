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

log = out / "cot_faith_arr.log"
if log.exists():
    t = log.read_text(errors="replace")
    over = re.findall(r'^Overfull.*$', t, re.M)
    miss = re.findall(r'^(LaTeX Warning: (?:Citation|Reference).*)$', t, re.M)
    err = re.findall(r'^! .*$', t, re.M)
    print(f"[arr] {len(err)} errors, {len(over)} overfull boxes, "
          f"{len(miss)} unresolved refs/cites")
    for e in err[:20]:
        print(f"[arr]   {e}")
    for m in miss[:20]:
        print(f"[arr]   {m}")
    (out / "warnings.txt").write_text("\n".join(err + over + miss))
PY

cp -v cot_faith_arr.tex arr_appendix.tex arr_bib.tex "$OUT/" 2>/dev/null

if [ -n "$FAILED" ]; then
    echo "[arr] stages failed:$FAILED"
    exit 1
fi
echo "[arr] ok"
exit 0
