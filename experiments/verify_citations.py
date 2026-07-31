#!/usr/bin/env python3
"""Check every bibliography entry in cot_faith_iclr.tex against a live registry.

Why this exists. A fabricated reference is not a small error in a benchmark
paper -- it is grounds for rejection on its own, and it is the single easiest
mistake for an LLM-assisted workflow to make, because a plausible-looking
BibTeX entry costs nothing to emit and reads exactly like a real one. This
repository's rule is that citations come from a registry rather than from
memory, and this script is what enforces it: it parses the actual bibliography
out of the manuscript, resolves each entry against a registry, and reports per
entry which fields were confirmed and which were not.

What it can and cannot confirm. The authoring environment reaches
export.arxiv.org but not doi.org, api.crossref.org, or dblp.org (all three
return "CONNECT tunnel failed, 403"), so entries with an arXiv id verify
locally while journal-only entries need a networked host. Rather than skip what
it cannot reach, this script records per-source reachability in the report and
labels every field CONFIRMED / MISMATCH / UNVERIFIED. An UNVERIFIED entry is
not a pass: scripts/verify_paper_numbers.py asserts that the count of
unverified entries matches the count the manuscript admits to, so silently
gaining an unverifiable reference fails the audit.

Field comparison is deliberately loose on formatting and strict on substance.
Titles are compared after case-folding and stripping LaTeX braces, accents and
punctuation, because "{CoT-VLA}: Visual Chain-of-Thought..." and the registry's
plain form are the same title. Author lists are compared on surnames only, in
order, because the manuscript abbreviates given names ("M.~Zawalski") and
registries do not -- but surname order is substance: a reordered author list is
a different citation. An "et al." in the manuscript truncates the comparison at
that point rather than counting as a mismatch.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "cot_faith_iclr.tex")

ARXIV_API = "http://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works/"
DBLP_API = "https://dblp.org/search/publ/api"

TIMEOUT = 30


# ---------------------------------------------------------------------------
# text normalisation
# ---------------------------------------------------------------------------

def strip_latex(s: str) -> str:
    """Reduce a LaTeX fragment to comparable plain text.

    Handles the specific constructs this bibliography uses: brace groups for
    capitalisation protection ({LIBERO}), tilde hard spaces, backslash accent
    macros (\\'e), and \\emph. Unicode is folded to ASCII afterwards so that
    "Daum\\'e" and the registry's "Daumé" compare equal.
    """
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)          # remaining macros
    s = s.replace("~", " ").replace("{", "").replace("}", "")
    s = s.replace("\\'", "").replace("\\`", "").replace('\\"', "")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def norm_title(s: str) -> str:
    s = strip_latex(s).casefold()
    # Registries disagree on hyphens, colons and dashes in titles; none of
    # those distinctions can make two different papers look like one.
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def surname(author: str) -> str:
    """Last whitespace-separated token, minus a generational suffix.

    "H.~Daum\\'e~III" -> "daume", "J.~Wortman~Vaughan" -> "vaughan". The latter
    is imperfect for compound surnames, which is why a surname mismatch is
    reported rather than treated as fatal by this script -- the audit reads the
    report, and a human reads a MISMATCH.
    """
    a = strip_latex(author).strip().rstrip(".,")
    toks = [t for t in a.split() if t]
    while toks and toks[-1].casefold().strip(".") in ("jr", "sr", "ii", "iii", "iv"):
        toks.pop()
    return toks[-1].casefold() if toks else ""


# ---------------------------------------------------------------------------
# manuscript parsing
# ---------------------------------------------------------------------------

def parse_bibliography(tex: str) -> list[dict]:
    """Pull (key, year, authors, title, arxiv_id, doi) out of each \\bibitem.

    The manuscript uses a hand-written thebibliography rather than a .bib file,
    so this parses the rendered form. The author/title split is the first
    sentence-ending period that is not part of an initial or an abbreviation --
    in practice, a period followed by a space and a capital, after the closing
    bracket of the \\bibitem label.
    """
    out = []
    for line in tex.splitlines():
        if not line.startswith(r"\bibitem"):
            continue
        m = re.match(r"\\bibitem\[([^\]]*)\]\{(\w+)\}\s*(.*)", line)
        if not m:
            out.append({"key": None, "raw": line, "parse_error":
                        "could not split \\bibitem label from body"})
            continue
        label, key, body = m.groups()
        year = re.search(r"\((\d{4})\)", label)

        # Split author list from title. An explicit "et~al." ends the author
        # list outright, and has to be handled before the general rule: the
        # period in "et~al." is an abbreviation, so the general rule skips it
        # and then cuts at the end of the TITLE instead, swallowing the title
        # into the author list. That is how this parser first read
        # lanham2023's title as its sole author.
        etal = re.search(r"\bet~?al\.\s*", body)
        if etal:
            authors_raw, rest = body[:etal.start()], body[etal.end():]
        else:
            cut = None
            for mm in re.finditer(r"\.\s+", body):
                before = body[:mm.start()]
                # A single capital letter before the period is an initial, not
                # a sentence end. Same for a generational suffix.
                tail = re.split(r"[\s~]", before)[-1]
                if len(tail.rstrip(".")) <= 1:
                    continue
                if tail.rstrip(".").casefold() in ("jr", "sr", "ii", "iii", "iv"):
                    continue
                cut = mm
                break
            if cut is None:
                out.append({"key": key, "raw": line, "parse_error":
                            "could not find the author/title boundary"})
                continue
            authors_raw, rest = body[:cut.start()], body[cut.end():]

        # Title runs to the next period that ends a sentence, by the same rule.
        tm = re.search(r"\.(\s|$)", rest)
        title = rest[:tm.start()] if tm else rest

        authors = [a for a in re.split(r",\s*|\s+and\s+", authors_raw) if a.strip()]
        truncated = bool(etal)
        authors = [a for a in authors
                   if strip_latex(a).split()[:1] != ["et"]]

        aid = re.search(r"arXiv:(\d{4}\.\d{4,5})", body)
        doi = re.search(r"doi:\s*(10\.\S+)", body)
        out.append({
            "key": key, "label": label,
            "year": int(year.group(1)) if year else None,
            "authors": authors, "authors_truncated": truncated,
            "title": title.strip(), "arxiv_id": aid.group(1) if aid else None,
            "doi": doi.group(1) if doi else None,
        })
    return out


# ---------------------------------------------------------------------------
# registries
# ---------------------------------------------------------------------------

def fetch(url: str, accept: str | None = None) -> tuple[str | None, str]:
    """GET, returning (body, error). Never raises: an unreachable registry is a
    fact about the environment to record, not a crash."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "cot-faith-citation-check (mailto:anonymous@example.org)",
        **({"Accept": accept} if accept else {}),
    })
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.read().decode("utf-8", "replace"), ""
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001 - includes the sandbox's 403 tunnel
        return None, f"{type(e).__name__}: {e}"


def arxiv_lookup(arxiv_id: str) -> tuple[dict | None, str]:
    q = f"{ARXIV_API}?id_list={urllib.parse.quote(arxiv_id)}&max_results=1"
    body, err = fetch(q)
    if body is None:
        return None, err
    entries = re.findall(r"<entry>(.*?)</entry>", body, re.S)
    if not entries:
        return None, "no entry in the arXiv response (bad id?)"
    e = entries[0]

    def one(tag):
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", e, re.S)
        return " ".join(m.group(1).split()) if m else None

    return {
        "title": one("title"),
        "authors": re.findall(r"<name>(.*?)</name>", e, re.S),
        "published": one("published"),
        "updated": one("updated"),
        "comment": one("arxiv:journal_ref") or one("arxiv:comment"),
        "doi": one("arxiv:doi"),
    }, ""


def arxiv_title_search(title: str) -> tuple[dict | None, str]:
    """Resolve an entry that prints no arXiv id, by searching arXiv for its title.

    Why this exists. The five venue-only entries (CACM, CVPR, NeurIPS x2, RSS)
    were UNVERIFIED for one reason: this script only queried arXiv when the
    bibitem itself printed an id, and it printed none for them. That made
    verifiability a property of our own formatting rather than of the citation,
    which is backwards -- and it was expensive, because DBLP is unreachable both
    from the authoring network (403 CONNECT tunnel) and from bolt task
    qrpd3f8z58 (SSL handshake timeout), so "fall back to DBLP" resolved nothing
    from anywhere. Most robotics and NLP venue papers have an arXiv preprint,
    and arXiv IS reachable, so searching it by title is the fallback that
    actually works.

    The match is exact on the normalized title, never on rank. arXiv's relevance
    ordering will happily return a related paper for a query that has no true
    hit, and confirming a citation against a different paper is a worse outcome
    than leaving it unverified.

    Two queries, not one. The phrase query is tried first and is the precise
    one, but it fails on any title containing punctuation that normalization
    splits and arXiv's index does not: turpin2023's "Language Models Don't
    Always Say..." normalizes to "... don t always ...", which matches nothing.
    The second query ANDs the title's long words instead, which skips the broken
    short token ("dont") and keeps the discriminative ones. Loosening the query
    is safe precisely because the accept test is unchanged -- a looser query can
    only surface more candidates for the same exact-title check to reject.
    """
    want = norm_title(title)
    long_words = [w for w in want.split() if len(w) >= 6][:6]
    queries = [f"ti:%22{urllib.parse.quote(want)}%22"]
    if long_words:
        queries.append("+AND+".join(f"ti:{urllib.parse.quote(w)}"
                                    for w in long_words))

    errs = []
    for query in queries:
        body, err = fetch(f"{ARXIV_API}?search_query={query}&max_results=20")
        if body is None:
            return None, err
        entries = re.findall(r"<entry>(.*?)</entry>", body, re.S)
        if not entries:
            errs.append("no arXiv hit for this title")
            continue
        for e in entries:
            m = re.search(r"<title[^>]*>(.*?)</title>", e, re.S)
            got = " ".join(m.group(1).split()) if m else ""
            if norm_title(got) != want:
                continue

            def one(tag, blk=e):
                mm = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", blk, re.S)
                return " ".join(mm.group(1).split()) if mm else None

            aid = one("id") or ""
            return {
                "title": got,
                "authors": re.findall(r"<name>(.*?)</name>", e, re.S),
                "published": one("published"),
                "updated": one("updated"),
                "comment": one("arxiv:journal_ref") or one("arxiv:comment"),
                "doi": one("arxiv:doi"),
                # Recorded so a reader can check the match themselves; the entry
                # in the manuscript deliberately still cites the venue, not this.
                "resolved_arxiv_id": aid.rsplit("/", 1)[-1] if aid else None,
            }, ""
        errs.append(f"arXiv returned {len(entries)} hit(s), none with a "
                    f"title-exact match")
    return None, "; ".join(errs)


def crossref_lookup(doi: str) -> tuple[dict | None, str]:
    body, err = fetch(CROSSREF_API + urllib.parse.quote(doi))
    if body is None:
        return None, err
    try:
        msg = json.loads(body)["message"]
    except Exception as e:  # noqa: BLE001
        return None, f"unparseable CrossRef response: {type(e).__name__}: {e}"
    parts = msg.get("published-print", msg.get("published", {})) \
        .get("date-parts", [[None]])
    return {
        "title": (msg.get("title") or [None])[0],
        "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in msg.get("author", [])],
        "year": parts[0][0] if parts and parts[0] else None,
        "container": (msg.get("container-title") or [None])[0],
        "volume": msg.get("volume"), "issue": msg.get("issue"),
        "page": msg.get("page"),
    }, ""


def dblp_lookup(title: str) -> tuple[dict | None, str]:
    q = f"{DBLP_API}?q={urllib.parse.quote(norm_title(title))}&format=json&h=5"
    body, err = fetch(q)
    if body is None:
        return None, err
    try:
        hits = json.loads(body)["result"]["hits"].get("hit", [])
    except Exception as e:  # noqa: BLE001
        return None, f"unparseable DBLP response: {type(e).__name__}: {e}"
    want = norm_title(title)
    for h in hits:
        info = h.get("info", {})
        if norm_title(info.get("title", "")) == want:
            au = info.get("authors", {}).get("author", [])
            au = au if isinstance(au, list) else [au]
            return {
                "title": info.get("title"),
                "authors": [a.get("text", a) if isinstance(a, dict) else a
                            for a in au],
                "year": int(info["year"]) if info.get("year") else None,
                "venue": info.get("venue"), "volume": info.get("volume"),
                "pages": info.get("pages"), "doi": info.get("doi"),
            }, ""
    return None, f"no exact title match among {len(hits)} DBLP hits"


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def compare(entry: dict, reg: dict, reg_name: str) -> dict:
    """Field-by-field verdict for one entry against one registry record."""
    fields = {}
    if reg.get("title"):
        same = norm_title(entry["title"]) == norm_title(reg["title"])
        fields["title"] = {
            "verdict": "CONFIRMED" if same else "MISMATCH",
            "manuscript": entry["title"], "registry": reg["title"],
        }
    if reg.get("authors"):
        ours = [surname(a) for a in entry["authors"]]
        theirs = [surname(a) for a in reg["authors"]]
        # "et al." licenses a prefix comparison; a full list must match in full.
        cmp_theirs = theirs[:len(ours)] if entry["authors_truncated"] else theirs
        same = ours == cmp_theirs
        fields["authors"] = {
            "verdict": "CONFIRMED" if same else "MISMATCH",
            "manuscript": ours, "registry": theirs,
            "compared_as_prefix": entry["authors_truncated"],
        }
    ryear = reg.get("year")
    if ryear is None and reg.get("published"):
        ryear = int(reg["published"][:4])
    if ryear and entry.get("year"):
        # A manuscript year later than the registry's first-posted year is
        # normal (preprint 2018, CACM 2021), so only an EARLIER year is wrong.
        ok = entry["year"] >= ryear
        fields["year"] = {
            "verdict": "CONFIRMED" if ok else "MISMATCH",
            "manuscript": entry["year"], "registry": ryear,
            "note": "manuscript year may postdate the preprint",
        }
    return {"registry": reg_name, "fields": fields, "record": reg}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="citation_check.json")
    p.add_argument("--tex", default=TEX)
    args = p.parse_args()

    tex = open(args.tex).read()
    entries = parse_bibliography(tex)
    print(f"[cite] parsed {len(entries)} \\bibitem entries from {args.tex}")

    reachability: dict[str, str] = {}
    report = {"n_entries": len(entries), "entries": []}

    for e in entries:
        if e.get("parse_error"):
            print(f"[cite] PARSE-ERROR {e.get('key')}: {e['parse_error']}")
            report["entries"].append({**e, "status": "PARSE_ERROR"})
            continue

        results, errors = [], {}
        if e["arxiv_id"]:
            rec, err = arxiv_lookup(e["arxiv_id"])
            reachability["arxiv"] = "reachable" if (rec or "HTTP" in err) \
                else err or "reachable"
            if rec:
                results.append(compare(e, rec, "arxiv"))
            else:
                errors["arxiv"] = err
        if e["doi"]:
            rec, err = crossref_lookup(e["doi"])
            reachability["crossref"] = "reachable" if rec else err
            if rec:
                results.append(compare(e, rec, "crossref"))
            else:
                errors["crossref"] = err
        if not results:
            # Title-search arXiv before DBLP, because it is the registry that
            # answers: DBLP times out from the authoring network and from bolt
            # alike (qrpd3f8z58), so ordering it first only spent 15 timeouts to
            # learn nothing. DBLP is still tried afterwards -- a venue-only
            # entry with no preprint has nowhere else to go.
            rec, err = arxiv_title_search(e["title"])
            if rec:
                reachability["arxiv"] = "reachable"
                results.append(compare(e, rec, "arxiv_title_search"))
            else:
                errors["arxiv_title_search"] = err
        if not results:
            rec, err = dblp_lookup(e["title"])
            reachability["dblp"] = "reachable" if rec else err
            if rec:
                results.append(compare(e, rec, "dblp"))
            else:
                errors["dblp"] = err

        verdicts = [f["verdict"] for r in results for f in r["fields"].values()]
        if not results:
            status = "UNVERIFIED"
        elif "MISMATCH" in verdicts:
            status = "MISMATCH"
        else:
            status = "CONFIRMED"
        print(f"[cite] {status:10s} {e['key']:16s} "
              + (", ".join(f"{k}={v['verdict']}"
                           for r in results for k, v in r["fields"].items())
                 or "; ".join(f"{k}: {v}" for k, v in errors.items())))
        report["entries"].append({**e, "status": status,
                                  "checks": results, "errors": errors})

    counts: dict[str, int] = {}
    for x in report["entries"]:
        counts[x["status"]] = counts.get(x["status"], 0) + 1
    report["status_counts"] = counts
    report["registry_reachability"] = reachability
    report["unverified_keys"] = sorted(x["key"] for x in report["entries"]
                                       if x["status"] == "UNVERIFIED")
    report["mismatch_keys"] = sorted(x["key"] for x in report["entries"]
                                     if x["status"] == "MISMATCH")

    print(f"\n[cite] {counts}")
    print(f"[cite] registries: {reachability}")
    if report["mismatch_keys"]:
        print(f"[cite] MISMATCH on: {report['mismatch_keys']}")
    if report["unverified_keys"]:
        print(f"[cite] unverified (registry unreachable or no match): "
              f"{report['unverified_keys']}")

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[cite] wrote {args.out}")

    # A mismatch is a hard failure: it means the manuscript states something the
    # registry contradicts. An unverified entry is not, because unreachability
    # is a property of the network rather than of the citation -- the audit is
    # what holds the unverified count to the number the manuscript discloses.
    return 1 if report["mismatch_keys"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
