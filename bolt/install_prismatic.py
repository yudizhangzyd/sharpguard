#!/usr/bin/env python3
"""Install whatever `prismatic` OFT's remote code actually needs, and record it.

Both stages of bolt `ga4eejnxnn` failed identically:

    ImportError: This modeling file requires the following packages that were
    not found in your environment: prismatic. Run `pip install prismatic`

That is a missing dependency, not an incompatibility, and limitation (ix) cannot
cite one as the other. So this resolves the dependency instead of concluding
from its absence -- and reports what the resolution cost, because "OFT needs N
extra packages, one of which moves transformers" would itself be a legitimate
reason to scope it out, whereas "unloadable" is not.

Two kinds of source are tried in order:

  1. `pip install prismatic` -- literally what the exception suggests. PyPI has
     an unrelated project under that name, so this is accepted only if it
     actually provides `prismatic.extern.hf.modeling_prismatic`; otherwise it is
     uninstalled again, because leaving it would shadow the real package.
     Measured (bolt `9n7zqdxy9b`): it installs and does NOT provide it --
     `ModuleNotFoundError: No module named 'prismatic.extern'`.
  2. the OFT / OpenVLA source trees, whose package IS `prismatic`. Both install
     cleanly and move no pins.

Every install uses --no-deps: the pinned torch 2.4.1+cu118 is the environment
the paper names, and a transitive upgrade would silently make the load result
describe a different stack. Missing transitive imports are then resolved one at
a time, by reading the module name out of the ImportError and installing it --
each one logged, so the output doubles as the answer to "what would supporting
OFT take". So far that answer is: isodate, draccus, mergedeep, typing_inspect,
mypy_extensions, and then whatever the logging handler needs (see
`target_importable` on why that one was invisible at first).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import re
import subprocess
import sys
import traceback

# What OFT's remote code needs to exist for `check_imports` to pass and for the
# modeling file to actually import.
TARGET = "prismatic.extern.hf.modeling_prismatic"

SOURCES = [
    ("pypi:prismatic", ["prismatic"]),
    ("git:moojink/openvla-oft", ["git+https://github.com/moojink/openvla-oft.git"]),
    ("git:openvla/openvla", ["git+https://github.com/openvla/openvla.git"]),
]

# pip name != import name for a few of the things these trees pull in. Every
# entry here is one the resolver actually hit, or one that `--no-deps` makes
# unavoidable: installing `rich` without its dependencies means the next round
# asks for `markdown_it`, and `pip install markdown_it` does not exist.
IMPORT_TO_PIP = {
    "cv2": "opencv-python-headless",
    "PIL": "Pillow",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "google": "protobuf",
    "markdown_it": "markdown-it-py",
    "dateutil": "python-dateutil",
    "pkg_resources": "setuptools",
    "json_numpy": "json-numpy",
    # Not on PyPI at all -- the ECoT/OpenVLA data pipeline depends on it by git
    # URL. `pip install dlimp` fails outright, so without this entry the
    # resolver stops on the one name whose fix is a URL rather than a rename.
    # This is what stopped the 4.40.1 path in bolt fshsqxp53m.
    "dlimp": "git+https://github.com/kvablack/dlimp.git",
}

MAX_TRANSITIVE = 24


def pip(*args) -> tuple[bool, str]:
    cp = subprocess.run([sys.executable, "-m", "pip", *args],
                        capture_output=True, text=True)
    return cp.returncode == 0, (cp.stdout + cp.stderr)[-2000:]


def pins() -> dict:
    out = {}
    for mod in ("torch", "transformers", "tokenizers", "timm"):
        try:
            out[mod] = importlib.import_module(mod).__version__
        except Exception as e:                              # noqa: BLE001
            out[mod] = f"<absent: {type(e).__name__}>"
    return out


def target_importable() -> tuple[bool, str | None]:
    """Can the OFT modeling module be imported right now?

    `find_spec` is not enough -- it answers "is it on the path", not "does it
    import", and the transitive-dependency failures this script exists to fix
    are all in the second category.

    The failure is returned as the full formatted traceback, not `str(e)`.
    Bolt `9n7zqdxy9b` is why: after prismatic installed, the import died with

        ValueError: Unable to configure handler 'console'

    which names nothing installable. `prismatic.overwatch` configures logging
    through `logging.config.dictConfig`, and dictConfig catches the handler's
    own ImportError and re-raises this instead -- so the module name survives
    only in the `__cause__` chain. Reading just the top-level message made a
    missing `rich` look like a broken package.
    """
    for name in list(sys.modules):
        if name == "prismatic" or name.startswith("prismatic."):
            del sys.modules[name]
    try:
        importlib.invalidate_caches()
        importlib.import_module(TARGET)
        return True, None
    except Exception:                                       # noqa: BLE001
        return False, traceback.format_exc()


def missing_module(err: str) -> str | None:
    """The top-level module name an ImportError is complaining about.

    Searched over the whole chained traceback, so a name buried under a
    dictConfig ValueError is still found (see `target_importable`).
    """
    for pat in (r"No module named '([^']+)'",
                r"not found in your environment: ([A-Za-z0-9_.]+)",
                r"cannot import name '[^']+' from '([^']+)'"):
        m = re.search(pat, err or "")
        if m:
            return m.group(1).split(".")[0]
    return None


def version_skew(err: str) -> tuple[str, str] | None:
    """The (pip_name, required_version) a version-skew error is complaining about.

    `--no-deps` is the right default here -- the pins ARE the environment the
    paper names -- but it has a failure mode this repo just paid for. Installing
    `pydantic_core` alone to satisfy a missing-module error pulled 2.47.0 next to
    the image's pydantic, which requires 2.46.4, and every OFT checkpoint then
    died at the processor stage on:

        SystemError: The installed pydantic-core version (2.47.0) is
        incompatible with the current pydantic version, which requires 2.46.4.

    Measured in bolt `gvvhgg4d4c`, all 5 checkpoints, both transformers pins.
    Not a missing module, so `missing_module` returns None and the resolver
    stopped and reported it as the blocker -- when the blocker was the resolver's
    own install. The exception prints the version it wants, so read that rather
    than guessing a pin: same principle as reading the module name out of an
    ImportError instead of maintaining a dependency list by hand.
    """
    m = re.search(r"installed (\S+?) version \(\S+?\) is incompatible with the "
                  r"current \S+ version, which requires (\d\S*)", err or "")
    # The version is matched greedily and then stripped: a non-greedy match stops
    # at the first dot INSIDE the version and yields "2" for "2.46.4".
    return (m.group(1), m.group(2).rstrip(".,;")) if m else None


def resolve() -> dict:
    log: dict = {"target": TARGET, "pins_before": pins(), "attempts": [],
                 "transitive_installs": [], "resolved_by": None,
                 "importable": False}

    ok, err = target_importable()
    if ok:
        log.update(importable=True, resolved_by="already_present")
        log["pins_after"] = pins()
        return log

    for label, spec in SOURCES:
        entry = {"source": label, "spec": spec}
        installed, tail = pip("install", "--quiet", "--no-deps", *spec)
        entry["pip_ok"] = installed
        if not installed:
            entry["pip_tail"] = tail[-600:]
            log["attempts"].append(entry)
            continue

        ok, err = target_importable()
        # Resolve transitive imports one at a time. Each is a real fact about
        # what OFT needs; guessing a list up front would hide which ones.
        seen: set[str] = set()
        for _ in range(MAX_TRANSITIVE):
            if ok:
                break
            mod = missing_module(err or "")
            if not mod or mod.startswith("prismatic"):
                break
            if mod in seen:
                # The pip install reported success and the import still asks
                # for the same module, so installing it again cannot help.
                # Without this the loop burns every remaining iteration on one
                # name and the log reads as if 24 dependencies were needed.
                entry["stuck_on"] = mod
                break
            seen.add(mod)
            pkg = IMPORT_TO_PIP.get(mod, mod)
            got, _tail = pip("install", "--quiet", "--no-deps", pkg)
            log["transitive_installs"].append(
                {"import": mod, "pip": pkg, "ok": got})
            if not got:
                break
            ok, err = target_importable()

        entry["importable"] = ok
        entry["error"] = None if ok else (err or "")[-3000:]
        log["attempts"].append(entry)
        if ok:
            log.update(importable=True, resolved_by=label)
            break
        # A source that did not work must not be left shadowing the next one.
        pip("uninstall", "-y", "-q", "prismatic")

    log["pins_after"] = pins()
    log["pins_moved"] = {k: [v, log["pins_after"].get(k)]
                         for k, v in log["pins_before"].items()
                         if log["pins_after"].get(k) != v}
    return log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="where to write the resolution log (JSON)")
    args = ap.parse_args()

    log = resolve()
    with open(args.out, "w") as f:
        json.dump(log, f, indent=2)

    print(f"[prismatic] importable={log['importable']} "
          f"resolved_by={log['resolved_by']}")
    for a in log["attempts"]:
        # The TAIL of the traceback, not the head: the head is our own import
        # frames and the exception is on the last line.
        tail = str(a.get("error") or "").strip().splitlines()[-1:] or [""]
        print(f"[prismatic]   {a['source']}: pip_ok={a.get('pip_ok')} "
              f"importable={a.get('importable')} {tail[0][:200]}")
        if a.get("stuck_on"):
            print(f"[prismatic]     stuck: installing {a['stuck_on']} did not "
                  f"satisfy the import that asked for it")
    if log["transitive_installs"]:
        print("[prismatic]   transitive: " + ", ".join(
            f"{t['import']}({'ok' if t['ok'] else 'FAILED'})"
            for t in log["transitive_installs"]))
    if log["pins_moved"]:
        # Loud, because it invalidates the comparison the stages exist to make.
        print(f"[prismatic] WARNING pins moved: {log['pins_moved']}")
    # Exit 0 either way: an unresolvable dependency is a measurement, and the
    # caller decides what to do with it. Only a crash here is a failure.
    return 0


if __name__ == "__main__":
    sys.exit(main())
