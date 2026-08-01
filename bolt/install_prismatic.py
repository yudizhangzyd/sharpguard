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

Two sources are tried in order:

  1. `pip install prismatic` -- literally what the exception suggests. PyPI has
     an unrelated project under that name, so this is accepted only if it
     actually provides `prismatic.extern.hf.modeling_prismatic`; otherwise it is
     uninstalled again, because leaving it would shadow the real package.
  2. the OFT / OpenVLA source trees, whose package IS `prismatic`.

Every install uses --no-deps: the pinned torch 2.4.1+cu118 is the environment
the paper names, and a transitive upgrade would silently make the load result
describe a different stack. Missing transitive imports are then resolved one at
a time, by reading the module name out of the ImportError and installing it --
each one logged, so the output doubles as the answer to "what would supporting
OFT take".
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import re
import subprocess
import sys

# What OFT's remote code needs to exist for `check_imports` to pass and for the
# modeling file to actually import.
TARGET = "prismatic.extern.hf.modeling_prismatic"

SOURCES = [
    ("pypi:prismatic", ["prismatic"]),
    ("git:moojink/openvla-oft", ["git+https://github.com/moojink/openvla-oft.git"]),
    ("git:openvla/openvla", ["git+https://github.com/openvla/openvla.git"]),
]

# pip name != import name for a few of the things these trees pull in.
IMPORT_TO_PIP = {
    "cv2": "opencv-python-headless",
    "PIL": "Pillow",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "google": "protobuf",
}

MAX_TRANSITIVE = 12


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
    """
    for name in list(sys.modules):
        if name == "prismatic" or name.startswith("prismatic."):
            del sys.modules[name]
    try:
        importlib.invalidate_caches()
        importlib.import_module(TARGET)
        return True, None
    except Exception as e:                                  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def missing_module(err: str) -> str | None:
    """The top-level module name an ImportError is complaining about."""
    for pat in (r"No module named '([^']+)'",
                r"cannot import name '[^']+' from '([^']+)'"):
        m = re.search(pat, err or "")
        if m:
            return m.group(1).split(".")[0]
    return None


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
        for _ in range(MAX_TRANSITIVE):
            if ok:
                break
            mod = missing_module(err or "")
            if not mod or mod.startswith("prismatic"):
                break
            pkg = IMPORT_TO_PIP.get(mod, mod)
            got, _tail = pip("install", "--quiet", "--no-deps", pkg)
            log["transitive_installs"].append(
                {"import": mod, "pip": pkg, "ok": got})
            if not got:
                break
            ok, err = target_importable()

        entry["importable"] = ok
        entry["error"] = None if ok else err
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
        print(f"[prismatic]   {a['source']}: pip_ok={a.get('pip_ok')} "
              f"importable={a.get('importable')} {str(a.get('error'))[:140]}")
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
