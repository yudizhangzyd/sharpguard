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

The prismatic tree itself is installed with --no-deps, and only it: its setup
pins a torch of its own, and letting the package under test re-pin the stack
would make the load result describe an environment the paper does not name.

Its ordinary PyPI dependencies are installed *with* their own dependencies --
see `install`. Round 4 (bolt `s6arkzytjr`) is why. Suppressing deps for every
package produced two different broken partial installs in one job: a
pydantic/pydantic-core version skew, and

    AttributeError: partially initialized module 'wandb' has no attribute
    'errors' (most likely due to a circular import)

which is what `wandb` does when its own requirements are absent. Both were
self-inflicted, and neither is evidence about OFT. So deps are resolved
normally and the four pins the paper names are re-asserted afterwards, which
protects the stack without breaking the packages -- and any pin that moved is
reported rather than quietly restored.

Missing transitive imports are resolved one at a time, by reading the module
name out of the ImportError and installing it -- each one logged, so the output
doubles as the answer to "what would supporting OFT take". So far that answer
is: isodate, draccus, mergedeep, typing_inspect, mypy_extensions, and then
whatever the logging handler needs (see `target_importable` on why that one was
invisible at first).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
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

# A pip constraints file, applied to EVERY install this module runs. Set by
# `--write-constraints`, read from the environment so the runner's own shell pip
# calls can pass the same file with `-c`.
#
# Round 5 (bolt `8ejjyfkzq8`) is why this exists. Two things moved that nobody
# asked to move:
#
#   * torch went from 2.4.1+cu118 to 2.2.0+cu121 between stages 1 and 3 -- the
#     runner's own pin-restore resolved torch from PyPI. Stages 3 and 4 were
#     therefore not measuring the stack the paper names, and nothing said so.
#   * protobuf oscillated 4.x <-> 5.x for four rounds, because `tensorflow`
#     requires <5, recent `wandb` requires >=5, and `tensorflow_metadata`'s
#     generated stubs import `google.protobuf.runtime_version`, which only
#     exists in 5.x. Whichever generation won, one of the three broke: the job
#     ended on protobuf 4.x and `cannot import name 'Imports' from
#     wandb.proto.wandb_telemetry_pb2`.
#
# A constraints file is the right shape for both: it is a statement about the
# whole environment rather than a flag on one install, so a resolution that would
# violate it fails and gets recorded instead of silently winning.
CONSTRAINTS_ENV = "SG_PIP_CONSTRAINTS"

# The coherent protobuf-4 world, chosen rather than discovered: the image ships
# tensorflow (needs <5) and the paper's rollout numbers depend on it, so the two
# packages that can move are the ones that get pinned back.
CONSTRAINT_LINES = [
    "protobuf<5",
    # Recent wandb ships stubs generated against protobuf 5. wandb is only here
    # because prismatic's logging imports it, so it is the cheapest thing to
    # move: <0.18 is the last line whose stubs work on protobuf 4.
    "wandb<0.18",
    # Same story one layer down: tensorflow_metadata >=1.15 calls
    # google.protobuf.runtime_version, added in protobuf 5.
    "tensorflow-metadata<1.15",
]


def constraints_path() -> str | None:
    """The constraints file to apply, if one has been written."""
    p = os.environ.get(CONSTRAINTS_ENV)
    return p if p and os.path.exists(p) else None


def write_constraints(path: str, freeze: tuple = ("torch",)) -> dict:
    """Freeze part of the paper's stack into a pip constraints file.

    `freeze` names the pins that must not move *at all*, and it defaults to torch
    alone. torch is the one no stage varies and the one PyPI cannot put back: the
    image ships `2.4.1+cu118`, a local build PyPI does not serve, so pinning that
    exact string turns round 5's silent downgrade (2.4.1+cu118 -> 2.2.0+cu121, via
    the runner's own pin-restore) into a resolution failure that gets recorded.

    transformers/tokenizers/timm are deliberately NOT frozen by default: the 2x2
    moves them on purpose, and a constraint on a variable of the experiment would
    make the stage that varies it fail rather than measure. They are held by the
    explicit `pip install` each stage runs, and any drift from that is caught by
    `reassert_pins`.
    """
    lines = list(CONSTRAINT_LINES)
    now = pins()
    for name in freeze:
        v = now.get(name, "")
        if v and not str(v).startswith("<absent"):
            lines.append(f"{name}=={v}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return {"path": path, "lines": lines, "frozen": list(freeze),
            "pins_at_write": now}


def pip(*args) -> tuple[bool, str]:
    # Every install carries the constraints file when one exists. Applied here
    # rather than at each call site so a new call site cannot forget it -- that
    # is exactly how round 5's pin-restore moved torch.
    if args and args[0] == "install" and constraints_path():
        args = (args[0], "-c", constraints_path(), *args[1:])
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


# Which of the four the paper names can be RESTORED by a plain PyPI install if a
# dependency resolution moves it. torch is deliberately absent: the image's build
# carries a local version suffix (2.4.1+cu118) that PyPI does not serve, so a
# "restore" would install a different build than the paper names -- worse than
# reporting the move. If torch moves, the stage is not comparable and says so.
REPINNABLE = ("transformers", "tokenizers", "timm")


def reassert_pins(before: dict) -> dict:
    """Put back any of the paper's pins that an install moved, and report both.

    Restoring silently would be the wrong shape: "we installed OFT's deps and
    nothing moved" and "we installed them, transformers jumped two minor
    versions, and we forced it back" are different facts about what supporting
    OFT costs, and only the first one lets a stage's load result be read as being
    about the pinned stack. So `moved` is kept even when `restored` succeeds.
    """
    after = pins()
    moved = {k: [v, after.get(k)] for k, v in before.items()
             if after.get(k) != v and not str(v).startswith("<absent")}
    rec: dict = {"moved": moved, "restored": {}, "unrestorable": []}
    for name, (want, saw) in moved.items():
        if name not in REPINNABLE:
            rec["unrestorable"].append({"pin": name, "want": want, "saw": saw})
            continue
        ok, tail = pip("install", "--quiet", "--no-deps", f"{name}=={want}")
        rec["restored"][name] = {"want": want, "saw": saw, "repin_ok": ok,
                                 "pip_tail": None if ok else tail[-300:]}
    return rec


def install(pkg: str, no_deps: bool = False) -> dict:
    """Install one dependency, repair what it broke, then re-assert the pins.

    `no_deps=True` is for the prismatic tree itself -- the package under test
    must not be allowed to re-pin the stack it is being tested against. For
    everything else deps are resolved normally, because round 4 showed what
    suppressing them costs: `wandb` installed without its requirements raised

        AttributeError: partially initialized module 'wandb' has no attribute
        'errors' (most likely due to a circular import)

    and a lone `pydantic_core` landed a version its sibling `pydantic` rejects.
    Both look exactly like an OFT incompatibility in a log, and neither is one.

    A with-deps install can still fail where --no-deps would have worked -- pip
    refuses the resolution rather than moving the installed torch. That is the
    only case that falls back, and the fallback is recorded, because a partial
    install is then a known risk for that package rather than the default.
    """
    before = pins()
    ok, tail = pip("install", "--quiet", *(("--no-deps",) if no_deps else ()),
                   pkg)
    rec: dict = {"pip": pkg, "ok": ok, "no_deps": no_deps}
    if not ok and not no_deps:
        ok, tail = pip("install", "--quiet", "--no-deps", pkg)
        rec.update(ok=ok, no_deps=True, fallback="no_deps_after_resolution_failed")
    if not ok:
        rec["pip_tail"] = tail[-400:]
        return rec
    rec["pins"] = reassert_pins(before)
    rec["consistency"] = pip_check_repair()
    return rec


# `pip check` says exactly two things, and each needs a different repair.
_RE_MISSING = re.compile(
    r"^(\S+) (\S+) requires ([A-Za-z0-9_.\-]+), which is not installed", re.M)
_RE_CONFLICT = re.compile(
    r"^(\S+) (\S+) has requirement (\S+), but you have (\S+) (\S+)", re.M)

# Never repaired automatically. These four ARE the environment the paper names,
# so a dependency that wants a different one is a fact to report, not a number
# to change underneath the measurement.
PROTECTED = {"torch", "transformers", "tokenizers", "timm"}

_NORM = re.compile(r"[-_.]+")


def _norm_dist(name: str) -> str:
    return _NORM.sub("-", name.strip().lower())


def _req_name(requirement: str) -> str:
    """`pydantic-core==2.46.4` -> `pydantic-core`; `numpy<2` -> `numpy`."""
    return re.split(r"[<>=!~\[;]", requirement, 1)[0].strip()


def parse_pip_check(text: str) -> list[dict]:
    """The unsatisfied requirements `pip check` found, as repair instructions.

    Split by kind because the fix differs: a missing requirement is installed by
    name and pip picks a compatible version, whereas a conflict must be installed
    to the exact specifier `pip check` printed -- installing `pydantic-core` by
    name is what produced the conflict in the first place.
    """
    found = []
    for holder, hver, dep in _RE_MISSING.findall(text or ""):
        found.append({"kind": "missing", "holder": holder,
                      "holder_version": hver, "requirement": dep, "name": dep})
    for holder, hver, req, have, hav in _RE_CONFLICT.findall(text or ""):
        found.append({"kind": "conflict", "holder": holder,
                      "holder_version": hver, "requirement": req,
                      "name": _req_name(req),
                      # pip check ends the line with a full stop, and the
                      # version's own `\S+` happily takes it for a version.
                      "installed": f"{have} {hav.rstrip('.')}"})
    return found


def pip_check_repair(max_rounds: int = 4) -> dict:
    """Make the installed set self-consistent, and report what it took.

    This is the fix for round 4 (bolt `s6arkzytjr`), whose two failures were both
    a broken install set rather than anything about OFT:

        SystemError: The installed pydantic-core version (2.47.0) is
        incompatible with the current pydantic version, which requires 2.46.4
        AttributeError: partially initialized module 'wandb' has no attribute
        'errors' (most likely due to a circular import)

    `pip check` names both -- as a conflict and as a missing requirement -- and
    it names them BEFORE anything imports them, which is the only point at which
    the first one is fixable: `pydantic_core` is a compiled extension, so once it
    is imported no on-disk reinstall and no `invalidate_caches()` can swap the
    loaded object. Round 4's in-process repair ran, succeeded, and changed
    nothing for exactly that reason.

    Bounded, and bounded against the right thing. The first version stopped when
    a round produced the SAME finding set twice in a row, which round 5 (bolt
    `8ejjyfkzq8`) showed is not enough: `tensorflow` requires protobuf <5, recent
    `wandb` requires >=5, so each round "fixed" the other one's complaint and the
    set ALTERNATED rather than repeated. Four rounds burned, ended unclean, and
    the last install decided which package was broken -- protobuf 4.x won and
    stage 3 died on `cannot import name 'Imports' from
    wandb.proto.wandb_telemetry_pb2`. So two guards now:

      * every finding set seen so far is remembered, and a repeat of ANY of them
        is an oscillation, not progress;
      * a dist that two holders want at incompatible versions in the SAME round
        is `unsatisfiable` and is not touched at all. Installing either side is
        picking a loser silently, and which side loses should be a decision in
        the constraints file, not a side effect of iteration order.

    Repairs that would touch the paper's pins are refused, not applied.

    `max_rounds=0` is report-only, and stage 1 uses it. That stage exists to
    measure the environment the paper names, so it must record whether the image
    arrived consistent without changing it -- otherwise "our installs broke this"
    and "it was already broken" become indistinguishable, and the baseline stops
    being the thing that keeps limitation (ix) falsifiable.
    """
    rec: dict = {"rounds": [], "refused": [], "unsatisfiable": [],
                 "oscillated": False, "clean": None}
    seen_sets: list[frozenset] = []
    protected = {_norm_dist(p) for p in PROTECTED}
    for _ in range(max_rounds):
        _ok, text = pip("check")
        found = parse_pip_check(text)
        if not found:
            rec["clean"] = True
            break
        key = frozenset((f["kind"], f["holder"], f["requirement"]) for f in found)
        if key in seen_sets:            # repeating, in any order: not progress
            rec["oscillated"] = True
            break
        seen_sets.append(key)

        # Group by target dist first: the conflicts worth refusing are the ones
        # where two holders disagree, and that is invisible finding-by-finding.
        by_dist: dict[str, list] = {}
        for f in found:
            by_dist.setdefault(_norm_dist(f["name"]), []).append(f)
        contested = {d for d, fs in by_dist.items()
                     if len({f["requirement"] for f in fs}) > 1}
        for d in sorted(contested):
            rec["unsatisfiable"].append(
                {"dist": d, "wanted_by": [{"holder": f["holder"],
                                           "requirement": f["requirement"]}
                                          for f in by_dist[d]]})

        rnd: dict = {"findings": found, "repairs": []}
        for f in found:
            dist = _norm_dist(f["name"])
            if dist in protected:
                rec["refused"].append(f)
                continue
            if dist in contested:
                continue                # reported above; picking a side is not ours
            spec = f["requirement"] if f["kind"] == "conflict" else f["name"]
            got, tail = pip("install", "--quiet", spec)
            rnd["repairs"].append({"spec": spec, "ok": got,
                                   "pip_tail": None if got else tail[-300:]})
        rec["rounds"].append(rnd)
        if not rnd["repairs"]:          # everything left is refused or contested
            break
    if rec["clean"] is None:
        _ok, text = pip("check")
        remaining = parse_pip_check(text)
        rec["clean"] = not remaining
        rec["remaining"] = remaining
    rec["constraints"] = constraints_path()
    return rec


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

    Installing `pydantic_core` alone to satisfy a missing-module error pulled
    2.47.0 next to the image's pydantic, which requires 2.46.4, and every OFT
    checkpoint then died at the processor stage on:

        SystemError: The installed pydantic-core version (2.47.0) is
        incompatible with the current pydantic version, which requires 2.46.4.

    Measured in bolt `gvvhgg4d4c`, all 5 checkpoints, both transformers pins.
    Not a missing module, so `missing_module` returns None and the resolver
    stopped and reported it as the blocker -- when the blocker was the resolver's
    own install. The exception prints the version it wants, so read that rather
    than guessing a pin: same principle as reading the module name out of an
    ImportError instead of maintaining a dependency list by hand.

    Round 5 demoted this from the fix to a detector. `pip_check_repair` now
    prevents the skew before anything imports the module, which is the only point
    at which it is repairable -- round 4 (bolt `s6arkzytjr`) installed the right
    pin, in-process, and hit the identical SystemError, because `pydantic_core`
    is a compiled extension and an already-imported one cannot be swapped. So a
    skew reaching here means the repair either did not run or could not fix it,
    and the caller re-execs rather than retrying in place.
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
        # --no-deps, and only here: prismatic's own setup pins a torch of its
        # own, so resolving ITS dependencies would re-pin the stack it is being
        # tested against. Its ordinary PyPI dependencies go through `install`.
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
            rec = install(pkg)
            rec["import"] = mod
            log["transitive_installs"].append(rec)
            if not rec["ok"]:
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
    # Last word on the install set, after every source and every dependency:
    # the next process to start is the one that loads a 7B checkpoint, and it
    # should not discover a conflict we could have named here.
    log["consistency_final"] = pip_check_repair()
    return log


def _print_consistency(label: str, c: dict) -> None:
    n_rep = sum(len(r["repairs"]) for r in c.get("rounds", []))
    print(f"[{label}] pip check: clean={c.get('clean')} repairs={n_rep}")
    for r in c.get("rounds", []):
        for f in r["findings"]:
            print(f"[{label}]   {f['kind']}: {f['holder']} "
                  f"{f['holder_version']} wants {f['requirement']}"
                  + (f" (have {f['installed']})" if f.get("installed") else ""))
        for rep in r["repairs"]:
            print(f"[{label}]   -> pip install {rep['spec']}: ok={rep['ok']}")
    for f in c.get("refused", []):
        # Refused rather than applied, so it has to be visible: a dependency
        # that wants a different torch/transformers than the paper names is a
        # finding about what supporting OFT costs.
        print(f"[{label}]   REFUSED (paper pin): {f['holder']} wants "
              f"{f['requirement']}")
    for u in c.get("unsatisfiable", []) or []:
        # No install can satisfy this, so the fix is a decision (a constraints
        # line), not another round. Round 5 spent four rounds not knowing that.
        want = ", ".join(f"{w['holder']} wants {w['requirement']}"
                         for w in u["wanted_by"])
        print(f"[{label}]   UNSATISFIABLE {u['dist']}: {want}")
    if c.get("oscillated"):
        print(f"[{label}]   OSCILLATED: a finding set repeated, so the repairs "
              f"were undoing each other. Pin the loser in CONSTRAINT_LINES.")
    for f in c.get("remaining", []) or []:
        print(f"[{label}]   UNRESOLVED: {f['holder']} wants {f['requirement']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="where to write the resolution log (JSON)")
    ap.add_argument("--pip-check-only", action="store_true",
                    help="only run pip_check_repair and exit. The runner calls "
                         "this before each probe stage, in its own process, "
                         "because that is the only place a compiled-extension "
                         "conflict can be fixed: once pydantic_core is imported "
                         "no reinstall can swap the loaded object (bolt "
                         "s6arkzytjr repaired it in-process and changed nothing).")
    ap.add_argument("--report-only", action="store_true",
                    help="with --pip-check-only, record the consistency of the "
                         "install set without repairing it. Used for stage 1, "
                         "which must measure the image the paper names rather "
                         "than an image we tidied first.")
    ap.add_argument("--write-constraints", metavar="PATH",
                    help="freeze the currently installed torch (see --freeze-pins) "
                         "plus the protobuf-4 pins round 5 showed are needed into "
                         "a pip constraints file at PATH; then exit. Export "
                         "SG_PIP_CONSTRAINTS=PATH (and pass -c PATH to any shell "
                         "pip) so every later install fails loudly instead of "
                         "moving a pin silently.")
    ap.add_argument("--freeze-pins", default="torch",
                    help="comma-separated pins to freeze exactly in the "
                         "constraints file (default: torch, the one no stage "
                         "varies and the one PyPI cannot restore).")
    args = ap.parse_args()

    if args.write_constraints:
        freeze = tuple(p for p in args.freeze_pins.split(",") if p.strip())
        rec = write_constraints(args.write_constraints, freeze=freeze)
        with open(args.out, "w") as f:
            json.dump({"mode": "write_constraints", **rec}, f, indent=2)
        print(f"[constraints] {rec['path']}")
        for line in rec["lines"]:
            print(f"[constraints]   {line}")
        return 0

    if args.pip_check_only:
        c = pip_check_repair(max_rounds=0 if args.report_only else 4)
        with open(args.out, "w") as f:
            json.dump({"mode": "pip_check_report" if args.report_only
                       else "pip_check_only",
                       "pins": pins(), "consistency": c}, f, indent=2)
        _print_consistency("pin-doctor", c)
        return 0

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
        # What the with-deps policy actually moved, per dependency. Round 4's
        # log could not show this: every install was --no-deps, so nothing ever
        # moved and nothing was ever consistent either.
        for t in log["transitive_installs"]:
            mv = (t.get("pins") or {}).get("moved") or {}
            if mv:
                print(f"[prismatic]     {t['import']} moved pins: {mv} "
                      f"restored={list((t['pins'].get('restored') or {}))}")
            if t.get("fallback"):
                print(f"[prismatic]     {t['import']}: {t['fallback']} -- "
                      f"partial install possible for this one")
    if log["pins_moved"]:
        # Loud, because it invalidates the comparison the stages exist to make.
        print(f"[prismatic] WARNING pins moved: {log['pins_moved']}")
    if log.get("consistency_final"):
        _print_consistency("prismatic", log["consistency_final"])
    # Exit 0 either way: an unresolvable dependency is a measurement, and the
    # caller decides what to do with it. Only a crash here is a failure.
    return 0


if __name__ == "__main__":
    sys.exit(main())
