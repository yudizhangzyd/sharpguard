#!/usr/bin/env python3
"""Offline checks on the OFT probe's install policy (no pip, no network, no GPU).

Rounds 1-4 of the OFT load probe all failed on this repo's own dependency
handling rather than on OpenVLA-OFT, and each round the failure looked like
evidence for limitation (ix). The functions under test here are the ones that
decide what gets installed and what gets reported, so they are the ones whose
mistakes turn into a false claim in the paper:

  * `parse_pip_check` -- the round-5 fix reads `pip check`, so a format it does
    not parse is a conflict silently left in place;
  * `PROTECTED` -- a repair that moves torch/transformers/tokenizers/timm would
    make the stage's load result describe a stack the paper does not name;
  * `version_skew` -- returned "2" instead of "2.46.4" for one round because a
    non-greedy match stopped inside the version;
  * `reassert_pins` -- must keep `moved` even after a successful restore, since
    "nothing moved" and "we forced it back" are different facts.

Both round-4 failure strings are kept verbatim as regression fixtures. They are
the two things this round exists to fix, and a refactor that stops recognising
them would re-open exactly the misdiagnosis that cost four rounds.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHECKS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(cond), detail))


def load_module():
    """Load install_prismatic.py by path, as the probe itself does."""
    path = ROOT / "bolt" / "install_prismatic.py"
    spec = importlib.util.spec_from_file_location("_ip_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Verbatim `pip check` output. The wandb line is the one that matters: round 4's
# wandb failure surfaced as an AttributeError deep inside an import, and this is
# the form in which it is visible BEFORE anything imports wandb.
PIP_CHECK_REAL = """\
pydantic 2.11.7 has requirement pydantic-core==2.46.4, but you have pydantic-core 2.47.0.
wandb 0.18.5 requires sentry-sdk, which is not installed.
tensorflow-datasets 4.9.3 requires array-record, which is not installed.
"""

# The exceptions round 4 actually died with, unedited.
ROUND4_SKEW = (
    "SystemError: The installed pydantic-core version (2.47.0) is incompatible "
    "with the current pydantic version, which requires 2.46.4. Check for "
    "version mismatches."
)
ROUND4_WANDB = (
    "AttributeError: partially initialized module 'wandb' has no attribute "
    "'errors' (most likely due to a circular import)"
)


def test_parse_pip_check(m) -> None:
    found = m.parse_pip_check(PIP_CHECK_REAL)
    check("pip_check: all three findings parsed", len(found) == 3, str(found))

    by_name = {f["name"]: f for f in found}
    check("pip_check: pydantic-core seen as a conflict",
          by_name.get("pydantic-core", {}).get("kind") == "conflict")
    check("pip_check: conflict keeps the EXACT specifier",
          by_name.get("pydantic-core", {}).get("requirement")
          == "pydantic-core==2.46.4",
          # Installing it by bare name is what created the conflict, so the
          # specifier is the whole repair.
          str(by_name.get("pydantic-core")))
    check("pip_check: conflict records what is installed",
          by_name.get("pydantic-core", {}).get("installed") == "pydantic-core 2.47.0")
    check("pip_check: wandb's absent requirement seen as missing",
          by_name.get("sentry-sdk", {}).get("kind") == "missing")
    check("pip_check: holder of the missing requirement is wandb",
          by_name.get("sentry-sdk", {}).get("holder") == "wandb")
    check("pip_check: hyphenated dist names survive",
          by_name.get("array-record", {}).get("holder") == "tensorflow-datasets")
    check("pip_check: clean output yields no findings",
          m.parse_pip_check("No broken requirements found.") == [])
    check("pip_check: empty/None input is not an error",
          m.parse_pip_check("") == [] and m.parse_pip_check(None) == [])

    # A specifier that is not `==`: the repair must still target the right dist.
    other = m.parse_pip_check(
        "foo 1.0 has requirement numpy<2,>=1.21, but you have numpy 2.1.0.\n")
    check("pip_check: range specifier keeps its name",
          other and other[0]["name"] == "numpy", str(other))
    check("pip_check: range specifier is installed as printed",
          other and other[0]["requirement"] == "numpy<2,>=1.21")


def test_req_name(m) -> None:
    cases = {
        "pydantic-core==2.46.4": "pydantic-core",
        "numpy<2,>=1.21": "numpy",
        "torch": "torch",
        "timm>=1.0.11": "timm",
        "uvicorn[standard]>=0.30": "uvicorn",
        "pytest; python_version>'3.8'": "pytest",
        "transformers!=4.53.0": "transformers",
        "tokenizers~=0.19": "tokenizers",
    }
    for spec, want in cases.items():
        check(f"req_name({spec!r}) -> {want}", m._req_name(spec) == want,
              m._req_name(spec))


def test_protected(m) -> None:
    # The four the paper names. If a repair could move one of them, a stage's
    # load result would silently be about a different stack -- which is the
    # failure mode this whole probe exists to avoid asserting.
    for pin in ("torch", "transformers", "tokenizers", "timm"):
        check(f"PROTECTED contains {pin}", pin in m.PROTECTED)
    check("PROTECTED does not over-reach to pydantic",
          "pydantic" not in m.PROTECTED and "pydantic-core" not in m.PROTECTED)

    # pip check prints distribution names, which normalize - and _ freely. A
    # repair spelled `Torch_` must still be refused.
    for spelling in ("torch", "Torch", "TORCH"):
        check(f"protected match is case-insensitive: {spelling}",
              m._norm_dist(spelling) in {m._norm_dist(p) for p in m.PROTECTED})
    check("dist normalization folds - . and _",
          m._norm_dist("pydantic_core") == m._norm_dist("pydantic-core")
          == m._norm_dist("pydantic.core") == "pydantic-core")

    # REPINNABLE must exclude torch: the image's build is 2.4.1+cu118 and PyPI
    # does not serve that local version, so "restoring" it would install a
    # DIFFERENT build while reporting success.
    check("torch is not repinnable by a plain PyPI install",
          "torch" not in m.REPINNABLE)
    for pin in ("transformers", "tokenizers", "timm"):
        check(f"{pin} is repinnable", pin in m.REPINNABLE)


def test_version_skew(m) -> None:
    got = m.version_skew(ROUND4_SKEW)
    check("version_skew: round-4 SystemError parsed", got is not None, str(got))
    check("version_skew: package name", got and got[0] == "pydantic-core", str(got))
    # The bug that shipped for one round: a non-greedy match stopped at the
    # version's own first dot and pinned "2".
    check("version_skew: FULL version, not truncated at the first dot",
          got and got[1] == "2.46.4", str(got))
    check("version_skew: unrelated error is not mistaken for a skew",
          m.version_skew(ROUND4_WANDB) is None)
    check("version_skew: a missing module is not a skew",
          m.version_skew("ModuleNotFoundError: No module named 'jsonlines'")
          is None)


def test_missing_module_still_works(m) -> None:
    # Not new in round 5, but the round-5 edits moved the call site, and this is
    # the function whose earlier version read only the top-level message.
    check("missing_module: plain ModuleNotFoundError",
          m.missing_module("ModuleNotFoundError: No module named 'jsonlines'")
          == "jsonlines")
    check("missing_module: submodule reduced to its top-level package",
          m.missing_module("No module named 'prismatic.extern.hf'") == "prismatic")
    check("missing_module: transformers' check_imports phrasing",
          m.missing_module("requires the following packages that were not "
                           "found in your environment: prismatic") == "prismatic")
    check("missing_module: round-4 wandb AttributeError names nothing to install",
          m.missing_module(ROUND4_WANDB) is None,
          # It is an import failure, but installing "wandb" again is not the fix
          # -- pip check is. If this returned "wandb" the loop would reinstall it
          # and report `stuck_on: wandb`, which is round 4's wrong conclusion.
          str(m.missing_module(ROUND4_WANDB)))


def test_reassert_pins(m) -> None:
    calls = []

    def fake_pip(*args):
        calls.append(args)
        return True, ""

    versions = {"torch": "2.4.1+cu118", "transformers": "4.53.2",
                "tokenizers": "0.21.0", "timm": "0.9.10"}
    orig_pip, orig_pins = m.pip, m.pins
    m.pip = fake_pip
    m.pins = lambda: dict(versions)
    try:
        before = {"torch": "2.4.1+cu118", "transformers": "4.40.1",
                  "tokenizers": "0.19.1", "timm": "0.9.10"}
        rec = m.reassert_pins(before)
    finally:
        m.pip, m.pins = orig_pip, orig_pins

    check("reassert: only the pins that changed are reported",
          set(rec["moved"]) == {"transformers", "tokenizers"}, str(rec["moved"]))
    check("reassert: the move is kept even though the restore succeeded",
          rec["moved"]["transformers"] == ["4.40.1", "4.53.2"],
          # "nothing moved" and "we forced it back" must not read alike.
          str(rec["moved"]))
    check("reassert: restore pins the version the paper names",
          ("install", "--quiet", "--no-deps", "transformers==4.40.1") in calls,
          str(calls))
    check("reassert: restore is --no-deps",
          all("--no-deps" in c for c in calls if c[0] == "install"), str(calls))
    check("reassert: an unchanged pin is not reinstalled",
          not any("timm" in str(c) for c in calls), str(calls))
    check("reassert: restore reported per pin",
          rec["restored"]["tokenizers"]["repin_ok"] is True, str(rec))

    # A moved torch cannot be restored, so it must be reported as unrestorable
    # rather than quietly "fixed" with a different build.
    versions["torch"] = "2.6.0"
    m.pip, m.pins = fake_pip, (lambda: dict(versions))
    try:
        rec2 = m.reassert_pins({"torch": "2.4.1+cu118"})
    finally:
        m.pip, m.pins = orig_pip, orig_pins
    check("reassert: a moved torch is unrestorable, not silently repinned",
          [u["pin"] for u in rec2["unrestorable"]] == ["torch"], str(rec2))
    check("reassert: no pip install is issued for torch",
          not any("torch" in str(c) for c in calls[len(calls):]), str(calls))

    # An absent pin (import failed) is not a "move" -- reporting it as one would
    # invent a version change out of an ImportError.
    m.pip, m.pins = fake_pip, (lambda: {"timm": "0.9.10"})
    try:
        rec3 = m.reassert_pins({"timm": "<absent: ModuleNotFoundError>"})
    finally:
        m.pip, m.pins = orig_pip, orig_pins
    check("reassert: absent-before is not counted as a moved pin",
          rec3["moved"] == {}, str(rec3))


def test_install_policy(m) -> None:
    """`install` must resolve deps by default and fall back only on failure."""
    seen = []

    def fake_pip(*args):
        seen.append(args)
        if args[0] == "check":
            return True, "No broken requirements found."
        # A with-deps resolution that pip refuses, exactly once, for "wandb".
        if "wandb" in args and "--no-deps" not in args:
            return False, "ResolutionImpossible: torch 2.4.1+cu118"
        return True, ""

    orig_pip, orig_pins = m.pip, m.pins
    m.pip = fake_pip
    m.pins = lambda: {"torch": "2.4.1+cu118", "transformers": "4.40.1",
                      "tokenizers": "0.19.1", "timm": "0.9.10"}
    try:
        ok_rec = m.install("jsonlines")
        wandb_rec = m.install("wandb")
        tree_rec = m.install("git+https://example/openvla-oft.git", no_deps=True)
    finally:
        m.pip, m.pins = orig_pip, orig_pins

    installs = [c for c in seen if c[0] == "install"]
    check("install: deps are NOT suppressed by default",
          ("install", "--quiet", "jsonlines") in installs, str(installs))
    check("install: default install reports no_deps False",
          ok_rec["ok"] and ok_rec["no_deps"] is False, str(ok_rec))
    check("install: runs pip check after a successful install",
          ("check",) in seen and ok_rec.get("consistency", {}).get("clean") is True,
          str(ok_rec.get("consistency")))
    check("install: falls back to --no-deps when the resolution fails",
          ("install", "--quiet", "--no-deps", "wandb") in installs, str(installs))
    check("install: the fallback is recorded, not hidden",
          wandb_rec.get("fallback") == "no_deps_after_resolution_failed"
          and wandb_rec["no_deps"] is True, str(wandb_rec))
    check("install: the package under test stays --no-deps",
          ("install", "--quiet", "--no-deps",
           "git+https://example/openvla-oft.git") in installs
          and tree_rec["no_deps"] is True, str(tree_rec))


def test_pip_check_repair(m) -> None:
    """The repair loop: fixes what it can, refuses the pins, and terminates."""
    state = {"round": 0}
    issued = []

    def fake_pip(*args):
        if args[0] == "check":
            state["round"] += 1
            if state["round"] == 1:
                return False, PIP_CHECK_REAL
            return True, "No broken requirements found."
        issued.append(args)
        return True, ""

    orig = m.pip
    m.pip = fake_pip
    try:
        rec = m.pip_check_repair()
    finally:
        m.pip = orig

    specs = [a[-1] for a in issued]
    check("repair: conflict repaired at the exact version",
          "pydantic-core==2.46.4" in specs, str(specs))
    check("repair: missing requirement installed by name",
          "sentry-sdk" in specs and "array-record" in specs, str(specs))
    check("repair: converges and reports clean", rec["clean"] is True, str(rec))
    check("repair: one round was enough", len(rec["rounds"]) == 1, str(rec))

    # Now a conflict that pip check keeps reporting: the loop must stop rather
    # than reinstall forever, and the leftover must be visible.
    stuck = "foo 1.0 has requirement pydantic-core==9.9.9, but you have pydantic-core 2.46.4.\n"
    m.pip = lambda *a: (False, stuck) if a[0] == "check" else (True, "")
    try:
        rec2 = m.pip_check_repair()
    finally:
        m.pip = orig
    check("repair: a finding that never clears stops the loop",
          len(rec2["rounds"]) == 1, str(len(rec2["rounds"])))
    check("repair: an unrepairable conflict is reported, not swallowed",
          rec2["clean"] is False and rec2["remaining"], str(rec2))

    # A requirement on one of the paper's pins must be refused and printed.
    protected = ("bar 1.0 has requirement transformers==4.53.2, but you have "
                 "transformers 4.40.1.\n")
    m.pip = lambda *a: (False, protected) if a[0] == "check" else (True, "")
    try:
        rec3 = m.pip_check_repair()
    finally:
        m.pip = orig
    check("repair: a repair that would move transformers is REFUSED",
          [f["name"] for f in rec3["refused"]] == ["transformers"], str(rec3))
    check("repair: refusing leaves the set unclean rather than pretending",
          rec3["clean"] is False, str(rec3))
    check("repair: nothing is installed when every finding is refused",
          rec3["rounds"] == [] or not rec3["rounds"][0]["repairs"], str(rec3))


def test_oscillation_and_unsatisfiable(m) -> None:
    """Round 6: the two ways round 5's repair loop misread its own output.

    Round 5 (bolt `8ejjyfkzq8`) hit both at once. `tensorflow` requires
    protobuf<5 and recent `wandb` requires >=5, so every round satisfied one and
    broke the other; the finding set alternated instead of repeating, the
    same-as-last-round guard never fired, four rounds were spent, and whichever
    install happened last decided which package was broken. Stage 3 then died on
    `cannot import name 'Imports' from wandb.proto.wandb_telemetry_pb2` -- a
    failure with our name on it, inside the stage that was supposed to answer
    whether OFT loads.
    """
    # Two holders, one dist, incompatible specifiers: no install satisfies both.
    contested = ("tensorflow 2.15.0 has requirement protobuf<5, but you have "
                 "protobuf 5.29.0.\n"
                 "wandb 0.19.1 has requirement protobuf>=5, but you have "
                 "protobuf 5.29.0.\n")
    issued = []

    def fake_pip(*args):
        if args[0] == "check":
            return False, contested
        issued.append(args)
        return True, ""

    orig = m.pip
    m.pip = fake_pip
    try:
        rec = m.pip_check_repair()
    finally:
        m.pip = orig

    check("unsatisfiable: the contested dist is reported",
          [u["dist"] for u in rec["unsatisfiable"]] == ["protobuf"], str(rec))
    check("unsatisfiable: BOTH holders are named",
          sorted(w["holder"] for w in rec["unsatisfiable"][0]["wanted_by"])
          == ["tensorflow", "wandb"], str(rec["unsatisfiable"]))
    check("unsatisfiable: nothing is installed for a contested dist",
          not issued,
          # Installing either side is choosing a loser by iteration order, and
          # round 5's loser was the one whose failure got read as OFT's.
          str(issued))
    check("unsatisfiable: the set is reported unclean, not clean",
          rec["clean"] is False, str(rec))

    # Now the alternation itself: two finding sets that swap every round. The
    # old guard compared against the previous round only and never matched.
    seq = [
        "tensorflow 2.15.0 has requirement protobuf<5, but you have protobuf 5.29.0.\n",
        "wandb 0.19.1 has requirement protobuf>=5, but you have protobuf 4.25.3.\n",
    ]
    state = {"i": 0}

    def flip_pip(*args):
        if args[0] == "check":
            text = seq[state["i"] % 2]
            state["i"] += 1
            return False, text
        return True, ""

    m.pip = flip_pip
    try:
        rec2 = m.pip_check_repair(max_rounds=8)
    finally:
        m.pip = orig

    check("oscillation: an alternating finding set is detected",
          rec2["oscillated"] is True, str(rec2))
    check("oscillation: it stops on the repeat, not at the round budget",
          len(rec2["rounds"]) == 2, str(len(rec2["rounds"])))
    check("oscillation: the outcome is reported unclean",
          rec2["clean"] is False, str(rec2))

    # A single non-clearing finding is NOT an oscillation -- it is one set seen
    # twice. Both must terminate, but they are different diagnoses and the fix
    # differs (pin the loser vs. the requirement is simply unmeetable).
    stuck = "foo 1.0 has requirement bar==9.9.9, but you have bar 1.0.\n"
    m.pip = lambda *a: (False, stuck) if a[0] == "check" else (True, "")
    try:
        rec3 = m.pip_check_repair()
    finally:
        m.pip = orig
    check("oscillation: a single non-clearing finding is flagged as such",
          rec3["oscillated"] is True and rec3["unsatisfiable"] == [], str(rec3))


def test_write_constraints(m, tmp) -> None:
    """The constraints file: torch exactly, protobuf world pinned, others free."""
    orig_pins = m.pins
    m.pins = lambda: {"torch": "2.4.1+cu118", "transformers": "4.40.1",
                      "tokenizers": "0.19.1", "timm": "0.9.10"}
    try:
        rec = m.write_constraints(str(tmp / "c.txt"))
        body = (tmp / "c.txt").read_text()
    finally:
        m.pins = orig_pins

    check("constraints: torch is pinned to the LOCAL build",
          "torch==2.4.1+cu118" in body,
          # PyPI does not serve +cu118, so this makes a downgrade a resolution
          # failure instead of round 5's silent 2.4.1+cu118 -> 2.2.0+cu121.
          body)
    check("constraints: the protobuf-4 world is pinned",
          "protobuf<5" in body and "wandb<0.18" in body
          and "tensorflow-metadata<1.15" in body, body)
    for free in ("transformers==", "tokenizers==", "timm=="):
        check(f"constraints: {free[:-2]} is NOT frozen by default",
              free not in body,
              # The 2x2 varies these on purpose; constraining a variable of the
              # experiment turns the cell that varies it into a pip failure.
              body)
    check("constraints: what was frozen is recorded",
          rec["frozen"] == ["torch"], str(rec))

    # An absent pin must not become the literal string `torch==<absent: ...>`.
    m.pins = lambda: {"torch": "<absent: ModuleNotFoundError>"}
    try:
        m.write_constraints(str(tmp / "d.txt"))
        body2 = (tmp / "d.txt").read_text()
    finally:
        m.pins = orig_pins
    check("constraints: an absent pin is skipped, not written as a bad spec",
          "absent" not in body2 and "torch" not in body2, body2)

    # And the whole point: an install must carry the file without the call site
    # asking. Round 5's pin restore is the call site that forgot.
    seen = []
    orig_pip_run = m.subprocess.run

    class _CP:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kw):
        seen.append(cmd)
        return _CP()

    import os as _os
    prev = _os.environ.get(m.CONSTRAINTS_ENV)
    _os.environ[m.CONSTRAINTS_ENV] = str(tmp / "c.txt")
    m.subprocess.run = fake_run
    try:
        m.pip("install", "--quiet", "wandb")
        m.pip("check")
    finally:
        m.subprocess.run = orig_pip_run
        if prev is None:
            _os.environ.pop(m.CONSTRAINTS_ENV, None)
        else:
            _os.environ[m.CONSTRAINTS_ENV] = prev

    check("constraints: every install carries -c automatically",
          "-c" in seen[0] and str(tmp / "c.txt") in seen[0], str(seen[0]))
    check("constraints: -c goes right after `install`, before the packages",
          seen[0][seen[0].index("install") + 1] == "-c", str(seen[0]))
    check("constraints: a non-install pip call is left alone",
          "-c" not in seen[1], str(seen[1]))


def test_runner_round6_policy() -> None:
    """The runner's own pip calls, which are the ones that escaped in round 5."""
    src = (ROOT / "bolt" / "run_oft_load_probe.sh").read_text()
    check("runner: a constraints file is written before any stage",
          "--write-constraints" in src and "SG_PIP_CONSTRAINTS" in src)
    check("runner: it is exported so install_prismatic picks it up",
          'export SG_PIP_CONSTRAINTS="$CONSTRAINTS"' in src)
    n_pip = src.count("pip install --quiet")
    n_constrained = src.count('pip install --quiet -c "$CONSTRAINTS"')
    check("runner: EVERY shell pip install is constrained",
          n_pip == n_constrained,
          # The stage-3 pin restore is the one that moved torch to 2.2.0+cu121.
          f"{n_constrained}/{n_pip} constrained")
    check("runner: stage 4 no longer forces a timm the checkpoint rejects",
          '"timm>=1.0.11"' not in src,
          # Quoted, so the header comment explaining WHY it was removed does not
          # itself satisfy the check.
          "modeling_prismatic.py raises NotImplementedError for timm >= 1.0.0")
    check("runner: timm stays inside the window the checkpoint declares",
          src.count('"timm>=0.9.10,<1.0.0"') == 2,
          "both upgrade stages must stay loadable to be about transformers")
    check("runner: torch movement is checked between stages",
          "torch_watch " in src and src.count("torch_watch ") >= 3)
    check("runner: the reader compares torch ACROSS stage reports",
          "torch is not the same in every 2x2 stage" in src,
          "each round-5 report looked fine alone; they disagreed with each other")
    check("runner: the reader surfaces unsatisfiable and oscillated",
          "UNSATISFIABLE" in src and "OSCILLATED" in src)


def test_runner_round7_policy() -> None:
    """Round 7's fifth stage: OFT's own declared pins, outside the 2x2.

    Round 6 (bolt dxb6wu9rxy) held torch at 2.4.1+cu118 in all four cells, which
    is what makes them comparable -- and that is exactly why no cell ran the
    torch==2.2.0 openvla-oft itself declares. This stage does, in its own venv.
    Three properties matter, and all three are the difference between a
    measurement and a second broken environment.
    """
    src = (ROOT / "bolt" / "run_oft_load_probe.sh").read_text()
    check("round7: the stage exists and is named in the probe's stage flag",
          "--stage oft_declared_pins" in src)
    check("round7: it runs in its own venv, not the job's interpreter",
          "python -m venv" in src and '"$VENV/bin/python" experiments/probe' in src,
          "installing torch 2.2.0 into the job's env would break stages 1-4")
    check("round7: the venv is NOT --system-site-packages",
          "venv --system-site-packages" not in src,
          "inheriting the job's torch is precisely what this stage must not do")
    check("round7: openvla-oft is installed WITH its declared deps",
          "git+https://github.com/moojink/openvla-oft.git" in src and
          '"$VENV/bin/pip" install --quiet --no-deps '
          '"git+https://github.com/moojink/openvla-oft.git"' not in src,
          "--no-deps would defeat the stage: the deps ARE the measurement")
    check("round7: the constraints file is not applied to that pip",
          '"$VENV/bin/pip" install --quiet -c' not in src,
          "the constraints file freezes torch, and moving torch is the stage")
    check("round7: the declared stage is excluded from the 2x2 torch check",
          'DECLARED = "oft_load_probe_oft_declared_pins.json"' in src and
          "for p in grid:" in src,
          "folding it in would fire the check on the one stage meant to differ")
    check("round7: a stage that silently ran OUR torch is reported inconclusive",
          "INCONCLUSIVE" in src,
          "a refused resolution leaves a cell that looks like a 5th data point "
          "and is a duplicate of the 2x2")
    check("round7: a missing stage-5 report is stated, not skipped",
          "stage 5 (oft_declared_pins) wrote NO report" in src)
    check("round7: the probe accepts the stage name",
          '"oft_declared_pins"' in
          (ROOT / "experiments" / "probe_openvla_oft_load.py").read_text(),
          "an unlisted --stage choice would exit 2 before measuring anything")


def test_report_only(m) -> None:
    """`max_rounds=0` observes without touching: stage 1 depends on it."""
    issued = []

    def fake_pip(*args):
        issued.append(args)
        return (False, PIP_CHECK_REAL) if args[0] == "check" else (True, "")

    orig = m.pip
    m.pip = fake_pip
    try:
        rec = m.pip_check_repair(max_rounds=0)
    finally:
        m.pip = orig

    check("report-only: nothing is installed",
          not [c for c in issued if c[0] == "install"], str(issued))
    check("report-only: the inconsistency is still reported",
          rec["clean"] is False and len(rec["remaining"]) == 3, str(rec))
    check("report-only: no repair rounds are claimed", rec["rounds"] == [],
          # Stage 1 must be able to say "the image arrived like this".
          str(rec["rounds"]))


def test_runner_stage_policy() -> None:
    """The runner repairs before stages 2-4 and only OBSERVES before stage 1."""
    src = (ROOT / "bolt" / "run_oft_load_probe.sh").read_text()
    check("runner: stage 1 pin doctor is report-only",
          "pin_doctor baseline --report-only" in src)
    for stage in ("upgraded", "prismatic_baseline", "prismatic_upgraded"):
        check(f"runner: {stage} repairs before the probe runs",
              f"pin_doctor {stage}\n" in src and
              f"pin_doctor {stage} --report-only" not in src)
    check("runner: the doctor runs in its own process, not inside the probe",
          "python bolt/install_prismatic.py --pip-check-only" in src,
          "a compiled extension can only be replaced before the importer starts")
    check("runner: pin doctor output is read back by the job",
          'glob("pin_doctor_*.json")' in src)


def test_probe_reexec_constants() -> None:
    """The probe's re-exec: bounded, and it carries the install log across."""
    src = (ROOT / "experiments" / "probe_openvla_oft_load.py").read_text()
    check("probe: re-exec exists", "def _reexec(" in src)
    check("probe: re-exec is bounded", "_MAX_REEXEC" in src and "os.execve" in src)
    check("probe: the skew branch re-execs instead of retrying in place",
          "_reexec(f\"{pkg}=={want} installed" in src,
          # Round 4 retried in-process against an already-loaded compiled module.
          "the skew branch must hand the fix to a fresh interpreter")
    check("probe: installs go through the with-deps policy",
          "res.install(pkg)" in src and 'res.pip("install", "--quiet", "--no-deps", pkg)' not in src,
          "the transitive loop must not use bare --no-deps any more")
    check("probe: the carried install log is bounded",
          "[:8000]" in src, "an env block is not a database")
    check("probe: re-exec count reaches the report",
          '"reexec_count"' in src)


def main() -> int:
    m = load_module()
    test_parse_pip_check(m)
    test_req_name(m)
    test_protected(m)
    test_version_skew(m)
    test_missing_module_still_works(m)
    test_reassert_pins(m)
    test_install_policy(m)
    test_pip_check_repair(m)
    test_oscillation_and_unsatisfiable(m)
    with tempfile.TemporaryDirectory() as td:
        test_write_constraints(m, Path(td))
    test_report_only(m)
    test_runner_stage_policy()
    test_runner_round6_policy()
    test_runner_round7_policy()
    test_probe_reexec_constants()

    bad = [c for c in CHECKS if not c[1]]
    for name, ok, detail in CHECKS:
        if not ok:
            print(f"FAIL {name}" + (f"  [{detail}]" if detail else ""))
    print(f"{len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
