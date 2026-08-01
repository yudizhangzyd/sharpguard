#!/usr/bin/env bash
# Measure the OpenVLA-OFT load claim (limitation (ix)) instead of asserting it.
#
# The manuscript says OFT "failed to load cleanly in our environment (Python
# 3.10, torch 2.2.0)" and "remains unloadable in our environment". That is the
# one claim in the paper with no artifact behind it, and it is the one that
# excuses a coverage gap -- so it gets measured like everything else.
#
# Five stages, in one job, deliberately in this order:
#   1. BASELINE: attempt the load in the environment the paper names, and record
#      the exception. Without this the claim stays unfalsifiable even if a later
#      stage succeeds, because we would not have shown what we originally hit.
#   2. UPGRADED: pip-install a transformers that postdates OFT's release and
#      retry. "Unloadable" is only honest if a current stack was tried; a stale
#      pin is our problem, not a property of the checkpoint.
#   3. PRISMATIC_BASELINE: stages 1 and 2 both failed with an ImportError naming
#      a package we had never installed (`prismatic`, the OpenVLA/OFT codebase).
#      A missing dependency is not an incompatibility, so it is installed and
#      the paper's own pins are restored before re-measuring.
#   4. PRISMATIC_UPGRADED: the last cell of the 2x2, which separates "OFT needs
#      a newer transformers than we pin" from "OFT does not load here at all".
#   5. OFT_DECLARED_PINS: outside the 2x2, added in round 7. The four cells above
#      all hold torch at ours so they stay comparable, and round 6 found that the
#      one torch OFT itself declares (2.2.0) is therefore the one torch none of
#      them ran. This cell runs the probe in a separate venv built from OFT's own
#      declared dependency set, so "unloadable" is tested at the pin the
#      checkpoint asks for and not only at ours.
#
# Every stage writes a report and the job exits 0 either way: a measured failure
# IS the deliverable here. What the job must never do is exit 0 having measured
# nothing, so the final check below fails loudly if no report exists.
#
# Round 5 adds `pin_doctor` before each probe stage. Round 4 (bolt s6arkzytjr)
# failed twice, and both failures were this job's own install set rather than
# anything about OFT: a pydantic/pydantic-core conflict and a wandb installed
# without its requirements. `pip check` names both. It runs in a SEPARATE process
# from the probe on purpose -- a compiled extension can only be replaced before
# the process that imports it starts, which is why round 4's in-process repair
# installed the right version and changed nothing.
#
# Round 6 fixes what round 5 (bolt 8ejjyfkzq8) measured. Round 5's own installs
# were clean, all four stages reported, and three of the four results were still
# not about OFT:
#
#   * a constraints file now holds torch and the protobuf-4 world across every
#     pip call, shell ones included. Round 5's stage-3 pin restore silently moved
#     torch 2.4.1+cu118 -> 2.2.0+cu121, so stages 3 and 4 measured a stack the
#     paper does not name;
#   * stage 4 no longer forces `timm>=1.0.11`. The checkpoint's remote code
#     raises `NotImplementedError: TIMM Version must be >= 0.9.10 and < 1.0.0`,
#     so round 5's last cell was unloadable by construction and its failure
#     measured our install, not transformers;
#   * `pip_check_repair` stops on an oscillation and refuses to pick a side
#     between two holders that can never both be satisfied. Round 5 ping-ponged
#     protobuf 4<->5 for four rounds and left stage 3 with a wandb that could not
#     import.
#
# Stage 3 is the cell that matters: round 5 proved OFT wants exactly the paper's
# own pins (transformers 4.40.1 / tokenizers 0.19.1 / timm 0.9.10 -- 20 REFUSED
# lines say so), and its only remaining blocker was the wandb/protobuf mismatch
# the constraints file now prevents.
set -x
cd "$(dirname "$0")/.."
if [ -f /tmp/sharpguard.env ]; then set -a; . /tmp/sharpguard.env; set +a; fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/cotfaith-oft-probe"
mkdir -p "$OUT_DIR"
nvidia-smi -L || true
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/tmp/hf}"

# Repair whatever the last pip command left inconsistent, in a fresh process,
# before the probe imports anything. Never repairs torch/transformers/tokenizers
# /timm -- those four ARE the environment the paper names, so a dependency that
# wants a different one is printed as a finding instead.
pin_doctor () {
    python bolt/install_prismatic.py --pip-check-only ${2:-} \
        --out "$OUT_DIR/pin_doctor_$1.json" || echo "[oft] pin doctor $1 crashed"
}

# ---- round 6: make a pin move a failure instead of a surprise ----------------
# Round 5 (bolt `8ejjyfkzq8`) ran all four stages and reported cleanly, and two
# things had still gone wrong underneath it:
#
#   * torch went 2.4.1+cu118 -> 2.2.0+cu121 between stages 1 and 3, because the
#     stage-3 pin restore below resolved torch from PyPI. Stages 3 and 4 were
#     measuring a different stack than the paper names and said nothing.
#   * protobuf oscillated 4.x <-> 5.x for four repair rounds (tensorflow wants
#     <5, recent wandb wants >=5, tensorflow_metadata's stubs need 5) and stage 3
#     died on `cannot import name 'Imports' from wandb.proto.wandb_telemetry_pb2`.
#
# One constraints file fixes both, and it is passed to EVERY pip call from here
# on -- including these shell ones, which is how torch escaped last time. A
# constraint is a statement about the whole environment, so a resolution that
# would break it fails and is recorded rather than silently winning.
CONSTRAINTS="$OUT_DIR/constraints.txt"
python bolt/install_prismatic.py --write-constraints "$CONSTRAINTS" \
    --freeze-pins torch --out "$OUT_DIR/constraints.json" \
    || echo "[oft] constraints write failed"
export SG_PIP_CONSTRAINTS="$CONSTRAINTS"
cat "$CONSTRAINTS" || true

# torch is the one pin no stage varies, so any movement invalidates the
# comparison the 2x2 exists to make. Checked between stages rather than at the
# end: knowing WHICH install moved it is the whole point.
TORCH_REF="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
echo "[oft] torch reference for this job: $TORCH_REF"
torch_watch () {
    now="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
    if [ "$now" != "$TORCH_REF" ]; then
        echo "[oft] FATAL-ISH torch moved before stage $1: $TORCH_REF -> $now."
        echo "[oft]   Everything from here on describes a stack the paper does"
        echo "[oft]   not name. The constraints file was supposed to prevent"
        echo "[oft]   this; treat any load result below as uncomparable."
    else
        echo "[oft] torch still $now before stage $1"
    fi
}

# The install policy under test here is the one that produced four rounds of
# results about our own environment rather than about OFT. It costs a second and
# it runs before anything is installed, while the image is still the one the
# paper names. Report-only: a failure here does not stop the measurement, but it
# tells the reader which of the two to trust.
python tests/test_install_policy.py || \
    echo "[oft] WARNING install-policy checks FAIL: read every verdict below as "\
         "provisional -- the code deciding what gets installed is not behaving "\
         "as its own tests specify."

# ---- stage 1: the environment the paper names -------------------------------
python -c "import torch, transformers, sys;
print('[oft] baseline python', sys.version.split()[0],
      'torch', torch.__version__, 'transformers', transformers.__version__)" || true

# Report-only here, on purpose: stage 1 measures the environment the paper names,
# so it records whether the image arrived consistent instead of tidying it first.
pin_doctor baseline --report-only
python experiments/probe_openvla_oft_load.py \
    --out   "$OUT_DIR" \
    --stage baseline \
    --dtype "${DTYPE:-bfloat16}" \
    ${OFT_REPOS:+--repos "$OFT_REPOS"} || echo "[oft] baseline probe exited nonzero"

# ---- stage 2: a stack that postdates the checkpoint -------------------------
# OFT's remote code needs a newer transformers than the image pins. Installed
# without --no-deps on purpose: the point is to try the combination upstream
# actually supports, not a hand-pinned one we invented. timm and tokenizers move
# with it because OpenVLA's remote code imports both.
#
# `-c "$CONSTRAINTS"` holds torch and the protobuf-4 world while transformers
# moves. Without it this install is free to drag torch along, which is what
# happened in round 5 -- and a transformers-vs-OFT result read off a stack whose
# torch also changed answers a question nobody asked.
pip install --quiet -c "$CONSTRAINTS" --upgrade \
    "transformers==4.53.2" "tokenizers>=0.21,<0.22" "timm>=0.9.10,<1.0.0" \
    "accelerate>=0.34" "peft>=0.13" || echo "[oft] upgrade pip install failed"

python -c "import torch, transformers, sys;
print('[oft] upgraded python', sys.version.split()[0],
      'torch', torch.__version__, 'transformers', transformers.__version__)" || true

torch_watch upgraded
pin_doctor upgraded
python experiments/probe_openvla_oft_load.py \
    --out   "$OUT_DIR" \
    --stage upgraded \
    --dtype "${DTYPE:-bfloat16}" \
    ${OFT_REPOS:+--repos "$OFT_REPOS"} || echo "[oft] upgraded probe exited nonzero"

# ---- stage 3: install the dependency the exception named, then retry ---------
# Stages 1 and 2 failed identically, and not with an incompatibility:
#
#   ImportError: This modeling file requires the following packages that were
#   not found in your environment: prismatic. Run `pip install prismatic`
#
# `prismatic` is the OpenVLA/OFT codebase's own package, and we had simply never
# installed it. Reporting that as "OFT remains unloadable" would be citing a
# missing dependency as a property of the checkpoint, so it gets resolved and
# re-measured. Back to the paper's pins first: stage 2 moved transformers, and a
# load result is only about the environment the paper names if that environment
# is the one in place.
#
# Round 5 ran this WITHOUT the constraints file and it is the line that moved
# torch to 2.2.0+cu121: resolving `transformers==4.40.1`'s dependency set was
# allowed to pick a torch, and it did.
pip install --quiet -c "$CONSTRAINTS" \
    "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10" \
    "peft==0.11.1" "accelerate==0.30.1" || echo "[oft] pin restore failed"

python bolt/install_prismatic.py --out "$OUT_DIR/prismatic_resolution.json" \
    || echo "[oft] prismatic resolver crashed"

python -c "import torch, transformers, sys;
print('[oft] prismatic_baseline python', sys.version.split()[0],
      'torch', torch.__version__, 'transformers', transformers.__version__)" || true

torch_watch prismatic_baseline
pin_doctor prismatic_baseline
python experiments/probe_openvla_oft_load.py \
    --out   "$OUT_DIR" \
    --stage prismatic_baseline \
    --resolve-deps \
    --dtype "${DTYPE:-bfloat16}" \
    ${OFT_REPOS:+--repos "$OFT_REPOS"} || echo "[oft] prismatic_baseline exited nonzero"

# ---- stage 4: dependency present AND a current stack ------------------------
# The last cell of the 2x2. If stage 3 still fails, this separates "OFT needs a
# newer transformers than our pin" from "OFT does not load here at all", and
# only the second supports the limitation as written.
#
# timm is held below 1.0 here, and that is a correction, not a compromise. Round
# 5 installed `timm>=1.0.11` in this cell and got timm 1.0.28, whereupon the
# checkpoint's own remote code refused before loading anything:
#
#   NotImplementedError: TIMM Version must be >= 0.9.10 and < 1.0.0
#     (modeling_prismatic.py:32)
#
# So the round-5 stage 4 could not have loaded for any transformers, and its
# failure said nothing about transformers -- it measured our own out-of-range
# install. `>=0.9.10,<1.0.0` is the window the checkpoint itself declares, which
# leaves transformers as the only thing this cell varies. That is what makes the
# cell a control instead of a second, differently-broken environment.
pip install --quiet -c "$CONSTRAINTS" --upgrade \
    "transformers==4.53.2" "tokenizers>=0.21,<0.22" "timm>=0.9.10,<1.0.0" \
    "accelerate>=0.34" "peft>=0.13" || echo "[oft] second upgrade failed"

torch_watch prismatic_upgraded
pin_doctor prismatic_upgraded
python experiments/probe_openvla_oft_load.py \
    --out   "$OUT_DIR" \
    --stage prismatic_upgraded \
    --resolve-deps \
    --dtype "${DTYPE:-bfloat16}" \
    ${OFT_REPOS:+--repos "$OFT_REPOS"} || echo "[oft] prismatic_upgraded exited nonzero"

# ---- stage 5: the pins OpenVLA-OFT itself declares --------------------------
# Round 6 (bolt dxb6wu9rxy) is what makes this stage necessary. Its constraints
# file worked -- torch held at 2.4.1+cu118 in all four cells, which round 5
# failed to do -- and with `prismatic` present both prismatic_* cells got past
# the processor to WEIGHT LOADING and died there with the same error:
#
#   TypeError: cumsum() received an invalid combination of arguments
#              - got (bool, dim=int)
#
# and the resolver's own output names the reason to suspect the pin rather than
# the checkpoint: `openvla-oft 0.0.1 wants torch==2.2.0` (also torchvision
# 0.17.0, torchaudio 2.2.0, timm 0.9.10, tokenizers 0.19.1), all REFUSED because
# holding torch is what makes the 2x2 comparable. So the one torch OFT names is
# the one torch no cell ran, and "remains unloadable in our environment" is not
# yet established at the pin the checkpoint asks for. That is the difference
# between a coverage gap we measured and one we declined to close.
#
# It runs in a SEPARATE venv, deliberately:
#   * `pip install git+...openvla-oft.git` there resolves OFT's own declared
#     dependency set with nothing hand-pinned by us, which is the purest form of
#     the question. In the main environment the same install would either be
#     refused by the constraints file or would break the four cells above.
#   * SG_PIP_CONSTRAINTS is NOT passed to it. The constraints file exists to
#     freeze torch, and moving torch is the entire content of this cell -- a
#     constraint here would turn the measurement into a resolution failure.
# Nothing in this stage can affect stages 1-4: different interpreter, different
# site-packages, and it comes last.
VENV=/tmp/oft_declared
python -m venv "$VENV" && "$VENV/bin/pip" install --quiet --upgrade pip
# No -c: see above. Failures are echoed, not fatal -- a stage that cannot be
# built is itself a recorded cost of supporting OFT.
"$VENV/bin/pip" install --quiet "git+https://github.com/moojink/openvla-oft.git" \
    || echo "[oft] declared-pin install of openvla-oft failed"
# The probe imports only torch, transformers and numpy beyond stdlib, so this is
# the whole extra surface it needs.
"$VENV/bin/pip" install --quiet "huggingface_hub>=0.23,<0.30" "pillow" \
    || echo "[oft] declared-pin probe deps failed"
"$VENV/bin/python" -c "import torch, transformers, sys;
print('[oft] oft_declared_pins python', sys.version.split()[0],
      'torch', torch.__version__, 'transformers', transformers.__version__)" || true
"$VENV/bin/python" -c "
import json
try:
    import importlib.metadata as md
    print('[oft] venv pins:', {p: md.version(p) for p in
          ('torch','torchvision','transformers','tokenizers','timm')})
except Exception as e:
    print('[oft] venv pin readout failed:', e)
" || true
"$VENV/bin/python" experiments/probe_openvla_oft_load.py \
    --out   "$OUT_DIR" \
    --stage oft_declared_pins \
    --dtype "${DTYPE:-bfloat16}" \
    ${OFT_REPOS:+--repos "$OFT_REPOS"} || echo "[oft] oft_declared_pins exited nonzero"

# ---- the job reads its own output -------------------------------------------
# A probe nobody reads is how two rollout defects in this paper survived: each
# exited 0 with a well-formed report. So the verdict is printed here, and a job
# that produced no report at all fails rather than passing quietly.
python - "$OUT_DIR" <<'PY' || exit 6
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
found = sorted(d.glob("oft_load_probe_*.json"))
# The declared-pin cell is outside the 2x2 BY CONSTRUCTION: it is the only cell
# that moves torch, so folding it into the cross-stage torch check below would
# make that check fire on the one stage whose whole purpose is to differ.
DECLARED = "oft_load_probe_oft_declared_pins.json"
grid = [p for p in found if p.name != DECLARED]
if not found:
    print("[FATAL] no stage wrote a report: this job measured nothing, "
          "which is worse than a failure it could have recorded.")
    sys.exit(1)
res = d / "prismatic_resolution.json"
if res.exists():
    r = json.loads(res.read_text())
    print(f"--- prismatic: importable={r.get('importable')} "
          f"via={r.get('resolved_by')} "
          f"transitive={[t['import'] for t in r.get('transitive_installs', [])]}")
    if r.get("pins_moved"):
        print(f"    WARNING pins moved during resolution: {r['pins_moved']}")
# What the install set looked like going into each stage. Round 4 had no such
# record, which is why two self-inflicted broken installs were read for a whole
# round as evidence about OFT.
for p in sorted(d.glob("pin_doctor_*.json")):
    c = json.loads(p.read_text()).get("consistency", {})
    fixed = [rep["spec"] for rnd in c.get("rounds", []) for rep in rnd["repairs"]]
    print(f"--- {p.stem}: clean={c.get('clean')} repaired={fixed}")
    for f in c.get("refused", []):
        print(f"    REFUSED (would move a paper pin): {f['holder']} "
              f"{f['holder_version']} wants {f['requirement']}")
    # Round 6. An unsatisfiable dist is a decision to make, not a round to run
    # again: round 5 spent its whole repair budget alternating between two
    # holders that can never both be satisfied.
    for u in c.get("unsatisfiable", []) or []:
        want = ", ".join(f"{w['holder']} wants {w['requirement']}"
                         for w in u["wanted_by"])
        print(f"    UNSATISFIABLE {u['dist']}: {want}")
    if c.get("oscillated"):
        print("    OSCILLATED: repairs were undoing each other; add the losing "
              "side to CONSTRAINT_LINES in bolt/install_prismatic.py")
    for f in c.get("remaining", []) or []:
        print(f"    UNRESOLVED: {f['holder']} wants {f['requirement']}")
# Did torch hold across the four 2x2 stages? Round 5's reports each looked fine
# on their own and only disagreed with each other, which no single report can
# show. `grid`, not `found`: stage 5 moves torch deliberately.
torches = {}
for p in grid:
    torches[p.stem] = (json.loads(p.read_text()).get("environment") or {}).get("torch")
if len(set(torches.values())) > 1:
    print(f"[FATAL-ISH] torch is not the same in every 2x2 stage: {torches}. The "
          f"2x2 compares cells that do not share a torch, so no cell's failure can "
          f"be attributed to the variable that cell was meant to vary.")
else:
    print(f"[oft] torch held across the 2x2 stages: {sorted(set(torches.values()))}")
# Round 7. The declared-pin cell is only worth reading if it actually got the
# torch OFT asks for -- if pip refused or the venv build failed, it is a second
# copy of the 2x2's torch and says nothing new. Reported either way; a stage that
# silently measured the wrong stack is exactly what round 5 did.
dp = d / DECLARED
if not dp.exists():
    print("[oft] stage 5 (oft_declared_pins) wrote NO report: the declared-pin "
          "environment could not be built or the probe died before writing. "
          "Limitation (ix) therefore still rests on our own pin only -- say so.")
else:
    e = (json.loads(dp.read_text()).get("environment") or {})
    gridt = sorted(set(torches.values()))
    print(f"--- oft_declared_pins: torch={e.get('torch')} "
          f"transformers={e.get('transformers')} (2x2 ran {gridt})")
    if e.get("torch") and e["torch"] in gridt:
        print("    INCONCLUSIVE: this cell ran the same torch as the 2x2, so it is "
              "not a test at OFT's declared pin. Check the install log above for a "
              "refused resolution before citing it.")
loaded_anywhere = False
for p in found:
    r = json.loads(p.read_text())
    env = r.get("environment", {})
    print(f"--- {p.name}: torch={env.get('torch')} "
          f"transformers={env.get('transformers')} "
          f"prismatic={env.get('prismatic')}")
    print(f"    existing={r.get('n_candidates_existing')}/"
          f"{r.get('n_candidates')} loaded={r.get('n_loaded')}"
          + (f" reexecs={r['reexec_count']}" if r.get("reexec_count") else ""))
    print(f"    verdict: {r.get('verdict')}")
    loaded_anywhere = loaded_anywhere or bool(r.get("any_loaded"))
    for repo, a in (r.get("load_attempts") or {}).items():
        print(f"    {repo}: stage={a.get('stage_reached')} ok={a.get('ok')} "
              f"{a.get('error_type', '')} {str(a.get('error', ''))[:160]}")
        # Round 3 added a retry loop at the load site (see try_load_resolving).
        # What it installed, and where it gave up, is the answer to "what would
        # supporting OFT cost" -- so it is printed, not just stored.
        inst = a.get("dependency_installs") or []
        if inst:
            print(f"      installed: "
                  f"{[(i['import'], i['ok']) for i in inst]}")
        if a.get("stuck_on"):
            print(f"      stuck_on: {a['stuck_on']} -- installing it did not "
                  f"clear the import, so this is the real blocker")
print(f"[oft] {len(found)}/5 stages reported ({len(grid)}/4 in the 2x2); "
      f"any_loaded={loaded_anywhere}")
if loaded_anywhere:
    print("[oft] ACTION REQUIRED: a checkpoint loaded. Limitation (ix) says OFT "
          "is unloadable in our environment and must be rewritten -- either the "
          "coverage gap closes, or it is re-scoped to the cost recorded in "
          "prismatic_resolution.json.")
else:
    print("[oft] limitation (ix) stands, now with an exception behind it. Cite "
          "this job id and the per-stage error_type.")
PY

echo "---- Done ----"
exit 0
