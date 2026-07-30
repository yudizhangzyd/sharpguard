"""Regression test for `_load_libero_init_states`.

Why this test exists. The decoder-validation gate (reviewer critical #2) read
Task SR = 0.08 on libero_goal and 0.00 on libero_object against a published
~85%, and both runs exited 0 and wrote a well-formed sr.json. The cause was not
the policy: LIBERO's `.pruned_init` files are PyTorch zip archives
(`archive/data.pkl` + `archive/version`), `np.load` opens the zip and returns
raw `bytes` for those entries, and the loader's `.tobytes()` call therefore
raised on every task of every suite. The loader then returned None and the
caller fell back to `env.reset()`, which places the objects at a random
initial state rather than the suite's canonical evaluation one.

So the failure mode being tested against is not "the loader is wrong" but "the
loader is wrong AND the run still looks finished". Test 4 is the important one:
an unparseable file must raise. Run this before the rollout, not after.
"""

import importlib.util
import os
import tempfile

import numpy as np
import torch

_SPEC = importlib.util.spec_from_file_location(
    "libero_sim_under_test",
    os.path.join(os.path.dirname(__file__), "..", "sharpguard", "libero_sim.py"),
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
load = _MOD._load_libero_init_states


def main() -> int:
    d = tempfile.mkdtemp()
    fails = []

    def ok(name, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    # 1. The actual on-disk format: torch.save of a (n_episodes, state_dim)
    #    tensor. This is what LIBERO ships and what the old code could not read.
    p = os.path.join(d, "t.pruned_init")
    torch.save(torch.randn(50, 79), p)
    a = load(p)
    ok("torch.save'd 2D tensor loads with the right shape",
        a is not None and a.shape == (50, 79),
        f"got {None if a is None else a.shape}")

    # 2. Some suites store a list of per-episode 1-D tensors instead.
    p = os.path.join(d, "l.pruned_init")
    torch.save([torch.randn(79) for _ in range(50)], p)
    a = load(p)
    ok("torch.save'd list of 1D tensors stacks to (50, 79)",
        a is not None and a.shape == (50, 79),
        f"got {None if a is None else a.shape}")

    # 3. The numpy path still works, so older suites are not regressed.
    p = os.path.join(d, "n.npz")
    np.savez(p, states=np.zeros((50, 79)))
    a = load(p)
    ok("npz with a 'states' key still loads",
        a is not None and a.shape == (50, 79),
        f"got {None if a is None else a.shape}")

    # 4. THE one that matters: an unreadable file must be fatal. If this ever
    #    returns None again, the gate can report a number it did not measure.
    p = os.path.join(d, "junk.pruned_init")
    with open(p, "wb") as fh:
        fh.write(b"not a checkpoint")
    try:
        load(p)
        ok("an unparseable init_states file RAISES rather than falling back",
           False, "it returned instead of raising")
    except RuntimeError as e:
        ok("an unparseable init_states file RAISES rather than falling back",
           True, str(e)[:70] + "...")

    # 5. A genuinely absent file is a different fact and stays non-fatal.
    a = load(os.path.join(d, "does_not_exist.pruned_init"))
    ok("a missing file returns None (suites that ship none use env.reset())",
       a is None, f"got {a!r}")

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("init_states loader: all 5 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
