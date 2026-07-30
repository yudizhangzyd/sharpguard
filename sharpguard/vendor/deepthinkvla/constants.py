"""LIBERO action-space constants for DeepThinkVLA.

Upstream (src/sft/constants.py) picks these by scanning sys.argv for "libero" /
"aloha" / "bridge" and defaulting to LIBERO. That indirection is a hazard in our
harness -- our argv contains neither word, so we would silently inherit a
default rather than an asserted value. We pin the LIBERO row explicitly and
assert against config.json at load time instead.

LIBERO_CONSTANTS, verbatim from upstream:
    NUM_ACTIONS_CHUNK = 10
    ACTION_DIM        = 7
    normalization     = QUANTILE   (q01/q99 -> [-1, 1])
    ACTION_MASK       = [True]*6 + [False]   (gripper dim excluded from masking)
"""

NUM_ACTIONS_CHUNK = 10
ACTION_DIM = 7

# Upstream normalizes actions with the QUANTILE scheme for LIBERO: q01 -> -1,
# q99 -> +1, then clips. norm_stats.json in each checkpoint carries q01/q99.
ACTION_NORMALIZATION = "QUANTILE"
