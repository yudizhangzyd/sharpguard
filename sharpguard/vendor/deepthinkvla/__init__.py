"""Vendored DeepThinkVLA model class plus the decode helpers our harness needs.

Import through `import_deepthinkvla()` rather than importing the module
directly: it checks the transformers pin first and raises a message that names
the fix, because the failure mode otherwise is an ImportError on a private
transformers symbol that says nothing about why.
"""
from __future__ import annotations

from .constants import ACTION_DIM, ACTION_NORMALIZATION, NUM_ACTIONS_CHUNK

# Special tokens, from the checkpoint's added_tokens.json. Hard-coded because
# the prompt assembly below has to emit them in a fixed order and a silent
# reordering upstream would be worse than a loud mismatch.
IMAGE_TOKEN = 257152
THINK_START = 257153
THINK_END = 257154
ACTION_START = 257155
ACTION_END = 257156

# Upstream's evaluation prompt prefix (src/experiments/deepthinkvla_utils.py).
# The model is trained with this exact sentence; changing it changes the
# measurement.
THINK_PREFIX = ("First output the thinking process in <think></think> tags and "
                "then output the final action in <action></action>.")

# The two ids upstream's model class looks for to locate the end of the prompt
# (`self.prompt_end_token_id`): ";" followed by the "\n" the PaliGemma processor
# appends. Everything after them is CoT-or-action, which is what makes the
# 4-bucket attention split well-defined for this architecture.
PROMPT_END_TOKEN_IDS = [235289, 108]

REQUIRED_TRANSFORMERS = "4.48"

__all__ = [
    "ACTION_DIM", "ACTION_NORMALIZATION", "NUM_ACTIONS_CHUNK",
    "IMAGE_TOKEN", "THINK_START", "THINK_END", "ACTION_START", "ACTION_END",
    "THINK_PREFIX", "PROMPT_END_TOKEN_IDS", "REQUIRED_TRANSFORMERS",
    "import_deepthinkvla",
]


def import_deepthinkvla():
    """Return the vendored DeepThinkVLA class, or raise explaining the pin.

    The vendored file targets the transformers version upstream's config.json
    records (4.48.1). Under 4.5x the names it imports from
    `transformers.models.paligemma.modeling_paligemma` no longer exist, and
    `_update_causal_mask` -- which is what implements the bidirectional action
    block -- has been removed. A bare ImportError on `PALIGEMMA_INPUTS_DOCSTRING`
    does not tell anyone that, so we say it here.
    """
    import transformers

    ver = transformers.__version__
    if not ver.startswith(REQUIRED_TRANSFORMERS):
        raise RuntimeError(
            f"DeepThinkVLA needs transformers=={REQUIRED_TRANSFORMERS}.x "
            f"(the checkpoint's config.json records 4.48.1); this environment "
            f"has {ver}. The vendored class calls PaliGemma internals "
            f"(_update_causal_mask, PALIGEMMA_INPUTS_DOCSTRING) that 4.5x "
            f"removed, and the bidirectional action-block mask is exactly what "
            f"those internals implement -- so running on the wrong pin would "
            f"either crash or silently decode actions under a causal mask. Pin "
            f"it in bolt/run_cotfaith_deepthink.sh.")

    from .modeling_deepthinkvla import DeepThinkVLA

    return DeepThinkVLA
