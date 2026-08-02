"""SharpGuard.

The public names below are resolved lazily (PEP 562) rather than imported at
package import time. They all live in modules that `import torch` at module
level, so an eager re-export meant that *any* `import sharpguard.<anything>`
paid for the whole GPU stack -- including `sharpguard.hf_retry`, which is
urllib-and-retry-loop and needs none of it.

That is not hypothetical: it killed the CPU-only edit-pair export
(bolt e8ge5734zc) with `ModuleNotFoundError: No module named 'torch'` on a pod
that had deliberately skipped bolt/setup.sh because nothing in the job touches
a GPU. Attribute access is where torch is actually needed, so attribute access
is where the import belongs.

`from sharpguard import epsilon_sharpness`, `import sharpguard` followed by
`sharpguard.measure_all(...)`, and `import sharpguard.hf_retry` all behave as
before; only the last one is now free.
"""
import importlib

# name -> submodule that defines it.
_EXPORTS = {
    "epsilon_sharpness": ".estimators.epsilon_sharpness",
    "lambda_max_power_iteration": ".estimators.power_iteration",
    "sam_perturbation_response": ".estimators.sam_response",
    "measure_global": ".measurement",
    "measure_sample_level": ".measurement",
    "measure_layerwise": ".measurement",
    "measure_all": ".measurement",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value          # resolve once, then it is a plain global
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
