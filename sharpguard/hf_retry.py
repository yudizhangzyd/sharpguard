"""Snapshot-level retry for Hub downloads that get rate limited.

Every Bolt config carries `HF_TOKEN: ${HF_TOKEN}` in `environment_variables`,
and Bolt does not expand `${VAR}` from the submitting shell -- the pods receive
the literal ten-character string. So every dataset pull this project has ever
done was **anonymous**, which worked only because the Hub's per-IP allowance was
never the binding constraint. It became one when eleven jobs ran at once:

    429 Client Error: Too Many Requests for url:
    .../embodied_features_and_demos_libero/resolve/.../
        libero_lm_90_openpi-train.tfrecord-00049-of-00128
    We had to rate limit your IP (18.246.171.23).

which killed `zbis2kbuin`, `gzst3cumj7`, `ukdgriqetw` and `bq7fsf5hpz` between
files 300 and 320 of a 392-file snapshot, ~40 minutes into each job. The 429
lands on a file's metadata HEAD, where `huggingface_hub` re-raises it as
`LocalEntryNotFoundError` ("check your connection") -- a misleading message for
a pod that had just pulled 857 MB of torch wheels at 15 MB/s.

`snapshot_download` retries an interrupted *transfer* but not a throttled
metadata call, so one 429 aborts the whole snapshot. The retry here is at the
snapshot level, which is the cheap place to put it: files already fetched are in
the cache, so each attempt resumes and only the throttled tail is re-requested.
`max_workers` is also lowered from the default 8, since eight concurrent
metadata calls per job is what tripped the limiter in the first place.
"""

from __future__ import annotations

import os
import sys
import time

# Roughly 15 minutes of total patience. The Hub's anonymous window is per-minute,
# so the first sleep alone clears an isolated burst; the long tail is for the
# case where several of our own jobs are contending for the same egress IP.
_RETRY_SLEEPS = (30, 60, 120, 240, 480)

# Anything the Hub raises when it is throttling us, including the
# LocalEntryNotFoundError it converts a metadata 429 into.
_THROTTLE_MARKERS = ("429", "too many requests", "rate limit",
                     "we cannot find the requested files in the local cache")


def _is_throttle(exc) -> bool:
    return any(m in str(exc).lower() for m in _THROTTLE_MARKERS)


def _drop_placeholder_token() -> None:
    """An unexpanded `${HF_TOKEN}` is worse than no token.

    `huggingface_hub` will happily send it as a Bearer credential, and a
    malformed credential is a 401 rather than the anonymous path that does
    work. Deleting it makes the anonymous download explicit instead of
    accidental.
    """
    tok = os.environ.get("HF_TOKEN", "")
    if tok.startswith("${") or (tok and not tok.startswith("hf_")):
        os.environ.pop("HF_TOKEN", None)
        print("[hf] HF_TOKEN was not expanded by the submitter (%r); "
              "downloading anonymously" % tok[:16], file=sys.stderr, flush=True)


def snapshot_with_retry(**kw):
    """`snapshot_download(**kw)`, retried through Hub rate limiting.

    Defaults `cache_dir` to `$HF_HOME` (what every call site here passed) and
    `max_workers` to 4. Non-throttle errors propagate unchanged: a missing repo
    or a bad revision must still fail immediately and loudly.
    """
    from huggingface_hub import snapshot_download
    _drop_placeholder_token()
    kw.setdefault("cache_dir", os.environ.get("HF_HOME"))
    kw.setdefault("max_workers", 4)
    last = None
    for attempt, nap in enumerate((0,) + _RETRY_SLEEPS):
        if nap:
            print("[hf] rate limited on %s; sleeping %ds before attempt %d/%d"
                  % (kw.get("repo_id"), nap, attempt + 1,
                     len(_RETRY_SLEEPS) + 1), file=sys.stderr, flush=True)
            time.sleep(nap)
        try:
            return snapshot_download(**kw)
        except Exception as e:                      # noqa: BLE001
            if not _is_throttle(e):
                raise
            last = e
    raise RuntimeError(
        "Hub rate limited every attempt for %s over %ds. This job downloads "
        "anonymously because Bolt passes the literal '${HF_TOKEN}'; either run "
        "fewer jobs concurrently or render a real token into the config the way "
        "bolt/submit_edit13.sh renders the S3 ones."
        % (kw.get("repo_id"), sum(_RETRY_SLEEPS))) from last
