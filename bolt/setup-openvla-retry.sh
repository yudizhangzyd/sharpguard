#!/usr/bin/env bash
# Retry wrapper around bolt/setup-openvla.sh.
#
# Across 7+ submissions of the DT-RL self-gen-edit job, setup-openvla.sh's
# own torch/CUDA reinstall has stalled at three different points (apt-get,
# a mid-size nvidia-*-cu11 wheel, the main 857.6MB torch wheel from
# download-r2.pytorch.org) on different attempts -- generic large-download
# flakiness on whatever node gets assigned, not a bug in this repo's own
# code (that one, a missing tensorflow_datasets install, was found and
# fixed separately in run_cotfaith_dt_selfgen_edit.sh). Cancelling the
# whole bolt task and resubmitting works but re-runs setup from a cold
# apt/pip cache every time. This wraps the SAME setup script in a bounded
# retry loop within one task instead: apt and pip are both idempotent
# (a package already installed or partially cached is a fast no-op or a
# resumed download on the next attempt), so retrying in place is strictly
# cheaper than a fresh task submission once the first attempt has made any
# progress at all.
set -u
cd "$(dirname "$0")/.."

MAX_ATTEMPTS="${SETUP_RETRY_MAX_ATTEMPTS:-4}"
PER_ATTEMPT_TIMEOUT="${SETUP_RETRY_TIMEOUT_S:-2700}"  # 45 min/attempt

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "[setup-retry] attempt $attempt/$MAX_ATTEMPTS (timeout ${PER_ATTEMPT_TIMEOUT}s)"
    if timeout "$PER_ATTEMPT_TIMEOUT" bash bolt/setup-openvla.sh; then
        echo "[setup-retry] succeeded on attempt $attempt"
        exit 0
    fi
    rc=$?
    echo "[setup-retry] attempt $attempt failed or timed out (rc=$rc)"
    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
        echo "[setup-retry] retrying in place (apt/pip caches persist within this task)"
    fi
done

echo "[setup-retry] all $MAX_ATTEMPTS attempts failed; giving up"
exit 1
