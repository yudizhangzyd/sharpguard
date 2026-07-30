#!/usr/bin/env python
"""Fail a job at setup time if the assigned GPU cannot actually run kernels.

Why this exists: the aws_10 pool is p6-b200 (NVIDIA B200, sm_100) while the
docker image's PyTorch is built for sm_50..sm_90. Every CUDA op on that pool
raises "no kernel image is available for execution on the device". Our probe
scripts catch per-sample exceptions so that one bad observation cannot kill a
100-sample run -- which meant a whole job on the wrong arch logged 100 caught
exceptions, wrote an empty report, and exited COMPLETED. Three DeepThinkVLA
edit jobs and three decoder-gate jobs were lost that way.

An arch mismatch is an environment fault, not a measurement, so it belongs in
setup where it fails the task loudly and immediately. Run this from
setup_command; a non-zero exit marks the task FAILED before any GPU-hours are
spent pretending to measure something.
"""
import sys

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("[preflight] FAIL: torch.cuda.is_available() is False")
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    print(f"[preflight] device      : {name}")
    print(f"[preflight] capability  : sm_{cap[0]}{cap[1]}")
    print(f"[preflight] torch        : {torch.__version__}")
    print(f"[preflight] torch archs : {' '.join(archs)}")

    # Do not trust the arch list alone -- a real op is the only proof. bf16
    # matmul is what every probe in this repo actually executes.
    try:
        a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        out = (a @ a).float().sum().item()
    except Exception as e:  # noqa: BLE001 -- any failure is disqualifying
        print(f"[preflight] FAIL: bf16 matmul on {name} raised "
              f"{type(e).__name__}: {e}")
        print(f"[preflight] this PyTorch supports {' '.join(archs)} but the "
              f"device is sm_{cap[0]}{cap[1]}. Submit to a cluster whose GPUs "
              f"this build covers (aws_2 p4de / aws_6 p4d are sm_80), or "
              f"install a PyTorch built for sm_{cap[0]}{cap[1]}.")
        return 1

    if out != out:  # NaN
        print("[preflight] FAIL: bf16 matmul returned NaN")
        return 1

    print("[preflight] OK: bf16 matmul executed on device")
    return 0


if __name__ == "__main__":
    sys.exit(main())
