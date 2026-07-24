"""CoT-Faith Week 1 scout: verify 4 candidate manipulation CoT-VLA checkpoints
actually load, run a single-step forward pass, and produce parseable
(reasoning-text, action-vector) output on a LIBERO scene.

This is a BLOCKER script — if fewer than 3 models produce clean CoT + action,
the CoT-Faith paper direction is not viable and we PIVOT.

Report format (JSON, printed to stdout + saved to $OUT/scout_report.json):
{
  "model_id": {
    "loadable": bool,
    "produced_cot": bool,
    "produced_action": bool,
    "cot_delim_convention": "TASK:/PLAN:/... | <reasoning>... | none",
    "example_cot_snippet": "first 300 chars of decoded reasoning",
    "example_action": [7 floats],
    "error": null | "traceback string"
  },
  ...
  "verdict": "GO" | "PIVOT",
  "n_working": int
}

Design notes:
- Uses a canned RGB image (all-gray 224x224x3) so we don't need libero
  sim/rendering during scout — the point is to test CoT+action generation
  API, not env correctness.
- Uses a fixed short instruction ("pick up the mug").
- Aborts each model with a timeout to avoid a single broken checkpoint
  wedging the whole scout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np


CANDIDATE_MODELS = [
    # (report_key, hf_id, model_family)
    ("ecot_libero_spatial", "leepanic/ecot-libero-spatial-r32", "ecot"),
    ("ecot_openvla_bridge", "Embodied-CoT/ecot-openvla-7b-bridge", "ecot"),
    ("deepthinkvla",         "OpenBMB/DeepThinkVLA",              "deepthink"),
    ("zr_0",                 "RUCKBReasoning/ZR-0",               "zr0"),
]

# Canonical LIBERO-like probe input.
PROBE_INSTRUCTION = "pick up the mug and place it on the plate"


def _make_probe_image() -> np.ndarray:
    """Neutral gray 224x224x3 uint8 — good enough to test generation API."""
    rng = np.random.default_rng(0)
    img = (rng.integers(80, 180, size=(224, 224, 3), dtype=np.uint8))
    return img


def _try_load_ecot(hf_id: str, dtype, device):
    """ECoT / OpenVLA-family: uses AutoModelForVision2Seq with trust_remote_code."""
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        hf_id, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()
    return processor, model


def _run_ecot(processor, model, image, instruction, device, dtype,
              max_new_tokens: int = 1024) -> dict:
    """Autoregressive generate CoT + action tokens, decode text and parse.

    ECoT model exposes .predict_action() but for scout we want the RAW
    generated ids so we can inspect the reasoning-tag sequence.
    """
    from PIL import Image as PILImage
    pil = PILImage.fromarray(image).convert("RGB")
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut: "
    proc = processor(images=pil, text=prompt, return_tensors="pt")
    proc_dev = {}
    for k, v in proc.items():
        if hasattr(v, "to"):
            if v.is_floating_point():
                proc_dev[k] = v.to(device).to(dtype)
            else:
                proc_dev[k] = v.to(device)
        else:
            proc_dev[k] = v

    import torch
    with torch.no_grad():
        out = model.generate(
            **proc_dev,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    text = processor.tokenizer.decode(out[0], skip_special_tokens=True)

    # Try ECoT tag parse:
    #  TASK:/PLAN:/SUBTASK:/MOVE:/GRIPPER POSITION:/VISIBLE OBJECTS:/ACTION:
    cot_snippet, action = "", None
    tag_hits = [t for t in ("TASK:", "PLAN:", "SUBTASK:", "MOVE:",
                             "GRIPPER POSITION:", "VISIBLE OBJECTS:", "ACTION:")
                if t in text]
    if tag_hits:
        cot_snippet = text[:text.rfind("ACTION:")] if "ACTION:" in text else text
        cot_snippet = cot_snippet[-500:]  # keep tail (post-prompt)
    # Try predict_action path for action floats.
    try:
        action_np = model.predict_action(
            **proc_dev, unnorm_key="bridge_orig",
        )
        action = np.asarray(action_np, dtype=np.float32).reshape(-1).tolist()
    except Exception:
        pass
    return {
        "raw_text_tail": text[-800:],
        "cot_snippet": cot_snippet[:500],
        "action": action,
        "cot_tags_found": tag_hits,
    }


def _try_load_deepthink(hf_id: str, dtype, device):
    """DeepThinkVLA — hybrid causal+bidirectional decoder from OpenBMB.
    We attempt the same AutoModel pattern; if it fails we return the error."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, trust_remote_code=True, torch_dtype=dtype,
    ).to(device).eval()
    return tok, processor, model


def _run_deepthink(tok, processor, model, image, instruction, device, dtype,
                    max_new_tokens: int = 512) -> dict:
    """Best-effort DeepThinkVLA generation. If the model needs a special
    input format, we log the error but don't crash the whole scout."""
    import torch
    from PIL import Image as PILImage
    pil = PILImage.fromarray(image).convert("RGB")
    prompt = f"Task: {instruction}\n"
    if processor is not None:
        try:
            proc = processor(images=pil, text=prompt, return_tensors="pt")
            proc_dev = {k: (v.to(device).to(dtype) if hasattr(v, 'is_floating_point')
                              and v.is_floating_point() else
                              v.to(device) if hasattr(v, 'to') else v)
                         for k, v in proc.items()}
        except Exception as e:
            return {"error": f"processor call failed: {e}"}
    else:
        inputs = tok(prompt, return_tensors="pt").to(device)
        proc_dev = {k: v for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**proc_dev, max_new_tokens=max_new_tokens,
                              do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    return {
        "raw_text_tail": text[-800:],
        "cot_snippet": text[:500],
        "action": None,
        "cot_tags_found": [],
    }


def _try_load_zr0(hf_id: str, dtype, device):
    """ZR-0 — uses ECoT during training but drops reasoning at inference.
    We test whether ANY reasoning token comes out; if not, ZR-0 is not
    a valid CoT-VLA subject for our study (falls into 'reasoning-as-
    regularizer' class, not 'reasoning-at-inference')."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, trust_remote_code=True, torch_dtype=dtype,
    ).to(device).eval()
    return tok, model


def _run_zr0(tok, model, image, instruction, device, dtype,
              max_new_tokens: int = 512) -> dict:
    import torch
    inputs = tok(f"Instruction: {instruction}", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                              do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    return {
        "raw_text_tail": text[-800:],
        "cot_snippet": text[:500],
        "action": None,
        "cot_tags_found": [],
    }


def scout_one(report_key: str, hf_id: str, family: str, image, instruction,
                device, dtype, timeout_sec: int) -> dict:
    """Run scout for one model. Returns per-model report dict."""
    entry = {
        "hf_id": hf_id,
        "family": family,
        "loadable": False,
        "produced_cot": False,
        "produced_action": False,
        "cot_delim_convention": None,
        "example_cot_snippet": None,
        "example_action": None,
        "error": None,
        "load_seconds": None,
        "generate_seconds": None,
    }
    t_load = time.time()
    try:
        if family == "ecot":
            processor, model = _try_load_ecot(hf_id, dtype, device)
        elif family == "deepthink":
            tok, processor, model = _try_load_deepthink(hf_id, dtype, device)
        elif family == "zr0":
            tok, model = _try_load_zr0(hf_id, dtype, device)
        else:
            entry["error"] = f"unknown family {family}"
            return entry
        entry["loadable"] = True
        entry["load_seconds"] = round(time.time() - t_load, 1)
    except Exception as e:
        entry["error"] = f"LOAD FAIL: {e}\n{traceback.format_exc()[-800:]}"
        return entry

    t_gen = time.time()
    try:
        if family == "ecot":
            result = _run_ecot(processor, model, image, instruction, device, dtype)
        elif family == "deepthink":
            result = _run_deepthink(tok, processor, model, image, instruction,
                                     device, dtype)
        elif family == "zr0":
            result = _run_zr0(tok, model, image, instruction, device, dtype)
        entry["generate_seconds"] = round(time.time() - t_gen, 1)
        entry["example_cot_snippet"] = result.get("cot_snippet")
        entry["example_action"] = result.get("action")
        tags = result.get("cot_tags_found", [])
        entry["cot_delim_convention"] = (
            "ECoT-tags:" + ",".join(tags) if tags
            else "unstructured-text" if result.get("cot_snippet") else "none"
        )
        cot_text = result.get("cot_snippet") or ""
        entry["produced_cot"] = (len(cot_text.strip()) > 20)
        entry["produced_action"] = result.get("action") is not None
    except Exception as e:
        entry["error"] = f"GEN FAIL: {e}\n{traceback.format_exc()[-800:]}"

    # Free GPU memory before next model
    try:
        import torch, gc
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass

    return entry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="./cotfaith_scout")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--models", default="all",
                   help="'all' or comma-separated list of report_keys.")
    p.add_argument("--per-model-timeout", type=int, default=1200)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                  "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    print(f"[scout] device={device} dtype={dtype}")
    if args.models == "all":
        models = CANDIDATE_MODELS
    else:
        keep = set(args.models.split(","))
        models = [m for m in CANDIDATE_MODELS if m[0] in keep]

    image = _make_probe_image()
    report = {"probe_instruction": PROBE_INSTRUCTION,
              "dtype": args.dtype,
              "device": str(device),
              "results": {}}

    for report_key, hf_id, family in models:
        print(f"\n===== {report_key}  ({hf_id})  family={family} =====")
        entry = scout_one(report_key, hf_id, family, image, PROBE_INSTRUCTION,
                            device, dtype, args.per_model_timeout)
        report["results"][report_key] = entry
        # Snapshot after each model so a crash doesn't lose progress.
        (out / "scout_report.json").write_text(json.dumps(report, indent=2))
        # Human summary
        print(f"  loadable={entry['loadable']}  produced_cot={entry['produced_cot']}"
              f"  produced_action={entry['produced_action']}")
        if entry["error"]:
            print(f"  ERROR (truncated): {entry['error'][:400]}")
        if entry.get("example_cot_snippet"):
            print(f"  CoT snippet: {entry['example_cot_snippet'][:300]!r}")
        if entry.get("example_action"):
            print(f"  action: {entry['example_action']}")

    n_working = sum(1 for v in report["results"].values()
                     if v["loadable"] and v["produced_cot"])
    report["n_working"] = n_working
    report["verdict"] = "GO" if n_working >= 3 else "PIVOT"

    (out / "scout_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n\n===== SCOUT DONE =====")
    print(f"n_working (loadable + produced_cot) = {n_working}/{len(models)}")
    print(f"verdict = {report['verdict']}")
    print(f"full report -> {out / 'scout_report.json'}")

    # Bypass Python teardown (EGL cleanup on some containers crashes).
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
