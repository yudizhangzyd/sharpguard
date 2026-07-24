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
    # (report_key, hf_id, model_family, unnorm_key)
    # ---- ECoT family (all use OpenVLA-Prismatic backbone, transformers 4.40.x)
    ("ecot_openvla_bridge", "Embodied-CoT/ecot-openvla-7b-bridge",
     "ecot", "bridge_orig"),
    ("ecot_libero_spatial_r32", "leepanic/ecot-libero-spatial-r32",
     "ecot", "libero_spatial_no_noops"),
    ("ecot_libero_full_ft", "Jiahao-Wang/ecot-libero-full-finetune-10k-resume-step-040000",
     "ecot", "libero_spatial_no_noops"),
    # Try the OXE variant too — different training mix might parse LIBERO
    # scenes better than bridge-only. Also gives us N=4 in the ECoT family.
    ("ecot_openvla_oxe", "Embodied-CoT/ecot-openvla-7b-oxe",
     "ecot", "bridge_orig"),
    # ---- DeepThinkVLA: PaliGemma-based, needs transformers>=4.42.
    #      Our current env pins 4.40.1 for OpenVLA compat. Handled in a
    #      SEPARATE scout run with an upgraded env; skipped here.
]

# Canonical LIBERO-like probe input.
PROBE_INSTRUCTION = "pick up the mug and place it on the plate"


def _make_probe_image() -> np.ndarray:
    """Neutral gray 224x224x3 uint8 — good enough to test generation API."""
    rng = np.random.default_rng(0)
    img = (rng.integers(80, 180, size=(224, 224, 3), dtype=np.uint8))
    return img


def _try_load_ecot(hf_id: str, dtype, device):
    """ECoT / OpenVLA-family: AutoModelForVision2Seq with trust_remote_code.

    Some third-party ECoT LIBERO fine-tunes (e.g. Jiahao-Wang/*full-finetune*)
    were uploaded without a `model_type` key in config.json, so plain
    AutoModelForVision2Seq raises "Unrecognized model". We attempt a
    fallback that (a) downloads the config, patches `model_type='openvla'`,
    and (b) explicitly loads via AutoConfig + trust_remote_code.
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            hf_id, trust_remote_code=True, torch_dtype=dtype,
            attn_implementation="eager", low_cpu_mem_usage=True,
        ).to(device).eval()
    except (ValueError, KeyError) as e:
        if "Unrecognized model" not in str(e) and "model_type" not in str(e):
            raise
        # Patch: load config explicitly, force model_type='openvla', retry.
        cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
        if getattr(cfg, "model_type", None) in (None, ""):
            cfg.model_type = "openvla"
        model = AutoModelForVision2Seq.from_pretrained(
            hf_id, config=cfg, trust_remote_code=True, torch_dtype=dtype,
            attn_implementation="eager", low_cpu_mem_usage=True,
        ).to(device).eval()
    return processor, model


# ECoT prompt/tag conventions (verbatim from
# github.com/MichalZawalski/embodied-CoT Example.ipynb).
ECOT_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

ECOT_TAGS = [
    "TASK:", "PLAN:", "VISIBLE OBJECTS:", "SUBTASK REASONING:", "SUBTASK:",
    "MOVE REASONING:", "MOVE:", "GRIPPER POSITION:", "ACTION:",
]

# Fallback unnorm_key order for LIBERO-family fine-tunes that didn't
# re-save dataset_statistics.json. ECoT/OpenVLA's `bridge_orig` is
# universally present since the base checkpoint ships with OXE stats.
UNNORM_FALLBACKS = ["bridge_orig", "austin_buds_dataset_converted_externally_to_rlds"]


def _run_ecot(processor, model, image, instruction, device, dtype,
              unnorm_key: str, max_new_tokens: int = 1024) -> dict:
    """Autoregressive CoT + action via ECoT's predict_action(). Uses the
    OFFICIAL prompt template (trailing ' TASK:' primes the CoT), positional
    processor call, and bf16 cast on inputs. Passing plain generate() with
    kwargs (as previous scout did) makes the model EOS in ~5 tokens with no
    reasoning."""
    import torch
    from PIL import Image as PILImage
    pil = PILImage.fromarray(image).convert("RGB")

    # Trailing " TASK:" is MANDATORY — primes the model to emit CoT tags.
    prompt = (f"{ECOT_SYSTEM_PROMPT} USER: What action should the robot take to "
              f"{instruction.lower()}? ASSISTANT: TASK:")

    # POSITIONAL args (prompt, image) — kwargs form skips image-token injection.
    inputs = processor(prompt, pil).to(device, dtype=dtype)

    action_arr = None
    generated_ids = None
    unnorm_key_used = None

    # Try the requested unnorm_key first, then fall back to universally-present
    # OXE keys. LIBERO LoRA fine-tunes commonly don't re-save LIBERO stats.
    candidate_keys = [unnorm_key] if unnorm_key else []
    candidate_keys += [k for k in UNNORM_FALLBACKS if k not in candidate_keys]
    last_err = None
    for k in candidate_keys:
        try:
            action_arr, generated_ids = model.predict_action(
                **inputs,
                unnorm_key=k,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
            unnorm_key_used = k
            break
        except (ValueError, KeyError) as e:
            last_err = e
            continue

    if generated_ids is None:
        # Every unnorm_key failed. Fall back to raw generate() and skip action
        # de-normalization — we still get the CoT text for faithfulness study.
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False)
        generated_ids = out
        unnorm_key_used = f"NO_UNNORM (last_err={str(last_err)[:120]})"

    text = processor.batch_decode(generated_ids)[0]

    # Parse: find CoT segment ending at ACTION: (or end of text).
    tag_hits = [t for t in ECOT_TAGS if t in text]
    if "ACTION:" in text:
        cot_end = text.rfind("ACTION:")
        cot_snippet = text[:cot_end]
    else:
        cot_snippet = text

    action_list = None
    if action_arr is not None:
        try:
            action_list = np.asarray(action_arr, dtype=np.float32) \
                .reshape(-1).tolist()
        except Exception:
            pass

    return {
        "raw_text_tail": text[-1200:],
        "cot_snippet": cot_snippet[-800:] if cot_snippet else "",
        "action": action_list,
        "cot_tags_found": tag_hits,
        "unnorm_key_used": unnorm_key_used,
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


def scout_one(report_key: str, hf_id: str, family: str, unnorm_key,
                image, instruction,
                device, dtype, timeout_sec: int) -> dict:
    """Run scout for one model. Returns per-model report dict."""
    entry = {
        "hf_id": hf_id,
        "family": family,
        "unnorm_key": unnorm_key,
        "loadable": False,
        "produced_cot": False,
        "produced_action": False,
        "cot_delim_convention": None,
        "example_cot_snippet": None,
        "example_action": None,
        "raw_text_tail": None,
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
            result = _run_ecot(processor, model, image, instruction, device, dtype,
                                unnorm_key=unnorm_key)
        elif family == "deepthink":
            result = _run_deepthink(tok, processor, model, image, instruction,
                                     device, dtype)
        entry["generate_seconds"] = round(time.time() - t_gen, 1)
        entry["example_cot_snippet"] = result.get("cot_snippet")
        entry["example_action"] = result.get("action")
        entry["raw_text_tail"] = result.get("raw_text_tail")
        entry["unnorm_key_used"] = result.get("unnorm_key_used")
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

    for report_key, hf_id, family, unnorm_key in models:
        print(f"\n===== {report_key}  ({hf_id})  family={family}  unnorm={unnorm_key} =====")
        entry = scout_one(report_key, hf_id, family, unnorm_key,
                            image, PROBE_INSTRUCTION,
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
            print(f"  CoT snippet: {entry['example_cot_snippet'][:400]!r}")
        if entry.get("example_action"):
            print(f"  action: {entry['example_action']}")
        if entry.get("raw_text_tail"):
            print(f"  raw text tail (last 300 chars): {entry['raw_text_tail'][-300:]!r}")

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
