"""Scout: does Embodied-CoT/embodied_features_and_demos_bridge (or similar
Bridge-annotated dataset) exist on HF? Also try DeepThinkVLA load with
upgraded transformers (paligemma requires 4.42+).
"""
import argparse, json, os, sys, traceback
from pathlib import Path
import numpy as np

def probe_bridge_dataset():
    """Try snapshot_download on Bridge-annotated ECoT dataset variants."""
    from huggingface_hub import HfApi
    api = HfApi()
    candidates = [
        "Embodied-CoT/embodied_features_and_demos_bridge",
        "Embodied-CoT/embodied_features_and_demos",
        "Embodied-CoT/embodied-CoT-bridge",
        "Embodied-CoT/bridge_reasoning",
    ]
    results = {}
    for repo_id in candidates:
        try:
            info = api.dataset_info(repo_id)
            files = list(api.list_repo_files(repo_id, repo_type="dataset"))[:20]
            results[repo_id] = {"exists": True, "files_sample": files}
            print(f"[bridge] FOUND {repo_id}: {len(files)} files, sample: {files[:3]}")
        except Exception as e:
            results[repo_id] = {"exists": False, "error": str(e)[:200]}
            print(f"[bridge] NOT FOUND {repo_id}: {str(e)[:100]}")
    return results


def probe_deepthinkvla():
    """Attempt to load DeepThinkVLA. Requires transformers >= 4.42 for PaliGemma."""
    # Try HF paths from earlier scout
    candidates = [
        "yinchenghust/deepthinkvla_libero_cot_rl",
        "yinchenghust/deepthinkvla_libero_cot_sft",
        "yinchenghust/deepthinkvla_base",
    ]
    results = {}
    for hf_id in candidates:
        try:
            from transformers import AutoModel, AutoTokenizer, AutoConfig
            # try config first (lightest)
            cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
            results[hf_id] = {"loadable": True,
                                "model_type": getattr(cfg, "model_type", "?"),
                                "arch": str(getattr(cfg, "architectures", "?"))[:100]}
            print(f"[deepthink] {hf_id}: CONFIG OK ({cfg.model_type})")
        except Exception as e:
            results[hf_id] = {"loadable": False, "error": str(e)[:200]}
            print(f"[deepthink] {hf_id}: FAIL — {str(e)[:150]}")
    return results


def probe_other_cotvlas():
    """Any other CoT-VLA on HF worth trying?"""
    candidates = [
        # Fast ECoT is by Duan et al 2506.07639
        "Duan-lab/Fast-ECoT",   # guess
        "Duan-lab/fast-ecot",
        # OpenVLA-OFT (Kim 2025)
        "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "moojink/openvla-7b-oft-finetuned-libero-object",
        "moojink/openvla-7b-oft-finetuned-libero-goal",
        "moojink/openvla-7b-oft-finetuned-libero-10",
        # ThinkingVLA
        "ThinkingVLA/thinking_vla",
        # ZR-0 (mentioned as ModelScope only earlier)
        "RUCKBReasoning/ZR-0",
        "seeklhy/ZR-0",
    ]
    results = {}
    for hf_id in candidates:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
            results[hf_id] = {"exists": True,
                                "model_type": getattr(cfg, "model_type", "?")}
            print(f"[other] {hf_id}: OK ({cfg.model_type})")
        except Exception as e:
            msg = str(e)[:150]
            results[hf_id] = {"exists": False, "error": msg}
            print(f"[other] {hf_id}: FAIL — {msg[:100]}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    report = {}
    report["bridge_dataset"] = probe_bridge_dataset()
    print()
    try:
        report["deepthinkvla"] = probe_deepthinkvla()
    except Exception as e:
        report["deepthinkvla"] = {"exception": str(e)}
    print()
    try:
        report["other_cotvlas"] = probe_other_cotvlas()
    except Exception as e:
        report["other_cotvlas"] = {"exception": str(e)}

    (out / "scout_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n===== SAVED -> {out / 'scout_report.json'} =====")
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
