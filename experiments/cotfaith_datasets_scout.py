"""Scout: additional dataset options for multi-dataset expansion.

Checks HF for other VLA-annotated datasets:
- Open-X-Embodiment shards (raw Bridge, RT-1, etc)
- RoboCasa demos
- CALVIN
- SimplerEnv reference tasks
- BridgeData V2 raw (no reasoning)
- Any other Embodied-CoT community-uploaded annotations
"""
import argparse, json, os, sys
from pathlib import Path
from huggingface_hub import HfApi


def probe_datasets():
    api = HfApi()
    candidates = [
        # Open-X-Embodiment
        "gresearch/rt-1-data-release",
        "openx-embodiment/bridge_orig",
        "openx-embodiment/rt_1",
        # Bridge V2 raw
        "yobibyte/bridge_v2",
        "IPEC-COMMUNITY/bridge_orig_lerobot",
        # CALVIN
        "IPEC-COMMUNITY/calvin_lerobot",
        "hf-vla-benchmark/calvin_a2b",
        # RoboCasa
        "amberxie/robocasa",
        "amberxie/robocasa_kitchen",
        # SimplerEnv reference
        "openai/simpler_env_tasks",
        # Other openvla-related
        "openvla/modified_libero_rlds",
        "moojink/libero_openvla_oft_lerobot",
        # Community CoT-annotated
        "OpenBMB/deepthinkvla_dataset",
        "OpenBMB/DeepThinkVLA_train",
        "OpenBMB/deepthinkvla-libero",
        "OpenBMB/deepthinkvla-bridge",
        "Weicheng-Gu1/embodied_features_and_demos_libero",  # variant of ECoT dataset
    ]
    results = {}
    for repo_id in candidates:
        try:
            info = api.dataset_info(repo_id)
            files = list(api.list_repo_files(repo_id, repo_type="dataset"))[:15]
            size_gb = "?"
            try:
                size_gb = f"{sum(getattr(f, 'size', 0) or 0 for f in info.siblings)/1e9:.2f}"
            except Exception:
                pass
            results[repo_id] = {"exists": True, "n_files_sample": len(files),
                                  "files_sample": files[:5], "size_gb_approx": size_gb}
            print(f"[ds] {repo_id}: FOUND ({size_gb} GB approx)")
        except Exception as e:
            msg = str(e)[:150]
            results[repo_id] = {"exists": False, "error": msg}
            print(f"[ds] {repo_id}: {msg[:80]}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    r = probe_datasets()
    (out / "datasets_scout.json").write_text(json.dumps(r, indent=2, default=str))
    print(f"\nsaved -> {out / 'datasets_scout.json'}")
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
