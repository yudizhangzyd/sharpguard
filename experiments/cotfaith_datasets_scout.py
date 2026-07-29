"""MEGA-Scout: probe 60+ HF dataset repositories to find at least 5 usable
VLA-compatible datasets for CoT-Faith cross-dataset evaluation.
"""
import argparse, json, os, sys
from pathlib import Path
from huggingface_hub import HfApi


def probe_datasets():
    api = HfApi()
    candidates = [
        # ==== Bridge V2 variants ====
        "IPEC-COMMUNITY/bridge_orig_lerobot",
        "IPEC-COMMUNITY/bridge_lerobot",
        "yobibyte/bridge_v2",
        "kdharmarajan/bridgev2_lerobot",
        "lerobot/bridge",
        # ==== LIBERO variants (baselines + reasoning-annotated) ====
        "openvla/modified_libero_rlds",
        "Embodied-CoT/embodied_features_and_demos_libero",
        "IPEC-COMMUNITY/libero_lerobot",
        "IPEC-COMMUNITY/libero_spatial_no_noops_lerobot",
        "IPEC-COMMUNITY/libero_object_no_noops_lerobot",
        "IPEC-COMMUNITY/libero_goal_no_noops_lerobot",
        "IPEC-COMMUNITY/libero_10_no_noops_lerobot",
        # ==== CALVIN ====
        "IPEC-COMMUNITY/calvin_lerobot",
        "oier-mees/calvin",
        "lerobot/calvin_debug_dataset",
        "lerobot/calvin",
        # ==== RT-1 / Fractal ====
        "gresearch/rt-1-data-release",
        "IPEC-COMMUNITY/fractal20220817_data_lerobot",
        "lerobot/rt1",
        # ==== RoboCasa ====
        "amberxie/robocasa",
        "IPEC-COMMUNITY/robocasa_lerobot",
        "lerobot/robocasa",
        # ==== SimplerEnv ====
        "openai/simpler_env_tasks",
        "IPEC-COMMUNITY/simplerenv_lerobot",
        # ==== Physical Intelligence π₀ datasets ====
        "physical-intelligence/aloha_transfer_cube",
        "physical-intelligence/aloha_sim",
        # ==== Aloha / bimanual ====
        "lerobot/aloha_sim_transfer_cube_human",
        "lerobot/aloha_sim_insertion_human",
        "lerobot/aloha_sim_transfer_cube_scripted",
        "lerobot/aloha_static_battery",
        "lerobot/aloha_static_cups_open",
        # ==== DROID / RT-2 / OpenX ====
        "droid-dataset/droid_100",
        "IPEC-COMMUNITY/droid_100_lerobot",
        "IPEC-COMMUNITY/utaustin_mutex_lerobot",
        "IPEC-COMMUNITY/austin_buds_dataset_converted_externally_to_rlds_lerobot",
        "IPEC-COMMUNITY/austin_sailor_dataset_converted_externally_to_rlds_lerobot",
        "IPEC-COMMUNITY/austin_sirius_dataset_converted_externally_to_rlds_lerobot",
        "IPEC-COMMUNITY/bc_z_lerobot",
        "IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot",
        "IPEC-COMMUNITY/berkeley_cable_routing_lerobot",
        "IPEC-COMMUNITY/berkeley_fanuc_manipulation_lerobot",
        "IPEC-COMMUNITY/cmu_stretch_lerobot",
        "IPEC-COMMUNITY/roboturk_lerobot",
        "IPEC-COMMUNITY/stanford_kuka_multimodal_dataset_lerobot",
        # ==== Community-uploaded CoT-annotated ====
        "OpenBMB/deepthinkvla_dataset",
        "OpenBMB/DeepThinkVLA_train",
        "Weicheng-Gu1/embodied_features_and_demos_libero",
        # ==== MetaWorld ====
        "lerobot/metaworld_mt10",
        # ==== ManiSkill ====
        "haosulab/ManiSkill2",
        # ==== Simple robot demos ====
        "lerobot/pusht",
        "lerobot/xarm_lift_medium",
    ]
    results = {}
    for repo_id in candidates:
        try:
            info = api.dataset_info(repo_id)
            files = list(api.list_repo_files(repo_id, repo_type="dataset"))[:8]
            results[repo_id] = {
                "exists": True,
                "sibling_count": len(info.siblings) if info.siblings else 0,
                "files_sample": files,
            }
            print(f"[ds] OK {repo_id}")
        except Exception as e:
            msg = str(e)[:120]
            results[repo_id] = {"exists": False, "error": msg}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    r = probe_datasets()
    (out / "datasets_scout.json").write_text(json.dumps(r, indent=2, default=str))
    n_found = sum(1 for v in r.values() if v.get("exists"))
    print(f"\n{n_found}/{len(r)} datasets found")
    print("\nFOUND:")
    for k, v in r.items():
        if v.get("exists"):
            print(f"  {k}  ({v.get('sibling_count', '?')} files)")
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()

