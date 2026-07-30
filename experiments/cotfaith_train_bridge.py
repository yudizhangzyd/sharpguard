"""F4 deconfound: LoRA r=32 fine-tune of ECoT-bridge on Bridge V2 4k subset.

Uses embodied-CoT/embodied_features_bridge reasoning annotations + Bridge V2
raw trajectories from IPEC-COMMUNITY/bridge_orig_lerobot (lerobot v2.0 format).
Same rank/steps/base as cotfaith_train.py's LIBERO-90 fine-tune.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from huggingface_hub import hf_hub_download, HfApi


def load_bridge_reasoning(hf_repo="Embodied-CoT/embodied_features_bridge"):
    """Load Embodied-CoT Bridge V2 reasoning annotations JSON (1.4GB)."""
    path = hf_hub_download(hf_repo, "embodied_features_bridge.json",
                              repo_type="dataset")
    print(f"[bridge-train] loading reasoning from {path}")
    with open(path) as f:
        return json.load(f)


class BridgeV2CotDataset(IterableDataset):
    """Stream (image, instruction, 9-tag CoT text, action) tuples from
    Bridge V2 lerobot + reasoning JSON."""
    def __init__(self, reasoning_map, dataset_repo="IPEC-COMMUNITY/bridge_orig_lerobot",
                    n_trajectories=4000, processor=None):
        self.reasoning = reasoning_map
        self.dataset_repo = dataset_repo
        self.n_trajectories = n_trajectories
        self.processor = processor
        api = HfApi()
        files = list(api.list_repo_files(dataset_repo, repo_type="dataset"))
        self.parquets = sorted([f for f in files if f.endswith(".parquet") and "data/" in f])[:n_trajectories]
        self.video_dirs = sorted(set(v.rsplit("/",1)[0] for v in files if v.endswith(".mp4") and "videos/" in v))
        print(f"[bridge-train] {len(self.parquets)} parquet files, {len(self.video_dirs)} video streams")

    def __iter__(self):
        import pyarrow.parquet as pq
        import av
        from PIL import Image as PILImage
        for parquet_path in self.parquets:
            try:
                pp = hf_hub_download(self.dataset_repo, parquet_path, repo_type="dataset")
                table = pq.read_table(pp)
                ep_col = table.column("episode_index").to_pylist()
                act_col = table.column("action").to_pylist()
                for i, ep_idx in enumerate(ep_col):
                    r = self.reasoning.get(str(ep_idx))
                    if r is None: continue
                    # Get first frame of episode video
                    for vd in self.video_dirs:
                        vname = f"{vd}/episode_{ep_idx:06d}.mp4"
                        try:
                            vp = hf_hub_download(self.dataset_repo, vname, repo_type="dataset")
                            container = av.open(vp)
                            frame = next(container.decode(video=0))
                            img = frame.to_image()
                            container.close()
                            break
                        except Exception: continue
                    else:
                        continue
                    instr = r.get("task", r.get("instruction", ""))
                    cot_text = self._build_cot_text(r)
                    action = np.asarray(act_col[i], dtype=np.float32)[:7]
                    yield {"image": img, "instruction": instr, "cot": cot_text, "action": action}
            except Exception as e:
                print(f"[bridge-train] {parquet_path}: {e}")
                continue

    def _build_cot_text(self, r):
        """Format 9-tag ECoT text from reasoning dict."""
        parts = [f"TASK: {r.get('task', '')}"]
        plan = r.get("plan", "")
        if isinstance(plan, dict):
            plan = " ".join(f"{k}. {v}" for k, v in sorted(plan.items()))
        parts.append(f"PLAN: {plan}")
        parts.append(f"SUBTASK: {r.get('subtask', '')}")
        parts.append(f"MOVE: {r.get('movement', r.get('move', ''))}")
        parts.append(f"GRIPPER POSITION: {r.get('gripper', '')}")
        return " ".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Embodied-CoT/ecot-openvla-7b-bridge")
    p.add_argument("--reasoning-repo", default="Embodied-CoT/embodied_features_bridge")
    p.add_argument("--dataset-repo", default="IPEC-COMMUNITY/bridge_orig_lerobot")
    p.add_argument("--n-trajectories", type=int, default=4000)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"[bridge-train] loading reasoning...")
    reasoning = load_bridge_reasoning(args.reasoning_repo)
    print(f"[bridge-train] {len(reasoning)} episodes in reasoning map")

    ds = BridgeV2CotDataset(reasoning, args.dataset_repo,
                              n_trajectories=args.n_trajectories)
    # Streaming smoke test: just print first 3 samples to verify pipeline
    print(f"[bridge-train] === Streaming pipeline smoke test ===")
    it = iter(ds)
    for i in range(3):
        try:
            s = next(it)
            print(f"[bridge-train] sample {i}: instr='{s['instruction'][:80]}' img={s['image'].size} action={s['action'][:3]}")
            print(f"                cot='{s['cot'][:100]}'")
        except StopIteration:
            print(f"[bridge-train] pipeline yielded {i} samples then stopped")
            break

    # NOTE: full LoRA training loop with peft/AutoModelForVision2Seq is deferred to
    # cotfaith_train.py-parity implementation. This smoke test validates that the
    # (image, instr, cot, action) pipeline works end-to-end on Bridge V2 + reasoning
    # JSON, which is the key F4-deconfound infrastructure piece.
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_report.json").write_text(json.dumps({
        "status": "smoke_test_only",
        "n_trajectories_requested": args.n_trajectories,
        "n_parquets_found": len(ds.parquets),
        "n_video_streams": len(ds.video_dirs),
        "n_reasoning_episodes": len(reasoning),
    }, indent=2))
    print(f"[bridge-train] smoke report -> {out}/smoke_report.json")
    print(f"[bridge-train] full training loop NOT run in this smoke pass; see note in cotfaith_train.py")


if __name__ == "__main__":
    main()
