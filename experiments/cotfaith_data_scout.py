"""Data-side scout for the Embodied-CoT LIBERO dataset.

Before we invest a week in training our own ECoT-LIBERO, we need to
confirm:
  1. `Embodied-CoT/embodied_features_and_demos_libero` downloads via HF.
  2. Its samples contain (image, instruction, action, reasoning_json) tuples.
  3. The reasoning field is structured (task/plan/subtask/move dicts)
     that we can turn into TASK: ... PLAN: ... ACTION: ... training targets
     compatible with the ECoT prompt convention.
  4. Reasoning is per-timestep (aligns with actions) or per-episode.

Output: /artifacts/cotfaith-data-scout/data_report.json + a few
example (image, instr, action, reasoning_text) rows dumped to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

def scan_dataset(local_dir: Path, out: Path, max_samples: int = 5) -> dict:
    """Walk the downloaded dataset; report file tree, per-file sample
    structure, and reasoning field format. Try 3 loader strategies:
      (a) tensorflow_datasets (RLDS) if `dataset_info.json` present
      (b) HuggingFace `datasets` library (parquet / arrow shards)
      (c) raw json / npz / pickle walk

    We don't need FULL loading — just first 5 samples across whichever
    strategy works, so we can see field names + reasoning structure.
    """
    report = {"local_dir": str(local_dir), "file_tree": [],
              "loader_strategy": None, "example_samples": [],
              "reasoning_structure_hints": [], "error": None}

    # 1. File tree walk
    for root, dirs, files in os.walk(local_dir):
        rel = os.path.relpath(root, local_dir)
        for f in files:
            path = os.path.join(rel, f) if rel != "." else f
            size_mb = round(os.path.getsize(os.path.join(root, f)) / 1e6, 1)
            report["file_tree"].append({"path": path, "size_mb": size_mb})
        if len(report["file_tree"]) > 200:
            report["file_tree"].append({"path": "...(truncated)", "size_mb": 0})
            break

    # 2. Try RLDS / tfds
    has_dataset_info = any(f["path"].endswith("dataset_info.json")
                            for f in report["file_tree"])
    has_features = any(f["path"].endswith("features.json")
                        for f in report["file_tree"])
    if has_dataset_info or has_features:
        try:
            import tensorflow_datasets as tfds
            builder = tfds.builder_from_directory(str(local_dir))
            print(f"[scout] tfds builder OK: {builder.info}")
            report["loader_strategy"] = "tfds"
            report["tfds_info"] = str(builder.info)[:2000]
            ds = builder.as_dataset(split="train")
            for i, ex in enumerate(ds.take(max_samples)):
                sample_summary = {}
                # ex is a dict of tf.Tensor / nested dicts. Convert to
                # python types + record shapes.
                def _summarize(obj, prefix=""):
                    if hasattr(obj, "numpy"):
                        arr = obj.numpy()
                        try:
                            if hasattr(arr, "shape"):
                                return {"shape": tuple(arr.shape),
                                          "dtype": str(arr.dtype)}
                            else:
                                s = str(arr)[:400]
                                return {"scalar": s}
                        except Exception as e:
                            return {"err": str(e)}
                    elif isinstance(obj, dict):
                        return {k: _summarize(v, prefix + "/" + k)
                                 for k, v in obj.items()}
                    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
                        return [_summarize(v, prefix + "/*") for v in obj]
                    else:
                        return {"repr": str(obj)[:200]}
                sample_summary = _summarize(ex)
                report["example_samples"].append(sample_summary)
                # dig for reasoning-like keys
                def _find_reasoning(obj, path=""):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            lk = k.lower()
                            if any(w in lk for w in ("reason", "cot", "task", "plan", "subtask",
                                                       "move", "language", "instr")):
                                report["reasoning_structure_hints"].append(
                                    {"path": path + "/" + k, "type": type(v).__name__})
                            _find_reasoning(v, path + "/" + k)
                _find_reasoning(sample_summary)
                if i == 0:
                    # Also dump the raw sample keys structure
                    (out / "sample_0_summary.json").write_text(
                        json.dumps(sample_summary, indent=2, default=str)[:20000])
        except Exception as e:
            report["error"] = f"tfds path failed: {e}\n" + traceback.format_exc()[-800:]

    # 3. Try HF `datasets` on parquet shards
    if report["loader_strategy"] is None:
        parquets = [f["path"] for f in report["file_tree"]
                     if f["path"].endswith(".parquet")]
        if parquets:
            try:
                import datasets as hfd
                ds = hfd.load_dataset("parquet",
                                        data_files=[str(local_dir / p) for p in parquets[:3]],
                                        split="train")
                report["loader_strategy"] = "hf_datasets_parquet"
                report["hf_features"] = str(ds.features)[:1200]
                for i, ex in enumerate(ds):
                    if i >= max_samples: break
                    keys = list(ex.keys())
                    sample_summary = {}
                    for k in keys:
                        v = ex[k]
                        if hasattr(v, "shape"):
                            sample_summary[k] = {"shape": v.shape,
                                                    "dtype": str(v.dtype)}
                        elif isinstance(v, (list, tuple)):
                            sample_summary[k] = {"len": len(v),
                                                    "first_repr": str(v[:1])[:200]}
                        else:
                            sample_summary[k] = {"repr": str(v)[:300]}
                    report["example_samples"].append(sample_summary)
                    for k in keys:
                        if any(w in k.lower() for w in ("reason", "cot", "task", "plan")):
                            report["reasoning_structure_hints"].append(
                                {"path": k, "type": type(ex[k]).__name__,
                                  "example": str(ex[k])[:600]})
            except Exception as e:
                report["error"] = f"hf_datasets path failed: {e}\n" + \
                                    traceback.format_exc()[-800:]

    # 4. Fallback: json / npz walk
    if report["loader_strategy"] is None:
        report["loader_strategy"] = "raw_walk"
        for f in report["file_tree"][:20]:
            if f["path"].endswith((".json", ".jsonl")):
                try:
                    p = local_dir / f["path"]
                    with open(p) as fp:
                        content = fp.read(4000)
                    report["example_samples"].append(
                        {"path": f["path"], "head": content[:2000]})
                except Exception as e:
                    pass

    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="./cotfaith-data-scout")
    p.add_argument("--repo-id", default="Embodied-CoT/embodied_features_and_demos_libero")
    p.add_argument("--max-samples", type=int, default=5)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[data-scout] downloading {args.repo_id} ...")

    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            cache_dir=os.environ.get("HF_HOME"),
        )
        print(f"[data-scout] downloaded to {local_dir}")
    except Exception as e:
        (out / "data_report.json").write_text(json.dumps(
            {"download_error": str(e), "traceback": traceback.format_exc()},
            indent=2))
        print(f"[data-scout] DOWNLOAD FAILED: {e}")
        sys.stdout.flush(); os._exit(0)

    report = scan_dataset(Path(local_dir), out, max_samples=args.max_samples)
    report["repo_id"] = args.repo_id
    (out / "data_report.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"\n\n===== DATA SCOUT DONE =====")
    print(f"loader_strategy: {report.get('loader_strategy')}")
    print(f"n_files: {len(report.get('file_tree', []))}")
    print(f"n_samples inspected: {len(report.get('example_samples', []))}")
    print(f"reasoning_hints: {len(report.get('reasoning_structure_hints', []))}")
    if report.get("error"):
        print(f"ERROR: {report['error'][:500]}")
    print(f"full report -> {out / 'data_report.json'}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
