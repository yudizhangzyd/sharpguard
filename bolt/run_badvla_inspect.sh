#!/usr/bin/env bash
# Inspect the pre-trained BadVLA repo and report what we got. After this
# completes we'll know exactly what files exist and can wire the right
# checkpoint into our SharpGuard eval pipeline.
set -e -x

cd "$(dirname "$0")/.."

if [ -f /tmp/sharpguard.env ]; then
    set -a; . /tmp/sharpguard.env; set +a
fi

OUT_DIR="${BOLT_ARTIFACT_DIR:-./artifacts}/badvla-inspect"
mkdir -p "$OUT_DIR"

echo "==== BADVLA_CKPT_DIR contents ===="
ls -la "$BADVLA_CKPT_DIR" 2>&1 | tee "$OUT_DIR/listing.txt"

echo "==== Recursive layout ===="
find "$BADVLA_CKPT_DIR" -maxdepth 5 -type f -size +1c 2>&1 \
    | head -200 | tee "$OUT_DIR/files.txt"

echo "==== Total size ===="
du -sh "$BADVLA_CKPT_DIR" 2>&1 | tee "$OUT_DIR/size.txt"

echo "==== Try to load each safetensors / pt file as a state dict ===="
python - <<PY 2>&1 | tee "$OUT_DIR/inspect.txt"
import os, glob, json
root = os.environ["BADVLA_CKPT_DIR"]
print(f"root = {root}")

# Look for HF-style model dirs (with config.json).
for root_dir, dirs, files in os.walk(root):
    if "config.json" in files:
        cfg_path = os.path.join(root_dir, "config.json")
        try:
            cfg = json.load(open(cfg_path))
            print(f"\n=== {root_dir}")
            print(f"  config.json: {cfg.get('_class_name', cfg.get('model_type', 'unknown'))}")
            print(f"             arch: {cfg.get('architectures', '?')}")
            for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                       "vocab_size", "torch_dtype"):
                if k in cfg:
                    print(f"             {k}: {cfg[k]}")
        except Exception as e:
            print(f"  config.json parse error: {e}")
        # List sibling files
        for f in sorted(files):
            full = os.path.join(root_dir, f)
            sz = os.path.getsize(full)
            print(f"  {f}  ({sz / 1e6:.1f} MB)")

# Also list adapter_config.json (LoRA)
for root_dir, dirs, files in os.walk(root):
    if "adapter_config.json" in files:
        ac_path = os.path.join(root_dir, "adapter_config.json")
        try:
            ac = json.load(open(ac_path))
            print(f"\n=== LoRA adapter at {root_dir}")
            print(f"  base_model_name_or_path: {ac.get('base_model_name_or_path', '?')}")
            print(f"  r: {ac.get('r', '?')}, alpha: {ac.get('lora_alpha', '?')}")
            print(f"  target_modules: {ac.get('target_modules', '?')}")
        except Exception as e:
            print(f"  adapter_config parse error: {e}")
PY

echo "Done. Artifacts in $OUT_DIR"
ls -la "$OUT_DIR"
