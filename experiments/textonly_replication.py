#!/usr/bin/env python3
"""Stage 5 (optional): does the two-floors pathology reproduce outside a VLA's
action space?

Reviewer's ask (Round 2, Stage 5): one plain 7-8B LM, a text reasoning
dataset, 3 conditions (unedited / paraphrase floor / syntactic-scramble
floor). If a meaning-preserving edit to the reasoning still shifts the
model's own next-token distribution at the point it commits to its answer --
the same pathology Table 1 reports for VLA actions -- the contribution
reframes from "a VLA benchmark problem" to "a general CoT-faithfulness
metrology problem". If not, that is a legitimate scoping finding: the
pathology is specific to a continuous, de-quantized action readout.

Dataset: GSM8K test split (openai/gsm8k, 1319 items), gold rationale as the
teacher-forced CoT (same protocol choice as the paper's primary results,
not Stage 2's self-generated one -- Stage 5 is explicitly the cheap, optional
check, not a second construct-validity fix).

Metric: TV distance between the model's next-token softmax at the position
where it is about to write the final numeric answer, under the original vs.
edited rationale -- literally the same statistic and formula
(0.5 * sum|p-q|) Stage 1 uses for VLA action bins, just over the LM's own
vocabulary instead of a 256-bin action grid, plus whether the argmax token
changes at all (the discrete analogue of Delta_inf > tau).

Edit generators (simpler and less validated than the VLA study's LLM-judged
ones -- this is the explicitly-optional, cheap check, not a construct-
validity fix, and is reported as such):
  paraphrase_null:    word-level synonym substitution from a small hand-built
                       table of common GSM8K connector/math words. Numbers
                       and operators are never touched.
  syntactic_scramble: shuffles the order of the newline-separated computation
                       lines when there are 2+ of them (skipped, not forced,
                       otherwise -- exactly how the VLA study skips a family
                       with no applicable edit rather than fabricate one).

Usage:
    python3 experiments/textonly_replication.py --out <dir> --n-samples 500
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch

SYNONYMS = {
    "total": "sum", "totals": "sums", "left": "remaining", "leftover": "remaining",
    "each": "every", "there are": "there exist", "there is": "there exists",
    "gets": "receives", "get": "receive", "buys": "purchases", "buy": "purchase",
    "sells": "vends", "sell": "vend", "makes": "earns", "make": "earn",
    "spends": "uses up", "spend": "use up", "more": "additional",
    "first": "initially", "then": "next", "also": "additionally",
    "so": "thus", "now": "at this point", "still": "yet",
    "how many": "what number of", "how much": "what amount of",
    "altogether": "in all", "total of": "sum of", "originally": "at first",
}
# Longest keys first so multi-word phrases match before their single-word
# substrings do (e.g. "there are" before "are" would not exist anyway, but
# keeps the substitution order well-defined).
SYN_KEYS = sorted(SYNONYMS, key=len, reverse=True)

CALC_TAG = re.compile(r"<<[^>]*>>")


def strip_calc_tags(rationale):
    return CALC_TAG.sub("", rationale)


def paraphrase_null(rationale, rng):
    text = rationale
    applied = []
    for k in SYN_KEYS:
        pat = re.compile(r"(?<![\w-])" + re.escape(k) + r"(?![\w-])", re.IGNORECASE)
        if pat.search(text):
            text = pat.sub(SYNONYMS[k], text, count=1)
            applied.append(k)
    if not applied:
        return None
    return text


def syntactic_scramble(rationale, rng):
    lines = [l for l in rationale.split("\n") if l.strip()]
    if len(lines) < 2:
        return None
    shuffled = lines[:]
    for _ in range(20):
        rng.shuffle(shuffled)
        if shuffled != lines:
            break
    if shuffled == lines:
        return None
    return "\n".join(shuffled)


def load_gsm8k(n_samples, seed):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n_samples]
    items = []
    for i in idx:
        q, a = ds[i]["question"], ds[i]["answer"]
        if "####" not in a:
            continue
        rationale, final = a.split("####")
        rationale = strip_calc_tags(rationale).strip()
        items.append({"idx": i, "question": q, "rationale": rationale,
                      "final_answer": final.strip()})
    return items


def next_token_dist(model, tokenizer, question, rationale, device):
    prompt = (f"Question: {question}\n\nReasoning: {rationale}\n"
              f"The final answer is:")
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
    logits = out.logits[0, -1, :].to(torch.float32).cpu().numpy()
    logits = logits - logits.max()
    p = np.exp(logits)
    p = p / p.sum()
    return p, int(np.argmax(p))


def total_variation(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    crash_path = os.environ.get("TEXTONLY_CRASH_LOG",
                                 os.path.join(args.out, "crash.log"))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    print(f"[textonly] loading {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()

    items = load_gsm8k(args.n_samples, args.seed)
    print(f"[textonly] loaded {len(items)} GSM8K items", flush=True)

    rng = random.Random(args.seed)
    per_sample = []
    families = {"paraphrase_null": paraphrase_null,
                 "syntactic_scramble": syntactic_scramble}
    t0 = time.time()
    for i, item in enumerate(items):
        p_orig, argmax_orig = next_token_dist(
            model, tokenizer, item["question"], item["rationale"], device)
        for fam, fn in families.items():
            edited = fn(item["rationale"], rng)
            if edited is None:
                per_sample.append({"sample": item["idx"], "family": fam,
                                    "skipped": True, "reason": "no applicable edit"})
                continue
            p_edit, argmax_edit = next_token_dist(
                model, tokenizer, item["question"], edited, device)
            per_sample.append({
                "sample": item["idx"], "family": fam, "skipped": False,
                "tv": total_variation(p_orig, p_edit),
                "argmax_changed": argmax_orig != argmax_edit,
                "argmax_orig_token": tokenizer.decode([argmax_orig]),
                "argmax_edit_token": tokenizer.decode([argmax_edit]),
            })
        if (i + 1) % 20 == 0:
            print(f"[textonly] {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)

    aggregate = {}
    for fam in families:
        rows = [r for r in per_sample if r["family"] == fam and not r["skipped"]]
        n_skipped = sum(1 for r in per_sample if r["family"] == fam and r["skipped"])
        if rows:
            tvs = [r["tv"] for r in rows]
            aggregate[fam] = {
                "n": len(rows), "n_skipped": n_skipped,
                "tv_mean": float(np.mean(tvs)), "tv_std": float(np.std(tvs)),
                "tv_median": float(np.median(tvs)),
                "argmax_changed_rate": float(np.mean(
                    [r["argmax_changed"] for r in rows])),
            }
        else:
            aggregate[fam] = {"n": 0, "n_skipped": n_skipped}

    report = {
        "model": args.model, "n_samples_requested": args.n_samples,
        "n_items_loaded": len(items), "seed": args.seed,
        "aggregate": aggregate, "per_sample": per_sample,
    }
    with open(os.path.join(args.out, "textonly_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(aggregate, indent=2))
    print(f"[textonly] -> {args.out}/textonly_report.json")


if __name__ == "__main__":
    crash_path = os.environ.get("TEXTONLY_CRASH_LOG", "/tmp/textonly_crash.log")
    try:
        main()
    except BaseException:
        import traceback
        try:
            with open(crash_path, "a") as fh:
                fh.write(f"\n=== crash at pid {os.getpid()} ===\n")
                traceback.print_exc(file=fh)
                fh.flush()
                os.fsync(fh.fileno())
        finally:
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
        raise
