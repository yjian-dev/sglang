"""
LongBench-v2 Hard evaluation.

Dataset: THUDM/LongBench-v2 (503 total, 311 hard)
Format: Multiple choice A/B/C/D with long context.

Context length constraints:
  - Model max: 131072 tokens
  - short: ~15k-50k tokens  (121 hard problems)
  - medium: ~50k-150k tokens (127 hard, many exceed 128k limit)
  - long: >150k tokens       (63 hard, all exceed 128k limit)

By default we only test "short" hard problems to avoid OOM.
Use --length-filter all to attempt all (medium/long may fail).

Usage:
  python scripts/eval_longbench_hard.py                        # short hard (121 problems)
  python scripts/eval_longbench_hard.py --length-filter all    # all 311 hard
  python scripts/eval_longbench_hard.py --ports 30010 ...      # Qwen3-8B
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset


def strip_thinking(text):
    if not text:
        return ""
    return re.sub(r"^.*?</think>\s*", "", text, count=1, flags=re.DOTALL)


def extract_choice(text):
    """Extract A/B/C/D from model output (first_option_postprocess style)."""
    text = strip_thinking(text)
    patterns = [
        r"[Tt]he correct answer is\s*\(?([ABCD])\)?",
        r"[Tt]he answer is\s*\(?([ABCD])\)?",
        r"[Aa]nswer:\s*\(?([ABCD])\)?",
        r"[Cc]hoice\s*\(?([ABCD])\)?",
        r"^\s*\(?([ABCD])\)?[\s\.\)]",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m and m.group(1) in "ABCD":
            return m.group(1)
    # Fallback: first uppercase option letter
    found = re.findall(r"\b([ABCD])\b", text)
    return found[0] if found else "?"


def build_prompt(item):
    """Matching OpenCompass longbenchv2 prompt template."""
    return (
        "Please read the following text and answer the questions below.\n"
        "<text>\n"
        f"{item['context']}\n"
        "</text>\n\n"
        f"What is the correct answer to this question: {item['question']}\n\n"
        "Choices:\n"
        f"(A) {item['choice_A']}\n"
        f"(B) {item['choice_B']}\n"
        f"(C) {item['choice_C']}\n"
        f"(D) {item['choice_D']}\n\n"
        "Let's think step by step. Based on the above, what is the single, "
        "most likely answer choice? Format your response as follows: "
        "\"The correct answer is (insert answer here)\""
    )


def run_one(args):
    idx, prompt, gold, meta, port, max_tokens, timeout, temperature, top_p, top_k = args
    try:
        r = requests.post(f"http://localhost:{port}/v1/chat/completions", json={
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }, timeout=timeout).json()
        content = r["choices"][0]["message"]["content"]
        finish = r["choices"][0]["finish_reason"]
        comp = r["usage"]["completion_tokens"]
        pred = extract_choice(content)
        return idx, pred, gold, finish, comp, meta, None
    except Exception as e:
        return idx, "?", gold, "error", 0, meta, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length-filter", type=str, default="short",
                        choices=["short", "medium", "short+medium", "all"],
                        help="Which length subset to test (default: short only)")
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max generation tokens (answers are short, 4k is enough)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-request timeout (long context needs more time)")
    parser.add_argument("--max-workers", type=int, default=32,
                        help="Parallel workers (keep low for long context)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    print("Loading LongBench-v2 dataset...")
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    all_hard = [item for item in ds if item["difficulty"] == "hard"]

    # Filter by length
    if args.length_filter == "short":
        problems = [item for item in all_hard if item["length"] == "short"]
    elif args.length_filter == "medium":
        problems = [item for item in all_hard if item["length"] == "medium"]
    elif args.length_filter == "short+medium":
        problems = [item for item in all_hard if item["length"] in ("short", "medium")]
    else:
        problems = all_hard

    print(f"LongBench-v2 Hard ({args.length_filter}): {len(problems)} problems, {len(args.ports)} servers")
    print(f"Config: max_tokens={args.max_tokens}, timeout={args.timeout}s")
    print("Note: context can be 10k-150k tokens, each request will be slow.\n")

    ports = args.ports
    tasks = [
        (i, build_prompt(item), item["answer"], {
            "domain": item.get("domain", ""),
            "length": item.get("length", ""),
            "_id": item.get("_id", str(i)),
        }, ports[i % len(ports)],
         args.max_tokens, args.timeout, args.temperature, args.top_p, args.top_k)
        for i, item in enumerate(problems)
    ]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        results = []
        done = 0
        for f in as_completed(futures):
            done += 1
            results.append(f.result())
            if done % max(len(tasks) // 5, 1) == 0:
                elapsed = time.time() - t0
                print(f"  {done}/{len(tasks)} done ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    results.sort(key=lambda x: x[0])

    correct = total_tok = errors = truncated = no_extract = 0
    by_domain = {}
    wrong_examples = []

    for idx, pred, gold, finish, comp, meta, err in results:
        domain = meta.get("domain", "unknown")
        if domain not in by_domain:
            by_domain[domain] = {"correct": 0, "total": 0}
        by_domain[domain]["total"] += 1

        if err:
            errors += 1
            continue
        total_tok += comp
        if finish == "length":
            truncated += 1
        if pred == "?":
            no_extract += 1
        if pred == gold:
            correct += 1
            by_domain[domain]["correct"] += 1
        else:
            if len(wrong_examples) < 8:
                wrong_examples.append((idx, pred, gold, finish, meta))

    N = len(problems)
    acc = correct / N * 100

    print(f"\n{'='*60}")
    print(f"LongBench-v2 Hard ({args.length_filter}): {N} problems")
    print(f"{'='*60}")
    print(f"Accuracy:      {correct}/{N} ({acc:.1f}%)")
    print(f"Truncated:     {truncated}")
    print(f"No extraction: {no_extract}")
    print(f"Errors:        {errors}")
    print(f"Total tokens:  {total_tok:,}")
    print(f"Wall time:     {elapsed:.1f}s")

    print(f"\n--- By Domain ---")
    for domain, s in sorted(by_domain.items(), key=lambda x: -x[1]["total"]):
        d_acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {domain:35s} {s['correct']:3d}/{s['total']:3d} = {d_acc:5.1f}%")

    if wrong_examples:
        print(f"\n--- Sample Wrong Answers ---")
        for idx, pred, gold, finish, meta in wrong_examples[:5]:
            print(f"  [{idx}] pred={pred} gold={gold} finish={finish} domain={meta['domain']}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "longbench_hard_summary.json"), "w") as f:
            json.dump({"accuracy": acc, "correct": correct, "total": N,
                       "length_filter": args.length_filter,
                       "truncated": truncated, "by_domain": by_domain}, f, indent=2)
        print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
