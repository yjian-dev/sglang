"""
MMLU-Pro evaluation across multiple sglang servers.

MMLU-Pro has 10 choices (A-J) instead of 4, covering 14 disciplines.
Uses TIGER-Lab/MMLU-Pro from HuggingFace.

Usage:
  python scripts/eval_mmlu_pro.py                          # 1000 problems, 8 GPUs
  python scripts/eval_mmlu_pro.py --num-problems 500
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
    """Extract single letter choice (A-J) from model output."""
    text = strip_thinking(text)
    # Look for "answer is (X)" or "answer is X"
    m = re.search(r"[Aa]nswer is:?\s*\(?([A-Ja-j])\)?", text)
    if m:
        return m.group(1).upper()
    # Look for boxed answer
    m = re.search(r"\\boxed\{([A-Ja-j])\}", text)
    if m:
        return m.group(1).upper()
    # Last standalone letter
    m = re.findall(r'\b([A-Ja-j])\b', text)
    if m:
        return m[-1].upper()
    return "?"


def format_choices(options):
    """Format choices as A. xxx  B. xxx  ..."""
    letters = "ABCDEFGHIJ"
    lines = []
    for i, opt in enumerate(options):
        if i < len(letters):
            lines.append(f"{letters[i]}. {opt}")
    return "\n".join(lines)


def run_one(args):
    idx, question, choices_str, gold, port, max_tokens, timeout, temperature, top_p, top_k = args
    prompt = (
        f"{question}\n\n{choices_str}\n\n"
        "Think step by step, then give your answer as \"The answer is (X)\"."
    )
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
        comp = r["usage"]["completion_tokens"]
        pred = extract_choice(content)
        return idx, pred, gold, comp, None
    except Exception as e:
        return idx, "?", gold, 0, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=0, help="0 = all")
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    if args.num_problems > 0 and args.num_problems < len(ds):
        import random
        random.seed(42)
        indices = random.sample(range(len(ds)), args.num_problems)
        subset = ds.select(indices)
    else:
        subset = ds
    N = len(subset)
    ports = args.ports

    LETTERS = "ABCDEFGHIJ"
    problems = []
    for item in subset:
        q = item["question"]
        options = item["options"]
        choices_str = format_choices(options)
        gold_idx = item["answer_index"]
        gold = LETTERS[gold_idx] if isinstance(gold_idx, int) else str(item["answer"])
        problems.append((q, choices_str, gold))

    print(f"MMLU-Pro eval: {N} problems, {len(ports)} servers")

    tasks = [
        (i, q, c, g, ports[i % len(ports)], args.max_tokens, args.timeout,
         args.temperature, args.top_p, args.top_k)
        for i, (q, c, g) in enumerate(problems)
    ]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        done = 0
        for f in as_completed(futures):
            done += 1
            if done % max(N // 5, 1) == 0:
                print(f"  {done}/{N} done ({time.time() - t0:.0f}s)")
        results = [f.result() for f in futures]

    elapsed = time.time() - t0
    correct = total_tok = errors = 0
    for idx, pred, gold, comp, err in results:
        if err:
            errors += 1
            continue
        total_tok += comp
        correct += (pred == gold)

    acc = correct / N * 100
    print(f"\n{'=' * 60}")
    print(f"MMLU-Pro {N} problems, {len(ports)} GPUs")
    print(f"{'=' * 60}")
    print(f"Accuracy:       {correct}/{N} ({acc:.1f}%)")
    print(f"Total tokens:   {total_tok:,}")
    print(f"Wall time:      {elapsed:.1f}s")
    print(f"Throughput:     {total_tok / elapsed:.1f} tok/s")
    if errors:
        print(f"Errors:         {errors}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "mmlu_pro_summary.json"), "w") as f:
            json.dump({"accuracy": acc, "correct": correct, "total": N,
                       "tokens": total_tok}, f, indent=2)


if __name__ == "__main__":
    main()
