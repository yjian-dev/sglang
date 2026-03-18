"""
MathBench evaluation across multiple sglang servers.

Uses MENTOR-RL/math-bench dataset (4077 problems, numeric/short answers).

Usage:
  python scripts/eval_mathbench.py                          # 1000 random problems, 8 GPUs
  python scripts/eval_mathbench.py --num-problems 500       # quick
"""
import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset


def strip_thinking(text):
    if not text:
        return ""
    return re.sub(r"^.*?</think>\s*", "", text, count=1, flags=re.DOTALL)


def extract_answer(text):
    """Extract numeric/short answer from model output."""
    text = strip_thinking(text)
    # boxed answer
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    # "answer is X"
    m = re.search(r"[Aa]nswer is:?\s*(.+?)(?:\.|,|\n|$)", text)
    if m:
        ans = m.group(1).strip()
        if ans:
            return ans
    # Last line
    lines = text.strip().split('\n')
    if lines:
        return lines[-1].strip()
    return "?"


def normalize_answer(s):
    """Normalize numeric answer for comparison."""
    s = s.strip().rstrip(".")
    # Remove $ signs, commas
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    # Try to convert to float for numeric comparison
    try:
        return str(float(s))
    except ValueError:
        return s.lower()


def run_one(args):
    idx, question, gold, port, max_tokens, timeout, temperature, top_p, top_k = args
    prompt = (
        f"{question}\n\n"
        "Think step by step, then give your final answer. "
        "Put your answer in \\boxed{} format."
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
        finish = r["choices"][0].get("finish_reason", "")
        comp = r["usage"]["completion_tokens"]
        pred = extract_answer(content)
        return idx, pred, gold, comp, finish, content, None
    except Exception as e:
        return idx, "?", gold, 0, "", "", str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=1000)
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    print("Loading MathBench dataset...")
    ds = load_dataset("MENTOR-RL/math-bench", split="train")
    print(f"Total MathBench problems: {len(ds)}")

    random.seed(42)
    N = min(args.num_problems, len(ds))
    indices = random.sample(range(len(ds)), N)
    subset = ds.select(indices)
    ports = args.ports

    problems = []
    for item in subset:
        q = item["question"]
        gold = str(item["answer"]).strip()
        problems.append((q, gold))

    print(f"MathBench eval: {N} problems, {len(ports)} servers")

    tasks = [
        (i, q, g, ports[i % len(ports)], args.max_tokens, args.timeout,
         args.temperature, args.top_p, args.top_k)
        for i, (q, g) in enumerate(problems)
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
    correct = total_tok = errors = truncated = no_extract = wrong = 0
    wrong_examples = []
    for idx, pred, gold, comp, finish, content, err in results:
        if err:
            errors += 1
            continue
        total_tok += comp
        pred_norm = normalize_answer(pred)
        gold_norm = normalize_answer(gold)
        if pred_norm == gold_norm:
            correct += 1
        else:
            if finish == "length":
                truncated += 1
            elif pred == "?":
                no_extract += 1
            else:
                wrong += 1
            if len(wrong_examples) < 10:
                wrong_examples.append({
                    "idx": idx, "pred": pred, "gold": gold,
                    "finish": finish, "output": content[:200]
                })

    acc = correct / N * 100
    print(f"\n{'=' * 60}")
    print(f"MathBench {N} problems, {len(ports)} GPUs")
    print(f"{'=' * 60}")
    print(f"Accuracy:       {correct}/{N} ({acc:.1f}%)")
    print(f"Truncated:      {truncated}")
    print(f"No extraction:  {no_extract}")
    print(f"Wrong answer:   {wrong}")
    print(f"Total tokens:   {total_tok:,}")
    print(f"Wall time:      {elapsed:.1f}s")
    print(f"Throughput:     {total_tok / elapsed:.1f} tok/s")
    if errors:
        print(f"Errors:         {errors}")

    if wrong_examples:
        print(f"\n--- Sample wrong answers ---")
        for ex in wrong_examples[:5]:
            print(f"  [{ex['idx']}] pred={ex['pred']} gold={ex['gold']} finish={ex['finish']}")
            print(f"    {ex['output'][:150]}...")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "mathbench_summary.json"), "w") as f:
            json.dump({"accuracy": acc, "correct": correct, "total": N,
                       "tokens": total_tok, "truncated": truncated,
                       "no_extract": no_extract, "wrong": wrong}, f, indent=2)


if __name__ == "__main__":
    main()
