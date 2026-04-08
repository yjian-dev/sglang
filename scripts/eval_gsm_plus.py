"""
GSM-Plus evaluation across multiple sglang servers.

GSM-Plus is an augmented version of GSM8K with perturbed questions.
Uses qintongli/GSM-Plus from HuggingFace.

Usage:
  python scripts/eval_gsm_plus.py                          # 1000 problems, 8 GPUs
  python scripts/eval_gsm_plus.py --num-problems 200       # quick
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


def extract_number(text):
    """Extract numerical answer from model output."""
    text = strip_thinking(text)
    # Try \boxed{} first
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        raw = boxed[-1].replace(",", "").replace("$", "").replace("\\", "").strip()
        try:
            return float(raw)
        except ValueError:
            pass
    # Fallback: last number in text
    nums = re.findall(r'-?[\d,]+\.?\d*', text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def run_one(args):
    idx, question, gold, port, max_tokens, timeout, temperature, top_p, top_k = args
    prompt = f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
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
        pred = extract_number(content)
        return idx, pred, gold, comp, None
    except Exception as e:
        return idx, None, gold, 0, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=0,
                        help="0 = all problems")
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    ds = load_dataset("qintongli/GSM-Plus", split="test")
    N = min(args.num_problems, len(ds)) if args.num_problems > 0 else len(ds)
    ports = args.ports

    problems = []
    for item in ds.select(range(N)):
        q = item["question"] if "question" in item else item["perturbation_question"]
        gold_str = str(item["answer"]).replace(",", "").strip()
        try:
            gold = float(gold_str)
        except ValueError:
            gold = gold_str
        problems.append((q, gold))

    print(f"GSM-Plus eval: {N} problems, {len(ports)} servers")

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
    correct = total_tok = errors = 0
    for idx, pred, gold, comp, err in results:
        if err:
            errors += 1
            continue
        total_tok += comp
        if pred is not None:
            try:
                correct += (abs(float(pred) - float(gold)) < 1e-3)
            except (ValueError, TypeError):
                correct += (str(pred) == str(gold))

    acc = correct / N * 100
    print(f"\n{'=' * 60}")
    print(f"GSM-Plus {N} problems, {len(ports)} GPUs")
    print(f"{'=' * 60}")
    print(f"Accuracy:       {correct}/{N} ({acc:.1f}%)")
    print(f"Total tokens:   {total_tok:,}")
    print(f"Wall time:      {elapsed:.1f}s")
    print(f"Throughput:     {total_tok / elapsed:.1f} tok/s")
    if errors:
        print(f"Errors:         {errors}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "gsm_plus_summary.json"), "w") as f:
            json.dump({"accuracy": acc, "correct": correct, "total": N,
                       "tokens": total_tok}, f, indent=2)


if __name__ == "__main__":
    main()
