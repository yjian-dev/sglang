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
    """Extract single letter choice (A-P) from model output."""
    text = strip_thinking(text)
    # OC primary: ANSWER: X
    m = re.search(r"(?i)ANSWER\s*:\s*([A-Pa-p])", text)
    if m:
        return m.group(1).upper()
    # Look for "answer is (X)" or "answer is X"
    m = re.search(r"[Aa]nswer is:?\s*\(?([A-Pa-p])\)?", text)
    if m:
        return m.group(1).upper()
    # Look for boxed answer
    m = re.search(r"\\boxed\{([A-Pa-p])\}", text)
    if m:
        return m.group(1).upper()
    # Last standalone letter (limited to J for MMLU-Pro's 10 choices)
    m = re.findall(r'\b([A-Ja-j])\b', text)
    if m:
        return m[-1].upper()
    return "?"


def format_choices(options):
    """Format choices as A. xxx  B. xxx  ... (OC format, period separator)."""
    letters = "ABCDEFGHIJKLMNOP"
    lines = []
    for i, opt in enumerate(options):
        if i < len(letters) and opt != "N/A":
            lines.append(f"{letters[i]}. {opt}")
    return "\n".join(lines)


def run_one(args):
    idx, question, choices_str, gold, port, max_tokens, timeout, temperature, top_p, top_k = args
    prompt = (
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\n"
        f"Question:\n\n{question}\n\nOptions:\n\n{choices_str}"
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
        pred = extract_choice(content)
        return idx, pred, gold, comp, finish, content, None
    except Exception as e:
        return idx, "?", gold, 0, "", "", str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=0, help="0=full dataset")
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=32768)
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
    correct = total_tok = errors = truncated = no_extract = wrong = 0
    wrong_examples = []
    for idx, pred, gold, comp, finish, content, err in results:
        if err:
            errors += 1
            continue
        total_tok += comp
        if pred == gold:
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
    print(f"MMLU-Pro {N} problems, {len(ports)} GPUs")
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
        with open(os.path.join(args.output_dir, "mmlu_pro_summary.json"), "w") as f:
            json.dump({"accuracy": acc, "correct": correct, "total": N,
                       "tokens": total_tok}, f, indent=2)


if __name__ == "__main__":
    main()
