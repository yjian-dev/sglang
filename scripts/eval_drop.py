"""
DROP (Discrete Reasoning Over Paragraphs) evaluation.

Reading comprehension requiring discrete reasoning (counting, sorting, arithmetic).
Uses F1 score matching (standard for DROP).

Usage:
  python scripts/eval_drop.py                          # 1000 problems, 8 GPUs
  python scripts/eval_drop.py --num-problems 500
"""
import argparse
import json
import os
import re
import string
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset


def strip_thinking(text):
    if not text:
        return ""
    return re.sub(r"^.*?</think>\s*", "", text, count=1, flags=re.DOTALL)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    return " ".join(s.split())


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def run_one(args):
    idx, passage, question, gold_answers, port, max_tokens, timeout, temperature, top_p, top_k = args
    prompt = (
        f"Read the passage and answer the question. Give a short, direct answer.\n\n"
        f"Passage: {passage}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
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
        content = strip_thinking(content)
        comp = r["usage"]["completion_tokens"]
        return idx, content, gold_answers, comp, None
    except Exception as e:
        return idx, "", gold_answers, 0, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=1000)
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    ds = load_dataset("ucinlp/drop", split="validation")
    N = min(args.num_problems, len(ds))
    ports = args.ports

    problems = []
    for item in ds.select(range(N)):
        passage = item["passage"]
        question = item["question"]
        # Extract gold answers: spans + number + date
        golds = []
        for span in item["answers_spans"]["spans"]:
            golds.append(span)
        if not golds:
            golds = [""]
        problems.append((passage, question, golds))

    print(f"DROP eval: {N} problems, {len(ports)} servers")

    tasks = [
        (i, p, q, g, ports[i % len(ports)], args.max_tokens, args.timeout,
         args.temperature, args.top_p, args.top_k)
        for i, (p, q, g) in enumerate(problems)
    ]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        results = [f.result() for f in futures]

    elapsed = time.time() - t0
    total_f1 = total_em = total_tok = errors = 0
    for idx, pred, golds, comp, err in results:
        if err:
            errors += 1
            continue
        total_tok += comp
        total_f1 += metric_max_over_ground_truths(f1_score, pred, golds)
        em = max(1 if normalize_answer(pred) == normalize_answer(g) else 0 for g in golds)
        total_em += em

    avg_f1 = total_f1 / N * 100
    avg_em = total_em / N * 100
    print(f"\n{'=' * 60}")
    print(f"DROP {N} problems, {len(ports)} GPUs")
    print(f"{'=' * 60}")
    print(f"F1 Score:       {avg_f1:.1f}%")
    print(f"Exact Match:    {total_em}/{N} ({avg_em:.1f}%)")
    print(f"Total tokens:   {total_tok:,}")
    print(f"Wall time:      {elapsed:.1f}s")
    print(f"Throughput:     {total_tok / elapsed:.1f} tok/s")
    if errors:
        print(f"Errors:         {errors}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "drop_summary.json"), "w") as f:
            json.dump({"f1": avg_f1, "em": avg_em, "total": N, "tokens": total_tok}, f, indent=2)


if __name__ == "__main__":
    main()
