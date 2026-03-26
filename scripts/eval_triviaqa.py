"""
TriviaQA evaluation across multiple sglang servers.

Uses the rc.nocontext split (no supporting document, pure knowledge).
Checks if gold answer appears in model output (standard exact-match + contains).

Usage:
  python scripts/eval_triviaqa.py                          # 1000 problems, 8 GPUs
  python scripts/eval_triviaqa.py --num-problems 200       # quick
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


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return " ".join(s.split())


def extract_answer(text):
    """Extract answer after 'The answer is' (OC-style extraction)."""
    m = re.search(r"[Tt]he answer is\s+(.+?)(?:[.\n]|$)", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def check_answer(pred, gold_answers):
    """Check if extracted answer matches any gold answer."""
    pred_norm = normalize_answer(pred)
    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        if gold_norm == pred_norm or gold_norm in pred_norm:
            return True
    return False


def run_one(args):
    idx, question, gold_answers, port, max_tokens, timeout, temperature, top_p, top_k = args
    prompt = f"Answer these questions, your answer should be as simple as possible, start your answer with the prompt 'The answer is '.\nQ: {question}?"
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
        # Extract just the answer part after "The answer is"
        answer = extract_answer(content)
        comp = r["usage"]["completion_tokens"]
        return idx, answer, gold_answers, comp, None
    except Exception as e:
        return idx, "", gold_answers, 0, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=0, help="0=full dataset")
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="4096 enough for thinking models; DLLM can use 256")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    N = min(args.num_problems, len(ds)) if args.num_problems > 0 else len(ds)
    ports = args.ports

    problems = []
    for item in ds.select(range(N)):
        q = item["question"]
        # Gold answers: main answer + aliases
        golds = list(set([item["answer"]["value"]] + item["answer"].get("aliases", [])))
        problems.append((q, golds))

    print(f"TriviaQA eval: {N} problems, {len(ports)} servers")

    tasks = [
        (i, q, g, ports[i % len(ports)], args.max_tokens, args.timeout,
         args.temperature, args.top_p, args.top_k)
        for i, (q, g) in enumerate(problems)
    ]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        results = [f.result() for f in futures]

    elapsed = time.time() - t0
    correct = total_tok = errors = 0
    for idx, pred, golds, comp, err in results:
        if err:
            errors += 1
            continue
        correct += check_answer(pred, golds)
        total_tok += comp

    acc = correct / N * 100
    print(f"\n{'=' * 60}")
    print(f"TriviaQA {N} problems, {len(ports)} GPUs")
    print(f"{'=' * 60}")
    print(f"Accuracy:       {correct}/{N} ({acc:.1f}%)")
    print(f"Total tokens:   {total_tok:,}")
    print(f"Wall time:      {elapsed:.1f}s")
    print(f"Throughput:     {total_tok / elapsed:.1f} tok/s")
    if errors:
        print(f"Errors:         {errors}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "triviaqa_summary.json"), "w") as f:
            json.dump({"accuracy": acc, "correct": correct, "total": N,
                       "tokens": total_tok}, f, indent=2)


if __name__ == "__main__":
    main()
