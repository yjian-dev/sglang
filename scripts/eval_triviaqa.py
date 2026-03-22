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


def general_postprocess(text):
    """OC-compatible postprocessing: cut at first newline/period/comma, remove punctuation and articles."""
    # Cut off at first newline, period, or comma
    truncated = re.split(r'[\n.,]', text, 1)[0]
    # Remove punctuation
    no_punct = re.sub(r'[^\w\s]', '', truncated)
    # Remove articles
    no_articles = re.sub(r'\b(a|an|the)\b', '', no_punct, flags=re.IGNORECASE)
    # Collapse whitespace
    return re.sub(r'\s+', ' ', no_articles).strip()


def extract_and_postprocess(text):
    """Extract answer matching OC TriviaQAEvaluator logic."""
    # Take first line (OC: prediction.strip().split('\\n')[0])
    text = text.strip().split('\n')[0].lower()
    # Split on known answer prefixes (OC order)
    text = text.split('answer is')[-1]
    text = text.split('a:')[-1]
    text = text.split('answer:')[-1]
    text = text.strip()
    return general_postprocess(text)


def check_answer(pred_processed, gold_answers):
    """Check if processed prediction contains any gold answer (OC: cand in pred)."""
    for gold in gold_answers:
        gold_processed = general_postprocess(gold).lower()
        if gold_processed in pred_processed:
            return True
    return False


def run_one(args):
    idx, question, gold_answers, port, max_tokens, timeout, temperature, top_p, top_k = args
    user_msg = f"Answer these questions, your answer should be as simple as possible, start your answer with the prompt 'The answer is '.\nQ: {question}?"
    # Use /generate with raw prompt to include "A:" bot prefix (matching OC)
    raw_prompt = (
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\nA:"
    )
    try:
        r = requests.post(f"http://localhost:{port}/generate", json={
            "text": raw_prompt,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
        }, timeout=timeout).json()
        content = r["text"]
        content = strip_thinking(content)
        # OC-compatible extraction: first line, split on prefixes, postprocess
        answer = extract_and_postprocess(content)
        comp = r.get("meta_info", {}).get("completion_tokens", len(content.split()))
        return idx, answer, gold_answers, comp, None
    except Exception as e:
        return idx, "", gold_answers, 0, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=0, help="0=full dataset")
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="OC uses max_out_len=50; 256 gives room for thinking")
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
