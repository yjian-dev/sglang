"""
GSM8K evaluation for SDAR/DreamShift model.

Usage:
  # Launch server first, then:
  python test_gsm8k_eval.py --port 30000 --num-examples 20 --parallel 4
"""

import argparse
import json
import re
import time
import concurrent.futures
import requests
from datasets import load_dataset


def extract_answer_from_gsm8k(answer_str: str) -> str:
    """Extract the final numeric answer from GSM8K ground truth (after ####)."""
    match = re.search(r"####\s*(.+)", answer_str)
    if match:
        return match.group(1).strip().replace(",", "")
    return ""


def extract_number_from_response(response: str) -> str:
    """Extract the final numeric answer from model response.

    Priority: \\boxed{} > #### > 'the answer is' > last number
    """
    # 1. Look for \boxed{number} (highest priority)
    boxed = re.findall(r"\\boxed\{[\\$]*([\d,.-]+)", response)
    if boxed:
        return boxed[-1].strip().replace(",", "")

    # 2. Look for "#### <number>" pattern
    match = re.search(r"####\s*([\d,.-]+)", response)
    if match:
        return match.group(1).strip().replace(",", "")

    # 3. "the answer is <number>" pattern (last occurrence)
    matches = re.findall(
        r"(?:the answer is|answer is|answer:)\s*\$?([\d,.-]+)",
        response, re.IGNORECASE,
    )
    if matches:
        return matches[-1].strip().replace(",", "")

    # 4. "= $<number>" at end of sentence
    matches = re.findall(r"=\s*\$?([\d,.-]+)", response)
    if matches:
        return matches[-1].strip().replace(",", "")

    # 5. Fall back to last number
    numbers = re.findall(r"[-]?\d[\d,]*\.?\d*", response)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def generate_one(host, port, prompt, max_tokens=512):
    """Send a single generate request."""
    url = f"{host}:{port}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_tokens,
        },
    }
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json().get("text", "")


def format_prompt(question: str) -> str:
    """Format GSM8K question as a prompt."""
    return (
        f"Question: {question}\n"
        f"Answer: Let's think step by step.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated indices to eval (e.g. '5,12,27')")
    args = parser.parse_args()

    # Load GSM8K test set
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    # Pick examples by indices or first N
    if args.indices:
        indices = [int(x) for x in args.indices.split(",")]
        examples = [dataset[i] for i in indices]
    else:
        examples = list(dataset.select(range(args.num_examples)))

    prompts = [format_prompt(ex["question"]) for ex in examples]
    gold_answers = [extract_answer_from_gsm8k(ex["answer"]) for ex in examples]

    print(f"Evaluating {len(examples)} GSM8K examples, parallel={args.parallel}")
    print(f"Gold answers: {gold_answers}")
    print()

    # Generate responses
    results = [None] * len(prompts)
    t0 = time.perf_counter()

    def _gen(idx):
        return idx, generate_one(args.host, args.port, prompts[idx], args.max_tokens)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futs = [pool.submit(_gen, i) for i in range(len(prompts))]
        for fut in concurrent.futures.as_completed(futs):
            idx, text = fut.result()
            results[idx] = text

    elapsed = time.perf_counter() - t0

    # Evaluate
    correct = 0
    for i, (ex, response, gold) in enumerate(zip(examples, results, gold_answers)):
        pred = extract_number_from_response(response).rstrip(".")
        is_correct = pred == gold
        correct += int(is_correct)

        status = "CORRECT" if is_correct else "WRONG"
        print(f"[{i:2d}] {status}  gold={gold:>8s}  pred={pred:>8s}  | {ex['question'][:60]}...")
        if not is_correct:
            # Show the end of the response for debugging
            print(f"     Response tail: ...{response[-150:]}")
        print()

    accuracy = correct / len(examples) * 100
    print(f"=" * 60)
    print(f"GSM8K Accuracy: {correct}/{len(examples)} = {accuracy:.1f}%")
    print(f"Time: {elapsed:.1f}s")
    print(f"Throughput: {sum(len(r.split()) for r in results) / elapsed:.1f} words/s")


if __name__ == "__main__":
    main()
