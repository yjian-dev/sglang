"""
BIG-Bench Hard (BBH) evaluation across multiple sglang servers.

Uses official 3-shot CoT prompts from Joschka/big_bench_hard.
Extracts answer after "ANSWER:" or "the answer is" pattern.

Usage:
  python scripts/eval_bbh.py                          # all tasks, 8 GPUs
  python scripts/eval_bbh.py --subtasks boolean_expressions date_understanding
  python scripts/eval_bbh.py --ports 30000 30001
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


def extract_answer(text):
    """Extract answer from model output.
    Official BBH format ends with 'So the answer is X.' or 'ANSWER: X'
    """
    text = strip_thinking(text)

    # 1. Look for "ANSWER: X" (official CoT format)
    m = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()

    # 2. Look for "the answer is X" (with optional parentheses/period)
    m = re.search(r"[Tt]he answer is\s*(.+?)(?:\.|$)", text, re.MULTILINE)
    if m:
        return m.group(1).strip()

    # 3. Fallback: last non-empty line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def normalize(s):
    """Normalize for comparison: extract core answer, case-insensitive."""
    s = s.strip().lower()
    # Normalize letter answers: "(b)", "b)", "(b" → "b"
    m = re.match(r'^\(?([a-j])\)?', s)
    if m:
        return m.group(1)
    return s


def run_one(args):
    idx, prompt, gold, port, max_tokens, timeout, temperature, top_p, top_k = args
    try:
        r = requests.post(f"http://localhost:{port}/v1/completions", json={
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }, timeout=timeout).json()
        content = r["choices"][0]["text"]
        comp = r["usage"]["completion_tokens"]
        pred = extract_answer(content)
        return idx, pred, gold, comp, content, None
    except Exception as e:
        return idx, "?", gold, 0, "", str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtasks", type=str, nargs="+", default=None,
                        help="Specific subtasks to run (default: all)")
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Load official 3-shot CoT prompts
    print("Loading BBH CoT prompts...")
    prompts_ds = load_dataset("Joschka/big_bench_hard", "few_shot_prompts")
    cot_prompts = {}
    for item in prompts_ds["few_shot_prompts"]:
        cot_prompts[item["dataset_name"]] = item["chain_of_thought_prompt"]

    subtasks = args.subtasks or sorted(cot_prompts.keys())
    ports = args.ports

    print(f"BBH eval: {len(subtasks)} subtasks, {len(ports)} servers")
    print(f"Config: max_tokens={args.max_tokens}, temp={args.temperature}")

    all_correct = all_total = all_tokens = 0
    subtask_results = {}
    t0 = time.time()

    for task_name in subtasks:
        if task_name not in cot_prompts:
            print(f"  Skipping {task_name} (no CoT prompt)")
            continue

        try:
            task_ds = load_dataset("lukaemon/bbh", task_name, split="test")
        except Exception as e:
            print(f"  Skipping {task_name} ({e})")
            continue

        cot_prefix = cot_prompts[task_name]

        # Build prompts: 3-shot CoT prefix + test question
        problems = []
        for item in task_ds:
            q = item["input"]
            gold = item["target"]
            # Append test question in same format as few-shot examples
            prompt = f"{cot_prefix}\n\nQUESTION: {q}\nLet's think step by step.\n"
            problems.append((prompt, gold))

        tasks = [
            (i, p, g, ports[i % len(ports)], args.max_tokens, args.timeout,
             args.temperature, args.top_p, args.top_k)
            for i, (p, g) in enumerate(problems)
        ]

        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = [pool.submit(run_one, t) for t in tasks]
            results = [f.result() for f in futures]

        correct = sum(1 for _, pred, gold, _, _, _ in results
                      if normalize(pred) == normalize(gold))
        total = len(results)
        tokens = sum(comp for _, _, _, comp, _, _ in results)
        errors = sum(1 for _, _, _, _, _, e in results if e)
        acc = correct / total * 100 if total > 0 else 0

        subtask_results[task_name] = {"accuracy": acc, "correct": correct, "total": total}
        all_correct += correct
        all_total += total
        all_tokens += tokens

        print(f"  {task_name:45s} {correct:3d}/{total:3d} = {acc:5.1f}%"
              + (f"  ({errors} errors)" if errors else ""))

    elapsed = time.time() - t0
    overall_acc = all_correct / all_total * 100 if all_total > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"BBH Overall: {all_correct}/{all_total} = {overall_acc:.1f}%")
    print(f"Total tokens: {all_tokens:,}, Wall time: {elapsed:.1f}s")
    print(f"Throughput: {all_tokens / elapsed:.1f} tok/s")
    print(f"{'=' * 60}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "bbh_summary.json"), "w") as f:
            json.dump({"overall_accuracy": overall_acc, "correct": all_correct,
                       "total": all_total, "subtasks": subtask_results}, f, indent=2)


if __name__ == "__main__":
    main()
