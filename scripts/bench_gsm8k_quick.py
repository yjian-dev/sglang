#!/usr/bin/env python3
"""Quick GSM8K benchmark: 5 problems, measure TPS and accuracy.

Usage:
    python scripts/bench_gsm8k_quick.py --port 31003
    python scripts/bench_gsm8k_quick.py --port 31003 --lora b3lora
    python scripts/bench_gsm8k_quick.py --port 31003 --num-problems 10
"""
import argparse
import re
import time

import requests


PROBLEMS = [
    ("Janet ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers market daily for 2 dollars. How much in dollars does she make every day at the farmers market?", 18),
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", 72),
    ("Weng earns 12 dollars an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", 10),
    ("Betty is saving money for a new wallet which costs 100 dollars. Betty has only half of the money she needs. Her parents decided to give her 15 dollars for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?", 5),
    ("Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?", 48),
    ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", 624),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", 3),
    ("James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?", 540),
    ("Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?", 260),
    ("Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?", 42),
]


def run_benchmark(port: int, lora: str = None, num_problems: int = 5):
    url = f"http://localhost:{port}/v1/chat/completions"
    problems = PROBLEMS[:num_problems]
    total_tokens = 0
    total_time = 0
    correct = 0

    for i, (q, expected) in enumerate(problems):
        body = {
            "model": "base:b3lora" if lora else "test",
            "messages": [{"role": "user", "content": q}],
            "max_tokens": 8192,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
        }
        if lora:
            body["lora_path"] = lora

        t0 = time.time()
        r = requests.post(url, json=body)
        elapsed = time.time() - t0

        d = r.json()
        tokens = d["usage"]["completion_tokens"]
        content = d["choices"][0]["message"]["content"]
        total_tokens += tokens
        total_time += elapsed

        match = re.findall(r"boxed\{([^}]+)\}", content)
        got_str = match[-1].replace(",", "").strip() if match else ""
        try:
            got = int(float(got_str))
        except (ValueError, IndexError):
            got = -1
        ok = got == expected
        if ok:
            correct += 1
        status = "Y" if ok else "N"
        print(
            f"Q{i+1:>2d}: exp={expected:<6} got={got:<6} {status} | "
            f"{tokens:>4d} tok {elapsed:>5.1f}s {tokens/elapsed:>5.0f} tok/s",
            flush=True,
        )

    avg_tps = total_tokens / total_time
    print(flush=True)
    print(f"Accuracy: {correct}/{len(problems)} ({100*correct/len(problems):.1f}%)", flush=True)
    print(f"Total: {total_tokens} tokens in {total_time:.1f}s", flush=True)
    print(f"Average TPS: {avg_tps:.1f}", flush=True)
    return avg_tps, correct, len(problems)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=31003)
    parser.add_argument("--lora", type=str, default=None, help="LoRA adapter name (e.g. b3lora)")
    parser.add_argument("--num-problems", type=int, default=5)
    args = parser.parse_args()
    run_benchmark(args.port, args.lora, args.num_problems)
