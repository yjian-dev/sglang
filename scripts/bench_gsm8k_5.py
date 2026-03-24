#!/usr/bin/env python3
"""Send 5 GSM8K problems sequentially and measure TPS from responses."""
import argparse
import json
import time
import requests

PROBLEMS = [
    "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers\u2019 market daily for $2 each. How much in dollars does she make every day at the farmers\u2019 market?",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
    "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
]

def run(port, model="test", lora_path=None, max_tokens=2048):
    url = f"http://localhost:{port}/v1/chat/completions"
    total_tokens = 0
    total_time = 0.0

    for i, q in enumerate(PROBLEMS):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": q}],
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
        }
        if lora_path:
            body["lora_path"] = lora_path

        t0 = time.time()
        resp = requests.post(url, json=body, headers={"Content-Type": "application/json"})
        elapsed = time.time() - t0

        data = resp.json()
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        tps = completion_tokens / elapsed if elapsed > 0 else 0

        print(f"[{i+1}/5] tokens={completion_tokens}, time={elapsed:.2f}s, tps={tps:.1f}")
        total_tokens += completion_tokens
        total_time += elapsed

    avg_tps = total_tokens / total_time if total_time > 0 else 0
    print(f"\n=== SUMMARY ===")
    print(f"Total tokens: {total_tokens}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average TPS: {avg_tps:.1f}")
    return avg_tps

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=31003)
    p.add_argument("--model", default="test")
    p.add_argument("--lora-path", default=None)
    p.add_argument("--max-tokens", type=int, default=2048)
    args = p.parse_args()
    run(args.port, args.model, args.lora_path, args.max_tokens)
