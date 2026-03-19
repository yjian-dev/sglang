"""Quick TPF diagnostic: send a few GSM8K problems and measure tokens per forward."""
import argparse
import json
import requests
import time

PROBLEMS = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
    "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
    "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
]

def test_tpf(base_url: str, n_problems: int = 5, max_tokens: int = 512):
    url = f"{base_url}/v1/chat/completions"
    results = []
    for i, problem in enumerate(PROBLEMS[:n_problems]):
        messages = [{"role": "user", "content": problem}]
        payload = {
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
        }
        t0 = time.time()
        resp = requests.post(url, json=payload, timeout=120)
        elapsed = time.time() - t0
        data = resp.json()
        usage = data.get("usage", {})
        output_tokens = usage.get("completion_tokens", 0)
        text = data["choices"][0]["message"]["content"]
        results.append({
            "problem": i,
            "output_tokens": output_tokens,
            "elapsed_s": round(elapsed, 2),
            "tok_per_s": round(output_tokens / elapsed, 1) if elapsed > 0 else 0,
            "answer_snippet": text[:100],
        })
        print(f"Problem {i}: {output_tokens} tokens in {elapsed:.2f}s ({output_tokens/elapsed:.1f} tok/s)")
        print(f"  Answer: {text[:120]}...")
        print()

    total_tokens = sum(r["output_tokens"] for r in results)
    total_time = sum(r["elapsed_s"] for r in results)
    print(f"=== Summary: {total_tokens} tokens, {total_time:.1f}s, avg {total_tokens/total_time:.1f} tok/s ===")
    print("Check server logs for [DreamShiftBlockN] stats (tok/fwd, round_accept, spec_accept)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:30000")
    parser.add_argument("--n-problems", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    test_tpf(args.base_url, args.n_problems, args.max_tokens)
