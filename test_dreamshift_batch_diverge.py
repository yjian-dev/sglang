"""
Test script to identify batching divergence in DreamShift DLLM.

Strategy:
  1. Launch server with max_running_requests=1 and generate outputs for N prompts
  2. Launch server with max_running_requests=K (K>1) and generate the same outputs
  3. Compare: with greedy decoding (temperature=0), identical prompts should
     produce identical outputs regardless of batch size.

Usage:
  # Step 1: Launch server (in another terminal or background)
  python -m sglang.launch_server \
    --model /data/cxu/dllm_experiments/sdar_qwen3_8b_dreamshift_rkl_both_8b_teacher_backup33250 \
    --dllm-algorithm DreamShift \
    --dllm-algorithm-config dreamshift_config.yaml \
    --trust-remote-code \
    --max-running-requests 1 \
    --port 30000

  # Step 2: Run this test with bs=1 server
  python test_dreamshift_batch_diverge.py --port 30000 --save-prefix bs1

  # Step 3: Restart server with max-running-requests 4 (or more)
  # Step 4: Run again
  python test_dreamshift_batch_diverge.py --port 30000 --save-prefix bs4

  # Step 5: Compare
  python test_dreamshift_batch_diverge.py --compare bs1 bs4
"""

import argparse
import json
import os
import time

import requests


# Simple GSM8K-style prompts for testing
TEST_PROMPTS = [
    "Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?\nAnswer:",
    "Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?\nAnswer:",
    "Question: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?\nAnswer:",
    "Question: James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?\nAnswer:",
]


def generate_one(host, port, prompt, max_tokens=256):
    """Send a single generate request and return the output."""
    url = f"{host}:{port}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_tokens,
            "stop": ["Question", "\n\n"],
        },
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("text", "")


def generate_batch(host, port, prompts, max_tokens=256, parallel=1):
    """Send prompts one at a time (sequential) and return outputs."""
    import concurrent.futures

    results = [None] * len(prompts)

    def _gen(idx):
        return idx, generate_one(host, port, prompts[idx], max_tokens)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        futs = [pool.submit(_gen, i) for i in range(len(prompts))]
        for fut in concurrent.futures.as_completed(futs):
            idx, text = fut.result()
            results[idx] = text

    return results


def run_generation(args):
    prompts = TEST_PROMPTS[: args.num_prompts]
    print(f"Generating {len(prompts)} prompts with parallel={args.parallel} ...")

    results = generate_batch(
        args.host, args.port, prompts,
        max_tokens=args.max_tokens,
        parallel=args.parallel,
    )

    # Save results
    out_file = f"{args.save_prefix}_results.json"
    data = {
        "prompts": prompts,
        "outputs": results,
        "parallel": args.parallel,
        "max_tokens": args.max_tokens,
    }
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(results)} results to {out_file}")

    for i, (p, o) in enumerate(zip(prompts, results)):
        print(f"\n--- Prompt {i} ---")
        print(p[:80] + "...")
        print(f"Output: {o[:200]}")


def compare_results(args):
    files = args.compare
    assert len(files) == 2, "Need exactly 2 result files to compare"

    with open(f"{files[0]}_results.json") as f:
        data1 = json.load(f)
    with open(f"{files[1]}_results.json") as f:
        data2 = json.load(f)

    assert data1["prompts"] == data2["prompts"], "Prompt mismatch!"

    n_match = 0
    n_total = len(data1["outputs"])
    for i in range(n_total):
        o1 = data1["outputs"][i]
        o2 = data2["outputs"][i]
        match = o1 == o2
        n_match += int(match)
        if not match:
            print(f"\n=== DIVERGENCE at prompt {i} ===")
            print(f"Prompt: {data1['prompts'][i][:80]}...")
            print(f"[{files[0]}]: {o1[:300]}")
            print(f"[{files[1]}]: {o2[:300]}")
            # Find first divergence point
            for j in range(min(len(o1), len(o2))):
                if o1[j] != o2[j]:
                    print(f"  First char diff at position {j}: '{o1[max(0,j-20):j+20]}' vs '{o2[max(0,j-20):j+20]}'")
                    break
            else:
                print(f"  Length diff: {len(o1)} vs {len(o2)}")

    print(f"\n=== Summary: {n_match}/{n_total} outputs match ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--parallel", type=int, default=4,
                        help="Number of concurrent requests (set to 1 for sequential)")
    parser.add_argument("--save-prefix", type=str, default="test")
    parser.add_argument("--compare", nargs=2, type=str,
                        help="Compare two saved result prefixes")
    args = parser.parse_args()

    if args.compare:
        compare_results(args)
    else:
        run_generation(args)
