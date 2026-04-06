#!/usr/bin/env python3
"""Collect per-second TPS time series from DreamShift SGLang server.

Sends MBPP prompts at fixed concurrency and counts output tokens from
streaming responses, bucketing into per-second TPS.
"""

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

import aiohttp
from datasets import load_dataset


def format_mbpp_prompt(item):
    tests = "\n".join(item["test_list"])
    return (
        f"You are an expert Python programmer. Here is your task:\n"
        f"{item['prompt']}\n"
        f"Your code should pass these tests:\n{tests}"
    )


def load_mbpp_prompts(n=None):
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    prompts = [format_mbpp_prompt(item) for item in ds]
    if n:
        while len(prompts) < n:
            prompts = prompts + prompts
        prompts = prompts[:n]
    return prompts


# Shared state for token counting
token_buckets = defaultdict(int)  # second -> token count
start_time = 0.0


async def send_request(session, url, prompt, max_tokens, semaphore):
    """Send a single streaming request and count tokens per second."""
    global token_buckets, start_time
    async with semaphore:
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": 0.95,
            "stream": True,
        }
        try:
            async with session.post(url, json=payload) as resp:
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                if delta.get("content"):
                                    # Count usage if available, else estimate 1 token per chunk
                                    # SSE chunks typically correspond to 1 token
                                    elapsed = time.time() - start_time
                                    bucket = int(elapsed)
                                    token_buckets[bucket] += 1
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            print(f"Request error: {e}")


async def run_workload(server_url, prompts, concurrency, max_tokens, duration):
    """Send requests at fixed concurrency for the given duration."""
    global start_time
    url = f"{server_url}/v1/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)
    start_time = time.time()

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        tasks = []
        prompt_idx = 0

        # Keep launching requests until duration expires
        while time.time() - start_time < duration:
            if prompt_idx >= len(prompts):
                prompt_idx = 0
            task = asyncio.create_task(
                send_request(session, url, prompts[prompt_idx], max_tokens, semaphore)
            )
            tasks.append(task)
            prompt_idx += 1
            await asyncio.sleep(0.01)

        # Wait for remaining in-flight requests
        if tasks:
            await asyncio.wait(tasks, timeout=60)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:30000")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--duration", type=int, default=90, help="Duration in seconds")
    parser.add_argument("--output", default="scripts/demo/results/dreamshift_tps.json")
    parser.add_argument("--num-prompts", type=int, default=500)
    args = parser.parse_args()

    print(f"Loading MBPP prompts...")
    prompts = load_mbpp_prompts(args.num_prompts)
    print(f"Loaded {len(prompts)} prompts")

    print(f"Starting workload: concurrency={args.concurrency}, duration={args.duration}s")
    print(f"Server: {args.server_url}")

    await run_workload(args.server_url, prompts, args.concurrency, args.max_tokens, args.duration)

    # Convert buckets to time series
    if token_buckets:
        max_sec = max(token_buckets.keys())
        tps_series = []
        for sec in range(max_sec + 1):
            tps_series.append({"time": sec, "tps": token_buckets.get(sec, 0)})
    else:
        tps_series = []

    total_tokens = sum(token_buckets.values())

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "engine": "dreamshift",
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "duration": args.duration,
        "tps_series": tps_series,
        "total_tokens": total_tokens,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    tps_values = [p["tps"] for p in tps_series if p["tps"] > 0]
    if tps_values:
        avg_tps = sum(tps_values) / len(tps_values)
        max_tps = max(tps_values)
        print(f"\nDreamShift Results:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Average TPS:  {avg_tps:.0f}")
        print(f"  Peak TPS:     {max_tps:.0f}")
        print(f"  Data points:  {len(tps_series)}")
    else:
        print("\nWARNING: No tokens recorded!")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
