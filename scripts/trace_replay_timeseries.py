#!/usr/bin/env python3
"""
Azure trace replay with time-series TPS recording.
Outputs per-second TPS for plotting throughput over time.
"""
import argparse
import asyncio
import json
import time
from collections import defaultdict

import aiohttp
import pandas as pd


def load_and_sample_trace(trace_path, duration, peak, target_peak_qps, max_context):
    df = pd.read_csv(trace_path)
    df['ts'] = pd.to_datetime(df['TIMESTAMP'])

    if peak:
        from datetime import timedelta
        df['hour'] = df['ts'].dt.floor('h')
        hourly = df.groupby('hour').size()
        peak_hour = hourly.idxmax()
        center = peak_hour + timedelta(minutes=30)
        t0 = center - timedelta(seconds=duration / 2)
        t1 = center + timedelta(seconds=duration / 2)
        window = df[(df['ts'] >= t0) & (df['ts'] < t1)].copy()
    else:
        window = df.head(int(duration * 100))  # rough estimate

    window['unix'] = window['ts'].apply(lambda x: x.timestamp())
    t_start = window['unix'].min()
    window['relative_time'] = window['unix'] - t_start

    if target_peak_qps:
        window['sec'] = window['ts'].dt.floor('s')
        max_qps = window.groupby('sec').size().max()
        if max_qps > target_peak_qps:
            ratio = target_peak_qps / max_qps
            window = window.sample(frac=ratio, random_state=42).sort_values('relative_time').reset_index(drop=True)

    if max_context:
        window['ContextTokens'] = window['ContextTokens'].clip(upper=max_context)

    return window


async def send_one(session, url, prompt, max_tokens, req_id, lora_path, replay_start):
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0, "top_p": 0.95,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
        "ignore_eos": True,
    }
    if lora_path:
        body["lora_path"] = lora_path

    send_time = time.time() - replay_start
    t0 = time.time()
    try:
        async with session.post(url, json=body) as resp:
            data = await resp.json()
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            done_time = time.time() - replay_start
            return {"req_id": req_id, "tokens": tokens, "send_time": send_time,
                    "done_time": done_time, "latency": time.time() - t0, "error": None}
    except Exception as e:
        return {"req_id": req_id, "tokens": 0, "send_time": send_time,
                "done_time": time.time() - replay_start, "latency": time.time() - t0,
                "error": str(e)[:100]}


import random
import os

# Load MATH-500 prompts if available, otherwise use fallback
_MATH500_PATH = "/tmp/math500.jsonl"
if os.path.exists(_MATH500_PATH):
    import json as _json
    with open(_MATH500_PATH) as _f:
        PROMPTS = [_json.loads(line)["prompt"] for line in _f]
    print(f"Loaded {len(PROMPTS)} MATH-500 prompts")
else:
    PROMPTS = [
        "Write a Python function that implements a binary search tree with insert, delete, and search operations.",
        "Explain the difference between TCP and UDP protocols. When would you use each one?",
        "Design a REST API for a social media platform with users, posts, comments.",
        "Implement a thread-safe singleton pattern in C++. Explain double-checked locking.",
        "Write a recursive Fibonacci function, then optimize with dynamic programming.",
        "Debug this merge sort implementation and explain the fix.",
        "Explain garbage collection in Java vs Python with trade-offs.",
        "Solve: find all integer solutions to x^2 + y^2 = z^2 where 1 <= x,y,z <= 1000.",
    ]
    print(f"Using {len(PROMPTS)} fallback prompts")


async def run(args):
    window = load_and_sample_trace(
        args.trace, args.duration, args.peak_hour,
        args.target_peak_qps, args.max_context,
    )
    n = len(window)
    print(f"Loaded {n} requests, duration={args.duration}s, target_peak={args.target_peak_qps}")

    ctx_tokens = window['ContextTokens'].tolist()
    rel_times = window['relative_time'].tolist()

    url = f"http://localhost:{args.port}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=600)
    tasks = []
    replay_start = time.time()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Warmup
        body = {"model": "test", "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 64, "temperature": 1.0}
        if args.lora_path:
            body["lora_path"] = args.lora_path
        async with session.post(url, json=body) as r:
            await r.json()

        replay_start = time.time()
        for i in range(n):
            # Wait for replay time
            target = rel_times[i] / args.accelerate
            actual = time.time() - replay_start
            if target > actual:
                await asyncio.sleep(target - actual)

            # Pick a MATH-500 prompt (cycle through)
            prompt = PROMPTS[i % len(PROMPTS)]

            task = asyncio.create_task(
                send_one(session, url, prompt, args.max_tokens, i, args.lora_path, replay_start)
            )
            tasks.append(task)

            if (i + 1) % 200 == 0:
                done = sum(1 for t in tasks if t.done())
                print(f"  Sent {i+1}/{n}, {done} done, {time.time()-replay_start:.1f}s")

        print(f"All {n} sent. Waiting...")
        results = await asyncio.gather(*tasks)
        total_wall = time.time() - replay_start

    # Compute per-second TPS time series
    ok = [r for r in results if r["error"] is None and r["tokens"] > 0]
    errors = [r for r in results if r["error"] is not None]

    # Method: for each second, sum tokens of requests that completed in that second
    bucket_size = 5  # 5-second buckets for smoother curve
    max_time = max(r["done_time"] for r in ok) if ok else 0
    n_buckets = int(max_time / bucket_size) + 1
    tps_series = []
    for b in range(n_buckets):
        t0 = b * bucket_size
        t1 = (b + 1) * bucket_size
        tokens_in_bucket = sum(r["tokens"] for r in ok if t0 <= r["done_time"] < t1)
        tps = tokens_in_bucket / bucket_size
        tps_series.append({"time": t0 + bucket_size / 2, "tps": tps})

    # Also compute active requests over time (requests sent but not yet done)
    active_series = []
    for b in range(n_buckets):
        t = b * bucket_size + bucket_size / 2
        active = sum(1 for r in results if r["send_time"] <= t < r["done_time"])
        active_series.append({"time": t, "active": active})

    # Summary
    total_tokens = sum(r["tokens"] for r in ok)
    job_tps = total_tokens / total_wall if total_wall > 0 else 0
    per_req = [r["tokens"] / r["latency"] for r in ok if r["latency"] > 0]

    print(f"\n{'='*50}")
    print(f"Requests: {len(ok)} ok, {len(errors)} errors")
    print(f"Total tokens: {total_tokens}, Wall: {total_wall:.1f}s")
    print(f"Job TPS: {job_tps:.1f}")
    if per_req:
        print(f"Per-req TPS: mean={sum(per_req)/len(per_req):.1f}")

    # Per-request TPS time series: average per-req TPS of requests completing in each bucket
    per_req_tps_series = []
    for b in range(n_buckets):
        t0_b = b * bucket_size
        t1_b = (b + 1) * bucket_size
        reqs_in_bucket = [r for r in ok if t0_b <= r["done_time"] < t1_b and r["latency"] > 0]
        if reqs_in_bucket:
            avg_per_req = sum(r["tokens"] / r["latency"] for r in reqs_in_bucket) / len(reqs_in_bucket)
        else:
            avg_per_req = 0
        per_req_tps_series.append({"time": t0_b + bucket_size / 2, "per_req_tps": avg_per_req})

    per_req_mean = sum(r["tokens"] / r["latency"] for r in ok if r["latency"] > 0) / len(ok) if ok else 0

    out = {
        "label": args.label,
        "config": {"port": args.port, "duration": args.duration,
                   "peak_qps": args.target_peak_qps, "max_tokens": args.max_tokens},
        "summary": {"ok": len(ok), "errors": len(errors), "total_tokens": total_tokens,
                     "wall_time": total_wall, "job_tps": job_tps, "per_req_tps_mean": per_req_mean},
        "tps_series": tps_series,
        "per_req_tps_series": per_req_tps_series,
        "active_series": active_series,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--peak-hour", action="store_true")
    parser.add_argument("--target-peak-qps", type=float, default=24)
    parser.add_argument("--accelerate", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--label", default="model")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
