#!/usr/bin/env python3
"""
Azure trace-driven traffic replay for sglang servers.
Reads Azure LLM Inference Trace CSV for request arrival pattern and token lengths.
Generates synthetic prompts matching the trace's context/generation token distribution.
Replays at real-time (or accelerated) pace.

Usage:
    python scripts/trace_replay.py \
        --trace assets/AzureLLMInferenceTrace_code_1week.csv \
        --port 37001 \
        --duration 300 \
        --peak-hour \
        --accelerate 1 \
        --output /tmp/trace_replay_results.json
"""

import argparse
import asyncio
import json
import random
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import aiohttp
import pandas as pd


def sample_trace_window(trace_path: str, duration: int, peak: bool = False,
                        offset_hour: Optional[int] = None) -> pd.DataFrame:
    """Sample a time window from the Azure trace."""
    df = pd.read_csv(trace_path)
    df['ts'] = pd.to_datetime(df['TIMESTAMP'])
    df['unix'] = df['ts'].apply(lambda x: x.timestamp())

    if offset_hour is not None:
        # Start at specific hour offset from beginning
        t0 = df['ts'].min() + timedelta(hours=offset_hour)
        t1 = t0 + timedelta(seconds=duration)
        window = df[(df['ts'] >= t0) & (df['ts'] < t1)].copy()
    elif peak:
        # Find peak hour, sample from its center
        df['hour'] = df['ts'].dt.floor('h')
        hourly = df.groupby('hour').size()
        peak_hour = hourly.idxmax()
        center = peak_hour + timedelta(minutes=30)
        t0 = center - timedelta(seconds=duration / 2)
        t1 = center + timedelta(seconds=duration / 2)
        window = df[(df['ts'] >= t0) & (df['ts'] < t1)].copy()
    else:
        # Random window
        min_t = df['ts'].min()
        max_t = df['ts'].max() - timedelta(seconds=duration)
        start = min_t + timedelta(seconds=random.random() * (max_t - min_t).total_seconds())
        window = df[(df['ts'] >= start) & (df['ts'] < start + timedelta(seconds=duration))].copy()

    # Normalize timestamps to start from 0
    t_start = window['unix'].min()
    window['relative_time'] = window['unix'] - t_start

    return window


def generate_prompt(target_tokens: int) -> List[dict]:
    """Generate a synthetic prompt of approximately target_tokens length.
    Uses a mix of reasoning and code prompts to simulate real traffic."""
    prompts_pool = [
        "Write a Python function that implements a binary search tree with insert, delete, and search operations. Include proper error handling and type hints.",
        "Explain the difference between TCP and UDP protocols. When would you use each one? Provide examples.",
        "Solve this math problem step by step: Find all integer solutions to x^3 + y^3 = z^3 + w^3 where 1 <= x,y,z,w <= 100.",
        "Debug this code and explain what's wrong:\n```python\ndef merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n```",
        "Design a REST API for a social media platform. Include endpoints for users, posts, comments, and likes. Provide OpenAPI specification.",
        "Write a recursive function to compute the nth Fibonacci number. Then optimize it using dynamic programming. Compare the time complexity.",
        "Explain how garbage collection works in Java vs Python. What are the trade-offs?",
        "Implement a thread-safe singleton pattern in C++. Explain why double-checked locking is needed.",
    ]

    # ~4 chars per token rough estimate
    target_chars = target_tokens * 4
    prompt = random.choice(prompts_pool)

    # Pad with context if needed
    if len(prompt) < target_chars:
        padding = "\n\nAdditional context: " + " ".join(
            random.choices(
                ["implement", "optimize", "debug", "explain", "design", "refactor",
                 "the", "function", "class", "method", "algorithm", "data structure",
                 "performance", "memory", "complexity", "async", "parallel", "distributed"],
                k=(target_chars - len(prompt)) // 8
            )
        )
        prompt = prompt + padding[:target_chars - len(prompt)]

    return [{"role": "user", "content": prompt[:target_chars]}]


async def send_request(session, url, messages, max_tokens, req_id, port,
                       lora_path=None):
    """Send a streaming request and collect metrics."""
    body = {
        "model": "test",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": True},
        "min_tokens": max_tokens,  # force generate exactly max_tokens
    }
    if lora_path:
        body["lora_path"] = lora_path

    t0 = time.time()
    ttft = 0
    token_count = 0
    completion_tokens = 0

    try:
        async with session.post(url, json=body) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {
                    "req_id": req_id, "port": port, "error": f"HTTP {resp.status}",
                    "completion_tokens": 0, "ttft": 0, "latency": time.time() - t0, "tps": 0,
                }

            first_token = True
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content") or delta.get("reasoning_content"):
                            if first_token:
                                ttft = time.time() - t0
                                first_token = False
                            token_count += 1
                    usage = chunk.get("usage")
                    if usage:
                        completion_tokens = usage.get("completion_tokens", 0)
                except json.JSONDecodeError:
                    pass

        if completion_tokens == 0:
            completion_tokens = token_count
        latency = time.time() - t0
        tps = completion_tokens / latency if latency > 0 and completion_tokens > 0 else 0

        return {
            "req_id": req_id, "port": port, "error": None,
            "completion_tokens": completion_tokens, "ttft": ttft,
            "latency": latency, "tps": tps,
        }
    except Exception as e:
        return {
            "req_id": req_id, "port": port, "error": str(e)[:200],
            "completion_tokens": 0, "ttft": 0, "latency": time.time() - t0, "tps": 0,
        }


async def run_trace_replay(args):
    print(f"Loading trace from {args.trace}...")
    window = sample_trace_window(
        args.trace, args.duration,
        peak=args.peak_hour,
        offset_hour=args.offset_hour,
    )
    print(f"Sampled {len(window)} requests over {args.duration}s window")
    print(f"QPS: {len(window)/args.duration:.1f}")
    print(f"Context tokens: mean={window.ContextTokens.mean():.0f}, median={window.ContextTokens.median():.0f}")
    print(f"Generated tokens: mean={window.GeneratedTokens.mean():.0f}, median={window.GeneratedTokens.median():.0f}")

    # Downsample to target peak QPS (preserves arrival pattern)
    if args.target_peak_qps:
        window['sec'] = window['ts'].dt.floor('s')
        max_qps = window.groupby('sec').size().max()
        if max_qps > args.target_peak_qps:
            ratio = args.target_peak_qps / max_qps
            window = window.sample(frac=ratio, random_state=42).sort_values('relative_time').reset_index(drop=True)
            new_max = window.groupby(window['ts'].dt.floor('s')).size().max()
            print(f"Downsampled {ratio:.2f}x for target peak QPS {args.target_peak_qps} "
                  f"(actual peak: {new_max})")

    if args.max_requests and len(window) > args.max_requests:
        # Subsample while preserving temporal distribution
        window = window.sample(n=args.max_requests).sort_values('relative_time').reset_index(drop=True)
        print(f"Subsampled to {len(window)} requests")

    # Cap context tokens to max_context and gen tokens to max_tokens
    if args.max_context:
        window['ContextTokens'] = window['ContextTokens'].clip(upper=args.max_context)

    if args.fixed_output_len:
        gen_tokens = [args.max_tokens] * len(window)
        print(f"Fixed output length: {args.max_tokens} tokens for all requests")
    else:
        gen_tokens = window['GeneratedTokens'].clip(upper=args.max_tokens).tolist()
    ctx_tokens = window['ContextTokens'].tolist()
    rel_times = window['relative_time'].tolist()

    ports = args.port
    urls = [f"http://localhost:{p}/v1/chat/completions" for p in ports]

    timeout = aiohttp.ClientTimeout(total=600)
    tasks = []

    print(f"\nStarting replay (accelerate={args.accelerate}x, {len(ports)} port(s))...")
    replay_start = time.time()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i in range(len(window)):
            # Wait for correct time
            target_elapsed = rel_times[i] / args.accelerate
            actual_elapsed = time.time() - replay_start
            wait = target_elapsed - actual_elapsed
            if wait > 0:
                await asyncio.sleep(wait)

            # Generate prompt matching trace's context token length
            messages = generate_prompt(ctx_tokens[i])
            max_tok = min(int(gen_tokens[i]), args.max_tokens)
            max_tok = max(max_tok, 16)  # minimum 16 tokens

            port = ports[i % len(ports)]
            url = urls[i % len(urls)]

            task = asyncio.create_task(
                send_request(session, url, messages, max_tok, i, port, args.lora_path)
            )
            tasks.append(task)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - replay_start
                done = sum(1 for t in tasks if t.done())
                print(f"  Sent {i+1}/{len(window)}, {done} done, {elapsed:.1f}s elapsed, "
                      f"QPS={i/elapsed:.1f}")

        print(f"All {len(window)} requests sent. Waiting for completions...")
        results = await asyncio.gather(*tasks)
        total_wall = time.time() - replay_start

    # Stats
    ok = [r for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]
    total_tokens = sum(r["completion_tokens"] for r in ok)
    job_tps = total_tokens / total_wall if total_wall > 0 else 0

    print(f"\n{'='*60}")
    print(f"Trace Replay Complete")
    print(f"{'='*60}")
    print(f"Duration: {total_wall:.1f}s (target: {args.duration/args.accelerate:.1f}s)")
    print(f"Requests: {len(ok)} ok, {len(errors)} errors")
    print(f"Total output tokens: {total_tokens}")
    print(f"Job-level TPS: {job_tps:.1f}")

    if ok:
        tps_list = [r["tps"] for r in ok if r["tps"] > 0]
        ttft_list = [r["ttft"] * 1000 for r in ok if r["ttft"] > 0]
        lat_list = [r["latency"] for r in ok]

        if tps_list:
            tps_sorted = sorted(tps_list)
            print(f"Per-request TPS: mean={sum(tps_list)/len(tps_list):.1f}, "
                  f"p50={tps_sorted[len(tps_sorted)//2]:.1f}, "
                  f"min={min(tps_list):.1f}")
        if ttft_list:
            ttft_sorted = sorted(ttft_list)
            print(f"TTFT (ms): mean={sum(ttft_list)/len(ttft_list):.1f}, "
                  f"p50={ttft_sorted[len(ttft_sorted)//2]:.1f}, "
                  f"p99={ttft_sorted[int(len(ttft_sorted)*0.99)]:.1f}")
        if lat_list:
            lat_sorted = sorted(lat_list)
            print(f"E2E latency (s): mean={sum(lat_list)/len(lat_list):.2f}, "
                  f"p50={lat_sorted[len(lat_sorted)//2]:.2f}, "
                  f"p99={lat_sorted[int(len(lat_sorted)*0.99)]:.2f}")

    if errors:
        from collections import Counter
        err_types = Counter(e["error"][:80] for e in errors)
        print(f"\nTop errors:")
        for msg, c in err_types.most_common(3):
            print(f"  {c}x: {msg}")

    if args.output:
        out = {
            "config": {
                "trace": args.trace,
                "duration": args.duration,
                "accelerate": args.accelerate,
                "peak_hour": args.peak_hour,
                "max_tokens": args.max_tokens,
                "ports": ports,
            },
            "summary": {
                "wall_time": total_wall,
                "total_requests": len(results),
                "ok": len(ok),
                "errors": len(errors),
                "total_tokens": total_tokens,
                "job_tps": job_tps,
                "per_req_tps_mean": sum(tps_list) / len(tps_list) if tps_list else 0,
                "ttft_mean_ms": sum(ttft_list) / len(ttft_list) if ttft_list else 0,
            },
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Azure trace-driven traffic replay")
    parser.add_argument("--trace", required=True, help="Azure trace CSV file")
    parser.add_argument("--port", type=int, action="append", required=True, help="Server port(s)")
    parser.add_argument("--duration", type=int, default=300, help="Window duration in seconds (default: 300)")
    parser.add_argument("--accelerate", type=float, default=1.0, help="Replay speed (1.0 = realtime)")
    parser.add_argument("--peak-hour", action="store_true", help="Sample from peak traffic hour")
    parser.add_argument("--offset-hour", type=int, default=None, help="Start at specific hour offset from trace start")
    parser.add_argument("--max-requests", type=int, default=None, help="Max requests to replay")
    parser.add_argument("--target-peak-qps", type=float, default=None, help="Downsample to this peak QPS")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument("--max-context", type=int, default=4096, help="Max context tokens per request")
    parser.add_argument("--fixed-output-len", action="store_true", help="Use --max-tokens for all requests (ignore trace GeneratedTokens)")
    parser.add_argument("--lora-path", default=None, help="LoRA adapter name")
    parser.add_argument("--output", default=None, help="Save results to JSON")
    args = parser.parse_args()
    asyncio.run(run_trace_replay(args))


if __name__ == "__main__":
    main()
