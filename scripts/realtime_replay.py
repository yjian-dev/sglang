#!/usr/bin/env python3
"""
Realtime traffic replay for local sglang/vllm servers.
Reads JSONL with timestamps, replays requests at real-time pace (with optional acceleration).
Reports per-request TPS, TTFT, ITL, and job-level throughput.

Usage:
    python scripts/realtime_replay.py \
        --input /tmp/real_traffic_with_timestamps.jsonl \
        --port 37001 \
        --accelerate 10 \
        --num-requests 200 \
        --max-tokens 2048

    # Multiple ports (round-robin for comparison):
    python scripts/realtime_replay.py \
        --input /tmp/real_traffic_with_timestamps.jsonl \
        --port 37001 --port 37002 \
        --accelerate 5
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp


@dataclass
class RequestResult:
    req_id: int
    port: int
    completion_tokens: int
    prompt_tokens: int
    ttft: float  # time to first token (seconds)
    latency: float  # total e2e latency (seconds)
    tps: float  # tokens per second for this request
    error: Optional[str] = None


@dataclass
class Stats:
    results: List[RequestResult] = field(default_factory=list)

    def summary(self, label: str = ""):
        ok = [r for r in self.results if r.error is None]
        err = [r for r in self.results if r.error is not None]
        if not ok:
            print(f"  [{label}] No successful requests ({len(err)} errors)")
            return

        total_tokens = sum(r.completion_tokens for r in ok)
        wall_time = max(r.latency for r in ok)  # approximate
        tps_list = [r.tps for r in ok if r.tps > 0]
        ttft_list = [r.ttft * 1000 for r in ok if r.ttft > 0]  # ms
        lat_list = [r.latency for r in ok]

        # Job-level TPS: use actual wall time from first to last completion
        if len(ok) >= 2:
            job_tps = total_tokens / wall_time if wall_time > 0 else 0
        else:
            job_tps = tps_list[0] if tps_list else 0

        print(f"  [{label}] Requests: {len(ok)} ok, {len(err)} errors")
        print(f"  [{label}] Total tokens: {total_tokens}")
        print(f"  [{label}] Job-level TPS: {job_tps:.1f}")
        if tps_list:
            avg_tps = sum(tps_list) / len(tps_list)
            print(f"  [{label}] Per-request TPS: {avg_tps:.1f} (min={min(tps_list):.1f}, max={max(tps_list):.1f})")
        if ttft_list:
            avg_ttft = sum(ttft_list) / len(ttft_list)
            print(f"  [{label}] TTFT: {avg_ttft:.1f}ms (min={min(ttft_list):.1f}, max={max(ttft_list):.1f})")
        if lat_list:
            avg_lat = sum(lat_list) / len(lat_list)
            print(f"  [{label}] E2E latency: {avg_lat:.2f}s (min={min(lat_list):.2f}, max={max(lat_list):.2f})")


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    req: dict,
    req_id: int,
    port: int,
    max_tokens: int,
    lora_path: Optional[str] = None,
) -> RequestResult:
    body = {
        "model": req.get("model", "test"),
        "messages": req["messages"],
        "max_tokens": max_tokens,
        "temperature": req.get("temperature", 1.0),
        "top_p": req.get("top_p", 0.95),
        "stream": True,
    }
    if lora_path:
        body["lora_path"] = lora_path

    t0 = time.time()
    ttft = 0
    completion_tokens = 0
    prompt_tokens = 0

    try:
        async with session.post(url, json=body) as resp:
            if resp.status != 200:
                text = await resp.text()
                return RequestResult(
                    req_id=req_id, port=port,
                    completion_tokens=0, prompt_tokens=0,
                    ttft=0, latency=time.time() - t0, tps=0,
                    error=f"HTTP {resp.status}: {text[:200]}",
                )

            first_token = True
            token_count = 0
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    if first_token:
                        # Check if this chunk has actual content
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if delta.get("content") or delta.get("reasoning_content"):
                                ttft = time.time() - t0
                                first_token = False
                    # Count tokens from delta content
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "") or ""
                        reasoning = delta.get("reasoning_content", "") or ""
                        if content or reasoning:
                            token_count += 1
                    # Extract usage from final chunk
                    usage = chunk.get("usage")
                    if usage:
                        completion_tokens = usage.get("completion_tokens", 0)
                        prompt_tokens = usage.get("prompt_tokens", 0)
                except json.JSONDecodeError:
                    pass
            # Fallback: use counted tokens if usage not reported
            if completion_tokens == 0:
                completion_tokens = token_count

        latency = time.time() - t0
        tps = completion_tokens / latency if latency > 0 and completion_tokens > 0 else 0
        return RequestResult(
            req_id=req_id, port=port,
            completion_tokens=completion_tokens, prompt_tokens=prompt_tokens,
            ttft=ttft, latency=latency, tps=tps,
        )

    except Exception as e:
        return RequestResult(
            req_id=req_id, port=port,
            completion_tokens=0, prompt_tokens=0,
            ttft=0, latency=time.time() - t0, tps=0,
            error=str(e),
        )


async def run_replay(args):
    # Load requests
    with open(args.input) as f:
        requests = [json.loads(line) for line in f]

    # Filter by max prompt length
    if args.max_prompt_chars:
        requests = [
            r for r in requests
            if sum(len(m.get("content", "") or "") for m in r.get("messages", []))
            <= args.max_prompt_chars
        ]

    if args.num_requests:
        requests = requests[: args.num_requests]

    print(f"Loaded {len(requests)} requests")

    # Sort by timestamp
    requests.sort(key=lambda r: r.get("timestamp", 0))

    # Check timestamps
    has_timestamps = all("timestamp" in r for r in requests)
    if has_timestamps:
        t_start = requests[0]["timestamp"]
        t_end = requests[-1]["timestamp"]
        span = t_end - t_start
        print(f"Time span: {span:.1f}s, accelerate: {args.accelerate}x")
        print(f"Replay will take ~{span / args.accelerate:.1f}s")
    else:
        print("No timestamps found, sending at constant QPS")

    ports = args.port
    urls = [f"http://localhost:{p}/v1/chat/completions" for p in ports]

    timeout = aiohttp.ClientTimeout(total=600)

    # Per-port stats
    port_stats = {p: Stats() for p in ports}
    tasks = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        replay_start = time.time()

        for i, req in enumerate(requests):
            # Determine target port (round-robin)
            port = ports[i % len(ports)]
            url = urls[i % len(urls)]

            # Wait for correct replay time
            if has_timestamps and i > 0:
                req_time = req["timestamp"]
                prev_time = requests[0]["timestamp"]
                target_elapsed = (req_time - prev_time) / args.accelerate
                actual_elapsed = time.time() - replay_start
                wait = target_elapsed - actual_elapsed
                if wait > 0:
                    await asyncio.sleep(wait)

            # Launch request (don't await, let it run concurrently)
            task = asyncio.create_task(
                send_request(session, url, req, i, port, args.max_tokens, args.lora_path)
            )
            tasks.append(task)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - replay_start
                done = sum(1 for t in tasks if t.done())
                print(f"  Sent {i + 1}/{len(requests)}, {done} done, {elapsed:.1f}s elapsed")

        print(f"All {len(requests)} requests sent. Waiting for completions...")

        # Wait for all to finish
        results = await asyncio.gather(*tasks)
        total_wall = time.time() - replay_start

    # Collect per-port stats
    for r in results:
        port_stats[r.port].results.append(r)

    # Report
    print(f"\n{'='*60}")
    print(f"Replay complete: {len(requests)} requests in {total_wall:.1f}s")
    print(f"{'='*60}")

    all_ok = [r for r in results if r.error is None]
    all_err = [r for r in results if r.error is not None]
    total_tokens = sum(r.completion_tokens for r in all_ok)
    job_tps = total_tokens / total_wall if total_wall > 0 else 0

    print(f"\nGlobal: {len(all_ok)} ok, {len(all_err)} errors")
    print(f"Total output tokens: {total_tokens}")
    print(f"Job-level TPS: {job_tps:.1f}")
    if all_ok:
        tps_list = [r.tps for r in all_ok if r.tps > 0]
        ttft_list = [r.ttft * 1000 for r in all_ok if r.ttft > 0]
        if tps_list:
            print(f"Per-request TPS: mean={sum(tps_list)/len(tps_list):.1f}, "
                  f"p50={sorted(tps_list)[len(tps_list)//2]:.1f}, "
                  f"min={min(tps_list):.1f}")
        if ttft_list:
            print(f"TTFT (ms): mean={sum(ttft_list)/len(ttft_list):.1f}, "
                  f"p50={sorted(ttft_list)[len(ttft_list)//2]:.1f}, "
                  f"p99={sorted(ttft_list)[int(len(ttft_list)*0.99)]:.1f}")

    if len(ports) > 1:
        print(f"\nPer-port breakdown:")
        for p in ports:
            port_stats[p].summary(label=f"port={p}")

    # Save detailed results
    if args.output:
        out = []
        for r in results:
            out.append({
                "req_id": r.req_id,
                "port": r.port,
                "completion_tokens": r.completion_tokens,
                "prompt_tokens": r.prompt_tokens,
                "ttft": r.ttft,
                "latency": r.latency,
                "tps": r.tps,
                "error": r.error,
            })
        with open(args.output, "w") as f:
            json.dump({"wall_time": total_wall, "job_tps": job_tps, "results": out}, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Realtime traffic replay for sglang servers")
    parser.add_argument("--input", required=True, help="JSONL file with requests (needs 'messages' and 'timestamp')")
    parser.add_argument("--port", type=int, action="append", required=True, help="Server port(s)")
    parser.add_argument("--accelerate", type=float, default=1.0, help="Replay speed multiplier (default: 1.0 = realtime)")
    parser.add_argument("--num-requests", type=int, default=None, help="Max requests to send")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Override max_tokens")
    parser.add_argument("--max-prompt-chars", type=int, default=20000, help="Filter out prompts longer than this")
    parser.add_argument("--lora-path", default=None, help="LoRA adapter name to include in requests")
    parser.add_argument("--output", default=None, help="Save detailed results to JSON file")
    args = parser.parse_args()
    asyncio.run(run_replay(args))


if __name__ == "__main__":
    main()
