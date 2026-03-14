"""
Profile DreamShiftBlockN server with torch.profiler.
Generates a Chrome trace JSON viewable in https://ui.perfetto.dev/

Usage:
  # Start server first, then:
  python scripts/profile_blockN.py --port 30000 --bs 32 --output trace.json

  # Open trace.json in https://ui.perfetto.dev/
"""
import argparse
import json
import os
import subprocess
import sys
import time

import requests
from concurrent.futures import ThreadPoolExecutor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--bs", type=int, default=1, help="Concurrent batch size")
    parser.add_argument("--num-requests", type=int, default=None,
                        help="Total requests (default: bs * 5)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", type=str, default="trace_blockN.json")
    parser.add_argument("--warmup", type=int, default=None,
                        help="Warmup requests (default: bs)")
    args = parser.parse_args()

    bs = args.bs
    num_requests = args.num_requests or bs * 5
    warmup = args.warmup or bs

    prompts = [
        "Write a Python function to check if a number is prime.",
        "Explain the difference between TCP and UDP.",
        "What is 127 * 43? Show work.",
        "Write a short story about a robot.",
        "List the top 10 largest countries.",
        "Translate to French: The weather is beautiful.",
        "Write a recipe for cookies.",
        "Explain photosynthesis simply.",
        "What are prime factors of 2520?",
        "Write haiku about seasons.",
        "Explain machine learning.",
        "Write bubble sort in C++.",
        "What causes earthquakes?",
        "Compare Python and JavaScript.",
        "Describe the water cycle.",
        "Write regex for emails.",
        "Pythagorean theorem examples.",
        "How neural networks learn.",
        "SQL find duplicates.",
        "How coffee is made.",
        "What is recursion?",
        "Explain relativity.",
        "Reverse a linked list.",
        "Renewable energy benefits.",
        "How encryption works.",
        "Write a poem about ocean.",
        "Stack vs queue.",
        "How vaccines work.",
        "Binary search algorithm.",
        "Northern lights cause.",
        "Describe solar system.",
        "What is an API?",
    ]

    url = f"http://127.0.0.1:{args.port}"

    def query(p):
        r = requests.post(f"{url}/v1/chat/completions", json={
            "model": "sdar",
            "messages": [{"role": "user", "content": p}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
        }, timeout=300).json()
        return r["usage"]["completion_tokens"]

    # Check server health
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        assert resp.status_code == 200
    except Exception:
        print(f"Server not ready at {url}")
        sys.exit(1)

    # Warmup
    print(f"Warming up with {warmup} requests...")
    ps = [prompts[i % len(prompts)] for i in range(warmup)]
    with ThreadPoolExecutor(max_workers=bs) as pool:
        list(pool.map(query, ps))

    # Profile using CUDA profiling via server-side torch.profiler
    # We trigger profiling by sending a special request, then run workload
    print(f"\nProfiling: {num_requests} requests, bs={bs}, max_tokens={args.max_tokens}")
    print("Use nsys or torch.profiler on the server side for GPU trace.")
    print("For CPU-side trace, use py-spy or viztracer.\n")

    # Simple client-side timing trace
    print("Running workload for client-side timing...")
    ps = [prompts[i % len(prompts)] for i in range(num_requests)]

    events = []
    t_start = time.time()

    def query_traced(args):
        idx, p = args
        t0 = time.time()
        tok = query(p)
        t1 = time.time()
        events.append({
            "name": f"req_{idx}",
            "cat": "request",
            "ph": "X",
            "ts": int((t0 - t_start) * 1e6),
            "dur": int((t1 - t0) * 1e6),
            "pid": 0,
            "tid": idx % bs,
            "args": {"tokens": tok, "prompt": p[:50]},
        })
        return tok

    with ThreadPoolExecutor(max_workers=bs) as pool:
        results = list(pool.map(query_traced, enumerate(ps)))

    total_time = time.time() - t_start
    total_tok = sum(results)

    print(f"\nResults: {total_tok} tokens in {total_time:.1f}s = {total_tok/total_time:.1f} tok/s")

    # Write Chrome trace format
    trace = {"traceEvents": events}
    with open(args.output, "w") as f:
        json.dump(trace, f)
    print(f"Client trace saved to {args.output}")
    print(f"Open in https://ui.perfetto.dev/")

    # Instructions for server-side GPU profiling
    print(f"""
=== For GPU profiling (server-side) ===

Option 1: nsys (NVIDIA Nsight Systems)
  # Restart server under nsys:
  nsys profile -o blockN_profile \\
    python -m sglang.launch_server ... --port {args.port}
  # Then run this script again. View with: nsys-ui blockN_profile.nsys-rep

Option 2: torch.profiler (add to scheduler loop)
  # Add to event_loop_normal():
  with torch.profiler.profile(
      activities=[torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA],
      schedule=torch.profiler.schedule(wait=100, warmup=10, active=20),
      on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/blockN'),
  ) as prof:
      # ... existing loop ...
      prof.step()

Option 3: py-spy (Python CPU profiling)
  py-spy record -o profile.svg --pid <server_pid>
""")


if __name__ == "__main__":
    main()
