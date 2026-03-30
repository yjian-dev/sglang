#!/usr/bin/env python3
"""
Benchmark TPF (tokens per forward) vs TPS (tokens per second) for DLLM algorithms.

Compares DreamShift (ours) vs SDAR baseline (LowConfidence) on the same model.
DreamShift TPF varies by dataset; SDAR TPF is controlled by confidence threshold.

Usage:
  # Full automated sweep (starts/stops servers automatically):
  python scripts/bench_tpf_vs_tps.py --sweep-all \
      --model-path /data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b1-allmasked_fixed2 \
      --concurrency 4 --output tpf_results.json

  # Just DreamShift:
  python scripts/bench_tpf_vs_tps.py --sweep-dreamshift \
      --model-path /data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b1-allmasked_fixed2

  # Just SDAR baseline:
  python scripts/bench_tpf_vs_tps.py --sweep-sdar \
      --model-path /data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b1-allmasked_fixed2

  # Single data point against running server:
  python scripts/bench_tpf_vs_tps.py --port 30000 --concurrency 4 --label "DreamShift N=2"

  # Plot from saved results:
  python scripts/bench_tpf_vs_tps.py --plot tpf_results.json
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import concurrent.futures

import requests as http_requests


# ── Constants ────────────────────────────────────────────────────────────

MODEL_PATH = "/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b1-allmasked_fixed2"

# N=2: gen_block_size=2, block_size=2*2-1=3
DREAMSHIFT_N = 2
DREAMSHIFT_BLOCK_SIZE = 2 * DREAMSHIFT_N - 1  # 3

# SDAR baseline: LowConfidence uses the same block_size
SDAR_BLOCK_SIZE = DREAMSHIFT_BLOCK_SIZE  # 3

# Threshold sweep for SDAR (gives TPF from ~block_size/2 down to ~block_size/(block_size+1))
SDAR_THRESHOLDS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]

# Datasets for DreamShift (varying difficulty → varying TPF)
DREAMSHIFT_DATASETS = ["simple", "conversation", "math", "code"]

PROMPT_SETS = {
    "math": [
        "What is 127 * 453? Show your step-by-step calculation.",
        "Solve the quadratic equation: 3x^2 - 7x + 2 = 0. Show all work.",
        "A train travels 120 km at 60 km/h, then 80 km at 40 km/h. What is the average speed?",
        "Find the derivative of f(x) = x^3 * sin(x) using the product rule.",
        "If a rectangle has perimeter 36 cm and length is twice its width, find dimensions.",
        "Calculate the sum of the first 50 positive integers.",
        "A bag has 5 red and 3 blue balls. P(drawing 2 red without replacement)?",
        "Simplify: (2^3 * 3^2) / (2^2 * 3^3)",
    ],
    "code": [
        "Write a Python function to find the longest common subsequence of two strings.",
        "Implement a binary search tree in Python with insert, delete, and search.",
        "Write a Python function that checks if a string of parentheses is balanced.",
        "Implement merge sort in Python and explain its time complexity.",
        "Write a Python class for a LRU cache with get and put operations.",
        "Implement Dijkstra's shortest path algorithm in Python.",
        "Write a Python function to detect a cycle in a linked list.",
        "Implement a trie data structure in Python with insert and search.",
    ],
    "conversation": [
        "Tell me about the history of artificial intelligence in a few paragraphs.",
        "Explain how photosynthesis works to a 10-year-old.",
        "What are the main differences between Python and JavaScript?",
        "Describe the process of making bread from scratch.",
        "What are the key factors that led to the fall of the Roman Empire?",
        "Explain the concept of blockchain in simple terms.",
        "What are the benefits and drawbacks of remote work?",
        "Describe the water cycle and its importance to life on Earth.",
    ],
    "simple": [
        "Count from 1 to 100.",
        "List the months of the year and their number of days.",
        "Write the alphabet forwards and backwards.",
        "List the first 20 prime numbers.",
        "Name all 50 US states in alphabetical order.",
        "Write a simple nursery rhyme.",
        "List the planets in our solar system from closest to farthest from the sun.",
        "Write the multiplication table for 7.",
    ],
}


# ── Server helpers ───────────────────────────────────────────────────────


def wait_for_server(base_url: str, timeout: int = 600):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = http_requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"Server not ready after {timeout}s")


def start_server(model_path, algorithm, algorithm_config, port, extra_args=None):
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--dllm-algorithm", algorithm,
        "--dllm-algorithm-config", algorithm_config,
        "--tp-size", "1",
        "--trust-remote-code",
        "--mem-fraction-static", "0.85",
        "--max-running-requests", "24",
        "--attention-backend", "flashinfer",
        "--port", str(port),
        "--dtype", "bfloat16",
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["PATH"] = "/usr/local/cuda-12.9/bin:" + env.get("PATH", "")
    env["CUDA_HOME"] = "/usr/local/cuda-12.9"

    print(f"\n>>> Starting server: {algorithm} on port {port}")
    print(f"    Config: {algorithm_config}")
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        wait_for_server(f"http://localhost:{port}")
        print("    Server ready!")
    except TimeoutError:
        proc.kill()
        raise
    return proc


def stop_server(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print("    Server stopped.")


def write_yaml(path, data):
    with open(path, "w") as f:
        for k, v in data.items():
            if isinstance(v, bool):
                f.write(f"{k}: {'true' if v else 'false'}\n")
            else:
                f.write(f"{k}: {v}\n")


# ── Benchmark helpers ────────────────────────────────────────────────────


def get_dllm_stats(base_url):
    r = http_requests.get(f"{base_url}/get_server_info")
    info = r.json()
    states = info.get("internal_states", [{}])
    state = states[0] if isinstance(states, list) and states else states
    return state.get("dllm_stats", {})


def send_requests_batch(base_url, prompts, output_len, concurrency):
    """Send requests with fixed concurrency, return total output tokens and wall time."""
    total_output_tokens = 0

    def send_one(prompt):
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_len,
            "temperature": 1.0,
            "top_p": 0.95,
            "stream": False,
            "ignore_eos": True,
        }
        try:
            r = http_requests.post(
                f"{base_url}/v1/chat/completions",
                json=payload, timeout=600,
            )
            usage = r.json().get("usage", {})
            return usage.get("completion_tokens", 0)
        except Exception as e:
            print(f"    Request failed: {e}")
            return 0

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send_one, p) for p in prompts]
        for f in concurrent.futures.as_completed(futures):
            total_output_tokens += f.result()
    elapsed = time.perf_counter() - t0

    return total_output_tokens, elapsed


def run_one_point(base_url, num_requests, concurrency, output_len, dataset, label,
                  warmup=10):
    """Measure one (TPF, TPS) data point."""
    prompts_pool = PROMPT_SETS.get(dataset, PROMPT_SETS["math"])
    all_prompts = [prompts_pool[i % len(prompts_pool)]
                   for i in range(warmup + num_requests)]

    # Warmup
    print(f"  [{label}] Warming up ({warmup} reqs)...")
    send_requests_batch(base_url, all_prompts[:warmup], output_len, concurrency)
    time.sleep(2)

    # Snapshot stats before
    stats_before = get_dllm_stats(base_url)

    # Benchmark
    print(f"  [{label}] Benchmarking ({num_requests} reqs, C={concurrency})...")
    total_tokens, elapsed = send_requests_batch(
        base_url, all_prompts[warmup:], output_len, concurrency,
    )

    # Snapshot stats after
    stats_after = get_dllm_stats(base_url)

    # Compute TPF from delta
    d_fwd = stats_after.get("total_forwards", 0) - stats_before.get("total_forwards", 0)
    d_tok = stats_after.get("total_tokens", 0) - stats_before.get("total_tokens", 0)
    tpf = d_tok / max(d_fwd, 1)
    tps = total_tokens / max(elapsed, 1e-6)

    print(f"  [{label}] TPF={tpf:.3f}  TPS={tps:.1f}  "
          f"(fwd={d_fwd}, tok={d_tok}, elapsed={elapsed:.1f}s)")

    return {
        "label": label,
        "algorithm": "",  # filled by caller
        "dataset": dataset,
        "concurrency": concurrency,
        "num_requests": num_requests,
        "output_len": output_len,
        "tpf": round(tpf, 4),
        "tps": round(tps, 2),
        "total_output_tokens": total_tokens,
        "elapsed_seconds": round(elapsed, 2),
        "delta_forwards": d_fwd,
        "delta_tokens": d_tok,
    }


# ── Sweep modes ──────────────────────────────────────────────────────────


def sweep_sdar(args):
    """Sweep LowConfidence threshold → varying TPF."""
    results = []
    config_path = "/tmp/_sdar_lc_config.yaml"

    for thr in SDAR_THRESHOLDS:
        print(f"\n{'='*60}")
        print(f"SDAR LowConfidence  threshold={thr}  block_size={SDAR_BLOCK_SIZE}")
        print(f"{'='*60}")

        write_yaml(config_path, {
            "block_size": SDAR_BLOCK_SIZE,
            "threshold": thr,
        })

        proc = start_server(
            model_path=args.model_path,
            algorithm="LowConfidence",
            algorithm_config=config_path,
            port=args.port,
        )
        try:
            dp = run_one_point(
                base_url=f"http://localhost:{args.port}",
                num_requests=args.num_requests,
                concurrency=args.concurrency,
                output_len=args.output_len,
                dataset="math",
                label=f"SDAR thr={thr}",
            )
            dp["algorithm"] = "LowConfidence"
            dp["threshold"] = thr
            results.append(dp)
        finally:
            stop_server(proc)
            time.sleep(5)

    return results


def sweep_dreamshift(args):
    """Run DreamShift N=2 with different datasets → varying TPF."""
    results = []
    config_path = args.dreamshift_config or "/tmp/_ds_n2_config.yaml"

    if not args.dreamshift_config:
        write_yaml(config_path, {
            "block_size": DREAMSHIFT_BLOCK_SIZE,
            "gen_block_size": DREAMSHIFT_N,
            "confidence_threshold": 0.0,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
            "use_spec_verify": True,
        })

    proc = start_server(
        model_path=args.model_path,
        algorithm="DreamShiftBlockN",
        algorithm_config=config_path,
        port=args.port,
    )
    try:
        for ds in DREAMSHIFT_DATASETS:
            print(f"\n{'='*60}")
            print(f"DreamShift N={DREAMSHIFT_N}  dataset={ds}")
            print(f"{'='*60}")

            dp = run_one_point(
                base_url=f"http://localhost:{args.port}",
                num_requests=args.num_requests,
                concurrency=args.concurrency,
                output_len=args.output_len,
                dataset=ds,
                label=f"DreamShift {ds}",
            )
            dp["algorithm"] = "DreamShiftBlockN"
            results.append(dp)
    finally:
        stop_server(proc)
        time.sleep(5)

    return results


def single_benchmark(args):
    """Benchmark against an already-running server."""
    dp = run_one_point(
        base_url=f"http://localhost:{args.port}",
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        output_len=args.output_len,
        dataset=args.dataset,
        label=args.label,
    )
    return [dp]


# ── Plotting ─────────────────────────────────────────────────────────────


def plot_results(results_file, output_file=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(results_file) as f:
        all_results = json.load(f)

    # Group: SDAR vs DreamShift
    groups = {}
    for dp in all_results:
        algo = dp.get("algorithm", "")
        if "LowConfidence" in algo:
            key = "SDAR (LowConfidence)"
        elif "DreamShift" in algo:
            key = "DreamShift (Ours)"
        else:
            key = dp.get("label", "unknown")
        groups.setdefault(key, []).append(dp)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    styles = {
        "SDAR (LowConfidence)": {"color": "#2196F3", "marker": "o"},
        "DreamShift (Ours)":    {"color": "#E53935", "marker": "s"},
    }

    for name, points in sorted(groups.items()):
        # Sort by TPF
        points_sorted = sorted(points, key=lambda p: p["tpf"])
        xs = [p["tpf"] for p in points_sorted]
        ys = [p["tps"] for p in points_sorted]

        st = styles.get(name, {"color": "#666", "marker": "^"})
        ax.plot(xs, ys, marker=st["marker"], color=st["color"],
                linewidth=2.2, markersize=9, label=name, zorder=5)

        # Annotate each point
        for p in points_sorted:
            txt = p.get("dataset") or f"thr={p.get('threshold', '?')}"
            ax.annotate(txt, (p["tpf"], p["tps"]),
                        textcoords="offset points", xytext=(6, 8),
                        fontsize=7.5, color=st["color"], alpha=0.85)

    c = all_results[0].get("concurrency", "?")
    ax.set_xlabel("Tokens Per Forward (TPF)", fontsize=13)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=13)
    ax.set_title(f"Batching Efficiency: TPF vs TPS  (batch_size={c})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.5)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = output_file or results_file.replace(".json", ".png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Benchmark TPF vs TPS")
    p.add_argument("--sweep-all", action="store_true",
                   help="Run both SDAR + DreamShift sweeps")
    p.add_argument("--sweep-sdar", action="store_true")
    p.add_argument("--sweep-dreamshift", action="store_true")
    p.add_argument("--plot", type=str, help="Just plot from results JSON")

    p.add_argument("--model-path", type=str, default=MODEL_PATH)
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--dreamshift-config", type=str, default="")

    p.add_argument("--num-requests", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--output-len", type=int, default=2048)
    p.add_argument("--dataset", type=str, default="math")
    p.add_argument("--label", type=str, default="DLLM")

    p.add_argument("--output", type=str, default="tpf_vs_tps_results.json")
    p.add_argument("--append", action="store_true")
    args = p.parse_args()

    if args.plot:
        plot_results(args.plot)
        return

    results = []

    if args.sweep_all or args.sweep_sdar:
        results.extend(sweep_sdar(args))
    if args.sweep_all or args.sweep_dreamshift:
        results.extend(sweep_dreamshift(args))
    if not (args.sweep_all or args.sweep_sdar or args.sweep_dreamshift):
        results.extend(single_benchmark(args))

    # Save
    if args.append and os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        results = existing + results

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    if len(results) >= 2:
        plot_results(args.output)


if __name__ == "__main__":
    main()
