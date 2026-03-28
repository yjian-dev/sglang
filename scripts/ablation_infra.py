#!/usr/bin/env python3
"""
Infrastructure ablation experiments for the Strided DLLM COLM paper.

Measures tokens/second with each optimization toggled on/off incrementally.
Produces a JSON results file + LaTeX table for the paper.

Usage:
    # Full ablation (non-LoRA, N=3, C=1 and C=32)
    python scripts/ablation_infra.py \
        --model-path /data/cxu/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b1-allmasked_fixed2_backup8000 \
        --gpu 0

    # Quick test (single config)
    python scripts/ablation_infra.py \
        --model-path <model> --gpu 0 --configs cuda_graph --concurrency 1
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time

import aiohttp

# ── Ablation configurations ──
# Each config is a dict of env vars + CLI flags that DISABLE certain optimizations.
# We run them incrementally: first with everything disabled, then enabling one by one.

ABLATION_CONFIGS = [
    {
        "name": "Naive (eager, full sched)",
        "description": "No CUDA graph, no decode loop, no argmax proposals, no fused verify, no deferred stream",
        "cli_extra": ["--disable-cuda-graph"],
        "env": {
            "SGLANG_ABLATION_NO_DECODE_LOOP": "1",
            "SGLANG_ABLATION_NO_ARGMAX_PROPOSALS": "1",
            "SGLANG_ABLATION_NO_FUSED_VERIFY": "1",
            "SGLANG_ABLATION_NO_DEFERRED_STREAM": "1",
        },
    },
    {
        "name": "+ CUDA graph",
        "description": "CUDA graph enabled, but no decode loop, no argmax, no fused verify, no deferred stream",
        "cli_extra": [],
        "env": {
            "SGLANG_ABLATION_NO_DECODE_LOOP": "1",
            "SGLANG_ABLATION_NO_ARGMAX_PROPOSALS": "1",
            "SGLANG_ABLATION_NO_FUSED_VERIFY": "1",
            "SGLANG_ABLATION_NO_DEFERRED_STREAM": "1",
        },
    },
    {
        "name": "+ Decode inner loop",
        "description": "CUDA graph + decode loop, but no argmax, no fused verify, no deferred stream",
        "cli_extra": [],
        "env": {
            "SGLANG_ABLATION_NO_ARGMAX_PROPOSALS": "1",
            "SGLANG_ABLATION_NO_FUSED_VERIFY": "1",
            "SGLANG_ABLATION_NO_DEFERRED_STREAM": "1",
        },
    },
    {
        "name": "+ Argmax proposals",
        "description": "CUDA graph + decode loop + argmax proposals, no fused verify, no deferred stream",
        "cli_extra": [],
        "env": {
            "SGLANG_ABLATION_NO_FUSED_VERIFY": "1",
            "SGLANG_ABLATION_NO_DEFERRED_STREAM": "1",
        },
    },
    {
        "name": "+ Fused verify kernel",
        "description": "CUDA graph + decode loop + argmax + fused verify, no deferred stream",
        "cli_extra": [],
        "env": {
            "SGLANG_ABLATION_NO_DEFERRED_STREAM": "1",
        },
    },
    {
        "name": "Full system",
        "description": "All optimizations enabled",
        "cli_extra": [],
        "env": {},
    },
]

# Short AIME-style prompts for throughput measurement
BENCH_PROMPTS = [
    "Let $a_1, a_2, \\ldots$ be an infinite sequence of positive integers, and let $N$ be a positive integer. Suppose that, for each $n > N$, $a_n$ is equal to the number of times $a_{n-1}$ appears in the list $a_1, a_2, \\ldots, a_{n-1}$. Prove that at least one of the sequences $a_1, a_3, a_5, \\ldots$ and $a_2, a_4, a_6, \\ldots$ is eventually periodic.",
    "Find the number of ways to place 8 non-attacking rooks on a standard 8x8 chessboard such that no two rooks are in the same row or column, and exactly 3 rooks are on black squares. Show your work step by step.",
    "Let $f(x) = x^4 + ax^3 + bx^2 + cx + d$ be a polynomial with integer coefficients such that $f(1) = 10$, $f(2) = 20$, and $f(3) = 30$. Find all possible values of $f(10) + f(-6)$. Explain your reasoning in detail.",
    "In triangle ABC, the incircle touches sides BC, CA, and AB at points D, E, and F respectively. Let P be the intersection of lines AD and EF. Prove that AP/PD = (AF/FB) + (AE/EC). Show all steps.",
    "A frog starts at the origin on the number line. Each second, it jumps either +1 or -1 with equal probability. What is the expected number of seconds until the frog reaches either +10 or -10? Derive the answer from first principles.",
    "Find all functions $f: \\mathbb{R} \\to \\mathbb{R}$ satisfying $f(x^2 + f(y)) = y + f(x)^2$ for all real numbers $x, y$. Prove your answer is complete.",
    "Let $S = \\{1, 2, 3, \\ldots, 2024\\}$. Find the number of subsets $T$ of $S$ such that the sum of elements of $T$ is divisible by 5. Show detailed working.",
    "Evaluate the integral $\\int_0^1 \\frac{\\ln(1+x)}{x} \\, dx$ using series expansion and prove your answer equals $\\frac{\\pi^2}{12}$. Show all steps of the computation.",
]


def wait_for_server(port, timeout=300):
    """Wait for the SGLang server to be ready."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            url = f"http://localhost:{port}/health"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def launch_server(model_path, port, gpu, tp_size, config_name, ablation_cfg,
                  algorithm_config, max_running_requests):
    """Launch an SGLang server with the given ablation configuration."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    # Clear all ablation env vars first
    for key in ["SGLANG_ABLATION_NO_DECODE_LOOP", "SGLANG_ABLATION_NO_ARGMAX_PROPOSALS",
                "SGLANG_ABLATION_NO_FUSED_VERIFY", "SGLANG_ABLATION_NO_DEFERRED_STREAM"]:
        env.pop(key, None)
    # Set ablation env vars for this config
    env.update(ablation_cfg["env"])

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--trust-remote-code",
        "--tp-size", str(tp_size),
        "--port", str(port),
        "--mem-fraction-static", "0.85",
        "--max-running-requests", str(max_running_requests),
        "--attention-backend", "flashinfer",
        "--dllm-algorithm", "DreamShiftBlockN",
        "--dllm-algorithm-config", algorithm_config,
        "--dtype", "bfloat16",
    ] + ablation_cfg.get("cli_extra", [])

    print(f"\n{'='*70}")
    print(f"[{config_name}]")
    print(f"  Env: {ablation_cfg['env'] or '(none)'}")
    print(f"  Extra CLI: {ablation_cfg.get('cli_extra', []) or '(none)'}")
    print(f"  Cmd: {' '.join(cmd)}")
    print(f"{'='*70}")

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc


def kill_server(proc):
    """Kill the server process group."""
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
            except Exception:
                pass


async def send_request(session, url, prompt, max_tokens, semaphore):
    """Send a single request and return (output_tokens, latency)."""
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.95,
    }
    async with semaphore:
        t0 = time.time()
        try:
            async with session.post(url, json=body) as resp:
                data = await resp.json()
                elapsed = time.time() - t0
                usage = data.get("usage", {})
                return usage.get("completion_tokens", 0), elapsed
        except Exception as e:
            print(f"  Request failed: {e}")
            return 0, time.time() - t0


async def run_benchmark(port, concurrency, num_requests, max_tokens):
    """Run throughput benchmark at given concurrency."""
    url = f"http://localhost:{port}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        semaphore = asyncio.Semaphore(concurrency)
        tasks = []
        for i in range(num_requests):
            prompt = BENCH_PROMPTS[i % len(BENCH_PROMPTS)]
            tasks.append(send_request(session, url, prompt, max_tokens, semaphore))

        wall_start = time.time()
        results = await asyncio.gather(*tasks)
        wall_time = time.time() - wall_start

    total_tokens = sum(r[0] for r in results)
    failed = sum(1 for r in results if r[0] == 0)
    tps = total_tokens / wall_time if wall_time > 0 else 0

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "total_tokens": total_tokens,
        "wall_time": wall_time,
        "tps": tps,
        "failed": failed,
        "avg_tokens_per_req": total_tokens / max(num_requests - failed, 1),
    }


async def warmup(port, n=3):
    """Send warmup requests."""
    url = f"http://localhost:{port}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i in range(n):
            body = {
                "model": "test",
                "messages": [{"role": "user", "content": f"What is {i+1}+{i+1}?"}],
                "max_tokens": 128,
                "temperature": 1.0,
            }
            async with session.post(url, json=body) as resp:
                await resp.json()
    print("  Warmup done.")


def generate_latex_table(results, output_file=None):
    """Generate LaTeX ablation table from results."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Infrastructure optimization ablation} ($N{=}3$, 1$\times$H100, bf16). Each row cumulatively enables one optimization. $\Delta$ is relative to the previous row.}")
    lines.append(r"\label{tab:infra_ablation}")
    lines.append(r"\small")

    # Determine which concurrency levels we have
    concurrencies = sorted(set(r["concurrency"] for r in results))
    config_names = []
    seen = set()
    for r in results:
        if r["config"] not in seen:
            config_names.append(r["config"])
            seen.add(r["config"])

    if len(concurrencies) == 1:
        c = concurrencies[0]
        lines.append(r"\begin{tabular}{lcc}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Configuration} & \textbf{TPS} & \textbf{$\Delta$} \\")
        lines.append(r"\midrule")
        prev_tps = None
        for name in config_names:
            row = next(r for r in results if r["config"] == name and r["concurrency"] == c)
            tps = row["tps"]
            if prev_tps is None:
                delta = "---"
            else:
                pct = (tps - prev_tps) / prev_tps * 100
                delta = f"+{pct:.0f}\\%"
            lines.append(f"{name} & {tps:.0f} & {delta} \\\\")
            prev_tps = tps
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    else:
        # Multi-concurrency table
        c_headers = " & ".join([f"$C{{=}}{c}$" for c in concurrencies])
        n_cols = len(concurrencies)
        lines.append(f"\\begin{{tabular}}{{l{'c' * n_cols}}}")
        lines.append(r"\toprule")
        lines.append(f"\\textbf{{Configuration}} & {c_headers} \\\\")
        lines.append(r"\midrule")
        for name in config_names:
            vals = []
            for c in concurrencies:
                row = next((r for r in results if r["config"] == name and r["concurrency"] == c), None)
                if row:
                    vals.append(f"{row['tps']:.0f}")
                else:
                    vals.append("---")
            lines.append(f"{name} & {' & '.join(vals)} \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    if output_file:
        with open(output_file, "w") as f:
            f.write(latex)
        print(f"\nLaTeX table written to {output_file}")
    return latex


def main():
    parser = argparse.ArgumentParser(description="Strided DLLM infrastructure ablation")
    parser.add_argument("--model-path", required=True, help="Path to the DLLM model")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1')")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--port", type=int, default=31100, help="Server port")
    parser.add_argument("--algorithm-config", default="dreamshift_blockN3_config.yaml",
                        help="DLLM algorithm config YAML")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 32],
                        help="Concurrency levels to test")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens per request")
    parser.add_argument("--num-requests", type=int, default=None,
                        help="Requests per concurrency level (default: max(C*2, 16))")
    parser.add_argument("--max-running-requests", type=int, default=64)
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Run only specific configs (by index 0-5 or name substring)")
    parser.add_argument("--output", default="ablation_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--latex-output", default="ablation_table.tex",
                        help="Output LaTeX table file")
    parser.add_argument("--server-timeout", type=int, default=300,
                        help="Timeout waiting for server startup (seconds)")
    parser.add_argument("--warmup-requests", type=int, default=3,
                        help="Number of warmup requests before benchmark")
    args = parser.parse_args()

    # Filter configs if requested
    configs_to_run = ABLATION_CONFIGS
    if args.configs:
        filtered = []
        for spec in args.configs:
            if spec.isdigit():
                idx = int(spec)
                if 0 <= idx < len(ABLATION_CONFIGS):
                    filtered.append(ABLATION_CONFIGS[idx])
            else:
                for cfg in ABLATION_CONFIGS:
                    if spec.lower() in cfg["name"].lower():
                        filtered.append(cfg)
        configs_to_run = filtered
        if not configs_to_run:
            print(f"No configs matched: {args.configs}")
            print("Available configs:")
            for i, cfg in enumerate(ABLATION_CONFIGS):
                print(f"  {i}: {cfg['name']}")
            sys.exit(1)

    all_results = []

    for cfg in configs_to_run:
        config_name = cfg["name"]
        proc = None

        try:
            # Launch server
            proc = launch_server(
                args.model_path, args.port, args.gpu, args.tp_size,
                config_name, cfg, args.algorithm_config, args.max_running_requests,
            )

            # Wait for server
            print(f"  Waiting for server (timeout={args.server_timeout}s)...")
            if not wait_for_server(args.port, timeout=args.server_timeout):
                print(f"  ERROR: Server failed to start for config '{config_name}'")
                # Dump last 50 lines of output
                if proc.poll() is not None:
                    out = proc.stdout.read().decode("utf-8", errors="replace")
                    print("  Server output (last 50 lines):")
                    for line in out.strip().split("\n")[-50:]:
                        print(f"    {line}")
                continue

            print(f"  Server ready.")

            # Warmup
            asyncio.run(warmup(args.port, n=args.warmup_requests))

            # Run benchmarks at each concurrency level
            for c in args.concurrency:
                n = args.num_requests or max(c * 2, 16)
                n = min(n, 128)
                print(f"  Benchmarking C={c}, num_requests={n}, max_tokens={args.max_tokens}...")

                result = asyncio.run(run_benchmark(args.port, c, n, args.max_tokens))
                result["config"] = config_name
                result["env"] = cfg["env"]
                result["cli_extra"] = cfg.get("cli_extra", [])

                print(f"    TPS={result['tps']:.1f}, wall={result['wall_time']:.1f}s, "
                      f"tokens={result['total_tokens']}, "
                      f"avg_tok/req={result['avg_tokens_per_req']:.0f}, "
                      f"failed={result['failed']}")

                all_results.append(result)

        finally:
            # Kill server
            print(f"  Shutting down server...")
            kill_server(proc)
            # Wait a bit for port to be freed
            time.sleep(3)

    # Save results
    if all_results:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

        # Print summary table
        print(f"\n{'='*70}")
        print("ABLATION SUMMARY")
        print(f"{'='*70}")
        concurrencies = sorted(set(r["concurrency"] for r in all_results))
        for c in concurrencies:
            print(f"\n--- Concurrency = {c} ---")
            print(f"  {'Config':<35} {'TPS':>8} {'Δ':>8}")
            print(f"  {'-'*51}")
            prev_tps = None
            for cfg in configs_to_run:
                row = next((r for r in all_results
                           if r["config"] == cfg["name"] and r["concurrency"] == c), None)
                if row:
                    tps = row["tps"]
                    if prev_tps:
                        delta = f"+{(tps - prev_tps) / prev_tps * 100:.0f}%"
                    else:
                        delta = "---"
                    print(f"  {cfg['name']:<35} {tps:>8.1f} {delta:>8}")
                    prev_tps = tps

        # Generate LaTeX table
        latex = generate_latex_table(all_results, args.latex_output)
        print(f"\n--- LaTeX Table ---")
        print(latex)
    else:
        print("\nNo results collected.")


if __name__ == "__main__":
    main()
