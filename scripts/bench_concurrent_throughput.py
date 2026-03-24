#!/usr/bin/env python3
"""
Concurrent throughput benchmark for the Strided DLLM paper.
Measures tokens/second at various concurrency levels.
Uses math reasoning prompts that generate long outputs.
"""
import argparse
import asyncio
import json
import time
import aiohttp

# AIME-style math reasoning prompts that generate long chain-of-thought outputs
PROMPTS = [
    "Let $a_1, a_2, \\ldots$ be an infinite sequence of positive integers, and let $N$ be a positive integer. Suppose that, for each $n > N$, $a_n$ is equal to the number of times $a_{n-1}$ appears in the list $a_1, a_2, \\ldots, a_{n-1}$. Prove that at least one of the sequences $a_1, a_3, a_5, \\ldots$ and $a_2, a_4, a_6, \\ldots$ is eventually periodic.",
    "Find the number of ways to place 8 non-attacking rooks on a standard 8x8 chessboard such that no two rooks are in the same row or column, and exactly 3 rooks are on black squares. Show your work step by step.",
    "Let $f(x) = x^4 + ax^3 + bx^2 + cx + d$ be a polynomial with integer coefficients such that $f(1) = 10$, $f(2) = 20$, and $f(3) = 30$. Find all possible values of $f(10) + f(-6)$. Explain your reasoning in detail.",
    "In triangle ABC, the incircle touches sides BC, CA, and AB at points D, E, and F respectively. Let P be the intersection of lines AD and EF. Prove that AP/PD = (AF/FB) + (AE/EC). Show all steps.",
    "A frog starts at the origin on the number line. Each second, it jumps either +1 or -1 with equal probability. What is the expected number of seconds until the frog reaches either +10 or -10? Derive the answer from first principles.",
    "Find all functions $f: \\mathbb{R} \\to \\mathbb{R}$ satisfying $f(x^2 + f(y)) = y + f(x)^2$ for all real numbers $x, y$. Prove your answer is complete.",
    "Let $S = \\{1, 2, 3, \\ldots, 2024\\}$. Find the number of subsets $T$ of $S$ such that the sum of elements of $T$ is divisible by 5. Show detailed working.",
    "Evaluate the integral $\\int_0^1 \\frac{\\ln(1+x)}{x} \\, dx$ using series expansion and prove your answer equals $\\frac{\\pi^2}{12}$. Show all steps of the computation.",
    "In a convex polygon with 12 vertices, we draw all diagonals. Assuming no three diagonals meet at a single interior point, how many regions do the diagonals divide the interior into? Derive a general formula and compute the answer.",
    "A sequence is defined by $a_1 = 1$, $a_2 = 1$, and $a_{n+2} = a_{n+1} + a_n + \\lfloor\\sqrt{a_n}\\rfloor$ for $n \\geq 1$. Find $a_{50} \\pmod{1000}$. Show your approach and calculations.",
    "Let $p$ be a prime number greater than 3. Prove that $\\sum_{k=1}^{p-1} \\frac{1}{k} \\equiv 0 \\pmod{p^2}$ (Wolstenholme's theorem). Give a complete proof.",
    "A 10x10 grid is filled with the numbers 1 through 100, each used exactly once. For each row, we compute the product of the 10 numbers in that row. What is the minimum possible value of the largest row product? Justify your answer.",
    "Find the volume of the region in $\\mathbb{R}^3$ defined by $|x| + |y| + |z| \\leq 3$ and $x^2 + y^2 \\leq 4$. Set up and evaluate all necessary integrals.",
    "Let $G$ be a group of order 168. Prove that $G$ has a normal subgroup of order 8 or a normal subgroup of order 7. Show the Sylow theory analysis.",
    "Alice and Bob play a game. They alternate placing 1s and 0s in the cells of a 4x4 grid (Alice plays 1, Bob plays 0). Alice wins if any row, column, or diagonal sums to an even number; Bob wins otherwise. Who has a winning strategy? Prove it.",
    "Find the exact value of $\\prod_{n=1}^{\\infty} \\left(1 - \\frac{1}{4n^2}\\right)$ and prove your answer. Use multiple methods if possible.",
]

async def send_request(session, url, prompt, max_tokens, model, lora_path=None, semaphore=None):
    """Send a single request and return (output_tokens, latency)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.95,
    }
    if lora_path:
        body["lora_path"] = lora_path

    if semaphore:
        async with semaphore:
            return await _do_request(session, url, body)
    else:
        return await _do_request(session, url, body)

async def _do_request(session, url, body):
    t0 = time.time()
    try:
        async with session.post(url, json=body) as resp:
            data = await resp.json()
            elapsed = time.time() - t0
            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            return completion_tokens, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Request failed: {e}")
        return 0, elapsed

async def run_benchmark(port, concurrency, num_requests, max_tokens, model, lora_path=None):
    """Run benchmark at given concurrency level."""
    url = f"http://localhost:{port}/v1/chat/completions"

    timeout = aiohttp.ClientTimeout(total=600)  # 10 min timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        semaphore = asyncio.Semaphore(concurrency)

        # Create tasks - cycle through prompts
        tasks = []
        for i in range(num_requests):
            prompt = PROMPTS[i % len(PROMPTS)]
            tasks.append(send_request(session, url, prompt, max_tokens, model, lora_path, semaphore))

        # Run all tasks and measure wall time
        wall_start = time.time()
        results = await asyncio.gather(*tasks)
        wall_time = time.time() - wall_start

    total_tokens = sum(r[0] for r in results)
    total_tps = total_tokens / wall_time if wall_time > 0 else 0
    avg_latency = sum(r[1] for r in results) / len(results)

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "total_tokens": total_tokens,
        "wall_time": wall_time,
        "tps": total_tps,
        "avg_latency": avg_latency,
        "avg_tokens_per_request": total_tokens / num_requests if num_requests > 0 else 0,
    }

async def warmup(port, model, lora_path=None):
    """Send a warmup request."""
    url = f"http://localhost:{port}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 64,
        "temperature": 1.0,
    }
    if lora_path:
        body["lora_path"] = lora_path

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=body) as resp:
            await resp.json()
    print("Warmup done.")

def main():
    parser = argparse.ArgumentParser(description="Concurrent throughput benchmark")
    parser.add_argument("--port", type=int, default=31003)
    parser.add_argument("--model", default="test")
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16, 32, 48, 64])
    parser.add_argument("--num-requests", type=int, default=None,
                        help="Total requests per concurrency level. Default: max(concurrency*2, 16)")
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()

    async def run():
        if not args.skip_warmup:
            print("Warming up...")
            await warmup(args.port, args.model, args.lora_path)

        results = []
        for c in args.concurrency:
            n = args.num_requests or max(c * 2, 16)
            # For very high concurrency, cap requests to avoid excessive runtime
            n = min(n, 128)
            print(f"\n--- Concurrency={c}, num_requests={n}, max_tokens={args.max_tokens} ---")
            r = await run_benchmark(args.port, c, n, args.max_tokens, args.model, args.lora_path)
            results.append(r)
            print(f"  TPS={r['tps']:.1f}, wall={r['wall_time']:.1f}s, "
                  f"tokens={r['total_tokens']}, avg_lat={r['avg_latency']:.1f}s, "
                  f"avg_tok/req={r['avg_tokens_per_request']:.0f}")

        print("\n=== SUMMARY ===")
        print(f"{'C':>4} {'TPS':>8} {'Wall(s)':>8} {'Tokens':>8} {'AvgLat':>8} {'Tok/Req':>8}")
        for r in results:
            print(f"{r['concurrency']:>4} {r['tps']:>8.1f} {r['wall_time']:>8.1f} "
                  f"{r['total_tokens']:>8} {r['avg_latency']:>8.1f} "
                  f"{r['avg_tokens_per_request']:>8.0f}")

        return results

    return asyncio.run(run())

if __name__ == "__main__":
    main()
