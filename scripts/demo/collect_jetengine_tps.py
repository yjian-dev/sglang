#!/usr/bin/env python3
"""Collect per-batch TPS data from JetEngine SDAR baseline.

Must be run via: CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 collect_jetengine_tps.py
JetEngine is an offline batch engine — we measure tokens/time per batch.
"""

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/data/shared/huggingface/SDAR-8B-Chat")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--output", default="scripts/demo/results/sdar_tps.json")
    parser.add_argument("--num-prompts", type=int, default=500)
    args = parser.parse_args()

    print("Loading MBPP prompts...")
    prompts = load_mbpp_prompts(args.num_prompts)
    print(f"Loaded {len(prompts)} prompts")

    # Tokenize prompts using chat template
    print("Tokenizing prompts...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    prompt_ids_list = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        prompt_ids_list.append(ids)

    # Initialize JetEngine
    print("Initializing JetEngine...")
    from jetengine import LLM, SamplingParams

    llm = LLM(
        args.model_path,
        enforce_eager=False,
        tensor_parallel_size=1,
        mask_token_id=151669,
        block_length=4,
        max_num_seqs=128,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        topk=50,
        topp=0.95,
        max_tokens=args.max_tokens,
        remasking_strategy="low_confidence_dynamic",
        dynamic_threshold=0.9,
        block_length=4,
        denoising_steps=4,
    )

    # Warmup
    print("Warmup batch...")
    warmup_ids = prompt_ids_list[: min(4, args.batch_size)]
    _ = llm.generate(warmup_ids, sampling_params)
    print("Warmup done")

    # Run batches and measure
    tps_series = []
    cumulative_time = 0.0
    batch_idx = 0

    for i in range(args.num_batches):
        start_idx = (i * args.batch_size) % len(prompt_ids_list)
        batch_ids = []
        for j in range(args.batch_size):
            idx = (start_idx + j) % len(prompt_ids_list)
            batch_ids.append(prompt_ids_list[idx])

        print(f"Batch {i+1}/{args.num_batches} (size={len(batch_ids)})...")
        start = time.perf_counter()
        outputs = llm.generate(batch_ids, sampling_params)
        elapsed = time.perf_counter() - start

        # Count generated tokens (subtract prompt length — token_ids includes prompt)
        total_tokens = sum(
            len(o["token_ids"]) - len(batch_ids[j])
            for j, o in enumerate(outputs)
        )
        tps = total_tokens / elapsed

        # Create time series points spanning this batch's duration
        batch_start = cumulative_time
        # Add a few data points across the batch duration for smoother plotting
        num_points = max(1, int(elapsed / 0.5))
        for p in range(num_points):
            t = batch_start + (p + 1) * elapsed / num_points
            tps_series.append({"time": round(t, 1), "tps": round(tps, 1)})

        cumulative_time += elapsed
        print(f"  Tokens: {total_tokens}, Time: {elapsed:.1f}s, TPS: {tps:.0f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "engine": "sdar",
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "num_batches": args.num_batches,
        "tps_series": tps_series,
        "total_time": round(cumulative_time, 1),
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    tps_values = [p["tps"] for p in tps_series]
    if tps_values:
        avg_tps = sum(tps_values) / len(tps_values)
        print(f"\nSDAR Baseline Results:")
        print(f"  Average TPS: {avg_tps:.0f}")
        print(f"  Peak TPS:    {max(tps_values):.0f}")
        print(f"  Total time:  {cumulative_time:.1f}s")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
