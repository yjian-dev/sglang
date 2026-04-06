#!/usr/bin/env python3
"""Quick test: run one MBPP problem through JetEngine SDAR."""
import time
from transformers import AutoTokenizer
from jetengine import LLM, SamplingParams

model_path = "/data/shared/huggingface/SDAR-8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

prompt = """You are an expert Python programmer. Here is your task:
Write a python function to remove first and last occurrence of a given character from the string.
Your code should pass these tests:
assert remove_Occ("hello","l") == "helo"
assert remove_Occ("abcda","a") == "bcda"
"""

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
)
ids = tokenizer.encode(text, add_special_tokens=False)
print(f"Prompt tokens: {len(ids)}")

llm = LLM(
    model_path, enforce_eager=False, tensor_parallel_size=1,
    mask_token_id=151669, block_length=4, max_num_seqs=128,
    max_model_len=4096, gpu_memory_utilization=0.85,
)

sampling_params = SamplingParams(
    temperature=1.0, topk=50, topp=0.95, max_tokens=512,
    remasking_strategy="low_confidence_dynamic",
    dynamic_threshold=0.9, block_length=4, denoising_steps=4,
)

start = time.perf_counter()
outputs = llm.generate([ids], sampling_params)
elapsed = time.perf_counter() - start

out = outputs[0]
n_tokens = len(out["token_ids"])
print(f"Output tokens: {n_tokens}, Time: {elapsed:.2f}s, TPS: {n_tokens/elapsed:.0f}")
print(f"Output: {out['text'][:500]}")
