"""Run the reference generate.py on the same GSM8K problems and measure TPF."""
import sys
import os
import time
import torch

# Add reference code path
sys.path.insert(0, "/data/cxu/dllm-distillation")
from generate import causal_blockN_spec_verified_generate_with_shift

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_PATH = "/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont"

PROBLEMS = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
    "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
    "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
]


def count_forwards_wrapper(model, tokenizer, prompt_text, mask_id, block_size=3,
                           gen_length=512, temperature=1.0, top_k=50, top_p=0.95):
    """Run reference blockN generation and count forwards."""
    # Monkey-patch model.forward to count calls
    original_forward = model.forward
    call_count = [0]

    def counting_forward(*args, **kwargs):
        call_count[0] += 1
        return original_forward(*args, **kwargs)

    model.forward = counting_forward

    messages = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = tokenizer(text, return_tensors="pt").to(model.device)

    eos_token_id = tokenizer.eos_token_id
    stopping = [eos_token_id] if eos_token_id is not None else None

    t0 = time.time()
    output = causal_blockN_spec_verified_generate_with_shift(
        model, prompt, mask_id,
        block_size=block_size,
        gen_length=gen_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        confidence_threshold=0.0,
        stopping_criteria_idx=stopping,
    )
    elapsed = time.time() - t0

    model.forward = original_forward

    prompt_len = prompt["input_ids"].shape[1]
    output_tokens = output.shape[1] - prompt_len
    n_forwards = call_count[0]
    tpf = output_tokens / max(n_forwards, 1)

    decoded = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
    return {
        "output_tokens": output_tokens,
        "n_forwards": n_forwards,
        "tpf": tpf,
        "elapsed_s": elapsed,
        "answer": decoded[:120],
    }


def main():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Get mask_id from config
    mask_id = getattr(model.config, "mask_id", None) or 151669
    print(f"mask_id={mask_id}")

    total_tokens = 0
    total_forwards = 0
    for i, problem in enumerate(PROBLEMS):
        print(f"\n--- Problem {i} ---")
        result = count_forwards_wrapper(
            model, tokenizer, problem, mask_id,
            block_size=3, gen_length=512,
            temperature=1.0, top_k=50, top_p=0.95,
        )
        total_tokens += result["output_tokens"]
        total_forwards += result["n_forwards"]
        print(f"  tokens={result['output_tokens']}, forwards={result['n_forwards']}, "
              f"TPF={result['tpf']:.2f}, time={result['elapsed_s']:.1f}s")
        print(f"  Answer: {result['answer']}...")

    avg_tpf = total_tokens / max(total_forwards, 1)
    print(f"\n=== Reference avg TPF: {avg_tpf:.2f} ({total_tokens} tokens / {total_forwards} forwards) ===")


if __name__ == "__main__":
    main()
