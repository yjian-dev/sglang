import time, json, torch
from jetengine import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = "/data/shared/huggingface/SDAR-8B-Chat"
BLOCK_SIZE = 4
MASK_ID = 151669

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

llm = LLM(
    MODEL,
    enforce_eager=False,
    tensor_parallel_size=1,
    mask_token_id=MASK_ID,
    block_length=BLOCK_SIZE,
    max_num_seqs=128,
    max_model_len=4096,
    gpu_memory_utilization=0.85,
)

sampling_params = SamplingParams(
    temperature=1.0,
    topk=0, topp=1.0,
    max_tokens=256,
    remasking_strategy="low_confidence_dynamic",
    dynamic_threshold=0.9,
    block_length=BLOCK_SIZE,
    denoising_steps=BLOCK_SIZE,
)

prompt_text = "What is 127 * 453? Show your step-by-step calculation."
prompt_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt_text}],
    tokenize=True, add_generation_prompt=True, enable_thinking=True,
)

batch_sizes = [1, 2, 4, 8, 16, 32, 64]
results = []

for bs in batch_sizes:
    prompts = [prompt_ids] * bs

    # Warmup on first batch
    if bs == batch_sizes[0]:
        _ = llm.generate(prompts[:1], sampling_params, use_tqdm=False)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o['token_ids']) for o in outputs)
    tps = total_tokens / elapsed
    per_req = total_tokens / bs
    results.append({"bs": bs, "tps": round(tps, 1), "elapsed": round(elapsed, 2),
                    "total_tok": total_tokens, "per_req_tok": round(per_req, 1)})
    print(f"bs={bs:3d}  TPS={tps:7.1f}  elapsed={elapsed:.2f}s  total_tok={total_tokens}  per_req={per_req:.0f}")

print("\nDone!")
with open("jetengine_sdar8b_results.json", "w") as f:
    json.dump(results, f, indent=2)
