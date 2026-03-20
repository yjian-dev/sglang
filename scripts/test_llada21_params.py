"""Test LLaDA2.1-mini with different generation parameters using HF API."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ['HF_HOME'] = '/data/yjian/hf_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/data/yjian/hf_cache/hub'

model_path = '/data/yjian/hf_cache/hub/models--inclusionAI--LLaDA2.1-mini/snapshots/f21be037104f6e044e1a86b6d8864a6b85cc868e'

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("Loaded.\n")

# Test prompts covering different IFEval failure types
prompts = [
    ("Length control", "Write exactly 3 sentences about the ocean."),
    ("Bullet format", "List 3 benefits of exercise. Use bullet points starting with *."),
    ("JSON format", "Give me 3 colors in JSON format with key 'colors'."),
    ("Keyword", "Write a sentence about dogs. The word 'bark' must appear."),
    ("Complex", "Write a 300+ word summary about cats. Use at least 3 markdown section titles."),
]

# Parameter configurations to test
configs = [
    # name, block_length, threshold, editing_threshold, temperature, gen_length
    ("Speed  bl=32 t=0.5", 32, 0.5, 0.0, 0.0, 512),
    ("Quality bl=32 t=0.7", 32, 0.7, 0.5, 0.0, 512),
    ("Small   bl=4  t=0.95", 4, 0.95, 0.0, 1.0, 512),
    ("Small   bl=8  t=0.7", 8, 0.7, 0.3, 1.0, 512),
    ("Medium  bl=16 t=0.7", 16, 0.7, 0.3, 0.0, 512),
]

# Test each config on the first 2 prompts
for pname, prompt in prompts[:3]:
    print(f"\n{'='*70}")
    print(f"PROMPT: [{pname}] {prompt}")
    print('='*70)

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to('cuda:0')

    for cname, bl, thr, ethr, temp, glen in configs:
        try:
            with torch.no_grad():
                out = model.generate(
                    inputs=input_ids,
                    eos_early_stop=True,
                    gen_length=glen,
                    block_length=bl,
                    threshold=thr,
                    editing_threshold=ethr,
                    temperature=temp,
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            # Remove the prompt part
            if prompt in text:
                text = text[text.index(prompt)+len(prompt):].strip()
            print(f"\n[{cname}]")
            print(text[:300])
        except Exception as e:
            print(f"\n[{cname}] ERROR: {e}")
