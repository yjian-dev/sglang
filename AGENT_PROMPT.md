# Agent Prompt

## Role
You are a benchmark automation agent. Your job is to systematically measure throughput and quality for multiple DreamShift configurations and baselines, recording ALL results in `bench_results/comprehensive_benchmark.md`.

## Project Description
Run comprehensive TP=1 benchmarks across different model configs on 8x H100 GPUs. Record throughput (tore-speed-eval) and quality (GSM8K, HumanEval, IFEval, MBPP, MATH-500).

## Configurations to Test (8 total)

### DreamShift (6 configs, SDAR model)
1. N=3 sampling (temp=1.0) — `dreamshift_blockN3_config.yaml`
2. N=3 greedy (temp=0.0) — `dreamshift_blockN3_greedy_standard.yaml`
3. N=4 sampling (temp=1.0) — `dreamshift_blockN4_config.yaml`
4. N=4 greedy (temp=0.0) — `dreamshift_blockN4_greedy.yaml`
5. N=5 sampling (temp=1.0) — `dreamshift_blockN5_config.yaml`
6. N=5 greedy (temp=0.0) — `dreamshift_blockN5_greedy.yaml`

### Baselines (2 configs)
7. Qwen3-8B AR (vanilla autoregressive)
8. Qwen3-8B + EAGLE3 (`Tengyunw/qwen3_8b_eagle3`)

## What to Measure

### A. Throughput (tore-speed-eval, burst mode)
Concurrency: 1, 2, 4, 8, 16, 32, 48, 64

**Two datasets**:
1. **AIME**: `--dataset_type jsonl --jsonl_input_path /tmp/aime2024.jsonl --jsonl_convert_to_chat_request_format true --max_tokens 2048 --num_examples 90`
2. **ShareGPT**: `--dataset_type sharegpt --max_tokens 2048 --num_examples 200`

Both use `--temperature 1.0 --top_p 0.95 --traffic_pattern burst`

### B. Quality Benchmarks (all use max_tokens=16384)
For each config:
- GSM8K (200 problems)
- HumanEval (164 problems)
- IFEval (541 prompts)
- MBPP (257 problems)
- MATH-500 (500 problems)

For sampling configs (1,3,5), also test with **temperature=0.6** (create temp yaml with temp=0.6).

## GPU & Server Strategy

Launch 8x TP=1 servers (one per GPU, ports 30000-30007), all running the SAME config.
- Throughput: run tore-speed-eval against port 30000
- Quality: distribute across all 8 ports (`--ports 30000 30001 ... 30007`)
- When done, kill all, launch next config

### Server Launch Commands

**DreamShift** (all N values):
```bash
bash scripts/killall_sglang.sh && sleep 3
CONFIG=<yaml_file> bash scripts/launch_blockN_8gpu.sh
```

**Qwen3-8B AR**:
```bash
bash scripts/killall_sglang.sh && sleep 3
bash scripts/launch_qwen_8gpu.sh
```

**EAGLE3** (8x TP=1):
```bash
bash scripts/killall_sglang.sh && sleep 3
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
for gpu in 0 1 2 3 4 5 6 7; do
  port=$((30000 + gpu))
  CUDA_VISIBLE_DEVICES=$gpu nohup python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B --trust-remote-code --tp-size 1 \
    --mem-fraction-static 0.85 --max-running-requests 64 \
    --attention-backend flashinfer \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --dtype bfloat16 --port $port --chunked-prefill-size 4096 \
    > /tmp/sglang_gpu${gpu}.log 2>&1 &
done
for gpu in 0 1 2 3 4 5 6 7; do
  port=$((30000 + gpu))
  for i in $(seq 1 120); do
    curl -sf http://localhost:$port/health > /dev/null 2>&1 && echo "GPU $gpu ready" && break
    sleep 2
  done
done
```

### Throughput Benchmark
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
TOKENIZER=/home/yjian/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218

# AIME
for C in 1 2 4 8 16 32 48 64; do
  tore-speed-eval --provider sglang --base_url http://localhost:30000/v1 \
    --model_name default --tokenizer_name $TOKENIZER --traffic_pattern burst \
    --concurrency $C --num_examples 90 --max_tokens 2048 \
    --dataset_type jsonl --jsonl_input_path /tmp/aime2024.jsonl \
    --jsonl_convert_to_chat_request_format true \
    --temperature 1.0 --top_p 0.95
done

# ShareGPT
for C in 1 2 4 8 16 32 48 64; do
  tore-speed-eval --provider sglang --base_url http://localhost:30000/v1 \
    --model_name default --tokenizer_name $TOKENIZER --traffic_pattern burst \
    --concurrency $C --num_examples 200 --max_tokens 2048 \
    --dataset_type sharegpt \
    --temperature 1.0 --top_p 0.95
done
```

### Quality Benchmark
```bash
PORTS="30000 30001 30002 30003 30004 30005 30006 30007"
python scripts/eval_gsm8k.py --ports $PORTS --num-problems 200 --max-tokens 16384
python scripts/eval_humaneval.py --ports $PORTS --max-tokens 16384
python scripts/eval_ifeval.py --ports $PORTS --max-tokens 16384
python scripts/eval_mbpp.py --ports $PORTS --max-tokens 16384
python scripts/eval_math500.py --ports $PORTS --max-tokens 16384
```

## AIME JSONL Setup
Ensure `/tmp/aime2024.jsonl` exists before starting:
```python
import json
from datasets import load_dataset
ds = load_dataset('AI-MO/aimo-validation-aime', split='train')
with open('/tmp/aime2024.jsonl', 'w') as f:
    for item in ds:
        f.write(json.dumps({'prompt': item['problem']}) + '\n')
```

## Output
Write ALL results to `bench_results/comprehensive_benchmark.md` with clear tables. Update PLAN.md with progress after each config.

## Environment
- GPUs: 8x H100 80GB
- Conda: `source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang`
- CUDA: `/usr/local/cuda-12.9`
- SDAR model: `/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont`

## Constraints
- Kill old servers before launching new ones
- Wait for ALL 8 servers to be healthy before benchmarking
- If quality score is suspiciously low (< 50%), investigate max_tokens or parsing. Fix script if needed.
- Record results immediately after each measurement
- For sampling temp=0.6 tests, create a yaml with temperature: 0.6 (copy from sampling yaml, change temp)
