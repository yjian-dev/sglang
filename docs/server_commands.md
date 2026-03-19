# Server Launch & Benchmark Commands

## Environment Setup (always run first)
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
```

---

## DLLM Model

**Model path**: `/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont`

### 8× TP=1 N=3 Sampling (ports 30000-30007) — **current default**
```bash
MODEL=/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i nohup python -m sglang.launch_server \
    --model-path $MODEL --trust-remote-code --tp-size 1 \
    --mem-fraction-static 0.85 --max-running-requests 32 \
    --attention-backend flashinfer \
    --dllm-algorithm DreamShiftBlockN \
    --dllm-algorithm-config dreamshift_blockN3_config.yaml \
    --dtype bfloat16 --port $((30000+i)) --chunked-prefill-size 4096 \
    > /tmp/sglang_n3_gpu${i}.log 2>&1 &
done
# Wait for all 8 healthy
for i in $(seq 0 7); do
  for j in $(seq 1 60); do
    curl -sf http://localhost:$((30000+i))/health > /dev/null 2>&1 && echo "GPU $i ready" && break
    sleep 5
  done
done
```

### 8× TP=1 N=5 Sampling (ports 30000-30007)
```bash
MODEL=/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i nohup python -m sglang.launch_server \
    --model-path $MODEL --trust-remote-code --tp-size 1 \
    --mem-fraction-static 0.85 --max-running-requests 32 \
    --attention-backend flashinfer \
    --dllm-algorithm DreamShiftBlockN \
    --dllm-algorithm-config dreamshift_blockN5_config.yaml \
    --dtype bfloat16 --port $((30000+i)) --chunked-prefill-size 4096 \
    > /tmp/sglang_n5_gpu${i}.log 2>&1 &
done
```

### 1× TP=4 N=5 FP8 (port 30000, GPUs 0-3)
```bash
MODEL=/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m sglang.launch_server \
  --model-path $MODEL --trust-remote-code --tp-size 4 \
  --mem-fraction-static 0.85 --max-running-requests 128 \
  --attention-backend flashinfer \
  --dllm-algorithm DreamShiftBlockN \
  --dllm-algorithm-config dreamshift_blockN5_config.yaml \
  --dtype bfloat16 --port 30000 --chunked-prefill-size 4096 \
  --quantization fp8 \
  > /tmp/sglang_tp4_n5_fp8.log 2>&1 &
```

### Available configs
| File | N | temp | notes |
|------|---|------|-------|
| `dreamshift_blockN3_config.yaml` | 3 | 1.0 | sampling |
| `dreamshift_blockN3_config_t06.yaml` | 3 | 0.6 | low-temp sampling |
| `dreamshift_blockN4_config.yaml` | 4 | 1.0 | sampling |
| `dreamshift_blockN5_config.yaml` | 5 | 1.0 | sampling |
| `dreamshift_blockN5_greedy_config.yaml` | 5 | 0.0 | greedy |

---

## Qwen3-8B Baseline

### 8× TP=1 (ports 30010-30017)
```bash
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i nohup python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B --trust-remote-code --tp-size 1 \
    --mem-fraction-static 0.85 --max-running-requests 64 \
    --attention-backend flashinfer \
    --dtype bfloat16 --port $((30010+i)) --chunked-prefill-size 4096 \
    > /tmp/sglang_qwen_gpu${i}.log 2>&1 &
done
```

---

## Kill All Servers
```bash
ps aux | grep "launch_server" | grep -v grep | awk '{print $2}' | xargs kill 2>/dev/null
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

---

## Throughput Benchmarks (tore-speed-eval)

```bash
TOKENIZER=/home/yjian/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
```

### AIME (reasoning, long output)
```bash
for C in 1 2 4 8 16 32 48 64; do
  echo "=== Concurrency $C ==="
  tore-speed-eval --provider sglang --base_url http://localhost:30000/v1 \
    --model_name default --tokenizer_name $TOKENIZER \
    --traffic_pattern burst --concurrency $C \
    --num_examples 90 --max_tokens 16384 \
    --dataset_type jsonl --jsonl_input_path /tmp/aime2024.jsonl \
    --jsonl_convert_to_chat_request_format true \
    --temperature 1.0 --top_p 0.95
done
```

### ShareGPT (general chat)
```bash
for C in 1 2 4 8 16 32 48 64; do
  echo "=== Concurrency $C ==="
  tore-speed-eval --provider sglang --base_url http://localhost:30000/v1 \
    --model_name default --tokenizer_name $TOKENIZER \
    --traffic_pattern burst --concurrency $C \
    --num_examples 200 --max_tokens 2048 \
    --dataset_type sharegpt \
    --temperature 1.0 --top_p 0.95
done
```

### AIME jsonl setup (one-time)
```bash
python3 -c "
import json
from datasets import load_dataset
ds = load_dataset('AI-MO/aimo-validation-aime', split='train')
with open('/tmp/aime2024.jsonl', 'w') as f:
    for item in ds:
        f.write(json.dumps({'prompt': item['problem']}) + '\n')
print(f'Written {len(ds)} problems')
"
```

---

## Quality Benchmarks

```bash
PORTS="30000 30001 30002 30003 30004 30005 30006 30007"
# (use 30010..30017 for Qwen3-8B)
```

### Math & Reasoning
```bash
python scripts/eval_gsm8k.py --ports $PORTS --num-problems 1319
python scripts/eval_math500.py --ports $PORTS
python scripts/eval_aime.py --year 2025 --ports $PORTS
python scripts/eval_mathbench.py --ports $PORTS   # circular, 32k tokens
```

### Code
```bash
python scripts/eval_humaneval.py --ports $PORTS
python scripts/eval_mbpp.py --ports $PORTS
python scripts/eval_humaneval_x.py --langs python --ports $PORTS
python scripts/eval_lcb.py --version 6 --ports $PORTS
```

### Knowledge
```bash
python scripts/eval_gpqa.py --ports $PORTS          # needs HF_TOKEN
python scripts/eval_mmlu_pro.py --num-problems 2000 --ports $PORTS
python scripts/eval_mmlu.py --num-problems 2000 --ports $PORTS
python scripts/eval_arc_c.py --ports $PORTS
python scripts/eval_cmmlu.py --num-problems 2000 --ports $PORTS
python scripts/eval_triviaqa.py --ports $PORTS
```

### Instruction Following
```bash
python scripts/eval_ifeval.py --ports $PORTS
```

### Notes
- All scripts default to `--max-tokens 32768`
- `HF_TOKEN=<your_hf_token>` needed for GPQA (jianstreet account)
- MMLU-Pro/MMLU use random seed=42 sampling for reproducibility
