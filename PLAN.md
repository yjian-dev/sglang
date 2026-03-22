# Plan — b3 epoch2 lr1e-5 ckpt-35405 Benchmark Evaluation (N=4 Sampling)

## Status
Not started

## Model Under Test
**b3 epoch2 lr1e-5 checkpoint-35405 (N=4 sampling)**
- Path: `/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-causal-from-b2amc_fixed2_cont_epoch2_lr1e-5/checkpoint-35405`
- Trained block_size=3, use_regular_causal=True
- Running with: DreamShiftBlockN **N=4** (block_size=7, gen_block_size=4)
- Config: `dreamshift_blockN4_config.yaml`
- Servers: 8×TP=1, ports 30000-30007, max-running-requests=32
- **ALL evals must use --max-tokens 32768** (thinking model needs room)

## Reference Results (all N=4 unless noted)
| Benchmark | b2 N=3 | b3-backup8000 | b3-ckpt35405 | b3-e2-lr1e5-ckpt35405 |
|-----------|--------|--------------|--------------|----------------------|
| ARC-C | 95.3% | 95.0% | 95.5% | |
| IFEval | 87.4% | 82.3% | 82.4% | |
| GSM8K | 96% | 94.1% | 95.0% | |
| Math500 | 95.2% | 96.4% | 96.0% | |
| AIME-2025 | 61.0% | 56.7% | **63.3%** | |
| HumanEval | 93.9% | 94.5% | 92.1% | |
| MBPP | 91.8% | 92.6% | 91.1% | |
| LCB-v6 | 45.1% | 43.4% | 40.0% | |
| GPQA main | 54.5% | 48.0% | 50.9% | |
| MMLU-Pro | 65.5% | 62.1% | 64.0% | |
| MMLU | 82.4% | 77.6% | 81.0% | |
| TriviaQA | 63.2% | 57.8% | 58.4% | |
| CMMLU | 76.7% | 74.5% | 76.4% | |

## Launch Command
```bash
MODEL=/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-causal-from-b2amc_fixed2_cont_epoch2_lr1e-5/checkpoint-35405

source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache

for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache nohup python -m sglang.launch_server \
    --model-path $MODEL --trust-remote-code --tp-size 1 \
    --mem-fraction-static 0.85 --max-running-requests 32 \
    --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
    --dllm-algorithm-config dreamshift_blockN4_config.yaml \
    --dtype bfloat16 --port $((30000+i)) --chunked-prefill-size 4096 \
    --watchdog-timeout 1800 \
    > /tmp/sglang_b3e2_gpu${i}.log 2>&1 &
done
for i in $(seq 0 7); do
  for j in $(seq 1 60); do
    curl -sf http://localhost:$((30000+i))/health > /dev/null 2>&1 && echo "GPU $i ready" && break; sleep 10
  done
done
```

## Benchmark Execution Order (easy → hard)
**ALL scripts must use --max-tokens 32768**

```bash
PORTS="30000 30001 30002 30003 30004 30005 30006 30007"
mkdir -p bench_results/b3e2lr1e5
```

### Tier 1 (Fast)
```bash
python scripts/eval_arc_c.py --ports $PORTS --max-tokens 32768 --output-dir bench_results/b3e2lr1e5
python scripts/eval_ifeval.py --ports $PORTS --max-tokens 32768 --max-workers 8 --output-dir bench_results/b3e2lr1e5
python scripts/eval_gsm8k.py --ports $PORTS --max-tokens 32768 --output-dir bench_results/b3e2lr1e5
python scripts/eval_math500.py --ports $PORTS --max-tokens 32768 --output-dir bench_results/b3e2lr1e5
python scripts/eval_aime.py --year 2025 --ports $PORTS --max-tokens 32768 --output-dir bench_results/b3e2lr1e5
python scripts/eval_humaneval.py --ports $PORTS --max-tokens 32768 --output-dir bench_results/b3e2lr1e5
python scripts/eval_mbpp.py --ports $PORTS --max-tokens 32768 --output-dir bench_results/b3e2lr1e5
```

### Tier 2 (Medium)
```bash
python scripts/eval_lcb.py --version 6 --max-workers 16 --ports $PORTS --max-tokens 32768 --output-dir bench_results/b3e2lr1e5
HF_TOKEN=<your_hf_token> python scripts/eval_gpqa.py --subset main --ports $PORTS --max-tokens 32768 --output-dir bench_results/b3e2lr1e5
```

### Tier 3 (Slow)
```bash
python scripts/eval_mmlu_pro.py --ports $PORTS --max-tokens 32768 --output-dir bench_results/b3e2lr1e5
python scripts/eval_mmlu.py --ports $PORTS --max-tokens 32768 --output-dir bench_results/b3e2lr1e5
python scripts/eval_triviaqa.py --ports $PORTS --max-tokens 32768 --output-dir bench_results/b3e2lr1e5
python scripts/eval_cmmlu.py --ports $PORTS --max-tokens 32768 --output-dir bench_results/b3e2lr1e5
```

## Quality Check After Each Benchmark
1. Truncation < 10% at 32k
2. Extraction failures < 5%
3. Compare vs b3-ckpt35405 reference (most relevant comparison — same step, different lr/epoch)
4. If > 5pp different from b3-ckpt35405 → note as interesting (lr effect)
5. If > 15pp below b2 N=3 → debug

## Environment
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export HF_HOME=/data/yjian/hf_cache
export FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache
export HF_TOKEN=<your_hf_token>
```

## Progress Log
<!-- Agent updates this after each benchmark -->
