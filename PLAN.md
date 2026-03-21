# Plan — b3 Checkpoint-35405 Benchmark Evaluation (N=4 Sampling)

## Status
Not started

## Model Under Test
**b3 checkpoint-35405 (N=4 sampling)**
- Path: `/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-causal-from-b2amc_fixed2_cont/checkpoint-35405`
- Trained block_size=3, use_regular_causal=True
- Running with: DreamShiftBlockN **N=4** (block_size=7, gen_block_size=4)
- Config: `dreamshift_blockN4_config.yaml`
- Servers: 8×TP=1, ports 30000-30007

## Reference Results
| Benchmark | b2 N=3 | b3-backup8000 N=4 | Delta (b3 vs b2) |
|-----------|--------|-------------------|-----------------|
| ARC-C | 95.3% | 95.0% | -0.3 |
| IFEval | 87.4% | 82.3% | -5.1 |
| GSM8K | 96% | 94.1% | -1.9 |
| Math500 | 95.2% | 96.4% | +1.2 |
| AIME-2025 | 61.0% | 56.7% | -4.3 |
| HumanEval | 93.9% | 94.5% | +0.6 |
| MBPP | 91.8% | 92.6% | +0.8 |
| LCB-v6 | 45.1% | 43.4% | -1.7 |
| GPQA main | 54.5% | 48.0% | -6.5 |
| MMLU-Pro | 65.5% | 62.1% | -3.4 |
| MMLU | 82.4% | 77.6% | -4.8 |
| TriviaQA | 63.2% | 57.8% | -5.4 |
| CMMLU | 76.7% | 74.5% | -2.2 |

## Results Table (checkpoint-35405, N=4 Sampling)
| Benchmark | ckpt-35405 N=4 | b3-backup8000 N=4 | b2 N=3 | Notes |
|-----------|---------------|-------------------|--------|-------|
| ARC-C (1172) | | 95.0% | 95.3% | |
| IFEval (541) | | 82.3% | 87.4% | |
| GSM8K (1319) | | 94.1% | 96% | |
| Math500 (500) | | 96.4% | 95.2% | |
| AIME-2025 (30) | | 56.7% | 61.0% | |
| HumanEval (164) | | 94.5% | 93.9% | |
| MBPP (257) | | 92.6% | 91.8% | |
| LCB-v6 (175) | | 43.4% | 45.1% | OC eval |
| GPQA main (448) | | 48.0% | 54.5% | |
| MMLU-Pro (~12k) | | 62.1% | 65.5% | |
| MMLU (~14k) | | 77.6% | 82.4% | |
| TriviaQA (~17.9k) | | 57.8% | 63.2% | |
| CMMLU (~11.5k) | | 74.5% | 76.7% | |

## Launch Command
```bash
MODEL=/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-causal-from-b2amc_fixed2_cont/checkpoint-35405

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
    > /tmp/sglang_b3c35_gpu${i}.log 2>&1 &
done
# Wait for all 8
for i in $(seq 0 7); do
  for j in $(seq 1 60); do
    curl -sf http://localhost:$((30000+i))/health > /dev/null 2>&1 && echo "GPU $i ready" && break; sleep 10
  done
done
```

## Benchmark Execution Order (easy → hard)

```bash
PORTS="30000 30001 30002 30003 30004 30005 30006 30007"
mkdir -p bench_results/b3c35
```

### Tier 1 (Fast, < 10 min each)
```bash
python scripts/eval_arc_c.py --ports $PORTS --output-dir bench_results/b3c35
python scripts/eval_ifeval.py --ports $PORTS --max-tokens 32768 --max-workers 8 --output-dir bench_results/b3c35
python scripts/eval_gsm8k.py --ports $PORTS --output-dir bench_results/b3c35
python scripts/eval_math500.py --ports $PORTS --output-dir bench_results/b3c35
python scripts/eval_aime.py --year 2025 --ports $PORTS --output-dir bench_results/b3c35
python scripts/eval_humaneval.py --ports $PORTS --output-dir bench_results/b3c35
python scripts/eval_mbpp.py --ports $PORTS --output-dir bench_results/b3c35
```

### Tier 2 (Medium, 10-60 min)
```bash
python scripts/eval_lcb.py --version 6 --max-workers 16 --ports $PORTS --output-dir bench_results/b3c35
HF_TOKEN=<your_hf_token> python scripts/eval_gpqa.py --subset main --ports $PORTS --output-dir bench_results/b3c35
```

### Tier 3 (Slow, overnight)
```bash
python scripts/eval_mmlu_pro.py --ports $PORTS --output-dir bench_results/b3c35
python scripts/eval_mmlu.py --ports $PORTS --output-dir bench_results/b3c35
python scripts/eval_triviaqa.py --ports $PORTS --output-dir bench_results/b3c35
python scripts/eval_cmmlu.py --ports $PORTS --output-dir bench_results/b3c35
```

## Quality Check After Each Benchmark
1. Truncation < 10%, extraction failures < 5%
2. Compare vs b3-backup8000: if > 5pp different → note as interesting
3. Compare vs b2 N=3: if > 10pp lower → investigate

## Anomaly Detection
If any score is > 15pp below b2 N=3 reference:
1. Check server log: `tail -20 /tmp/sglang_b3c35_gpu0.log`
2. Test manually: `curl -s http://localhost:30000/v1/chat/completions ...`
3. Check model loaded correctly in log

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
