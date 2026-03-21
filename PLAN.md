# Plan — b3 Checkpoint Benchmark Evaluation (N=4 Sampling)

## Status
Not started

## Model Under Test
**b3 checkpoint (N=4 sampling)**
- Path: `/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-causal-from-b2amc_fixed2_cont_epoch2_lr1e-5_backup8000`
- Trained block_size=3, use_regular_causal=True
- Running with: DreamShiftBlockN **N=4** (block_size=7, gen_block_size=4)
- Config: `dreamshift_blockN4_config.yaml`
- Servers: 8×TP=1, ports 30000-30007

## N=3 Reference Results (b2-cont checkpoint)
| Benchmark | N=3 (b2-cont) | Notes |
|-----------|--------------|-------|
| ARC-C (1172) | 95.3% | |
| GPQA main (448) | 54.5% | |
| MMLU (~14k) | 82.4% | |
| MMLU-Pro (~12k) | 65.5% | |
| TriviaQA (~11k) | 63.2% | |
| GSM8K (1319) | 96% | |
| Math500 (500) | 95.2% | |
| AIME-2025 (30) | 61.0% | |
| IFEval (541) | 87.4% | |
| HumanEval (164) | 93.9% | |
| MBPP (257) | 91.8% | |
| LCB-v6 (175) | 45.1% | OC evaluator |
| CMMLU (~11.5k) | 76.7% | |

## Results Table (b3, N=4 Sampling)
| Benchmark | b3 N=4 | b2 N=3 | Delta | Notes |
|-----------|--------|--------|-------|-------|
| ARC-C (1172) | | 95.3% | | |
| IFEval (541) | | 87.4% | | |
| GSM8K (1319) | | 96% | | |
| Math500 (500) | | 95.2% | | |
| AIME-2025 (30) | | 61.0% | | |
| HumanEval (164) | | 93.9% | | |
| MBPP (257) | | 91.8% | | |
| LCB-v6 (175) | | 45.1% | | OC eval |
| GPQA main (448) | | 54.5% | | |
| MMLU-Pro (~12k) | | 65.5% | | |
| MMLU (~14k) | | 82.4% | | |
| TriviaQA (~11k) | | 63.2% | | |
| CMMLU (~11.5k) | | 76.7% | | |

## Launch Command (8×TP=1 N=4 Sampling)
```bash
MODEL=/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-causal-from-b2amc_fixed2_cont_epoch2_lr1e-5_backup8000

source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export HF_HOME=/data/yjian/hf_cache
export FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache

for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache nohup python -m sglang.launch_server \
    --model-path $MODEL \
    --trust-remote-code --tp-size 1 \
    --mem-fraction-static 0.85 --max-running-requests 32 \
    --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
    --dllm-algorithm-config dreamshift_blockN4_config.yaml \
    --dtype bfloat16 --port $((30000+i)) --chunked-prefill-size 4096 \
    --watchdog-timeout 1800 \
    > /tmp/sglang_b3n4_gpu${i}.log 2>&1 &
done
# Wait for all 8
for i in $(seq 0 7); do
  for j in $(seq 1 60); do
    curl -sf http://localhost:$((30000+i))/health > /dev/null 2>&1 && echo "GPU $i ready" && break; sleep 10
  done
done
```

## Benchmark Execution Order (easy → hard)

### Tier 1: Fast (< 10 min each) — run first
```bash
PORTS="30000 30001 30002 30003 30004 30005 30006 30007"

# ARC-C
python scripts/eval_arc_c.py --ports $PORTS --output-dir bench_results/b3n4

# IFEval
python scripts/eval_ifeval.py --ports $PORTS --max-tokens 32768 --max-workers 8 --output-dir bench_results/b3n4

# GSM8K
python scripts/eval_gsm8k.py --ports $PORTS --output-dir bench_results/b3n4

# Math500
python scripts/eval_math500.py --ports $PORTS --output-dir bench_results/b3n4

# AIME-2025
python scripts/eval_aime.py --year 2025 --ports $PORTS --output-dir bench_results/b3n4

# HumanEval
python scripts/eval_humaneval.py --ports $PORTS --output-dir bench_results/b3n4

# MBPP
python scripts/eval_mbpp.py --ports $PORTS --output-dir bench_results/b3n4
```

### Tier 2: Medium (10-60 min each)
```bash
# LCB-v6 (OC evaluator, accurate)
python scripts/eval_lcb.py --version 6 --max-workers 16 --ports $PORTS --output-dir bench_results/b3n4

# GPQA main (needs HF token)
HF_TOKEN=<your_hf_token> python scripts/eval_gpqa.py --subset main --ports $PORTS --output-dir bench_results/b3n4
```

### Tier 3: Slow (1-8 hrs each)
```bash
# MMLU-Pro (full ~12k)
python scripts/eval_mmlu_pro.py --ports $PORTS --output-dir bench_results/b3n4

# MMLU (full ~14k)
python scripts/eval_mmlu.py --ports $PORTS --output-dir bench_results/b3n4

# TriviaQA (full ~11k)
python scripts/eval_triviaqa.py --ports $PORTS --output-dir bench_results/b3n4

# CMMLU (full ~11.5k)
python scripts/eval_cmmlu.py --ports $PORTS --output-dir bench_results/b3n4
```

## Quality Check After Each Benchmark
1. Check truncation rate (finish_reason='length'). At 32k should be < 10%.
2. Check extraction failures (pred='?'). Should be < 5%.
3. **Compare vs N=3 reference**:
   - If b3 N=4 is within ±3% of b3 N=3: normal variation
   - If b3 N=4 is > 5% LOWER than b3 N=3: investigate (wrong checkpoint? wrong config?)
   - If b3 N=4 consistently HIGHER: good news, better checkpoint

## Anomaly Detection
If any benchmark score is > 10pp lower than N=3 reference, immediately:
1. Test a few sample prompts manually for quality
2. Verify the correct model path is loaded (check server log)
3. Verify dreamshift_blockN4_config.yaml is correct (block_size=7, gen_block_size=4)
4. Check if server crashed and restarted

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
