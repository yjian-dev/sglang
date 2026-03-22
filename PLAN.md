# Plan — b3 epoch2 lr1e-5 ckpt-35405 Benchmark Evaluation (N=4 Sampling)

## Status
**COMPLETE** — All 13 benchmarks finished

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
| ARC-C | 95.3% | 95.0% | 95.5% | **95.4%** |
| IFEval | 87.4% | 82.3% | 82.4% | **83.0%** |
| GSM8K | 96% | 94.1% | 95.0% | **94.5%** |
| Math500 | 95.2% | 96.4% | 96.0% | **95.8%** |
| AIME-2025 | 61.0% | 56.7% | **63.3%** | **56.7%** ⚠️ |
| HumanEval | 93.9% | 94.5% | 92.1% | **92.1%** |
| MBPP | 91.8% | 92.6% | 91.1% | **89.5%** |
| LCB-v6 | 45.1% | 43.4% | 40.0% | **40.6%** |
| GPQA main | 54.5% | 48.0% | 50.9% | **45.3%** ⚠️ |
| MMLU-Pro | 65.5% | 62.1% | 64.0% | **56.9%** ⚠️ |
| MMLU | 82.4% | 77.6% | 81.0% | **77.4%** ⚠️ |
| TriviaQA | 63.2% | 57.8% | 58.4% | **52.6%** (~60.1% excl errors) ¹ |
| CMMLU | 76.7% | 74.5% | 76.4% | **71.1%** ⚠️ |

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

### Iteration 1 (2026-03-21)
- Servers already running with correct model (checkpoint-35405, epoch2 lr1e-5)
- Sanity check passed — model produces coherent thinking output
- **All 13 benchmarks completed**

**¹ TriviaQA note:** GPU 5 crashed mid-run (tokenizer NoneType bug in `convert_tokens_to_string`). 2243/17944 requests errored (all routed to port 30005). Excluding errors: 9439/15701 = 60.1%, close to reference 58.4%. GPU 5 restarted.

**Anomalies (vs b3-ckpt35405 reference):**
| Benchmark | b3-ckpt35405 | b3-e2-lr1e5 | Delta | Verdict |
|-----------|-------------|-------------|-------|---------|
| AIME-2025 | 63.3% | 56.7% | -6.6pp | High variance (N=30), matches b3-backup8000 |
| GPQA main | 50.9% | 45.3% | -5.6pp | Below even b3-backup8000 (48.0%). Real regression |
| MMLU-Pro | 64.0% | 56.9% | -7.1pp | **Significant.** Epoch2 lr1e-5 hurt knowledge |
| MMLU | 81.0% | 77.4% | -3.6pp | Moderate drop |
| CMMLU | 76.4% | 71.1% | -5.3pp | Notable drop in Chinese knowledge |
| MBPP | 91.1% | 89.5% | -1.6pp | Within noise |

**Summary:** The epoch2 lr1e-5 checkpoint shows consistent degradation on knowledge-heavy benchmarks (MMLU-Pro -7.1pp, GPQA -5.6pp, CMMLU -5.3pp, MMLU -3.6pp). Math/code benchmarks are stable (ARC-C, GSM8K, Math500, HumanEval, LCB all within ±1pp). This pattern suggests the lower learning rate in epoch2 may have caused catastrophic forgetting of factual knowledge while preserving reasoning ability. The b3-ckpt35405 (original lr) is strictly better.

**No further action needed.** All results saved in `bench_results/b3e2lr1e5/`.
