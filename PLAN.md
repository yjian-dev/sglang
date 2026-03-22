# Plan — N=3 Sampling 5-Run Stability Benchmark

## Status
Not started — Run 1/5

## Model
**N=3 sampling** (b2-allmasked-causal_fixed2_cont)
- Path: `/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont`
- Config: `dreamshift_blockN3_config.yaml` (block_size=5, gen_block_size=3, temp=1.0)
- Servers: 8×TP=1, ports 30000-30007, max-running-requests=32
- **ALL evals: --max-tokens 32768**

## Goal
Run all 13 benchmarks **5 times** to get mean ± std. Temperature=1.0 (sampling), so scores have variance.

## Previous Single-Run Reference (N=3)
| Benchmark | Score |
|-----------|-------|
| ARC-C | 95.3% |
| IFEval | 87.4% |
| GSM8K | 96% |
| Math500 | 95.2% |
| AIME-2025 | 61.0% |
| HumanEval | 93.9% |
| MBPP | 91.8% |
| LCB-v6 | 45.1% |
| GPQA main | 54.5% |
| MMLU-Pro | 65.5% |
| MMLU | 82.4% |
| TriviaQA | 63.2% |
| CMMLU | 76.7% |

## Results (5 runs each)
| Benchmark | Run1 | Run2 | Run3 | Run4 | Run5 | Mean | Std |
|-----------|------|------|------|------|------|------|-----|
| ARC-C | | | | | | | |
| IFEval | | | | | | | |
| GSM8K | | | | | | | |
| Math500 | | | | | | | |
| AIME-2025 | | | | | | | |
| HumanEval | | | | | | | |
| MBPP | | | | | | | |
| LCB-v6 | | | | | | | |
| GPQA main | | | | | | | |
| MMLU-Pro | | | | | | | |
| MMLU | | | | | | | |
| TriviaQA | | | | | | | |
| CMMLU | | | | | | | |

## Execution Order Per Run

For each run N (1 to 5), run benchmarks easy → hard.
Save to `bench_results/n3_run{N}/`.

```bash
PORTS="30000 30001 30002 30003 30004 30005 30006 30007"

# Replace N with run number (1, 2, 3, 4, 5)
N=1
OUTDIR=bench_results/n3_run${N}
mkdir -p $OUTDIR
```

### Tier 1 (Fast, < 10 min each)
```bash
python scripts/eval_arc_c.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
python scripts/eval_ifeval.py --ports $PORTS --max-tokens 32768 --max-workers 8 --output-dir $OUTDIR
python scripts/eval_gsm8k.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
python scripts/eval_math500.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
python scripts/eval_aime.py --year 2025 --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
python scripts/eval_humaneval.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
python scripts/eval_mbpp.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
```

### Tier 2 (Medium)
```bash
python scripts/eval_lcb.py --version 6 --max-workers 16 --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
HF_TOKEN=<your_hf_token> python scripts/eval_gpqa.py --subset main --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
```

### Tier 3 (Slow — run all 3 concurrently to save time)
```bash
python scripts/eval_mmlu_pro.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR &
python scripts/eval_mmlu.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR &
python scripts/eval_triviaqa.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR &
wait
python scripts/eval_cmmlu.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
```

## After Each Run
1. Extract all scores and update the Results table above
2. Check for anomalies vs previous single-run reference (> 5pp off = investigate)
3. Check truncation and extraction rates
4. Start next run immediately — no need to restart servers

## After All 5 Runs
Compute mean ± std for each benchmark and update the table.
If std > 3pp on any benchmark, flag it as high variance.

## Quality Check
- Truncation < 10% at 32k
- Extraction failures < 5%
- If a run has obvious issues (crash, high error rate), mark and re-run that specific benchmark

## Server Launch
```bash
MODEL=/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache

for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache nohup python -m sglang.launch_server \
    --model-path $MODEL --trust-remote-code --tp-size 1 \
    --mem-fraction-static 0.85 --max-running-requests 32 \
    --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
    --dllm-algorithm-config dreamshift_blockN3_config.yaml \
    --dtype bfloat16 --port $((30000+i)) --chunked-prefill-size 4096 \
    --watchdog-timeout 1800 \
    > /tmp/sglang_n3_gpu${i}.log 2>&1 &
done
for i in $(seq 0 7); do
  for j in $(seq 1 60); do
    curl -sf http://localhost:$((30000+i))/health > /dev/null 2>&1 && echo "GPU $i ready" && break; sleep 10
  done
done
```

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
<!-- Agent updates after each run -->
