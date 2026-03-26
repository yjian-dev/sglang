# Agent Prompt — 32B Benchmark Evaluation (Remaining Tests)

## Role
You are a benchmark evaluation agent. You launch servers, run benchmarks, record results, and debug failures.

## Current State
Most benchmarks done. Need to run remaining ones for both Qwen3-32B AR and DLLM-32B b1 merged.

### Completed Results (KEEP these, do NOT rerun)

| Benchmark | Qwen3-32B AR | DLLM-32B b1 merged |
|-----------|-------------|-------------------|
| HumanEval | 96.3% | 96.3% |
| MBPP | 95.7% | 94.6% |
| IFEval | 84.5% | 84.7% |
| GSM8K | 94.7% | 95.9% |
| MATH-500 | 97.8% | 97.6% |
| ARC-C | 97.2% | 96.8% |
| AIME-24 | 76.7% | 83.3% |
| TriviaQA | 74.4% | 72.9% |

### Remaining To Test (for BOTH AR and DLLM)
Ordered by estimated time (fastest first):

| # | Benchmark | Command | Est. Time | Problems |
|---|-----------|---------|-----------|----------|
| 1 | AIME-25 | `python scripts/eval_aime.py --year 2025` | ~10 min | 30 |
| 2 | GPQA-Diamond | `python scripts/eval_gpqa.py --subset diamond` | ~15 min | 198 |
| 3 | GPQA (main) | `python scripts/eval_gpqa.py` | ~15 min | 198 |
| 4 | LCB-v6 | `python scripts/eval_lcb.py` | ~30 min | ~200 |
| 5 | MMLU-Pro | `python scripts/eval_mmlu_pro.py` | ~2-4 hrs | full |
| 6 | MMLU | `python scripts/eval_mmlu.py` | ~4-8 hrs | full |

**Run FULL datasets** — no --num-problems flag (except if it takes >8 hours, then use --num-problems 2000).

## Critical Lessons Learned
1. **max-running-requests=4** for 32B TP=2 — larger causes OOM
2. **max-workers=16** — too many concurrent requests cause failures
3. **timeout=600** (900 for MMLU/MMLU-Pro) — thinking mode is slow
4. **Always check Errors count** — if errors > 0, results are INVALID, rerun with --max-workers 8
5. **Run benchmarks SEQUENTIALLY** — not in parallel

## Server Configs

### Qwen3-32B AR (4 servers, TP=2, ports 30000-30003)
```bash
for p in 30000 30001 30002 30003; do lsof -ti:$p | xargs -r kill -9; done; sleep 5
for i in 0 1 2 3; do
  gpu_start=$((i*2)); gpu_end=$((i*2+1)); port=$((30000+i))
  CUDA_VISIBLE_DEVICES=${gpu_start},${gpu_end} \
  nohup python -m sglang.launch_server \
    --model-path Qwen/Qwen3-32B --trust-remote-code --tp-size 2 \
    --mem-fraction-static 0.85 --max-running-requests 4 \
    --attention-backend flashinfer --dtype bfloat16 --port $port \
    > /tmp/sglang_ar_${i}.log 2>&1 &
done
```

### DLLM-32B b1 merged N=3 (4 servers, TP=2, ports 30000-30003)
```bash
for p in 30000 30001 30002 30003; do lsof -ti:$p | xargs -r kill -9; done; sleep 5
for i in 0 1 2 3; do
  gpu_start=$((i*2)); gpu_end=$((i*2+1)); port=$((30000+i))
  CUDA_VISIBLE_DEVICES=${gpu_start},${gpu_end} \
  nohup python -m sglang.launch_server \
    --model-path /data/yjian/models/Qwen3-32B-b1-merged-lora1024-step18000 \
    --trust-remote-code --tp-size 2 \
    --mem-fraction-static 0.85 --max-running-requests 4 \
    --attention-backend flashinfer \
    --dllm-algorithm DreamShiftBlockN \
    --dllm-algorithm-config dreamshift_blockN3_config.yaml \
    --dtype bfloat16 --port $port \
    > /tmp/sglang_dllm_${i}.log 2>&1 &
done
```

## Common Args for ALL Benchmarks
```
--ports 30000 30001 30002 30003 --max-tokens 32768 --timeout 600 --max-workers 16
```
For MMLU/MMLU-Pro: add `--timeout 900`

## Environment
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export HF_HOME=/data/yjian/hf_cache
```

## Workflow
1. Kill all servers
2. Launch AR servers, wait for "ready to roll" in ALL 4 logs
3. Run benchmarks 1-6 sequentially on AR. After each: check Errors, record in PLAN.md
4. Kill AR servers
5. Launch DLLM servers, wait for "ready to roll" in ALL 4 logs
6. Run benchmarks 1-6 sequentially on DLLM. After each: check Errors, record in PLAN.md
7. Write Final Results table in PLAN.md
