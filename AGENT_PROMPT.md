# Agent Prompt — Benchmark Evaluation (DLLM vs Qwen3-8B)

## Role
You are a benchmark evaluation agent. Your job is to run missing quality benchmarks on the DLLM model (N=3 sampling), analyze wrong answers, fix evaluation issues, and then run the same benchmarks on Qwen3-8B for comparison.

## Project Description
We have a DLLM model (DreamShiftBlockN N=3 sampling) running on 8x TP=1 servers (ports 30000-30007). We need to complete the benchmark table in PLAN.md. For each benchmark:
1. Run it on DLLM (ports 30000-30007)
2. Check wrong answers — determine if failures are due to (a) extraction bug, (b) max_tokens truncation, or (c) actual model limitations
3. Fix the script if (a) or (b), re-run
4. Run same script on Qwen3-8B (launch fresh 8 servers on ports 30010-30017 using Qwen/Qwen3-8B)

## Current Model (already running)
- **Model**: `/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont`
- **Algorithm**: DreamShiftBlockN N=3 sampling (block_size=5, gen_block_size=3)
- **Config**: `dreamshift_blockN3_config.yaml`
- **Servers**: 8x TP=1, ports 30000-30007, max-running-requests=32
- **DO NOT KILL OR RESTART these servers unless they crash**

## Benchmarks to Complete (in priority order)

### 1. ARC-C (Quick, ~1.2k problems)
- Dataset: `allenai/ai2_arc`, config `ARC-Challenge`, split `test`
- Format: 4-choice (A/B/C/D), generate answer
- Script: `scripts/eval_arc_c.py` (create if not exists)

### 2. MMLU (57 subjects, ~14k problems — sample 2k randomly)
- Dataset: `cais/mmlu`, config `all`, split `test`
- Format: 4-choice, generate answer
- Script: `scripts/eval_mmlu.py` (create if not exists)

### 3. MMLU-Pro (already have script, fix sampling)
- Dataset: `TIGER-Lab/MMLU-Pro`, split `test`
- Issue: previous run used first 2k which was biased (all law). Fix: random sample.
- Script: `scripts/eval_mmlu_pro.py` — fix to use random seed=42 sampling, run 2000 problems

### 4. CMMLU (Chinese MMLU, ~11.5k problems — sample 2k)
- Dataset: `haonan-li/cmmlu`, all subjects
- Format: 4-choice Chinese multiple choice
- Script: `scripts/eval_cmmlu.py` (create)

### 5. MMMLU-lite (multilingual, sample from English + 5 key languages)
- Dataset: `openai/MMMLU`, language configs: `EN_US`, `ZH_CN`, `JA_JP`, `FR_FR`, `DE_DE`, `KO_KR`
- Sample ~200 per language
- Script: `scripts/eval_mmmlu.py` (create)

### 6. MathBench (math, ~3k — sample 1k)
- Dataset: `opencompass/mathbench` or check HF for available dataset
- Script: `scripts/eval_mathbench.py` (create)

### 7. TriviaQA (already have script)
- Script: `scripts/eval_triviaqa.py` — run with 1000 problems

### 8. LongBench-v2 Hard (subset with difficulty='hard')
- Dataset: `THUDM/LongBench-v2`, filter `difficulty == 'hard'`
- Format: multiple choice or short answer
- Script: `scripts/eval_longbench_hard.py` (create)
- **Note**: requires long context (up to 128k), may need special handling

## Qwen3-8B Comparison
After completing DLLM benchmarks, launch Qwen3-8B:
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
for gpu in 0 1 2 3 4 5 6 7; do
  port=$((30010 + gpu))
  CUDA_VISIBLE_DEVICES=$gpu nohup python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B --trust-remote-code --tp-size 1 \
    --mem-fraction-static 0.85 --max-running-requests 64 \
    --attention-backend flashinfer \
    --dtype bfloat16 --port $port --chunked-prefill-size 4096 \
    > /tmp/sglang_qwen_gpu${gpu}.log 2>&1 &
done
# Wait for all 8: check ports 30010-30017
```
Then run same scripts with `--ports 30010 30011 30012 30013 30014 30015 30016 30017`

## Script Template
All scripts should:
- Use chat completions API (`/v1/chat/completions`)
- Use `max_tokens=16384` (thinking models need long output)
- Strip `<think>...</think>` before extracting answers
- Distribute across all ports with `--ports 30000 30001 ... 30007`
- Print wrong examples at the end to diagnose failures

## Error Diagnosis Protocol
After each benchmark run:
1. Check if wrong answers are **truncated** (finish_reason='length') → increase max_tokens or reduce thinking
2. Check if extraction **failed** (pred='?' or empty) → fix regex/extraction logic
3. Check if model genuinely **wrong** → note as model limitation

## Environment
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
```

## Constraints
- Always use `--max-tokens 16384` or higher
- If >10% of answers are truncated, increase to 32768
- Record all results in PLAN.md results table
- Fix scripts before considering a benchmark "done"
- Skip LongBench-v2 if context length causes OOM
