# Agent Prompt — 8B b3 Benchmark Evaluation

## Role
You are a benchmark evaluation agent. You launch servers, run benchmarks, record results, and debug failures.

## Model Under Test
- Path: `/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-allmasked-causal_fixed2`
- Architecture: SDARForCausalLM, block_size=3, mask_token_id=151669
- Algorithm: DreamShiftBlockN (gen_block_size=3)
- This IS thinking mode (Qwen3-8B based) — use max-tokens=32768

## Reference Scores (from our baseline)
If any benchmark result deviates from these by >5pp, investigate: check wrong answers, look at whether prompt format or answer extraction is wrong.

| Benchmark | Reference |
|-----------|-----------|
| ARC-C | 95.5 |
| TriviaQA | 66.3 |
| MMLU | 82.4 |
| MMLU-Pro | 73.1 |
| GPQA-Diamond | 59.1 |
| GPQA | 54.5 |
| IFEval | 84.7 |
| GSM8K | 96.0 |
| MATH-500 | 95.2 |
| MathBench | 89.13 |
| AIME-24 | 72.5 |
| AIME-25 | 61.04 |
| HumanEval | 94.5 |
| MBPP | 92.8 |
| LCB-v6 | 45.1 |

## Benchmarks (ordered fast → slow)

| # | Benchmark | Command |
|---|-----------|---------|
| 1 | AIME-24 | `python scripts/eval_aime.py --year 2024` |
| 2 | AIME-25 | `python scripts/eval_aime.py --year 2025` |
| 3 | MATH-500 | `python scripts/eval_math500.py` |
| 4 | HumanEval | `python scripts/eval_humaneval.py` |
| 5 | MBPP | `python scripts/eval_mbpp.py` |
| 6 | ARC-C | `python scripts/eval_arc_c.py` |
| 7 | GSM8K | `python scripts/eval_gsm8k.py` |
| 8 | GPQA-Diamond | `python scripts/eval_gpqa.py --subset diamond` |
| 9 | GPQA | `python scripts/eval_gpqa.py` |
| 10 | IFEval | `python scripts/eval_ifeval.py` |
| 11 | TriviaQA | `python scripts/eval_triviaqa.py` |
| 12 | MathBench | `python scripts/eval_mathbench.py` |
| 13 | LCB-v6 | `python scripts/eval_lcb.py` |
| 14 | MMLU-Pro | `python scripts/eval_mmlu_pro.py` |
| 15 | MMLU | `python scripts/eval_mmlu.py` |

**Run FULL datasets.** Run benchmarks SEQUENTIALLY in the order above.

## Server Config (8 servers, TP=1, ports 30000-30007)
```bash
for p in 30000 30001 30002 30003 30004 30005 30006 30007; do lsof -ti:$p | xargs -r kill -9; done; sleep 5
for i in 0 1 2 3 4 5 6 7; do
  port=$((30000+i))
  CUDA_VISIBLE_DEVICES=${i} \
  nohup python -m sglang.launch_server \
    --model-path /data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-allmasked-causal_fixed2 \
    --trust-remote-code --tp-size 1 \
    --mem-fraction-static 0.85 --max-running-requests 64 \
    --attention-backend flashinfer \
    --dllm-algorithm DreamShiftBlockN \
    --dllm-algorithm-config dreamshift_blockN3_config.yaml \
    --dtype bfloat16 --port $port \
    > /tmp/sglang_8b_b3_${i}.log 2>&1 &
done
```

## Common Args for ALL Benchmarks
```
--ports 30000 30001 30002 30003 30004 30005 30006 30007 --max-tokens 32768 --timeout 1200 --max-workers 64
```

## Environment
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export HF_HOME=/data/yjian/hf_cache
```

## Critical Rules
1. **Check Errors count** after each benchmark — if errors > 0, results are INVALID, reduce --max-workers to 32 and rerun
2. **Run benchmarks SEQUENTIALLY** — one at a time in the order listed
3. **Record results in PLAN.md** after each benchmark completes
4. **Full datasets only** — no --num-problems flag
5. **If score deviates >5pp from reference**, investigate wrong answers before moving on: check prompt format, answer extraction, error count

## Workflow
1. Launch 8 servers, wait for ALL 8 to show "ready to roll" in logs
2. Run benchmarks 1-15 sequentially. After each: check Errors, compare to reference, record in PLAN.md
3. If a score is off by >5pp: read some wrong answers from the output, check if it's extraction or real model error
4. Kill servers when done
5. Write Final Results table in PLAN.md
