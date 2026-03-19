# Agent Prompt — Full Benchmark Suite (DLLM N=3 vs Qwen3-8B)

## Role
Run benchmarks for both models. After each benchmark: check truncation/extraction rates, fix issues if found, record results in PLAN.md.

## Models
- **DLLM N=3** (already running): ports 30000-30007
- **Qwen3-8B**: launch on ports 30010-30017 when needed

## Environment
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export HF_HOME=/data/yjian/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/yjian/hf_cache/hub
export FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache
# HF_TOKEN: needed only for GPQA. Get from user if needed or use env var.
```

## Launch Qwen3-8B (ports 30010-30017)
```bash
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache nohup python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B --trust-remote-code --tp-size 1 \
    --mem-fraction-static 0.85 --max-running-requests 64 \
    --attention-backend flashinfer --dtype bfloat16 \
    --port $((30010+i)) --chunked-prefill-size 4096 --watchdog-timeout 1800 \
    > /tmp/sglang_qwen_gpu${i}.log 2>&1 &
done
for i in $(seq 0 7); do
  for j in $(seq 1 60); do
    curl -sf http://localhost:$((30010+i))/health > /dev/null 2>&1 && echo "GPU $i ready" && break; sleep 10
  done
done
```

---

## DLLM N=3 — 6 Benchmarks (ports 30000-30007)

Run in order (fast first):

```bash
DLLM="30000 30001 30002 30003 30004 30005 30006 30007"
```

### 1. ARC-C (1172, ~5 min)
```bash
python scripts/eval_arc_c.py --ports $DLLM
```

### 2. GPQA main (448, ~10 min) — needs HF_TOKEN
```bash
HF_TOKEN=<token> python scripts/eval_gpqa.py --subset main --ports $DLLM
```

### 3. MMLU-Pro (full ~12k, ~5-7 hrs)
```bash
python scripts/eval_mmlu_pro.py --ports $DLLM
```

### 4. MMLU (full ~14k, ~6-8 hrs)
```bash
python scripts/eval_mmlu.py --ports $DLLM
```

### 5. TriviaQA (full ~11k, ~2-3 hrs)
```bash
python scripts/eval_triviaqa.py --ports $DLLM
```

### 6. CMMLU (full ~11.5k, ~5-7 hrs)
```bash
python scripts/eval_cmmlu.py --ports $DLLM
```

---

## Qwen3-8B — 16 Benchmarks (ports 30010-30017)

**Launch Qwen3-8B first** (see above). Then run:

```bash
QWEN="30010 30011 30012 30013 30014 30015 30016 30017"
```

Fast benchmarks first:

```bash
# ARC-C
python scripts/eval_arc_c.py --ports $QWEN

# GPQA Diamond (198)
HF_TOKEN=<token> python scripts/eval_gpqa.py --subset diamond --ports $QWEN

# GPQA main (448)
HF_TOKEN=<token> python scripts/eval_gpqa.py --subset main --ports $QWEN

# IFEval
python scripts/eval_ifeval.py --ports $QWEN

# GSM8K
python scripts/eval_gsm8k.py --ports $QWEN

# Math500
python scripts/eval_math500.py --ports $QWEN

# AIME-2024
python scripts/eval_aime.py --year 2024 --ports $QWEN

# AIME-2025
python scripts/eval_aime.py --year 2025 --ports $QWEN

# HumanEval
python scripts/eval_humaneval.py --ports $QWEN

# MBPP
python scripts/eval_mbpp.py --ports $QWEN

# LCB-v6
python scripts/eval_lcb.py --version 6 --max-workers 16 --ports $QWEN

# MathBench (circular, ~60 min)
python scripts/eval_mathbench.py --ports $QWEN

# TriviaQA
python scripts/eval_triviaqa.py --ports $QWEN

# MMLU-Pro
python scripts/eval_mmlu_pro.py --ports $QWEN

# MMLU
python scripts/eval_mmlu.py --ports $QWEN

# CMMLU
python scripts/eval_cmmlu.py --ports $QWEN
```

---

## Quality Check After EACH Benchmark
1. Check truncation rate (`finish_reason='length'`). If >15% at 32k → note in results, don't rerun
2. Check extraction failure rate (`pred='?'`). **If >5% → fix script and rerun**
3. Print 3 wrong examples — confirm it's model error not script bug
4. Update PLAN.md results table

## OC Alignment Reference
`/data/cxu/dllm-distillation/evaluation/opencompass/opencompass/configs/datasets/`
- TriviaQA prompt: "The answer is " prefix
- GPQA prompt: "ANSWER: $LETTER" (implemented ✓)
- MMLU/MMLU-Pro/CMMLU: standard multiple choice
- MathBench: circular perf_4 (implemented ✓)
- LCB: OC run_test evaluator (implemented ✓)

## Notes
- All scripts default to full dataset (--num-problems 0)
- All scripts default to --max-tokens 32768
- DLLM servers must NOT be killed (running on 30000-30007)
- If a server crashes, restart with: `CUDA_VISIBLE_DEVICES=$i FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache python -m sglang.launch_server ... --watchdog-timeout 1800`
