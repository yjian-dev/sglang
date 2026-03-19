# Plan — Full Benchmark Suite (DLLM N=3 vs Qwen3-8B)

## Status
Not started

## Models
- **DLLM N=3**: `sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont` — DreamShiftBlockN N=3 sampling, ports 30000-30007 (**already running**)
- **Qwen3-8B**: `Qwen/Qwen3-8B` — standard AR with thinking, ports 30010-30017 (launch when needed)

## Results Table

| Benchmark | DLLM N=3 | Qwen3-8B | Notes |
|-----------|----------|----------|-------|
| ARC-C (full) | | | |
| TriviaQA (1k) | | | |
| MMLU (2k rand) | | | |
| MMLU-Pro (2k rand) | | | |
| GPQA-Diamond (198) | | | |
| IFEval (541) | | | |
| GSM8K (1319) | | | |
| Math500 (500) | | | |
| MathBench (perf_4) | | | |
| AIME-2024 (30) | | | |
| AIME-2025 (30) | | | |
| HumanEval+ (164) | | | |
| MBPP+ (257) | | | |
| LCB-v6 (175) | | | |
| CMMLU (2k rand) | | | |

## Tasks

### Phase 1: Fix & Verify Scripts (do BEFORE running)
- [ ] Check `eval_triviaqa.py` prompt matches OC: "Answer these questions... start your answer with 'The answer is '"
- [ ] Check `eval_gpqa.py` prompt matches OC: "ANSWER: $LETTER" format at end, extract `ANSWER: [ABCD]`
- [ ] Verify `eval_mmlu.py` uses random seed=42 sampling
- [ ] Verify `eval_mmlu_pro.py` uses random seed=42 sampling
- [ ] Verify `eval_cmmlu.py` uses random seed=42 sampling
- [ ] Verify all scripts default to `--max-tokens 32768`

### Phase 2: DLLM N=3 Benchmarks (ports 30000-30007)
- [ ] ARC-C
- [ ] TriviaQA
- [ ] MMLU
- [ ] MMLU-Pro
- [ ] GPQA-Diamond
- [ ] IFEval
- [ ] GSM8K
- [ ] Math500
- [ ] MathBench (circular perf_4)
- [ ] AIME-2024
- [ ] AIME-2025
- [ ] HumanEval+
- [ ] MBPP+
- [ ] LCB-v6
- [ ] CMMLU

### Phase 3: Qwen3-8B Benchmarks (ports 30010-30017)
- [ ] Launch Qwen3-8B (8×TP=1)
- [ ] Run all 15 benchmarks with QWEN_PORTS

### Quality Check (after EACH benchmark)
For each result, check and record in notes column:
- truncation rate (finish_reason='length')
- extraction failure rate (pred='?')
- sample wrong answers reviewed

## OC Alignment Checklist
Reference: `/data/cxu/dllm-distillation/evaluation/opencompass/opencompass/configs/datasets/`

| Benchmark | OC Prompt Key | Our Script Status |
|-----------|--------------|-------------------|
| TriviaQA | "The answer is " prefix | check |
| GPQA | "ANSWER: $LETTER" at end | check |
| MMLU | standard 4-choice | check |
| MMLU-Pro | 10-choice (A-J) | check |
| GSM8K | \\boxed{} | check |
| Math500 | \\boxed{} | check |
| MathBench | circular perf_4 | ✓ implemented |
| LCB-v6 | OC run_test evaluator | ✓ implemented |
| CMMLU | 4-choice CN | check |
| IFEval | instruction format | check |

## Environment
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export HF_HOME=/data/yjian/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/yjian/hf_cache/hub
export FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache
export HF_TOKEN=<your_hf_token>
```

## Progress Log
<!-- Agent updates this after each step -->
