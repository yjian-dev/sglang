# Plan — Full Benchmark Suite (DLLM N=3 vs Qwen3-8B)

## Status
Not started

## Models
- **DLLM N=3**: `sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont` — DreamShiftBlockN N=3 sampling, ports 30000-30007 (**already running**)
- **Qwen3-8B**: `Qwen/Qwen3-8B` — standard AR with thinking, ports 30010-30017 (launch when needed)

## Results Table

| Benchmark | DLLM N=3 | Qwen3-8B | Notes |
|-----------|----------|----------|-------|
| ARC-C (1172) | | | |
| TriviaQA (full ~11k) | | | |
| MMLU (full ~14k) | | | |
| MMLU-Pro (full ~12k) | | | |
| GPQA-Diamond (198) | | | |
| IFEval (541) | | | |
| GSM8K (1319) | | | |
| Math500 (500) | | | |
| MathBench (full, perf_4) | | | |
| AIME-2024 (30) | | | |
| AIME-2025 (30) | | | |
| HumanEval+ (164) | | | |
| MBPP+ (257) | | | |
| LCB-v6 (175) | | | |
| CMMLU (full ~11.5k) | | | |

## Tasks

### Phase 1: Fix & Verify Scripts (do BEFORE running)
- [x] `eval_triviaqa.py` prompt: OC format "The answer is " ✓
- [x] `eval_gpqa.py` prompt: OC format "ANSWER: $LETTER" ✓
- [x] All scripts: default max_tokens=32768 ✓
- [ ] Remove ALL `--num-problems` limits: MMLU/MMLU-Pro/CMMLU/TriviaQA must run full dataset
- [ ] `eval_mmlu.py`: remove `--num-problems` default (or set 0=full)
- [ ] `eval_mmlu_pro.py`: remove `--num-problems` default
- [ ] `eval_cmmlu.py`: remove `--num-problems` default
- [ ] `eval_triviaqa.py`: run full validation set (no limit)

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
