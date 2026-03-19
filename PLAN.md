# Plan — Full Benchmark Suite (DLLM N=3 vs Qwen3-8B)

## Status
Not started

## Models
- **DLLM N=3**: `sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont` — DreamShiftBlockN N=3 sampling, ports 30000-30007 (**already running**)
- **Qwen3-8B**: `Qwen/Qwen3-8B` — standard AR with thinking, ports 30010-30017 (launch when needed)

## Results Table

| Benchmark | DLLM N=3 | Qwen3-8B |
|-----------|----------|----------|
| ARC-C (1172) | | ✓ run |
| TriviaQA (full ~11k) | | ✓ run |
| MMLU (full ~14k) | | ✓ run |
| MMLU-Pro (full ~12k) | | ✓ run |
| GPQA (448, main) | ✓ run | ✓ run |
| GPQA-Diamond (198) | — | ✓ run |
| IFEval (541) | — | ✓ run |
| GSM8K (1319) | — | ✓ run |
| Math500 (500) | — | ✓ run |
| MathBench (full, perf_4) | — | ✓ run |
| AIME-2024 (30) | — | ✓ run |
| AIME-2025 (30) | — | ✓ run |
| HumanEval (164) | — | ✓ run |
| MBPP (257) | — | ✓ run |
| LCB-v6 (175) | — | ✓ run |
| CMMLU (full ~11.5k) | ✓ run | ✓ run |

*(✓ run = needs to be run, — = skip)*

## Tasks

### Phase 1: Script Checks ✅ Done
- [x] All scripts: default max_tokens=32768
- [x] All scripts: default num_problems=0 (full dataset)
- [x] eval_triviaqa.py: OC prompt "The answer is " ✓
- [x] eval_gpqa.py: OC prompt "ANSWER: $LETTER" ✓
- [ ] eval_gpqa.py: add --subset flag to support both "main" (448) and "diamond" (198)

### Phase 2: DLLM N=3 (ports 30000-30007) — 6 benchmarks
Run in this order (fast → slow):

- [ ] ARC-C → `python scripts/eval_arc_c.py --ports 30000 30001 30002 30003 30004 30005 30006 30007`
- [ ] GPQA main → `HF_TOKEN=<token> python scripts/eval_gpqa.py --subset main --ports 30000..30007`
- [ ] MMLU-Pro → `python scripts/eval_mmlu_pro.py --ports 30000..30007`
- [ ] MMLU → `python scripts/eval_mmlu.py --ports 30000..30007`
- [ ] TriviaQA → `python scripts/eval_triviaqa.py --ports 30000..30007`
- [ ] CMMLU → `python scripts/eval_cmmlu.py --ports 30000..30007`

After each: check truncation %, extraction failure %, sample wrong answers.

### Phase 3: Qwen3-8B (ports 30010-30017) — 16 benchmarks
First launch Qwen3-8B:
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

Then run (fast → slow):
- [ ] ARC-C
- [ ] GPQA-Diamond → `HF_TOKEN=<token> python scripts/eval_gpqa.py --subset diamond --ports 30010..30017`
- [ ] GPQA main → `HF_TOKEN=<token> python scripts/eval_gpqa.py --subset main --ports 30010..30017`
- [ ] IFEval
- [ ] GSM8K
- [ ] Math500
- [ ] AIME-2024 → `python scripts/eval_aime.py --year 2024 --ports 30010..30017`
- [ ] AIME-2025 → `python scripts/eval_aime.py --year 2025 --ports 30010..30017`
- [ ] HumanEval → `python scripts/eval_humaneval.py --ports 30010..30017`
- [ ] MBPP → `python scripts/eval_mbpp.py --ports 30010..30017`
- [ ] LCB-v6 → `python scripts/eval_lcb.py --version 6 --max-workers 16 --ports 30010..30017`
- [ ] MathBench → `python scripts/eval_mathbench.py --ports 30010..30017`
- [ ] TriviaQA
- [ ] MMLU-Pro
- [ ] MMLU
- [ ] CMMLU

## Quality Check (after EACH benchmark)
1. Count truncated (finish_reason='length') — note if >10%
2. Count extraction failures (pred='?') — FIX script if >5%, rerun
3. Print 3 wrong answer samples — verify it's model error not script bug

## Environment
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export HF_HOME=/data/yjian/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/yjian/hf_cache/hub
export FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache
# HF_TOKEN: stored separately, ask user if needed (for GPQA)
```

## OC Reference
`/data/cxu/dllm-distillation/evaluation/opencompass/opencompass/configs/datasets/`

## Progress Log
<!-- Agent updates this after each benchmark -->
