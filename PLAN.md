# Plan — Benchmark Evaluation (DLLM N=3 vs Qwen3-8B)

## Status
In progress — running missing benchmarks

## Current Model (DO NOT RESTART)
- `/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont`
- N=3 sampling, 8×TP=1, ports 30000-30007

## Results Table

| Benchmark | Ours (DLLM N=3) | Qwen3-8B | Notes |
|-----------|-----------------|----------|-------|
| ARC-C | | | TODO |
| TriviaQA | | | TODO |
| MMLU | | | TODO |
| MMLU-Pro | | | TODO — prev run biased, need random 2k |
| GPQA-Diamond | 59.1% | 48.01% | ✓ done |
| IFEval | 87.4% | | ✓ done (DLLM), Qwen TODO |
| GSM8K | 96% | | ✓ done (DLLM), Qwen TODO |
| Math500 | 95.2% | | ✓ done (DLLM), Qwen TODO |
| MathBench | | | TODO |
| AIME-2025 | 61.04% | | ✓ done (DLLM), Qwen TODO |
| HumanEval+ | 93.9% | | ✓ done (DLLM), Qwen TODO |
| MBPP+ | 91.8% | | ✓ done (DLLM), Qwen TODO |
| HumanEval-X Python | 86.6% | | ✓ done (DLLM), Qwen TODO |
| LCB-v6 | 45.1% | | ✓ done (DLLM), Qwen TODO |
| CMMLU | | | TODO |
| MMMLU-lite | | | TODO |

## Tasks

### Phase 1: Complete missing DLLM benchmarks

- [ ] **ARC-C** — create `scripts/eval_arc_c.py`, run, check errors
- [ ] **MMLU** — create `scripts/eval_mmlu.py` (distributed version), run 2k random
- [ ] **MMLU-Pro** — fix random sampling in `scripts/eval_mmlu_pro.py`, run 2k random
- [ ] **CMMLU** — create `scripts/eval_cmmlu.py`, run 2k random
- [ ] **MMMLU-lite** — create `scripts/eval_mmmlu.py`, run 200/language × 6 languages
- [ ] **MathBench** — create `scripts/eval_mathbench.py`, run 1k
- [ ] **TriviaQA** — run `scripts/eval_triviaqa.py --num-problems 1000`
- [ ] **LongBench-v2 Hard** — create `scripts/eval_longbench_hard.py` (skip if OOM)

### Phase 2: Error diagnosis for each completed benchmark
For EACH benchmark:
- [ ] Check wrong examples (print first 10 failures)
- [ ] Count: (a) truncated (finish_reason=length), (b) no extraction (pred=?), (c) wrong answer
- [ ] If truncated >10%: increase max_tokens to 32768, re-run
- [ ] If no extraction >5%: fix regex/prompt, re-run
- [ ] Document root cause in notes column of results table

### Phase 3: Qwen3-8B comparison
Launch Qwen3-8B on ports 30010-30017:
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
# Wait for all 8 healthy (ports 30010-30017)
for i in $(seq 0 7); do
  port=$((30010+i))
  for j in $(seq 1 60); do
    curl -sf http://localhost:$port/health > /dev/null 2>&1 && echo "GPU $i ready" && break
    sleep 5
  done
done
```

Then run all benchmark scripts with `--ports 30010 30011 30012 30013 30014 30015 30016 30017`

**Already have Qwen3-8B reference numbers** (from LLaDA paper):
- GPQA: 48.01%, MMLU-Pro: 65.83%, IFEval: 84.29%, GSM+: 85.56%, LCB: 26.76%
- For benchmarks not in paper, measure directly

## Script Convention
All scripts must:
- Default `--max-tokens 16384`
- Use `/v1/chat/completions` with thinking model format
- Strip `<think>...</think>` before answer extraction
- Print sample wrong answers at end
- Support `--ports` for multi-GPU distribution

## Environment
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
HF_TOKEN=<your_hf_token>  # for gated datasets like GPQA
```

## Progress Log
- 2026-03-18: LCB-v6 45.1% confirmed with OC evaluator (reliability_guard fix)
- 2026-03-18: eval scripts pushed to jyq/dreamshift-blockN-accuracy-fix
- Started Phase 1 (missing benchmarks)


## Evaluator Feedback (Iteration 1)
Could not parse evaluator response.


## Evaluator Feedback (Iteration 2)
Could not parse evaluator response.


## Evaluator Feedback (Iteration 3)
Could not parse evaluator response.


## Evaluator Feedback (Iteration 4)
Could not parse evaluator response.


## Evaluator Feedback (Iteration 5)
Could not parse evaluator response.


## Evaluator Feedback (Iteration 6)
Could not parse evaluator response.


## Evaluator Feedback (Iteration 7)
Could not parse evaluator response.


## Evaluator Feedback (Iteration 8)
Could not parse evaluator response.


## Evaluator Feedback (Iteration 9)
Could not parse evaluator response.


## Evaluator Feedback (Iteration 10)
Could not parse evaluator response.


## Evaluator Feedback (Iteration 11)
Could not parse evaluator response.
