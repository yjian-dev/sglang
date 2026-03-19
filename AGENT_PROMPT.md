# Agent Prompt — Full Benchmark Suite (DLLM N=3 vs Qwen3-8B)

## Role
You are a benchmark evaluation agent. Run all benchmarks for both models, check output quality after each benchmark, fix issues, and record results in PLAN.md.

## Models
- **DLLM N=3** (already running): ports 30000-30007, 8×TP=1
- **Qwen3-8B**: launch on ports 30010-30017 when needed (see launch command below)

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
# Wait for all 8: check ports 30010-30017
for i in $(seq 0 7); do
  for j in $(seq 1 60); do
    curl -sf http://localhost:$((30010+i))/health > /dev/null 2>&1 && echo "GPU $i ready" && break
    sleep 10
  done
done
```

## Benchmark Scripts
All scripts are in `scripts/`. Default `--max-tokens 32768`. Use all 8 ports.

```bash
DLLM_PORTS="30000 30001 30002 30003 30004 30005 30006 30007"
QWEN_PORTS="30010 30011 30012 30013 30014 30015 30016 30017"
```

## Benchmark List & Commands

### 1. ARC-C (1172 problems, ~5 min)
```bash
python scripts/eval_arc_c.py --ports $PORTS
```

### 2. TriviaQA (1000 problems, ~3 min)
**OC prompt**: "Answer these questions... start your answer with 'The answer is '"
**Our prompt**: must match this pattern. Check `scripts/eval_triviaqa.py`.
```bash
python scripts/eval_triviaqa.py --num-problems 1000 --ports $PORTS
```

### 3. MMLU (random 2000 from 14k, ~10 min)
```bash
python scripts/eval_mmlu.py --num-problems 2000 --ports $PORTS
```

### 4. MMLU-Pro (random 2000 from 12k, ~15 min)
```bash
python scripts/eval_mmlu_pro.py --num-problems 2000 --ports $PORTS
```

### 5. GPQA-Diamond (198 problems, ~5 min)
**OC prompt**: "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER'... Think step by step before answering.\n\n{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
**Note**: Uses `jianstreet` HF account (HF_TOKEN set above).
```bash
HF_TOKEN=<your_hf_token> python scripts/eval_gpqa.py --ports $PORTS
```
**Check**: Compare prompt in `scripts/eval_gpqa.py` with OC format above. Fix if different.

### 6. IFEval (541 problems, ~5 min)
```bash
python scripts/eval_ifeval.py --ports $PORTS
```

### 7. GSM8K (full 1319 problems, ~5 min)
```bash
python scripts/eval_gsm8k.py --num-problems 1319 --ports $PORTS
```

### 8. Math500 (500 problems, ~5 min)
```bash
python scripts/eval_math500.py --ports $PORTS
```

### 9. MathBench (circular eval, ~30 min)
**Data**: `/data/cxu/dllm-distillation/evaluation/opencompass/.cache/opencompass/data/mathbench_v1/`
**Metric**: `perf_4` (all 4 circular shifts must be correct) — matches OC default
```bash
python scripts/eval_mathbench.py --ports $PORTS
```

### 10. AIME 2024 (30 problems, ~3 min)
```bash
python scripts/eval_aime.py --year 2024 --num-problems 30 --ports $PORTS
```

### 11. AIME 2025 (30 problems, ~3 min)
```bash
python scripts/eval_aime.py --year 2025 --num-problems 30 --ports $PORTS
```

### 12. HumanEval+ (164 problems, ~3 min)
```bash
python scripts/eval_humaneval.py --ports $PORTS
```

### 13. MBPP+ (257 problems, ~3 min)
```bash
python scripts/eval_mbpp.py --ports $PORTS
```

### 14. LCB-v6 (175 problems, ~15 min with OC evaluator)
```bash
python scripts/eval_lcb.py --version 6 --max-workers 16 --ports $PORTS --output-dir bench_results/lcb_$MODEL
```

### 15. CMMLU (random 2000 from 11.5k, ~15 min)
```bash
python scripts/eval_cmmlu.py --num-problems 2000 --ports $PORTS
```

## Execution Order
**Run DLLM first (ports 30000-30007), then launch Qwen3-8B and run same benchmarks.**

For DLLM:
1. ARC-C, TriviaQA, MMLU, MMLU-Pro (can run in parallel as separate scripts)
2. GPQA, IFEval, GSM8K, Math500
3. MathBench (long, circular), AIME-24, AIME-25
4. HumanEval+, MBPP+, LCB-v6, CMMLU

For Qwen3-8B: same order, same scripts with QWEN_PORTS.

## After Each Benchmark: Quality Check Protocol
For each completed benchmark, check:
1. **Truncation**: count finish_reason='length'. If >10% truncated → this is already at 32k so note as "model needs long thinking"
2. **No extraction**: count pred='?'. If >5% → fix extraction regex and rerun
3. **Sample wrong answers**: print 3-5 examples, check if failure is model error vs extraction error
4. **OC alignment**: verify prompt matches OpenCompass format (see OC configs at `/data/cxu/dllm-distillation/evaluation/opencompass/opencompass/configs/datasets/`)

If extraction failure >5%: FIX the script and rerun before moving to next benchmark.

## OC Prompt Alignment Notes
Key prompts to verify/fix before running:

**TriviaQA**: prompt should be:
`"Answer these questions, your answer should be as simple as possible, start your answer with the prompt 'The answer is '.\nQ: {question}?"`
Then extract "The answer is X" from response.

**GPQA**: prompt should end with `"ANSWER: $LETTER"` format instruction.
Extraction: look for `ANSWER: [ABCD]` at end of response.

**MMLU**: standard 4-choice format. Extract last letter A/B/C/D.

**MathBench**: use OC's circular eval. perf_4 = all 4 permutations correct.

**LCB**: use OC's `run_test` evaluator (already implemented in eval_lcb.py).

## Results Recording
Update PLAN.md results table after each benchmark completes.

## Important Constraints
- Always use `--max-tokens 32768`
- Do NOT kill DLLM servers (ports 30000-30007) unless explicitly needed
- Qwen3-8B runs on ports 30010-30017 (separate GPUs)
- If a server crashes, restart it before continuing
- Record both DLLM and Qwen3-8B results in the same table
