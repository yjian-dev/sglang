# Plan — Full OC-Aligned Benchmark: N=3 vs Qwen3-8B vs SDAR-8B-Chat (2 runs each)

## Status
Not started

## Models (run in this order)

### Model 1: DLLM N=3 (our model)
- Path: `/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont`
- Algorithm: DreamShiftBlockN, config: `dreamshift_blockN3_config.yaml`
- Launch: 8×TP=1, max-running-requests=32

### Model 2: Qwen3-8B (teacher/baseline AR)
- Path: `Qwen/Qwen3-8B`
- Standard AR, no dllm
- Launch: 8×TP=1, max-running-requests=64

### Model 3: SDAR-8B-Chat (JetLM, another DLLM)
- Path: `JetLM/SDAR-8B-Chat`
- Algorithm: LowConfidence
- Launch: 8×TP=1, max-running-requests=32

## Benchmarks to Run (8 OC-realigned scripts)
All use `--max-tokens 32768` except TriviaQA (256).

| # | Benchmark | Script | Est. Time | Notes |
|---|-----------|--------|-----------|-------|
| 1 | ARC-C | eval_arc_c.py | ~5 min | prompt: ANSWER: $LETTER |
| 2 | MMLU | eval_mmlu.py | ~6-8 hrs | prompt: ANSWER: $LETTER, A) format |
| 3 | MMLU-Pro | eval_mmlu_pro.py | ~5-7 hrs | prompt: Question:/Options: |
| 4 | CMMLU | eval_cmmlu.py | ~5-7 hrs | prompt: 答案: $选项, subject prefix |
| 5 | MBPP | eval_mbpp.py | ~5 min | 3-shot (was 0-shot) |
| 6 | TriviaQA | eval_triviaqa.py | ~30 min | /generate + A: prefix, max_tokens=256 |
| 7 | GSM8K | eval_gsm8k.py | ~10 min | math_verify scoring |
| 8 | AIME-2025 | eval_aime.py | ~5 min | no system msg |

## Results Table

| Benchmark | N=3 R1 | N=3 R2 | N=3 Mean | Qwen3 R1 | Qwen3 R2 | Qwen3 Mean | SDAR R1 | SDAR R2 | SDAR Mean |
|-----------|--------|--------|----------|----------|----------|------------|---------|---------|-----------|
| ARC-C | | | | | | | | | |
| MMLU | | | | | | | | | |
| MMLU-Pro | | | | | | | | | |
| CMMLU | | | | | | | | | |
| MBPP | | | | | | | | | |
| TriviaQA | | | | | | | | | |
| GSM8K | | | | | | | | | |
| AIME-2025 | | | | | | | | | |

## Server Launch Commands

### N=3
```bash
MODEL=/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont
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
```

### Qwen3-8B
```bash
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache nohup python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B --trust-remote-code --tp-size 1 \
    --mem-fraction-static 0.85 --max-running-requests 64 \
    --attention-backend flashinfer --dtype bfloat16 \
    --port $((30000+i)) --chunked-prefill-size 4096 --watchdog-timeout 1800 \
    > /tmp/sglang_qwen_gpu${i}.log 2>&1 &
done
```

### SDAR-8B-Chat
```bash
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache nohup python -m sglang.launch_server \
    --model-path JetLM/SDAR-8B-Chat --dllm-algorithm LowConfidence \
    --tp-size 1 --trust-remote-code \
    --mem-fraction-static 0.85 --max-running-requests 32 \
    --attention-backend flashinfer --dtype bfloat16 \
    --port $((30000+i)) --watchdog-timeout 1800 \
    > /tmp/sglang_sdar_gpu${i}.log 2>&1 &
done
```

## Execution Order Per Model

For each model, run 2 rounds. Each round runs fast → slow:

```bash
PORTS="30000 30001 30002 30003 30004 30005 30006 30007"
OUTDIR=bench_results/<model>_run<N>
mkdir -p $OUTDIR
```

### Fast (run sequentially, < 10 min each)
```bash
python scripts/eval_arc_c.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
python scripts/eval_mbpp.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
python scripts/eval_gsm8k.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
python scripts/eval_aime.py --year 2025 --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
python scripts/eval_triviaqa.py --ports $PORTS --max-tokens 256 --output-dir $OUTDIR
```

### Slow (run sequentially, hours each — NEVER concurrent)
```bash
python scripts/eval_mmlu_pro.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
python scripts/eval_mmlu.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
python scripts/eval_cmmlu.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
```

## Full Workflow
1. **Launch N=3 servers** → Run 8 benchmarks × 2 rounds → Kill servers
2. **Launch Qwen3-8B servers** → Run 8 benchmarks × 2 rounds → Kill servers
3. **Launch SDAR-8B-Chat servers** → Run 8 benchmarks × 2 rounds → Kill servers
4. **Compute means** and update results table

## Quality Check After Each Benchmark
1. Errors < 5% (if higher, increase timeout and rerun)
2. Truncation < 10%
3. Sample 3 wrong answers — confirm model error not script bug

## Important Notes
- **NEVER run Tier 3 (MMLU/MMLU-Pro/CMMLU) concurrently** — causes server OOM/hang
- TriviaQA uses `/generate` with raw ChatML + "A:" prefix — works for all 3 models (all use ChatML)
- For Qwen3-8B: thinking model, responses much longer, use --timeout 600 for slow benchmarks
- SDAR-8B-Chat: block_size=32, LowConfidence, much slower per-request than N=3

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
<!-- Agent updates after each benchmark -->
