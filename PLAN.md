# Plan — b3 Checkpoint Benchmark Evaluation (N=4 Sampling)

## Status
COMPLETE — All 13 benchmarks finished

## Model Under Test
**b3 checkpoint (N=4 sampling)**
- Path: `/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-causal-from-b2amc_fixed2_cont_epoch2_lr1e-5_backup8000`
- Trained block_size=3, use_regular_causal=True
- Running with: DreamShiftBlockN **N=4** (block_size=7, gen_block_size=4)
- Config: `dreamshift_blockN4_config.yaml`
- Servers: 8×TP=1, ports 30000-30007

## N=3 Reference Results (b2-cont checkpoint)
| Benchmark | N=3 (b2-cont) | Notes |
|-----------|--------------|-------|
| ARC-C (1172) | 95.3% | |
| GPQA main (448) | 54.5% | |
| MMLU (~14k) | 82.4% | |
| MMLU-Pro (~12k) | 65.5% | |
| TriviaQA (~11k) | 63.2% | |
| GSM8K (1319) | 96% | |
| Math500 (500) | 95.2% | |
| AIME-2025 (30) | 61.0% | |
| IFEval (541) | 87.4% | |
| HumanEval (164) | 93.9% | |
| MBPP (257) | 91.8% | |
| LCB-v6 (175) | 45.1% | OC evaluator |
| CMMLU (~11.5k) | 76.7% | |

## Results Table (b3, N=4 Sampling)
| Benchmark | b3 N=4 | b2 N=3 | Delta | Notes |
|-----------|--------|--------|-------|-------|
| ARC-C (1172) | 95.0% | 95.3% | -0.3 | OK |
| IFEval (541) | 82.3% | 87.4% | -5.1 | prompt-strict; loose=86.0% |
| GSM8K (1319) | 94.1% | 96% | -1.9 | 32k max_tokens |
| Math500 (500) | 96.4% | 95.2% | +1.2 | 32k max_tokens (was 87.6% at 8k) |
| AIME-2025 (30) | 56.7% | 61.0% | -4.3 | 17/30, small sample |
| HumanEval (164) | 94.5% | 93.9% | +0.6 | 32k max_tokens (was 76.8% at 4k) |
| MBPP (257) | 92.6% | 91.8% | +0.8 | 32k max_tokens (was 4.7% at 512!) |
| LCB-v6 (175) | 43.4% | 45.1% | -1.7 | OC eval |
| GPQA main (448) | 48.0% | 54.5% | -6.5 | **ANOMALY** — real drop, 32k default |
| MMLU-Pro (~12k) | 62.1% | 65.5% | -3.4 | |
| MMLU (~14k) | 77.6% | 82.4% | -4.8 | 32k default |
| TriviaQA (17944) | 57.8% | 63.2% | -5.4 | 32k rerun; ref used ~11k subset |
| CMMLU (~11.5k) | 74.5% | 76.7% | -2.2 | OK |

## Launch Command (8×TP=1 N=4 Sampling)
```bash
MODEL=/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-causal-from-b2amc_fixed2_cont_epoch2_lr1e-5_backup8000

source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export HF_HOME=/data/yjian/hf_cache
export FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache

for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache nohup python -m sglang.launch_server \
    --model-path $MODEL \
    --trust-remote-code --tp-size 1 \
    --mem-fraction-static 0.85 --max-running-requests 32 \
    --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
    --dllm-algorithm-config dreamshift_blockN4_config.yaml \
    --dtype bfloat16 --port $((30000+i)) --chunked-prefill-size 4096 \
    --watchdog-timeout 1800 \
    > /tmp/sglang_b3n4_gpu${i}.log 2>&1 &
done
# Wait for all 8
for i in $(seq 0 7); do
  for j in $(seq 1 60); do
    curl -sf http://localhost:$((30000+i))/health > /dev/null 2>&1 && echo "GPU $i ready" && break; sleep 10
  done
done
```

## Benchmark Execution Order (easy → hard)

### Tier 1: Fast (< 10 min each) — run first
```bash
PORTS="30000 30001 30002 30003 30004 30005 30006 30007"

# ARC-C
python scripts/eval_arc_c.py --ports $PORTS --output-dir bench_results/b3n4

# IFEval
python scripts/eval_ifeval.py --ports $PORTS --max-tokens 32768 --max-workers 8 --output-dir bench_results/b3n4

# GSM8K
python scripts/eval_gsm8k.py --ports $PORTS --output-dir bench_results/b3n4

# Math500
python scripts/eval_math500.py --ports $PORTS --output-dir bench_results/b3n4

# AIME-2025
python scripts/eval_aime.py --year 2025 --ports $PORTS --output-dir bench_results/b3n4

# HumanEval
python scripts/eval_humaneval.py --ports $PORTS --output-dir bench_results/b3n4

# MBPP
python scripts/eval_mbpp.py --ports $PORTS --output-dir bench_results/b3n4
```

### Tier 2: Medium (10-60 min each)
```bash
# LCB-v6 (OC evaluator, accurate)
python scripts/eval_lcb.py --version 6 --max-workers 16 --ports $PORTS --output-dir bench_results/b3n4

# GPQA main (needs HF token)
HF_TOKEN=<your_hf_token> python scripts/eval_gpqa.py --subset main --ports $PORTS --output-dir bench_results/b3n4
```

### Tier 3: Slow (1-8 hrs each)
```bash
# MMLU-Pro (full ~12k)
python scripts/eval_mmlu_pro.py --ports $PORTS --output-dir bench_results/b3n4

# MMLU (full ~14k)
python scripts/eval_mmlu.py --ports $PORTS --output-dir bench_results/b3n4

# TriviaQA (full ~11k)
python scripts/eval_triviaqa.py --ports $PORTS --output-dir bench_results/b3n4

# CMMLU (full ~11.5k)
python scripts/eval_cmmlu.py --ports $PORTS --output-dir bench_results/b3n4
```

## Quality Check After Each Benchmark
1. Check truncation rate (finish_reason='length'). At 32k should be < 10%.
2. Check extraction failures (pred='?'). Should be < 5%.
3. **Compare vs N=3 reference**:
   - If b3 N=4 is within ±3% of b3 N=3: normal variation
   - If b3 N=4 is > 5% LOWER than b3 N=3: investigate (wrong checkpoint? wrong config?)
   - If b3 N=4 consistently HIGHER: good news, better checkpoint

## Anomaly Detection
If any benchmark score is > 10pp lower than N=3 reference, immediately:
1. Test a few sample prompts manually for quality
2. Verify the correct model path is loaded (check server log)
3. Verify dreamshift_blockN4_config.yaml is correct (block_size=7, gen_block_size=4)
4. Check if server crashed and restarted

## Environment
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export HF_HOME=/data/yjian/hf_cache
export FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache
export HF_TOKEN=<your_hf_token>
```

## Progress Log

### Iteration 1 (2026-03-21)
- [x] Verified all 8 servers healthy (b3 N=4 config correct)
- [x] Sanity check: model produces coherent output
- [x] Tier 1 benchmarks complete (ARC-C, IFEval, GSM8K, Math500, AIME, HumanEval, MBPP)
- [x] Tier 2 benchmarks complete (LCB-v6, GPQA main)
- [x] MMLU-Pro complete
- [x] MMLU — 77.6% (-4.8pp)
- [x] TriviaQA — 57.8% at 32k (-5.4pp); was 29.5% at 4k default
- [x] CMMLU — 74.5% (-2.2pp)

**Key findings:**
1. **max_tokens truncation bug**: HumanEval (4k), MBPP (512), Math500 (8k), TriviaQA (4k) defaults too low for thinking model. Re-ran all with 32k → scores normalized.
2. **GPQA anomaly**: -6.5pp drop is real (already 32k max_tokens). Likely N=4 OOD effect on hard reasoning.
3. **IFEval**: -5.1pp on prompt-strict but only -1.4pp on loose. Borderline.
4. **MMLU**: -4.8pp real drop. MMLU-Pro -3.4pp.
5. **TriviaQA**: -5.4pp after fixing max_tokens. Different dataset size (17944 vs ~11k ref).
6. **Code benchmarks improved**: HumanEval +0.6pp, MBPP +0.8pp, Math500 +1.2pp (all with 32k).

### Summary
- **Improved (>+0.5pp)**: Math500 (+1.2), MBPP (+0.8), HumanEval (+0.6)
- **Within ±3pp**: ARC-C (-0.3), GSM8K (-1.9), LCB-v6 (-1.7), CMMLU (-2.2)
- **Moderate drop (3-5pp)**: MMLU-Pro (-3.4), AIME (-4.3), MMLU (-4.8), IFEval (-5.1)
- **Notable drop (>5pp)**: TriviaQA (-5.4), GPQA (-6.5)
- **Average delta across all 13 benchmarks**: ~-2.5pp

The b3 N=4 checkpoint shows modest overall degradation vs b2 N=3, concentrated in knowledge-heavy benchmarks (GPQA, MMLU, TriviaQA). Code/math benchmarks are equal or slightly improved. The N=4 OOD gap (trained bl=3, infer bl=7) likely explains the knowledge benchmark drops.

### Next Steps
- All benchmarks complete. No further action needed.
- Consider re-running GPQA/MMLU with N=3 config on b3 checkpoint to isolate OOD effect vs model difference.
