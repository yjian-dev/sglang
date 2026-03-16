# Plan

## Status
Phase 1-3 complete. TP=4 baselines measured, bottlenecks profiled, optimizations applied. Greedy mode TP=4 benchmarked. Quality verified.

## Goal
Maximize DreamShiftBlockN throughput in TP=4 (4-GPU) deployment while maintaining quality.

## Phase 1: Baseline Measurements (TP=4) ✅
- [x] Launch N=3 sample server with TP=4 (GPUs 0-3), measure tore-speed-eval at C=1,2,4,8,16,32,64
- [x] Launch Qwen-AR server with TP=4 (GPUs 4-7), measure same concurrency levels
- [x] Record single-request output tok/s (best case latency)
- [x] Record max throughput (saturated concurrency)
- [x] Compare TP=4 vs TP=1 scaling efficiency

### TP=4 Baseline Results (iteration 1)

| Conc | DreamShift N=3 sample | Qwen-AR TP=4 | DS/Qwen Ratio |
|------|---------------------|--------------|---------------|
| 1    | 363                 | 320          | 1.13x         |
| 2    | 665                 | 603          | 1.10x         |
| 4    | 1,201               | 1,180        | 1.02x         |
| 8    | 2,128               | 2,272        | 0.94x         |
| 16   | 3,560               | 4,191        | 0.85x         |
| 32   | 5,566               | 7,421        | 0.75x         |
| 64   | 7,528               | 12,016       | 0.63x         |

### TP=4 vs TP=1 Scaling Efficiency

| Conc | DS TP=1 | DS TP=4 | DS Scale | Qwen TP=1 | Qwen TP=4 | Qwen Scale |
|------|---------|---------|----------|-----------|-----------|------------|
| 1    | 247     | 363     | 1.47x    | 149       | 320       | 2.15x      |
| 4    | 846     | 1,201   | 1.42x    | 565       | 1,180     | 2.09x      |
| 8    | 1,550   | 2,128   | 1.37x    | 1,086     | 2,272     | 2.09x      |
| 16   | 2,675   | 3,560   | 1.33x    | 2,017     | 4,191     | 2.08x      |
| 32   | 4,209   | 5,566   | 1.32x    | 3,613     | 7,421     | 2.05x      |
| 64   | 5,708   | 7,528   | 1.32x    | 5,794     | 12,016    | 2.07x      |

## Phase 2: Profile TP=4 Bottlenecks ✅
- [x] Capture algorithm timing at C=1 and C=32
- [x] Measure step breakdown: forward, prep, process, algorithm phases
- [x] Identify root cause of poor TP=4 scaling
- [x] Determine compute-bound vs memory-bound crossover
- [x] **NEW: Micro-profiled verify_sample 0.66ms breakdown**

### Profiling Results (TP=4, server-side timing)

#### At C=1 (bs=1):
| Component | Time (ms) | % of step |
|-----------|-----------|-----------|
| **Total step** | **5.7** | 100% |
| Forward (GPU) | 4.2 | 74% |
| Algorithm: classify | 0.17 | 3% |
| Algorithm: verify_sample | 0.63 | 11% |
| Algorithm: trim_assemble | 0.28 | 5% |
| Scheduler: prep | 0.29 | 5% |
| Scheduler: process | 0.17 | 3% |
| **Total CPU overhead** | **~1.5** | **26%** |

#### verify_sample breakdown (0.66ms):
| Sub-phase | Time (ms) | % of verify_sample |
|-----------|-----------|-------------------|
| verify_gpu (fused Triton kernel + gather) | 0.21 | 32% |
| sample_gpu (flashinfer sampling + draft probs softmax) | 0.36 | 55% |
| sync (cat + tolist GPU→CPU) | 0.08 | 12% |

**Key insight**: The sync overhead is only 0.08ms. The bottleneck is GPU kernel launches for small tensors (verify=0.21ms, sample=0.36ms).

#### Greedy mode CPU overhead (0.81ms total vs 1.14ms sampling):
| Phase | Sampling | Greedy | Savings |
|-------|----------|--------|---------|
| classify | 0.21 | 0.15 | -0.06 |
| verify_sample | 0.65 | 0.43 | -0.22 |
| trim_assemble | 0.28 | 0.23 | -0.05 |
| **Total** | **1.14** | **0.81** | **-0.33** |

### Root Cause Analysis: Why TP=4 Scaling Is Poor

**Amdahl's Law with serial CPU overhead:**
- DreamShift step at TP=1 ≈ 8.8ms (forward ≈ 7.5ms + CPU ≈ 1.3ms)
- DreamShift step at TP=4 ≈ 5.7ms (forward ≈ 4.2ms + CPU ≈ 1.5ms)
- Forward speedup: 7.5/4.2 = 1.79x (near-ideal for memory-bound)
- Total speedup: 8.8/5.7 = 1.54x (limited by constant CPU overhead)

**Qwen-AR comparison:**
- Qwen step at TP=1 ≈ 6.7ms (forward ≈ 6.5ms + CPU ≈ 0.2ms with overlap)
- Qwen step at TP=4 ≈ 3.1ms (forward ≈ 3.0ms + CPU ≈ 0.1ms with overlap)
- Total speedup: 6.7/3.1 = 2.16x (overlap hides CPU overhead)

## Phase 3: Optimize TP=4 Performance ✅ (partial)
- [x] **Profile verify_sample 0.63ms**: verify_gpu=0.21, sample_gpu=0.36, sync=0.08
- [x] **Cache CPU values**: Skip GPU→CPU sync in classify (classify: 0.28→0.21ms, ~0.07ms saved)
- [x] **Measure greedy mode TP=4**: 10-13% faster than sampling at all concurrency levels
- [x] **Quality verification**: IFEval strict 84.1% ✅, GSM8K 93.9% ✅
- [ ] **[HIGH] Overlap prep with forward**: Move prepare_for_dllm_decode into overlap window
- [ ] **[MED] Pipeline process + next prep**: Restructure decode loop to overlap post-processing
- [ ] **[MED] Reduce GPU kernel launches**: Fuse verify+sample into single kernel
- [ ] Test N=5 fp8 at TP=4

### TP=4 Full Results (iteration 2, with CPU cache optimization)

| Conc | DS N=3 greedy | DS N=3 sample | Qwen-AR | Greedy/Qwen | Sample/Qwen |
|------|--------------|--------------|---------|-------------|-------------|
| 1    | **427**      | 383          | 325     | **1.31x**   | 1.18x       |
| 2    | **798**      | 709          | 617     | **1.29x**   | 1.15x       |
| 4    | **1,448**    | 1,278        | 1,204   | **1.20x**   | 1.06x       |
| 8    | **2,592**    | 2,293        | 2,321   | **1.12x**   | 0.99x       |
| 16   | 4,266        | 3,804        | 4,282   | 1.00x       | 0.89x       |
| 32   | 6,284        | 5,835        | 7,596   | 0.83x       | 0.77x       |
| 64   | 8,484        | 7,963        | 12,464  | 0.68x       | 0.64x       |

### Improvement from iteration 1 → 2 (sampling mode)

| Conc | Iter 1 | Iter 2 | Change |
|------|--------|--------|--------|
| 1    | 363    | 383    | +5.5%  |
| 4    | 1,201  | 1,278  | +6.4%  |
| 8    | 2,128  | 2,293  | +7.8%  |
| 16   | 3,560  | 3,804  | +6.9%  |
| 32   | 5,566  | 5,835  | +4.8%  |
| 64   | 7,528  | 7,963  | +5.8%  |

## Phase 4: Quality Verification ✅ (partial)
- [x] Quick check: IFEval strict = 84.1% (>= 80%) ✅
- [x] Quick check: GSM8K = 93.9% (>= 90%) ✅
- [ ] Full validation: HumanEval >= 85%, MBPP >= 80%
- [ ] Full validation: AIME 2024 >= 73% single-shot

## Phase 5: Final Comparison & Report
- [x] Full throughput table: TP=4 DreamShift (greedy + sample) vs Qwen-AR
- [ ] Test EAGLE3 at TP=4 for comparison
- [ ] Plot throughput curves
- [ ] Document optimal configuration and remaining bottlenecks

## TP=1 Baselines (for comparison)

### Throughput (tore-speed-eval, 1x H100, AIME burst mode)
| Conc | Qwen-AR | EAGLE3 | N=3 sample | N=3 greedy | N=5 fp8 |
|------|---------|--------|-----------|-----------|---------|
| 1    | 149     | 204    | 247       | 272       | 328     |
| 4    | 565     | 699    | 846       | 967       | 1,072   |
| 8    | 1,086   | 1,269  | 1,550     | 1,770     | 1,605   |
| 16   | 2,017   | 2,160  | 2,675     | 2,999     | 2,380   |
| 32   | 3,613   | 3,414  | 4,209     | 4,736     | 3,745   |
| 64   | 5,794   | 4,676  | 5,708     | 6,414     | 4,862   |

### Quality (DreamShift N=3, from previous iterations)
| Benchmark | BF16 | Threshold |
|-----------|------|-----------|
| IFEval strict | 84.1% | >= 80% |
| GSM8K | 93.9% | >= 90% |
| HumanEval | 92.7% | >= 85% |
| MBPP | 93.8% | >= 80% |
| MATH-500 | 89.6% | — |

## Next Steps
1. **Overlap post-processing with next forward**: The biggest remaining opportunity. Move verify_sample+trim_assemble+process_batch_result to run concurrently with the next step's forward pass. This requires restructuring the decode loop into a pipeline.
2. **Fuse verify+sample into single kernel**: Currently 2 separate GPU operations (verify Triton + flashinfer sampling). Fusing would save ~0.15ms kernel launch overhead.
3. **Test N=5 fp8 at TP=4**: N=5 fp8 was fastest at low concurrency in TP=1 (328 tok/s). May benefit more from TP=4 due to smaller per-token compute.
4. **Explore enabling overlap scheduler for DLLM**: Currently disabled (server_args.py:2856-2860). If enabled, could hide ~1ms CPU overhead per step.

## Progress Log

### Iteration 1 (2026-03-16)
**Accomplished:**
- Killed 8x TP=1 Qwen-AR servers
- Launched DreamShift N=3 TP=4 on GPUs 0-3 (port 30000) and Qwen-AR TP=4 on GPUs 4-7 (port 30004)
- Measured throughput at C=1,2,4,8,16,32,64 for both servers
- Profiled per-step timing breakdown at C=1 (bs=1) and C=32 (bs=32)
- Identified root cause of poor TP=4 scaling: 1.5ms/step constant CPU overhead (Amdahl's law)

**Key findings:**
- DreamShift TP=4 scaling is 1.3-1.5x (poor) vs Qwen-AR 2.0-2.1x (near-ideal)
- DreamShift wins at C=1-4 (1.13x over Qwen-AR at C=1)
- CPU overhead is ~1.5ms/step: verify_sample (0.63ms), trim_assemble (0.28ms), classify (0.17ms), prep (0.3-1.1ms), process (0.17-0.5ms)
- Forward pass scales well (1.79x at TP=4, near-ideal for memory-bound)
- Overlap scheduler is disabled for DLLM (server_args.py:2856-2860)
- CUDA graph is working for DLLM decode at TP=4

### Iteration 2 (2026-03-16)
**Accomplished:**
- Micro-profiled verify_sample 0.66ms: verify_gpu=0.21ms, sample_gpu=0.36ms, sync=0.08ms
- Implemented CPU cache optimization: skip GPU→CPU sync in classify phase by caching rpx and seq_lens from prepare_for_dllm_decode
  - Modified: schedule_batch.py (cache CPU values), forward_batch_info.py (pass through), dreamshift_blockN.py (use cached values)
  - Classify time: 0.28→0.21ms (saved ~0.07ms)
- Attempted pre-allocated GPU tensor approach — caused memory leak (available > max_total), reverted to safe approach
- Benchmarked N=3 greedy mode TP=4: 10-13% faster than sampling at all concurrency levels
  - Greedy CPU overhead: 0.81ms vs sampling 1.14ms (29% less)
  - Greedy accept rate: 76% vs sampling 66%
- Benchmarked N=3 sampling mode TP=4 with optimization: +5-8% improvement over iteration 1
- Quality verification: IFEval strict 84.1% ✅, GSM8K 93.9% ✅

**Key results:**
- DS N=3 greedy TP=4: 427 tok/s at C=1 (1.31x Qwen-AR), 8,484 tok/s at C=64
- DS N=3 sample TP=4: 383 tok/s at C=1 (1.18x Qwen-AR), 7,963 tok/s at C=64
- DreamShift beats Qwen-AR at C<=4 (greedy) or C<=2 (sampling)
- Crossover point: C=8 for greedy, C=4 for sampling

**Changes made:**
- `python/sglang/srt/managers/schedule_batch.py`: Added `_dllm_rpx_cpu`/`_dllm_seq_lens_cpu` cache in `prepare_for_dllm_decode`, added `dllm_rpx_cpu`/`dllm_seq_lens_cpu` fields to `ModelWorkerBatch`, passed through `get_model_worker_batch`
- `python/sglang/srt/model_executor/forward_batch_info.py`: Pass DLLM CPU cache from `ModelWorkerBatch` to `ForwardBatch` in `init_new`
- `python/sglang/srt/dllm/algorithm/dreamshift_blockN.py`: Use cached CPU values when available (skip GPU→CPU sync), added verify_sample micro-timing

**Server status:** DS sampling TP=4 on port 30000 (GPUs 0-3)
**Next iteration:** Implement overlap post-processing or fused verify+sample kernel
