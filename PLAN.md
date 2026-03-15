# Plan

## Status
Phase 1 complete, Phase 2 partially complete. N=2 identified as optimal config.

## Goal
Maximize sampling verify throughput to match/exceed EAGLE3 (~5100 tok/s at conc=32) while maintaining quality (GSM8K >= 90%, HumanEval >= 85%).

## Key Finding: N=2 is optimal for high-concurrency sampling verify

### Throughput by N (tore-speed-eval, sampling verify, 1 GPU H100)
| N | conc=1 | conc=32 | conc=64 | tok/fwd (bs=1) | Accept% | block_size |
|---|--------|---------|---------|----------------|---------|-----------|
| 2 | 181 | 3,272 | **5,350** | 1.64 | 78.5% | 3 |
| 3 | 205 | **3,509** | 5,109 | 1.84 | 50.5% | 5 |
| 4 | **227** | 3,461 | 4,690 | 2.05 | 37.3% | 7 |
| 5 | 220 | 3,244 | 4,222 | 1.99 | 19.2% | 9 |
| EAGLE3 | 228 | 5,103 | 5,569 | — | — | — |

- N=2 wins at conc=64 (5,350, only 4% behind EAGLE3)
- N=2's small block_size (3) is most compute-efficient at high batch sizes
- N=2's high accept rate (78.5%) means less rejection overhead

### After fused single-sync optimization (N=2)
| Concurrency | Baseline | Optimized | EAGLE3 | vs EAGLE3 |
|-------------|----------|-----------|--------|-----------|
| 1 | 181 | 177 | 228 | -22% |
| 32 | 3,272 | 3,188 | 5,103 | -37% |
| 64 | 5,350 | **5,451** | 5,569 | **-2.1%** |

### Profiling (N=2, conc=32, per step)
| Phase | Time (ms) | % |
|-------|-----------|---|
| Phase 1: classify + fill input_ids | 0.87 | 7.5% |
| Phase 2: GPU forward (model) | 1.34 | 11.5% |
| **Phase 3: verify + sample** | **9.27** | **79.5%** |
| Phase 4: trim + assemble | 0.31 | 2.7% |
| **Total** | **11.79** | 100% |

Phase 3 breakdown (9.27ms):
- softmax [32, 151k] for verify: ~1ms
- torch.stack [32, 151k] draft probs: ~0.5ms
- multinomial [32, 151k] for correction: ~2ms
- flashinfer sampling [~64, 151k]: ~2ms
- draft_probs softmax [32, 151k]: ~1ms
- GPU→CPU sync (.tolist()): ~1ms
- Various gathers, tensor creation: ~1.5ms

### Quality (N=3, sampling verify, 8 GPU) — N=2 quality TBD
| Benchmark | Score |
|---|---|
| GSM8K (1319Q) | 94.7% |
| MATH-500 | 88.4% |
| IFEval strict | 85.95% |
| HumanEval | 91.5% |
| MBPP (16k) | 93.0% |

## Tasks

### Phase 1: Find optimal N for sampling verify
- [x] Run tore-speed-eval for N=2, 3, 4, 5 at concurrency=1, 32, 64
- [x] Compare TPF, accept rate, and throughput for each N
- [ ] Run quality benchmarks (GSM8K 200Q, HumanEval) for N=2
- [x] Select best N for further optimization → **N=2**

### Phase 2: Reduce per-step overhead
- [x] Profile current bottlenecks with timing instrumentation
  - **Result**: 80% of step time is verify+sample (GPU compute + sync), NOT scheduler overhead
- [x] Fused single GPU→CPU sync (verify + sample in one .tolist())
  - **Result**: ~1.5% improvement at conc=64, neutral at conc=32
- [x] Tried pre-allocating batch tensors in prepare_for_dllm_decode
  - **Result**: Regression, reverted. Scheduler prep is only 3% of step time
- [x] Tried lazy correction (only compute multinomial for rejected)
  - **Result**: Extra sync cost offsets savings from smaller multinomial
- [x] Skip mask check in decode loop (use extend_lens heuristic)
  - **Result**: Saves ~0.1ms, minor improvement

### Phase 3: Fused verification kernel (CRITICAL for conc=32)
- [ ] **Design fused verify+sample Triton kernel**
  - The 9.27ms verify+sample phase has 5+ GPU kernel launches
  - Fusing softmax + p/q gather + ratio + accept/reject into one kernel could save 2-3ms
  - Fusing correction multinomial with the acceptance into a single kernel
- [ ] **Eliminate torch.stack for draft probs**
  - Pre-allocate [max_bs, num_masks, vocab] buffer
  - Write draft probs directly during sampling, read during verify
- [ ] Benchmark: target verify+sample < 5ms (from 9.27ms)

### Phase 4: Overlap scheduling
- [ ] **Overlap verify+sample (CPU-side) with next GPU forward**
  - Current pipeline: prep → forward → verify+sample → prep → forward
  - Key insight: verify+sample for step N can overlap with forward of step N+1
  - Requires double-buffered batch and careful state management
- [ ] This could effectively hide the 9ms verify+sample behind the forward

### Phase 5: Quality validation for N=2
- [ ] Run GSM8K 200Q with --max-tokens 8192
- [ ] Run HumanEval, MBPP
- [ ] Verify quality >= thresholds (GSM8K >= 90%, HumanEval >= 85%)

### Phase 6: Final benchmark sweep
- [ ] tore-speed-eval at concurrency=[1, 4, 8, 16, 32, 48, 64] for N=2
- [ ] Compare with EAGLE3 at all concurrency levels
- [ ] Profile final version

## Code Changes Made
1. `python/sglang/srt/dllm/algorithm/dreamshift_blockN.py`:
   - Fused verify + sample into single GPU→CPU sync
   - Skip mask check when extend_lens indicates decode requests
   - Added optional timing instrumentation (disabled by default)
   - `import time` added

## Next Steps
1. **Run quality benchmarks for N=2** (GSM8K 200Q, HumanEval) to validate quality
2. **Phase 3: Fused Triton kernel** for verify+sample — most impactful optimization
3. **Phase 4: Overlap scheduling** — hide verify+sample behind GPU forward

## Key Insight
The conc=32 gap vs EAGLE3 (~37%) is fundamentally because:
- EAGLE3 processes 1 input token per forward (AR-like) with draft verification
- DreamShiftBlockN N=2 processes 3 input tokens per forward
- At conc=32, the compute difference matters (3x more input tokens)
- At conc=64, the GPU is saturated either way, so the gap narrows to ~2%

To close the conc=32 gap, we need either:
1. Reduce verify+sample overhead significantly (currently 80% of step time)
2. Use overlap scheduling to hide this overhead behind GPU forward
3. Or accept that N=2 is optimal at conc>=48 and N=3 at conc=32

## Progress Log
### Iteration 1 (2026-03-15)
- Completed Phase 1: N sweep for N=2,3,4,5 at conc=1,32,64
- Found N=2 is optimal at high concurrency (5350 tok/s at conc=64)
- Profiled N=2 at conc=32: verify+sample is 80% of step time (9.27ms)
- Implemented fused single-sync optimization: +1.5% at conc=64 (5451 tok/s)
- Tried and reverted: pre-allocated batch tensors (regression), lazy correction (neutral)
- N=2 at conc=64 now within 2.1% of EAGLE3 (5451 vs 5569)
