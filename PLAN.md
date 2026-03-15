# Plan

## Status
Starting — Phase 1 ready

## Goal
Maximize sampling verify throughput to match/exceed EAGLE3 (~5100 tok/s at conc=32) while maintaining quality (GSM8K >= 90%, HumanEval >= 85%).

## Current Baseline (sampling verify, temp=1.0)

### Throughput (tore-speed-eval, N=3)
| concurrency | Current | EAGLE3 | Gap |
|---|---|---|---|
| 1 | 288 | 228 | +26% (ahead) |
| 32 | 3,568 | 5,103 | **-30%** |
| 64 | ~5,000 | 5,569 | -10% |

### Quality (N=3, sampling verify, 8 GPU)
| Benchmark | Score |
|---|---|
| GSM8K (1319Q) | 94.7% |
| MATH-500 | 88.4% |
| IFEval strict | 85.95% |
| HumanEval | 91.5% |
| MBPP (16k) | 93.0% |

### Profiling (conc=32, per step)
| Component | ms/step | % |
|---|---|---|
| GPU forward (CUDA graph) | 2.2 | 10% |
| tensor.tolist() | 7.8 | 36% |
| aten::copy_/to | 9.1 | 42% |
| recv_requests | 2.6 | 12% |
| softmax (correction) | 0.2 | 1% |
| **Total** | **21.8** | **100%** |

## Tasks

### Phase 1: Find optimal N for sampling verify
- [ ] Run tore-speed-eval for N=2, 3, 4, 5 at concurrency=1, 32, 64
- [ ] Compare TPF, accept rate, and throughput for each N
- [ ] Run quality benchmarks (GSM8K 200Q, HumanEval) for top 2 candidates
- [ ] Select best N for further optimization

### Phase 2: Reduce per-step CPU overhead
- [ ] **Profile current bottlenecks** with torch profiler at conc=32
- [ ] **Reduce tolist() calls** (currently 5/step, 7.8ms)
  - Combine multiple GPU tensors into single tolist()
  - Keep more data GPU-side, defer CPU sync
  - Pre-compute reusable indices
- [ ] **Reduce aten::copy_ ops** (currently 120/step, 9.1ms)
  - Pre-allocate input_ids buffer instead of rebuilding each step
  - Avoid tensor creation in prepare_for_dllm_decode hot path
  - Use in-place operations where possible
- [ ] **Reduce recv_requests overhead** (2.6ms/step)
  - Batch receive: collect all pending messages in one call
  - Adaptive polling frequency based on batch state

### Phase 3: Fused verification kernel
- [ ] **Design fused verify+sample kernel**
  - Input: logits [bs*vns, vocab], spec_token_ids [bs*vns], draft_probs [bs*vns, vocab]
  - Operations: softmax → gather p(x),q(x) → ratio → accept/reject → correction sample
  - Output: accepted [bs*vns], corrected_tokens [bs*vns]
  - Single kernel launch instead of 5+ separate ops
- [ ] **Implement using sglang JIT kernel infrastructure**
  - Use `add-jit-kernel` skill for lightweight Triton kernel
  - Or `add-sgl-kernel` for AOT CUDA kernel
- [ ] Benchmark: target 50%+ reduction in verify step time

### Phase 4: Overlap scheduling
- [ ] **Overlap CPU post-processing with GPU forward**
  - Current: forward → wait → verify+sample → prepare → forward
  - Target: forward(N) → verify+sample(N-1) overlapped with prepare(N) → forward(N+1)
  - Use double-buffering: two batch objects, alternate between them
- [ ] **Pipeline correction sampling**
  - Start next forward before correction finishes (corrections only affect rejected tokens)
  - Requires careful state management

### Phase 5: Benchmark sweep and quality validation
- [ ] tore-speed-eval at concurrency=[1, 4, 8, 16, 32, 48, 64] for best config
- [ ] Full quality suite: GSM8K 200Q, HumanEval, MBPP, IFEval, MATH-500
- [ ] Compare with EAGLE3 at all concurrency levels
- [ ] Profile final version to confirm overhead reduction

## Test Commands

### Launch servers
```bash
bash scripts/killall_sglang.sh && sleep 3
CONFIG=dreamshift_blockN3_config.yaml bash scripts/launch_blockN_8gpu.sh
```

### Quick throughput test (single server)
```bash
tore-speed-eval --provider sglang --base_url http://localhost:30000/v1 \
  --model_name default --tokenizer_name Qwen/Qwen3-8B --traffic_pattern concurrent \
  --concurrency 32 --num_examples 100 --dataset_type synthetic \
  --synthetic_input_length 256 --synthetic_output_length 1024 --max_tokens 2048
```

### Quality test (8 GPU)
```bash
PORTS="30000 30001 30002 30003 30004 30005 30006 30007"
python scripts/eval_gsm8k.py --ports $PORTS --num-problems 200 --max-tokens 8192
```

### Sanity check
```bash
python scripts/stream_demo.py --url http://localhost:30000 --prompt 'What is 15*23+7?' --max-tokens 256
```

## Next Steps
1. Start Phase 1: run tore-speed-eval for N=2,3,4,5 at conc=1,32,64
2. Compare results and pick best N
3. Profile the best N config to identify specific optimization targets

## Progress Log
(To be filled by agent)
