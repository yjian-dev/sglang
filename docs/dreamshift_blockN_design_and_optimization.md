# DreamShiftBlockN: Design, Optimization, and Performance

## 1. Algorithm Overview

DreamShiftBlockN is a **speculative self-drafting** decoding algorithm for SDAR (Speculative Diffusion Auto-Regressive) models. Unlike EAGLE3 which uses a separate small draft model, DreamShiftBlockN uses the **same model** for both drafting and verification, producing 1 to N tokens per forward pass.

### Core Idea

Each forward pass processes `block_size = 2N - 1` input tokens and produces up to `N` output tokens through speculative verification:

```
N=5 (block_size=9):

Cold Start:  [t0, M, M, M, M, M, M, M, M]  ->  t0(clean) + s1,s2,s3,s4(spec)
Verify:      [t0, s1, s2, s3, s4, M, M, M, M]  ->  verify s1..s4, sample new specs
```

Where `M` = mask token (model's special diffusion token), `t0` = pending clean token, `s_i` = speculative token.

![Algorithm Pipeline](fig_algorithm_pipeline.png)

### Two Operating Modes

**Verify Round** (steady state, ~78% of rounds for N=5):
1. Input `[pending, spec_0, ..., spec_{N-2}, M, M, ..., M]`
2. Model produces logits for all positions
3. Verify each spec token: accept if `p(x)/q(x) >= 1` or with probability `p(x)/q(x)`
4. On all-accept: output `spec_0...spec_{N-2}, clean_token` = **N tokens**
5. On reject at position i: output `spec_0...spec_{i-1}, corrected_token` and fall back to cold start

**Cold Start** (after rejection or first step):
1. Input `[t0, M, M, ..., M]`
2. Sample 1 clean token + (N-1) speculative tokens for next verify round
3. Output: 1-2 tokens

### Key Metric: Tokens Per Forward (TPF)

TPF measures average output tokens per model forward pass:

| Config | Accept Rate | TPF | Quality (GSM8K) |
|--------|------------|-----|-----------------|
| N=3 sample | 78% | 2.37 | 95.0% |
| N=4 sample | 59% | 2.58 | 96.0% |
| N=5 sample | 41% | 2.70 | 95.5% |

Higher N = higher TPF but lower accept rate. The sweet spot depends on the forward latency regime.

## 2. Architecture in SGLang

### Key Files

```
python/sglang/srt/dllm/algorithm/
  dreamshift_blockN.py          # Core algorithm: classify, verify, sample, trim
  fused_verify_kernel.py        # Triton kernel for fused spec verification

python/sglang/srt/managers/
  scheduler.py                  # _dllm_decode_loop(): fast inner decode loop
  schedule_batch.py             # prepare_for_dllm_decode(): lightweight batch prep
  scheduler_output_processor_mixin.py  # process_batch_result_dllm(): output handling

python/sglang/srt/layers/attention/
  flashinfer_backend.py         # Attention metadata for DLLM extend mode
```

### Decode Loop Structure

```python
# Simplified _dllm_decode_loop
while True:
    # 1. Scheduling (recv, filter, absorb)     ~0.01ms
    recv_requests()
    filter_batch()  # remove finished

    # 2. Batch preparation                     ~0.24ms
    batch.prepare_for_dllm_decode()  # KV alloc, rebuild tensors

    # 3. Algorithm run                          ~5.2ms
    #    - classify: read prev state, set input_ids     0.1ms
    #    - metadata: flashinfer page table rebuild       0.43ms
    #    - CUDA graph replay: GPU forward                3.5ms (GPU)
    #    - verify_sample: check specs, sample new        0.7ms
    #    - trim_assemble: KV trim, build output          0.3ms
    result = algorithm.run(model_runner, forward_batch)

    # 4. Output processing                     ~0.2ms
    process_batch_result(batch, result)  # KV free, output_ids, stream
```

### Attention Mode: DLLM Extend

Unlike Qwen-AR which uses **decode mode** (1 token per request, cheap metadata), DreamShift uses **extend mode** (block_size tokens per request). This is necessary because the block_size new tokens need **causal attention** among themselves.

The extend mode metadata setup costs ~0.43ms/step (vs ~0.07ms for decode mode), which is a significant overhead at TP=4 where GPU forward is only ~3.5ms.

## 3. Optimizations Implemented

### 3.1 Argmax-for-Specs

**Problem**: Speculative tokens are sampled with full top_k/top_p, producing diverse but often-rejected guesses.

**Solution**: Use argmax for speculative positions (clean position still uses proper sampling). Speculative tokens are just guesses verified next round anyway.

```python
# Before: expensive flashinfer sampling for ALL positions
all_sample_ids_gpu, _ = _batched_sample(
    all_sample_logits, temperature, top_k, top_p  # runs on 4+ positions
)

# After: sample clean position properly, argmax for specs
all_sample_ids_gpu, _ = _batched_sample(...)
# Overwrite spec positions with argmax (cheap, verified next round)
draft_logits = all_sample_logits[draft_indices]
all_sample_ids_gpu[draft_indices] = draft_logits.argmax(dim=-1)
```

**Impact**: Accept rate 68% -> 78% (N=3), TPF 2.17 -> 2.37. **+10% throughput.**

### 3.2 Fused Triton Verify Kernel

**Problem**: Standard verify requires: softmax + gather p(x) + gather q(x) + ratio + rand + accept/reject + correction sampling = 7+ GPU kernel launches.

**Solution**: Single Triton kernel with online softmax + Gumbel-max correction:

```python
@triton.jit
def _fused_verify_kernel(clean_logits, draft_probs, spec_vals, ...):
    # Pass 1: Online softmax of clean logits
    for block in range(0, V, BLOCK_V):
        # Streaming max + exp-sum (no materializing [nv, V] probs)
        ...

    # Gather p(spec) and q(spec) — just 2 loads
    p_spec = exp(spec_logit - max) / sum
    q_spec = load(draft_probs[spec_val])

    # Accept/reject
    if p_spec / (q_spec * alpha) >= rand:
        return ACCEPT  # Early exit: skip Pass 2

    # Pass 2 (rejected only): Gumbel-max correction sampling
    # Sample from max(0, p-q) using argmax(log(max(0,p-q)) + Gumbel)
    for block in range(0, V, BLOCK_V):
        ...
```

**Impact**: Eliminates 4+ kernel launches, ~0.15ms savings. Accept path (~78%) skips the expensive Pass 2.

### 3.3 Prep Fast Path

**Problem**: `prepare_for_dllm_decode()` calls `req.init_next_round_input()` which does Python list manipulation + unnecessary bookkeeping.

**Solution**: For pure-decode batches (no new/finished requests), inline minimal state update:

```python
if _pure_decode:
    # Fast path: skip init_next_round_input entirely
    advance = req.dllm_next_advance or block_size
    req.dllm_next_advance = None
    req.dllm_block_offset += advance
    req.dllm_ids += [mask_id] * block_size
    prefix_len = req.kv_committed_len
    req.extend_input_len = block_size
```

### 3.4 Buffer Reuse

**Problem**: Every step creates new GPU tensors for input_ids, kv_indices, etc.

**Solution**: Cache and reuse:

```python
# input_ids: fill with mask_id (algorithm overwrites specifics)
if _pure_decode and self._dllm_input_ids_buf is not None:
    self._dllm_input_ids_buf.fill_(mask_id)
    self.input_ids = self._dllm_input_ids_buf

# req_pool_indices: same requests each step
if _pure_decode and self._dllm_rpx_cache is not None:
    self.req_pool_indices = self._dllm_rpx_cache[1]

# kv_indices: pre-allocated buffer
if self._kv_indices_buf.shape[0] >= needed:
    kv_indices = self._kv_indices_buf  # no torch.empty()
```

### 3.5 Greedy Argmax Shortcut

**Problem**: For greedy (temp=0) mode, saving full logits [vocab_size] per request for next step's sampling is wasteful.

**Solution**: Store argmax token id (1 int) instead of full logits (600KB tensor):

```python
if self.temperature <= 0:
    # Just store the argmax token, skip clone
    _all_argmax = full_logits[_idx_t].argmax(dim=-1).tolist()
    for k, bid in enumerate(save_bids):
        _saved_argmax[bid] = _all_argmax[k]
else:
    # Sampling needs full logits for next round
    _all_saved = full_logits[_idx_t].detach().clone()
```

## 4. TP=4 Step Timeline Analysis

![Step Timeline](fig_step_timeline.png)

### Why Qwen-AR Scales Better at TP=4

| | Qwen-AR | DreamShift |
|---|---------|-----------|
| Attention mode | Decode (1 token) | Extend (block_size tokens) |
| Metadata cost | ~0.07ms | ~0.43ms |
| CPU overhead | ~0.2ms (hidden by overlap) | ~1.7ms (serial) |
| Overlap scheduler | Yes (CPU hidden during GPU) | No (strict dependency chain) |
| Step time | ~3.1ms (≈ GPU forward) | ~5.6ms (GPU + serial CPU) |
| TP=1→4 scaling | 2.15x | 1.5x |

### Dependency Chain (Why Overlap is Hard)

```
forward(N) →  logits(N)
                ↓
           verify(N) →  accept/reject decisions
                ↓
           trim(N)   →  KV trim indices, internal state
                ↓
           process(N) → free KV, update kv_committed_len
                ↓
           prep(N+1)  → allocate KV, rebuild tensors
                ↓
           classify(N+1) → set input_ids from verify(N) state
                ↓
           metadata(N+1) → flashinfer page tables
                ↓
           forward(N+1)
```

Every step depends on the PREVIOUS step's results. No CPU work can run during GPU forward.

## 5. Performance Results

### TP=1 (1x H100, AIME, burst mode, bf16)

![TP=1 Comparison](fig_tp1_comparison.png)

| Conc | Qwen-AR | EAGLE3 | N=3 sample | N=3 greedy | N=5 fp8 |
|------|---------|--------|-----------|-----------|---------|
| 1 | 149 | 204 | 247 | 272 | **328** |
| 2 | 290 | 378 | 464 | 521 | **594** |
| 4 | 565 | 699 | 846 | 967 | **1,072** |
| 8 | 1,086 | 1,269 | 1,550 | **1,770** | 1,605 |
| 16 | 2,017 | 2,160 | 2,675 | **2,999** | 2,380 |
| 32 | 3,613 | 3,414 | 4,209 | **4,736** | 3,745 |
| 64 | 5,794 | 4,676 | 5,708 | **6,414** | 4,862 |

**Key findings at TP=1:**
- DreamShift beats Qwen-AR at ALL concurrency levels
- N=3 greedy is fastest at C>=8 (6,414 tok/s at C=64)
- N=5 fp8 is fastest per-request at C<=4 (328 tok/s at C=1)
- EAGLE3 (Tengyunw) is slowest at high concurrency

### TP=4 (4x H100, bf16, sampling verify)

![TP=4 Comparison](fig_tp4_comparison.png)

| Conc | N=5 sample | N=3 sample | Qwen-AR | N=5/Qwen | N=3/Qwen |
|------|-----------|-----------|---------|----------|----------|
| 1 | **453** | 395 | 319 | **1.42x** | 1.24x |
| 2 | **824** | 739 | 605 | **1.36x** | 1.22x |
| 4 | **1,472** | 1,352 | 1,178 | **1.25x** | 1.15x |
| 8 | **2,602** | 2,418 | 2,270 | **1.15x** | 1.07x |
| 16 | 4,253 | 4,104 | 4,195 | 1.01x | 0.98x |
| 32 | 6,417 | 6,507 | **7,419** | 0.86x | 0.88x |
| 48 | 7,746 | 8,328 | **10,135** | 0.76x | 0.82x |
| 64 | 8,172 | 9,166 | **11,668** | 0.70x | 0.79x |

**Key findings at TP=4:**
- DreamShift N=5 is **42% faster** than Qwen-AR at C=1 (453 vs 319 tok/s)
- DreamShift wins at C<=8, Qwen-AR overtakes at C>=16
- N=5 is better at low concurrency (higher TPF), N=3 is better at high concurrency (smaller block_size)
- The crossover at C=16 is due to Qwen-AR's overlap scheduler hiding CPU overhead

### Optimization Progression (TP=4, C=1, sampling verify)

![Optimization Progression](fig_optimization_progression.png)

| Version | C=1 tok/s | Change | Key Optimization |
|---------|-----------|--------|-----------------|
| Baseline (N=3) | 363 | — | — |
| + argmax-for-specs | 420 | +16% | Higher accept rate → higher TPF |
| + N=4 | 438 | +4% | More tokens per forward |
| + N=5 (final) | 453 | +3% | Even higher TPF at memory-bound regime |
| **Total** | **453** | **+25%** | |

### Quality (GSM8K, 200 problems, max_tokens=8192)

| Config | GSM8K | IFEval | HumanEval | MBPP |
|--------|-------|--------|-----------|------|
| N=3 bf16 | 95.0% | 84.1% | 92.7% | 93.8% |
| N=5 bf16 | 95.5% | — | — | — |
| N=3 FP8 | 95.5% | 89.2% | 89.6% | 93.4% |
| Threshold | >=90% | >=80% | >=85% | >=80% |

All quality metrics pass thresholds with comfortable margins.

## 6. Profiling Breakdown (TP=4, bs=1)

### Step Time Breakdown

| Component | Time (us) | % | Notes |
|-----------|-----------|---|-------|
| **Total step** | **5,627** | 100% | |
| Scheduler (recv, filter, absorb) | 7 | 0.1% | Mostly hidden by overlap |
| prepare_for_dllm_decode | 235 | 4.2% | KV alloc, tensor rebuild |
| get_model_worker_batch | 29 | 0.5% | Python dict construction |
| **forward_call** | **5,190** | 92.2% | Contains all below: |
|  flashinfer metadata | 430 | 7.6% | Page table rebuild (extend mode) |
|  CUDA graph replay | ~3,500 | 62.2% | Actual GPU computation |
|  classify | 90 | 1.6% | Read prev state, set input_ids |
|  verify_sample | 700 | 12.4% | Triton verify + sampling |
|  trim_assemble | 270 | 4.8% | KV trim, output assembly |
| process_batch_result | 166 | 2.9% | KV free, output, stream |

### The Bottleneck: flashinfer Extend Metadata (430us/step)

Qwen-AR decode mode metadata: **74us** (6x cheaper).

DLLM uses extend mode because block_size tokens need causal attention. The metadata rebuild includes:
1. `torch.cumsum` for kv_indptr — 20us
2. Triton kernel for kv_indices — 50us
3. `torch.cumsum` for qo_indptr — 20us
4. `wrapper.begin_forward` (flashinfer plan) — 140us
5. Python overhead — 200us

Our inlined fast path saves ~100us of Python overhead by skipping the `update_single_wrapper` -> `call_begin_forward` call chain.

## 7. Remaining Bottlenecks and Future Work

### Short-term (< 1 week)
1. **Inline flashinfer metadata further**: Pre-compute `qo_indptr` (constant for uniform block_size), cache `kv_indptr` pattern
2. **CUDA stream overlap for verify+sample**: Launch verify/sample kernels on a separate stream concurrent with next step's metadata update
3. **Reduce `trim_assemble` Python overhead**: Pre-allocate output lists, vectorize KV trim index computation

### Medium-term (1-2 weeks)
4. **Speculative metadata pre-computation**: Predict seq_lens(N+1) during GPU forward(N) overlap window (correct 78% of time), pre-run flashinfer `plan()`. On misprediction, recompute (~22% penalty).
5. **Hybrid decode+extend attention**: Split attention into decode-like (prefix KV, 74us metadata) + local causal (7x7, cheap). Would reduce metadata from 430us to ~180us.

### Long-term (fundamental changes)
6. **Enable DLLM overlap scheduler**: Currently disabled (`server_args.py:2856-2860`). Requires breaking the strict dependency chain, possibly via speculative scheduling.
7. **Custom flashinfer wrapper for DLLM**: Incremental metadata update (only new KV slots change each step) instead of full rebuild.

### Theoretical Limits

At TP=4 bf16 with N=5:
- GPU forward: ~3.5ms (hardware limit)
- TPF: 2.70
- **Theoretical max**: TPF / GPU_forward = 2.70 / 3.5ms = **771 tok/s**
- Current: 453 tok/s = **59% of theoretical**
- Gap: 1.9ms CPU overhead that could be hidden with overlap scheduling
