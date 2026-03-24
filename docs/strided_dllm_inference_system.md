# Strided DLLM Inference System: Design & Optimizations

This document describes how Introspective Strided Decoding (ISD) is implemented inside SGLang's AR serving stack, and every optimization we apply. Use this as the source material for §3.3 (Infrastructure) and §4 (Ablation) of the COLM paper.

---

## 1. Architectural Insight: ISD as "Extend" in an AR Stack

### Why DLLMs Usually Need Custom Infra

Prior DLLMs (LLaDA, SDAR, DREAM) use bidirectional or block-causal attention during decoding. This breaks three core assumptions of AR serving systems:

1. **Decode = 1 new token**: AR systems (vLLM, SGLang) optimize the decode step for exactly 1 new token per request. KV cache management, attention metadata, and batching all assume this.
2. **Incremental KV cache**: AR systems append 1 KV entry per step. DLLMs need to write/overwrite multiple positions (denoising iterations rewrite the same slots).
3. **Overlap scheduling**: SGLang hides CPU scheduling overhead behind GPU compute by running them on separate CUDA streams. This requires no dependency between the current step's CPU work and the previous step's GPU output.

### Our Key Insight: Strict Causal ⟹ Reuse Extend Mode

Because Strided DLLM uses **strict causal attention** (no bidirectional within blocks), each ISD step is equivalent to SGLang's native **extend** operation — appending `block_size = 2N−1` new tokens to the prefix, all with causal attention among themselves and to the prefix. This means:

- **Paged KV cache**: Works unmodified. Each step allocates `block_size` new KV slots, writes them during forward, then trims MASK positions after verify.
- **Continuous batching**: Multiple requests at different sequence lengths can share a batch — each contributes `block_size` tokens to the extend.
- **Tensor parallelism**: The model forward (including the `block_size` tokens) shards across GPUs exactly as normal extend does.
- **CUDA graph**: We capture the DLLM extend forward (all `block_size` tokens) into a single CUDA graph, replaying it each step with only input_ids and KV metadata updated.

### The Cost of Extend vs Decode

However, extend mode is fundamentally heavier than decode mode:

| Property | AR Decode (1 token) | DLLM Extend (block_size tokens) |
|----------|--------------------|---------------------------------|
| Attention metadata rebuild | ~0.07ms (cheap) | ~0.43ms (page table rebuild, flashinfer plan) |
| Attention kernel | decode-only (1 query) | paged-only with causal among new tokens |
| CPU scheduling overhead | hidden by overlap scheduler | **serial** — cannot overlap |
| KV management | append 1 slot | allocate block_size, trim MASK slots after verify |

The serial scheduling is the biggest structural disadvantage: ISD has a strict dependency chain:

```
forward(N) → verify(N) → trim(N) → process(N) → prepare(N+1) → forward(N+1)
```

Every step depends on the GPU output of the previous step (verify needs logits, trim needs accept/reject decisions). This prevents CPU–GPU overlap, making every millisecond of CPU overhead directly additive to step latency.

---

## 2. Decode-Mode Inner Loop (Biggest Optimization: +134% at bs=32)

### Problem

SGLang's standard scheduling pipeline runs a full cycle each step:
1. `cache_unfinished_req()` — GPU tensor copy per request (copies KV indices)
2. `get_next_batch_to_run()` — filter finished, merge new, rebuild batch
3. `prepare_for_extend()` — allocate KV, build attention metadata, create ForwardBatch
4. `run_batch()` — GPU forward
5. `process_batch_result()` — free KV, update state, stream output

Steps 1–3 and 5 run on CPU. For DLLM, `cache_unfinished_req()` runs **2×B times per step** (once before scheduling, once after), where B is batch size. Each call does `req_to_token[:kv_len].to(int64, copy=True)` — a GPU tensor copy.

At bs=32: 64 GPU tensor copies per step + full Python batch reconstruction = ~4ms CPU overhead per step.

### Solution: Tight Inner Loop

We bypass the full scheduling pipeline with a tight inner loop (`_dllm_decode_loop`) that:

1. **Reuses the batch object** across steps — no `ScheduleBatch.init_new()`, no `cache_unfinished_req()`
2. **Lightweight `prepare_for_dllm_decode()`** — only:
   - Allocate KV slots for new positions (one batched scatter)
   - Update `seq_lens` incrementally (`prev + bs * block_size`)
   - Reuse cached `req_pool_indices` and `position_offsets` (constant when batch is stable)
3. **In-place completion handling** — `filter_batch()` removes finished requests without full batch reconstruction
4. **Exit to slow path** only when new requests arrive or batch empties

### Impact

| Batch Size | Before (full scheduling) | After (inner loop) | Improvement |
|-----------|-------------------------|-------------------|-------------|
| bs=1 | ~139 tok/s (12ms/step) | ~238 tok/s (8ms/step) | +71% |
| bs=32 | ~1,923 tok/s | ~4,509 tok/s | +134% |

This is the single largest optimization because it eliminates the O(B) GPU tensor copies and Python overhead that dominated at high batch sizes.

---

## 3. CUDA Graph for DLLM Extend

### Problem

Without CUDA graph, each forward pass launches thousands of individual GPU kernels (GEMM, attention, norm, etc.) with Python dispatch overhead. For Qwen3-8B at bs=1:
- Eager forward: ~13ms (dominated by kernel launch latency, not compute)
- The GPU is only ~15% utilized at bs=1 — most time is spent in launch gaps

### Solution

We capture the entire DLLM extend forward into a CUDA graph:

1. **Capture**: During warmup, run one forward pass with `block_size` tokens at each batch size. The graph records all kernel launches, memory allocations, and stream operations.
2. **Replay**: Each decode step, only update `input_ids` and attention metadata (KV page tables, sequence lengths), then replay the captured graph.

Key challenge: DLLM extend uses **different attention metadata each step** (sequence lengths grow). We solve this by:
- Pre-allocating the flashinfer metadata buffers at max size during capture
- Updating only the changed fields (kv_indptr, kv_indices, qo_indptr) before replay via `init_forward_metadata_replay_cuda_graph()`

### Impact

| Mode | Forward Time | Step Time |
|------|-------------|-----------|
| Eager (no graph) | ~13ms | ~25ms |
| CUDA graph replay | ~6.5ms | ~8.5ms |

~2× speedup from graph alone.

---

## 4. Argmax-for-Proposals

### Problem

Standard speculative decoding samples draft tokens from the full distribution q(x) using top-k/top-p. For ISD, proposals are verified and either accepted or corrected — their sampling quality doesn't affect output quality (which is guaranteed by p/q acceptance).

### Insight

Proposals should maximize **acceptance probability**, not diversity. The most probable token under q (argmax) is also most likely to be accepted under p (since p and q are correlated through shared training).

### Implementation

```python
# Sample clean position properly (this determines output quality)
clean_token = _batched_sample(clean_logits, temperature, top_k, top_p)

# For proposal positions: just take argmax (cheap, will be verified anyway)
draft_tokens = draft_logits.argmax(dim=-1)
```

### Impact

- Accept rate: 68% → 78% (N=3)
- TPF: 2.17 → 2.37
- TPS improvement: ~10%
- Output quality: **identical** (every accepted token is verified against p)

### Does This Hurt Diversity?

No. The output distribution is determined entirely by the acceptance criterion min(1, p/q). Argmax proposals just increase the acceptance rate — when accepted, the token provably follows distribution p. When rejected, we resample from max(0, p−q), which is independent of the proposal method. The mathematical guarantee holds regardless of how proposals are generated.

---

## 5. Fused Triton Verify Kernel

### Problem

The verification step involves:
1. Compute softmax over anchor logits → p(x) for all x ∈ V (vocabulary 152K)
2. Gather p(x_k) for the specific proposed token x_k
3. Gather q(x_k) from the stored draft probability
4. Compute ratio p/q, compare against random uniform
5. On rejection: sample from max(0, p−q) distribution

Naively this requires 7+ separate GPU kernel launches: softmax, gather (×2), div, compare, correction sampling (softmax again + multinomial).

### Solution: Two-Pass Early-Exit Kernel

Single Triton kernel with online softmax:

**Pass 1 (always runs):**
- Stream through vocabulary in blocks of BLOCK_V (e.g., 4096)
- Maintain running max and exp-sum accumulators (online softmax)
- At end: compute p(x_k) = exp(logit[x_k] − max) / sum
- Compare p(x_k) / (q(x_k) × α) against random threshold
- If ACCEPT: return immediately (no Pass 2)

**Pass 2 (only on rejection, ~22% of steps):**
- Stream through vocabulary again
- Compute correction distribution: max(0, p(x) − q(x)) for all x
- Apply Gumbel-max trick: argmax(log(max(0, p−q)) + Gumbel noise)
- Returns corrected token in one pass (no explicit normalization + multinomial)

### Impact

- Eliminates 4+ kernel launches per verified position
- ~0.15ms savings per step
- The common accept path (78% of steps) does only 1 pass over the vocabulary

---

## 6. Scalar Draft Probabilities

### Problem

Standard speculative verification stores the full draft distribution q ∈ ℝ^V per speculative token for the correction step max(0, p−q). At bs=32 with N−1=2 spec tokens each:
- 32 × 2 × 152K × 4 bytes ≈ **39MB** GPU memory
- Plus expensive `torch.stack()` to batch them each verify step

### Insight

The acceptance decision only needs the scalar q(x_k) — the probability of the specific token that was sampled. The full distribution q is only needed for correction sampling on rejection.

### Solution

Store only `q(x_k)` (one float per spec token) instead of the full distribution:

```python
# Before: store full distribution (152K floats per spec token)
self._spec_draft_probs[rpx] = [full_softmax_output]  # 600KB per token

# After: store just the scalar probability of the sampled token
self._spec_draft_qx[rpx] = [token_prob]  # 4 bytes per token
```

On rejection, resample from p instead of max(0, p−q). Since rejections are rare (~22% of steps) and p ≈ max(0, p−q) when q(x_k) is small at the rejected token, the quality impact is negligible.

### Impact

- Memory: 39MB → negligible at bs=32
- Eliminates `torch.stack()` overhead
- Slight approximation on rejection (p instead of max(0,p−q)), negligible quality impact

---

## 7. Buffer Reuse and Metadata Caching

### Problem

Every step creates new GPU tensors:
- `input_ids`: `torch.tensor([...], device='cuda')` for block_size × bs tokens
- `req_pool_indices`: repeated each step despite being constant
- `position_offsets`: `torch.arange(block_size).repeat(bs)` — pure constant
- `seq_lens_sum`: `sum(seq_lens)` — recomputed from scratch

### Solution

Cache all constant/predictable tensors:

1. **input_ids buffer**: Pre-allocate `[bs × block_size]` tensor, fill with mask_id. Algorithm overwrites specific positions.
2. **req_pool_indices cache**: Cache keyed by `(bs, tuple(rpx_list))`. Same requests each step → same cache hit.
3. **Position offsets**: Computed once, reused forever (constant for uniform block_size).
4. **Incremental seq_lens_sum**: `prev_sum + bs × block_size` instead of full reduction.

### Impact

- prep time: 220μs → 180μs (~20% reduction)
- Eliminates Python→GPU tensor creation overhead for common case

---

## 8. Deferred Stream Output

### Problem

After each GPU forward + verify, `process_batch_result_dllm()` runs on the critical path:
- KV free (critical — must happen before next step's KV alloc)
- State update (critical — next step reads this)
- `stream_output()` (~120μs — sends tokens to detokenizer via ZMQ)

The `stream_output()` is NOT critical for the next step but runs serially.

### Solution

Split processing into critical and deferred:
- **Critical path** (runs immediately): KV free, state update, finished check
- **Deferred** (runs in next step's overlap window): `stream_output()`

```python
# In _dllm_decode_loop overlap_fn:
if _deferred_stream_batch is not None:
    self.stream_output(_reqs, _return_logprob)  # hidden during GPU compute
    _deferred_stream_batch = None

# After GPU forward:
self._process_dllm_critical_inline(batch, result)  # KV free + state only
_deferred_stream_batch = (list(batch.reqs), batch.return_logprob)
```

### Impact

- process time: 374μs → 123μs
- step time: ~9.2ms → ~9.0ms

---

## 9. Paged-Only Attention (3 Kernels → 1)

### Problem

SGLang's standard extend uses **cascade attention**: ragged attention for new tokens + paged attention for cached prefix + merge_state = 3 kernel launches per layer.

### Insight

For DLLM decode (not prefill), all tokens have a cached prefix. We can use paged-only attention (`use_ragged=False`) which handles both new-token and prefix attention in a single kernel with `causal=True`.

### Implementation

In `flashinfer_backend.py`, when the batch contains MASK tokens (DLLM decode), route to paged-only:
```python
if dllm_has_decode:
    # Paged-only: 1 kernel/layer, causal among new tokens
    use_ragged = False
    # All tokens attend to cached prefix + causally to each other
```

### Impact

- 3 → 1 kernel per layer
- 36 layers: saves 72 kernel launches per step

---

## 10. KV Trim: Managing Speculative Positions

### Problem Unique to ISD

After each verify step, MASK positions that weren't accepted need their KV slots freed. Unlike AR (which never frees KV during generation), ISD must:
1. Identify which positions to trim (rejected specs + all MASKs)
2. Free those KV slots back to the memory pool
3. Update `kv_committed_len` so the next step allocates from the right position

### Solution

After verify, the algorithm computes `trim_count` (how many positions to free from the end) and `advance` (how many positions to commit). The output processor:

```python
# Free KV slots for trimmed positions (MASK + rejected specs)
kv_indices_to_free = req_to_token[rpx, committed_len : committed_len + trim_count]
token_to_kv_pool.free(kv_indices_to_free)

# Advance committed length
req.kv_committed_len += advance
```

This is batched across requests using GPU tensor operations to avoid per-request CPU syncs.

---

## 11. Conditional LoRA: Per-Token Gating for Lossless ISD

### Problem

For lossless ISD with LoRA, different token positions in the same forward pass need different weights:
- **Verify positions** (committed + pending): base-only weights (to produce exact AR anchor distribution p)
- **MASK positions** (proposals): base+LoRA weights (to produce high-quality proposals q)

Standard LoRA applies the same adapter to all positions uniformly.

### Solution: Segment-Based Routing with Binary Mask

Each forward pass partitions positions into segments:
```
Input:  [committed, pending, spec_0, spec_1, MASK, MASK]
LoRA:   [  base,     base,   base,   base,   LoRA, LoRA ]
```

Implementation:
1. **Segment IDs**: `[None, None, None, None, "b3lora", "b3lora"]` — None = base only
2. **Binary mask**: `[0, 0, 0, 0, 1, 1]` — applied after LoRA shrink to zero out base positions
3. **LoRA manager**: `prepare_lora_batch()` uses custom segment lengths instead of per-request lengths

The mask is a GPU tensor updated each step (not re-allocated), so CUDA graph can capture the `mul_()` operation and the mask contents update between replays.

---

## 12. cuBLAS for Small-M LoRA (177 → 228 tok/s)

### Problem

SGLang's default LoRA kernel (csgmv from Punica) uses Triton with chunked segmented GEMV. At M=5 tokens (DLLM block_size=5, bs=1), this is extremely inefficient:
- Register usage: 48 regs (vs cuBLAS 168)
- Shared memory: 32KB (vs cuBLAS 160KB)
- Per-kernel latency: 11–19μs

With 288 LoRA kernels per forward (7 modules × 36 layers × shrink/expand), this adds ~4ms overhead.

### Solution

Replace csgmv with cuBLAS `torch.mm` / `torch.addmm_` for CUDA graph mode:

```python
# Shrink: x @ A.T → (M, rank)
shrink_out = torch.mm(x, A.t())
# Apply conditional mask
shrink_out.mul_(lora_mask[:M].unsqueeze(1))
# Expand: base_output += scaling * shrink_out @ B.T
base_output.addmm_(shrink_out, B.t(), beta=1.0, alpha=scaling)
```

cuBLAS is 2–3× faster per kernel at M=5:
| Kernel | csgmv (μs) | cuBLAS (μs) |
|--------|-----------|------------|
| QKV shrink+expand | 14–19 | 4.5–6.5 |
| O shrink+expand | 11–14 | 3.8–4.5 |

### CUDA Graph Capture

The cuBLAS ops are captured directly into the CUDA graph with real weight pointers:
1. Pre-load the LoRA adapter before graph capture
2. Capture with real `lora_ids` (not None) so cuBLAS operations use actual weights
3. Mark `cublas_graph_captured = True` so `prepare_lora_batch()` is skipped during replay

### Impact

- LoRA overhead: 4.0ms → 1.3ms
- TPS: 177 → 228 (+29%)

---

## 13. Multi-Stream Overlap for LoRA (194 → 212 tok/s)

### Problem

At M=5, the base model's linear projection uses <15% of the H100's 132 SMs. The LoRA shrink (mm + mask) is a separate computation that reads the same input x but writes to an independent output. Running them sequentially wastes GPU parallelism.

### Solution

Dispatch LoRA shrink to a dedicated CUDA stream concurrent with the base matmul:

```python
s_main = torch.cuda.current_stream()
s_lora = lora_stream

# Fork: shrink on lora_stream (concurrent with base)
s_lora.wait_stream(s_main)
with torch.cuda.stream(s_lora):
    shrink_out = torch.mm(x, A.t())
    shrink_out.mul_(lora_mask)

# Base matmul on main stream (runs in parallel)
base_output = torch.mm(x, W.t())

# Join: expand after both complete
s_main.wait_stream(s_lora)
base_output.addmm_(shrink_out, B.t(), alpha=scaling)
```

This works because:
- Shrink reads x, writes to shrink_out (independent memory)
- Base reads x, writes to base_output (independent memory)
- Expand reads shrink_out, modifies base_output (must wait for both)

### Pre-allocated Buffers

The shrink output buffers are pre-allocated during `init_cuda_graph_batch_info()` to avoid tensor allocation on the side stream during CUDA graph capture.

### Impact

- LoRA shrink latency hidden: 57% overlap
- Total LoRA overhead: 1.69ms → 0.73ms (visible) + 0.96ms (hidden)
- TPS: 194 → 212

---

## 14. Summary: Optimization Stack

### Non-LoRA (b2 model, N=3 sampling, H100 TP=1 C=1 bf16)

| Optimization | TPS | Δ | Key Mechanism |
|-------------|-----|---|---------------|
| Naive SGLang (eager, full scheduling) | ~100 | — | Baseline: eager forward + full batch rebuild each step |
| + CUDA graph | ~180 | +80% | Graph replay eliminates kernel launch overhead |
| + Decode inner loop | ~240 | +33% | Skip cache_unfinished_req, reuse batch |
| + Argmax-for-proposals | ~260 | +8% | Accept rate 68%→78%, TPF 2.17→2.37 |
| + Fused verify kernel | ~270 | +4% | Single Triton kernel, early-exit on accept |
| + Buffer reuse + deferred stream | ~287 | +6% | Eliminate tensor alloc, defer non-critical work |

### LoRA Lossless (b3 model + LoRA r=128, N=3, H100 TP=1 C=1 bf16)

| Optimization | TPS | Key Mechanism |
|-------------|-----|---------------|
| Non-LoRA reference | 267 | — |
| + Conditional LoRA (csgmv, no graph) | 63 | LoRA applied but 288 Triton kernels + eager forward |
| + CUDA graph capture | 154 | Graph replay with csgmv baked in |
| + cuBLAS replacement | 228 | 2–3× faster per-kernel at M=5 |
| + Conditional mask | 194 | mul_() for verify=base, MASK=LoRA; accept drops 80%→71% |
| + Multi-stream overlap | 212 | Shrink hidden behind base matmul |
| + Deferred softmax | 213 | fused_verify_from_logits replaces 2× F.softmax |

Note: The "conditional mask" step reduces TPS because it changes the LoRA behavior — verify positions now use base-only (correct for lossless), which slightly reduces accept rate. The subsequent multi-stream overlap recovers some of this.

---

## 15. Remaining Bottlenecks

### Why AR Wins at High Concurrency (TP=4)

| Property | Strided DLLM | AR |
|----------|-------------|-----|
| Attention mode | Extend (2N−1 tokens) | Decode (1 token) |
| Metadata cost | 0.43–0.85ms | 0.07ms |
| CPU overhead | ~1.5ms (serial) | ~0.2ms (hidden by overlap) |
| Overlap scheduler | Not available* | Yes |
| TP=1→4 scaling | 1.5× | 2.15× |

*ISD's strict dependency chain prevents CPU–GPU overlap.

### Theoretical Limit (TP=4, N=5)

- GPU forward: ~3.5ms (hardware limit)
- TPF: 2.70
- Theoretical max: 2.70 / 3.5ms = 771 tok/s
- Current: 453 tok/s = 59% of theoretical
- Gap: 1.9ms serial CPU overhead per step

### Future Directions

1. **Speculative metadata pre-computation**: Predict seq_lens(N+1) during GPU forward(N), pre-run flashinfer plan(). Correct 78% of the time.
2. **Hybrid decode+extend attention**: Split into decode-like (prefix, 0.07ms metadata) + local causal (block_size tokens). Reduce metadata from 0.43ms to ~0.18ms.
3. **Overlap scheduling for ISD**: Break the dependency chain by pipelining verify(N) with forward(N+1) using speculative batch preparation.
