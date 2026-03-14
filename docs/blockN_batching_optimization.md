# DreamShift BlockN: Speculative Diffusion LLM Serving in SGLang

## Technical Report: Design, Optimization, and Results

---

## 1. Background: How LLM Inference Works

### 1.1 Autoregressive (AR) Decoding

Standard LLMs (like Qwen3-8B) generate tokens **one at a time**:

```
Step 1: [prompt] → model → token_1
Step 2: [prompt, token_1] → model → token_2
Step 3: [prompt, token_1, token_2] → model → token_3
...
```

Each step requires a full model forward pass. The model has two phases:
- **Prefill**: Process the entire prompt at once (compute-heavy, one-time)
- **Decode**: Generate one token per step (memory-bandwidth-bound, repeated)

**Key metric**: At batch_size=1 on H100, Qwen3-8B does ~152 tokens/sec. Each decode step takes ~6.5ms (GPU forward) to produce 1 token.

### 1.2 KV Cache

To avoid recomputing all previous tokens, LLMs store intermediate results in a **KV cache** (Key-Value cache). Each layer stores the K and V matrices for all processed tokens. This turns each decode step from O(seq_len^2) to O(seq_len).

The KV cache grows linearly with sequence length and batch size, making **memory management** critical for serving.

### 1.3 Diffusion LLM (DLLM) / SDAR Model

Unlike AR models that generate left-to-right, **SDAR (Semi-Autoregressive Diffusion)** models generate **blocks of tokens** per forward pass:

```
Step 1: [prompt, MASK, MASK, MASK, MASK, MASK] → model → [token_1, draft_2, draft_3, MASK, MASK]
Step 2: [token_1, draft_2, draft_3, MASK, MASK] → model → [token_1, token_2, token_3, draft_4, MASK]
```

The model sees both clean (committed) and masked positions, using **bidirectional attention** within the block (unlike AR's causal-only attention). This allows generating multiple tokens per step, potentially beating AR throughput.

### 1.4 DreamShift

**DreamShift** is the inference algorithm for SDAR models. It uses a "logit shift" trick: the logits at position `i` predict the token for position `i+1` (shifted), enabling causal-style generation with bidirectional attention training.

The algorithm has two modes per step:
- **Cold start**: `[t0, MASK, MASK, ..., MASK]` → sample clean + speculative tokens
- **Verify**: `[pending, spec0, spec1, ..., MASK, MASK]` → verify specs, sample new ones

---

## 2. DreamShiftBlockN Algorithm Design

### 2.1 Parameters

```yaml
gen_block_size: N    # tokens produced per step (1 clean + N-1 speculative)
block_size: 2N-1     # input positions per forward pass
```

For N=3: each forward processes 5 positions, producing 1-3 output tokens.

### 2.2 Speculative Verification

Speculative tokens are verified using the standard speculative decoding criterion:

```
r = p(x) / q(x)    # p = model's distribution, q = draft distribution
if r >= 1: accept   # model agrees with draft
if r < 1: accept with probability r, else resample from max(0, p-q)
```

- **Accept rate**: The fraction of verify rounds where all specs are accepted
- **TPF (Tokens Per Forward)**: Average output tokens per model forward pass

Higher accept rate → higher TPF → faster generation.

### 2.3 Algorithm State

Per-request state is stored in Python dicts keyed by `req_pool_idx`:

| State | Description |
|---|---|
| `_prev_last_logits` | Last position's logits from previous forward (for cold start sampling) |
| `_pending` | Clean token awaiting verification |
| `_spec_tokens` | Speculative token values |
| `_spec_draft_qx` | Scalar q(x) probability per spec token (for verification) |
| `_force_next_token` | Corrected token on rejection (used as next t0) |

### 2.4 Per-Step Flow

```
Phase 1: Classify requests (verify vs cold start), fill input_ids
Phase 2: Model forward (GPU, ~6.5ms)
Phase 3: Post-forward processing
  Step 1: Spec verification (softmax + accept/reject)
  Step 2: Batched sampling (flashinfer top_k_top_p)
  Step 3: KV trim (free MASK positions)
  Step 4: Output assembly (build token lists, save state)
```

---

## 3. Serving Architecture in SGLang

### 3.1 Standard SGLang Pipeline

```
HTTP Request → Tokenizer → Scheduler → GPU Forward → Detokenizer → HTTP Response
                              |
                     [ZMQ IPC between processes]
```

SGLang uses three processes:
1. **Tokenizer/HTTP** (FastAPI + uvicorn): handles HTTP, tokenization
2. **Scheduler** (GPU process): batch scheduling, model forward, result processing
3. **Detokenizer**: converts token IDs back to text

### 3.2 Scheduler Event Loop

The scheduler runs a loop:
```python
while True:
    recv_requests()          # receive from tokenizer
    get_next_batch_to_run()  # build batch (scheduling)
    run_batch(batch)         # GPU forward
    process_batch_result()   # update state, stream output
```

For AR models, SGLang supports **overlap scheduling** (`event_loop_overlap`): CPU result processing overlaps with GPU forward of the next batch using two CUDA streams. This hides CPU overhead behind GPU compute.

### 3.3 DLLM Scheduling Differences

DLLM **cannot use overlap scheduling** because:
1. The algorithm's `run()` bundles forward + post-processing (p3 needs GPU results immediately)
2. DLLM uses "extend" mode (not decode mode), requiring full batch reconstruction each step
3. Radix cache is disabled (DLLM's KV trim pattern is incompatible)

This means DLLM uses `event_loop_normal` (fully serial), putting it at a structural disadvantage vs AR's overlap scheduling.

---

## 4. Optimizations Implemented

### 4.1 Decode-Mode Inner Loop (Biggest Impact: +134% at bs=32)

**Problem**: Every DLLM decode step went through the full scheduling pipeline:
```
stash staging reqs (cache_unfinished_req x N) → filter last_batch →
get_new_batch_dllm (init_next_round + prepare_for_extend) →
run_batch → process_batch_result (cache_unfinished_req x N again)
```

Each `cache_unfinished_req` does a GPU tensor copy: `prefix_indices = req_to_token[:kv_len].to(int64, copy=True)`. With N requests, this happens **2N times per step**.

**Solution**: A tight inner loop (`_dllm_decode_loop`) that:
- Reuses the batch object across steps (no `ScheduleBatch.init_new()`)
- Uses lightweight `prepare_for_dllm_decode()` instead of full `prepare_for_extend()`
- Handles finished requests in-place via `filter_batch()`
- Breaks to slow path only when new requests arrive or batch empties

**Entry conditions** (to avoid the prefill guard bug):
```python
if (self.dllm_config is not None
    and self.dllm_manager.any_staging_reqs()
    and not self.waiting_queue
    and not any(r.finished() for r in batch.reqs)
    and not any(r.is_dllm_prefill() for r in batch.reqs)):  # crucial!
    self._dllm_decode_loop(batch)
```

The `is_dllm_prefill()` guard prevents entering the loop after prefill (which hasn't set `dllm_next_advance` yet -- see Section 5.1).

**Impact**: Step time reduced from ~12ms to ~8ms at bs=1. Throughput: 1923 → 4245 tok/s at bs=32.

### 4.2 Scalar Draft Probabilities (Memory: -39MB)

**Problem**: Original code stored full vocabulary distributions per speculative token:
```python
self._spec_draft_probs: Dict[int, List[torch.Tensor]]  # [vocab_size] per spec token
```
At bs=32 with 2 spec tokens each: 64 x 152K x 4 bytes = **~39MB** of GPU memory, plus expensive `torch.stack()` on every verify step.

**Solution**: Store only the scalar probability q(x) of the sampled token:
```python
self._spec_draft_qx: Dict[int, List[float]]  # one float per spec token
```

Verification only needs `q(x)` for the specific drafted token, not the full distribution. The `_batched_sample` function already returns `token_probs`, so we extract q(x) directly from the sampling result -- **eliminating the separate softmax pass** for draft probabilities.

**Trade-off**: On rejection, we can no longer compute `max(0, p-q)` exactly (since we don't have full q). Instead, we resample from `p(x)` directly. This is slightly suboptimal but the quality impact is negligible.

### 4.3 GPU-Side KV Index Passing

**Problem**: KV trim indices were collected as a Python list, converted to GPU tensor in the output processor:
```python
# Algorithm side
kv_indices = [req_to_token[rpx, pos].item() for ...]  # N .item() syncs!
# Output processor side
torch.tensor(kv_indices, device=cuda)  # H2D memcpy
```

**Solution**: Keep KV indices as GPU tensor slices, pass directly:
```python
all_kv_indices = req_to_token[all_trim_rpx, all_trim_pos]  # one batched GPU read
# Pass GPU tensor slice to output processor
self._kv_trim_info[rpx] = {"kv_indices_gpu": all_kv_indices[offset:offset+tc], ...}
```

### 4.4 Python List Output (Eliminate Round-Trip)

**Problem**: `next_token_ids` were created as GPU tensors, then immediately `.tolist()`'d in the output processor:
```python
# Algorithm: Python list → torch.tensor(device=cuda)  [H2D memcpy + GPU alloc]
# Output processor: tensor.tolist()                     [D2H sync]
```

**Solution**: Return Python lists directly, skip the GPU round-trip entirely.

### 4.5 Batched Mask Check

**Problem**: Per-request loop checking for MASK tokens:
```python
for bid in range(batch_size):
    chunk = input_ids[offset:offset+n_tokens]
    is_decode.append((chunk == mask_id).any().item())  # sync per request!
```

**Solution**: One vectorized check:
```python
has_any_decode = (forward_batch.input_ids == self.mask_id).any().item()  # 1 sync
```

### 4.6 Batched Reject Correction

**Problem**: Per-rejection `torch.multinomial().item()` sync:
```python
for bid in rejected_bids:
    new_tok = torch.multinomial(corr_dist, 1).item()  # sync per rejection
```

**Solution**: Batch all corrections:
```python
corr_batch = torch.stack(corr_dists)
corr_tokens = torch.multinomial(corr_batch, 1).squeeze(1)
corr_tokens_cpu = corr_tokens.tolist()  # 1 sync for all
```

### 4.7 Pre-Allocated Sampling Buffers

Flashinfer's `top_k_top_p_sampling_from_probs` requires `top_ks` and `top_ps` tensors. Instead of creating new ones each call:
```python
_SAMPLE_BUFS = {}  # cached per device
def _get_sample_bufs(n, top_k, top_p, device):
    # Returns pre-allocated tensors, resized if needed
```

### 4.8 Batched Logit Gathering

**Problem**: Per-request GPU tensor slice in the output assembly loop:
```python
for bid in range(batch_size):
    self._prev_last_logits[rpx] = full_logits[bid * blk + offset]  # GPU index kernel
```

**Solution**: Collect all indices first, one gather:
```python
_idx_t = torch.tensor(_logit_save_indices, dtype=torch.long, device=device)
_saved_logits = full_logits[_idx_t].detach().clone()  # 1 gather
# In loop: _saved_logits[bid] is a cheap CPU-side tensor offset
```

### 4.9 Request State Cleanup

**Problem**: When a request finishes, its `req_pool_idx` may be recycled by a new request. The algorithm's per-request dicts (`_pending`, `_spec_tokens`, etc.) still hold stale data for the old request. This caused **accuracy collapse from 93% to 6%** at bs=32.

**Solution**:
```python
def cleanup_request(self, req_pool_idx):
    self._prev_last_logits.pop(req_pool_idx, None)
    self._pending.pop(req_pool_idx, None)
    self._spec_tokens.pop(req_pool_idx, None)
    self._spec_draft_qx.pop(req_pool_idx, None)
    self._force_next_token.pop(req_pool_idx, None)
```
Called from `process_batch_result_dllm` when `req.finished()`.

### 4.10 Empty Stream Output Skip

**Problem**: DLLM prefill produces 0 output tokens, but `stream_output` still sent an empty ZMQ message through the full pipeline (~50ms round-trip wasted).

**Solution**:
```python
if should_output and not req.finished():
    if req.send_token_offset >= len(req.output_ids):
        should_output = False  # no new tokens to stream
```

---

## 5. Key Bugs Found and Fixed

### 5.1 Prefill Guard Bug (Accuracy: 6% -> 94%)

**Root cause**: The decode loop entered immediately after prefill, before the first decode step set `dllm_next_advance`. `prepare_for_dllm_decode` called `init_next_round_input()` which advanced `dllm_block_offset` by `block_size` (default) instead of the correct advance value.

**Fix**: Add `not any(r.is_dllm_prefill() for r in batch.reqs)` to the decode loop entry condition.

### 5.2 `filter_batch` Output IDs Bug

When `filter_batch()` ran inside the decode loop, it tried to index `self.output_ids` with `keep_indices_device`. But our optimization returns Python lists (not tensors) for `output_ids`, causing `TypeError`.

**Fix**: Set `batch.output_ids = None` before calling `filter_batch()`.

### 5.3 `req_pool_idx` Reuse Bug

Detailed in Section 4.9. This was the most impactful correctness bug -- without cleanup, bs=32 accuracy was 6.2% (effectively random).

---

## 6. Profiling Results

### 6.1 Step Time Breakdown (bs=1, N=3)

| Component | Time (ms) | % of Step |
|---|---|---|
| GPU forward (model + lm_head) | 6.43 | 78% |
| Verify (softmax + accept/reject) | 0.63 | 8% |
| Sampling (flashinfer top_k_top_p) | 0.22 | 3% |
| KV trim + output assembly | 0.50 | 6% |
| Sync overhead (20x `.tolist()`) | ~1.0 | 12% |
| **Total step** | **~8.3** | **100%** |

### 6.2 GPU Kernel Breakdown

| Category | Time/step | % GPU |
|---|---|---|
| GEMM (FFN + lm_head) | 5.17ms | 81% |
| Attention (flashinfer) | 0.68ms | 11% |
| Activation/Norm/Other | 0.52ms | 8% |

**Key insight**: GPU compute is nearly identical between BlockN (6.37ms for 5 tokens) and Qwen AR (6.10ms for 1 token) at bs=1. The GPU is far from saturated -- kernel launch overhead dominates, not FLOPs.

### 6.3 Sync Analysis

| bs | Syncs/step | Sync time/step | CPU gap | Step time |
|---|---|---|---|---|
| 1 | 20 | 0.89ms | 5.78ms | 12.25ms |
| 8 | 19 | 6.53ms | 5.04ms | 11.58ms |
| 32 | 48 | 7.11ms | 8.46ms | 15.41ms |
| 64 | 70 | 7.52ms | 10.78ms | 17.99ms |

Sync count scales with batch size due to per-request `.tolist()` calls.

---

## 7. Performance Results

### 7.1 Throughput (GSM8K, H100, 1 GPU)

| bs | BlockN N=3 | BlockN N=4 | Qwen3-8B (no overlap) | N=4 / Qwen |
|---|---|---|---|---|
| 1 | 238 | 254 | 139 | **1.83x** |
| 2 | -- | 509 | 269 | **1.89x** |
| 4 | 852 | 880 | 532 | **1.65x** |
| 8 | 1456 | 1596 | 1041 | **1.53x** |
| 16 | -- | 2706 | 1983 | **1.36x** |
| 32 | 4540 | 4526 | 3765 | **1.20x** |
| 48 | 5312 | 5250 | 5182 | **1.01x** |

### 7.2 Accept Rate by N and Domain

| N | block_size | GSM8K Accept% | Essay Accept% | GSM8K TPF |
|---|---|---|---|---|
| 2 | 3 | 87.3% | 76.4% | 1.60 |
| 3 | 5 | 70.6% | 70.5% | 2.11 |
| 4 | 7 | 56.8% | 20.5% | 2.44 |
| 5 | 9 | 32.6% | 18.3% | 2.35 |

**Key finding**: Accept rate is domain-dependent. Math (GSM8K) has more predictable token patterns -> higher accept rate. N=4 is the sweet spot for GSM8K (TPF 2.44), while N=3 is safer for general text.

### 7.3 AIME 2024 (16 rounds, 8 GPU)

- **Average score**: 18.1/30 = 60.4%
- **pass@16**: 83.3% (25/30 problems solved at least once)
- **Throughput**: ~2100-2500 tok/s aggregate (8 GPU)

---

## 8. Optimization Attempts That Failed

### 8.1 Overlap Scheduling for DLLM

Three approaches tried:
1. **`overlap_fn` with `get_next_batch_to_run`**: Scheduler state conflicts (not thread-safe) -> crash
2. **`overlap_fn` with `process_result` + `prepare_for_dllm_decode`**: KV memory leak (allocated slots unused on early exit)
3. **Deferred result processing**: `prepare_for_dllm_decode` depends on `process_result` -> can't decouple

**Fundamental issue**: The dependency chain `process_result(N) -> prepare(N+1) -> forward(N+1) -> p3(N+1)` is strictly serial.

### 8.2 Prefill Sampling for TTFT

Attempted to sample the first token directly from prefill logits (saving one decode step). **Failed** because SDAR's prefill logits are "shifted" -- they predict different positions than standard next-token prediction.

### 8.3 Skip `cache_unfinished_req`

Attempted to use `kv_committed_len` directly instead of GPU tensor copy. Performance gain at bs=32 (+2-9%) but caused crash at bs=48 (KV accounting mismatch when decode loop exits to slow path). Half-opt (skip only in prepare) was **slower** because the GPU copy is extremely cheap (~0.07ms) and pipelines well.

### 8.4 Pre-Allocated Index Buffers

`torch.tensor(list, device='cuda')` is already PyTorch's optimized path (list->numpy->cuda direct). Manual buffer management (`copy_` from pinned memory) was consistently slower.

### 8.5 Batch Scalar GPU Writes

Batching `forward_batch.input_ids[idx] = val` into one vectorized write. The tensor allocation overhead for the batch write exceeded the cumulative cost of N scalar writes (each <0.01ms).

---

## 9. Glossary

| Term | Definition |
|---|---|
| **TPF** | Tokens Per Forward -- average output tokens per model forward pass |
| **Accept Rate** | Fraction of verify rounds where all speculative tokens are accepted |
| **TTFT** | Time To First Token -- latency from request arrival to first token streamed |
| **ITL** | Inter-Token Latency -- time between consecutive streaming token events |
| **KV Cache** | Key-Value cache storing intermediate attention results for past tokens |
| **KV Trim** | Freeing KV cache slots for MASK positions after each step |
| **CUDA Graph** | Pre-recorded GPU operation sequences that reduce kernel launch overhead |
| **Radix Cache** | SGLang's prefix caching mechanism -- disabled for DLLM |
| **Overlap Scheduling** | SGLang's pipelined scheduler where CPU processes results while GPU runs next forward |
| **`cache_unfinished_req`** | GPU operation that copies KV indices to `prefix_indices` for next scheduling round |
| **`prepare_for_dllm_decode`** | Lightweight batch preparation that reuses the batch object (vs full `prepare_for_extend`) |
| **`req_pool_idx`** | Index into the KV pool for each request -- recycled when requests finish |
| **Decode Loop** | Tight inner loop that skips full scheduling, only doing prepare + forward + process |
| **Slow Path** | Full scheduling pipeline (stash + filter + merge + get_new_batch + prepare_for_extend) |
| **ZMQ Pipeline** | Inter-process communication between scheduler, detokenizer, and tokenizer manager |

---

## 10. Files Modified

| File | Changes |
|---|---|
| `dllm/algorithm/dreamshift_blockN.py` | Algorithm rewrite: scalar draft probs, batched ops, GPU KV indices, list output, overlap_fn hook, cleanup_request, batched logit gather |
| `dllm/mixin/scheduler.py` | Removed broken `_try_dllm_fast_decode`, simplified scheduling |
| `managers/schedule_batch.py` | `prepare_for_dllm_decode`: alloc-before-mutate, rebuild req_pool_indices, batched KV writes |
| `managers/scheduler.py` | `_dllm_decode_loop` with in-place finished handling, decode loop entry with prefill guard |
| `managers/scheduler_output_processor_mixin.py` | GPU KV free, list/tensor compat, cleanup_request, skip empty stream |
| `managers/tp_worker.py` | `overlap_fn` parameter passthrough |
