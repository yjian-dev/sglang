# DreamShift Speculative Decoding for SDAR Models in SGLang

## 1. Overview

This project implements speculative decoding for SDAR (Speculative Diffusion Auto-Regressive) models in SGLang, enabling up to **N tokens per forward pass** while maintaining lossless output quality through speculative verification.

Two algorithm variants are implemented:
- **DreamShiftBlock2** (N=2): 2 tokens/forward, proven stable, fully tested
- **DreamShiftBlockN** (N=3): 3 tokens/forward, generalized version, higher throughput

**Key Results (single H100 GPU, CUDA graph enabled):**

| Algorithm | Model | TPS (bs=1) | TPF | Accept Rate |
|-----------|-------|-----------|-----|-------------|
| DreamShiftBlock2 | SDAR-Qwen3-8B (b1) | 183 tok/s | 1.53 | 77.7% |
| DreamShiftBlockN (N=3) | SDAR-Qwen3-8B (b2) | 249 tok/s | 2.19 | 71.5% |
| Qwen3-8B AR baseline | Qwen3-8B | 151 tok/s | 1.0 | — |

**Quality (GSM8K, 0-shot):**

| Algorithm | Setup | Accuracy |
|-----------|-------|----------|
| DreamShiftBlock2 | 8-GPU, 1319 problems | 95.1% |
| DreamShiftBlockN (N=3) | 1-GPU, 10 problems | 100% (10/10) |
| OpenCompass reference (AR) | — | 90.75% |

**IFEval (541 prompts, 8-GPU):**

| Config | Prompt-strict | Inst-strict | Throughput |
|--------|-------------|------------|-----------|
| t=1.0, 4K | 80.41% | 86.09% | 12,731 tok/s |
| t=1.0, 16K | 81.33% | 86.93% | 5,641 tok/s |

**HumanEval (164 problems, 8-GPU):** pass@1 = 93.9%, 3,882 tok/s

## 2. Algorithm Design

### 2.1 Core Idea

SDAR models are trained with MASK tokens and can predict multiple tokens from a single input. We exploit this for speculative decoding:

1. **Forward pass:** Process N positions (real tokens + MASKs), get logits at each position
2. **Clean token:** Sample from the real token's logits (exact, no approximation)
3. **Speculative tokens:** Sample from MASK logits (draft quality, needs verification)
4. **Next round:** Verify speculative tokens using clean distributions, accept/reject

### 2.2 DreamShiftBlock2 (N=2, block_size=3)

Each forward processes 3 tokens with three cases:

```
Case A: [pending, carry, MASK]   → verify carry, sample new diff + carry
Case B: [pending, fresh, MASK]   → no carry to verify, sample fresh + new carry
Case C: [t0, MASK, MASK]         → cold start after reject
```

- **Accept:** output 2 tokens `[carry, diff]`, advance 2 positions
- **Reject:** output 1 corrected token, force as next t0

### 2.3 DreamShiftBlockN (N=3, block_size=5)

Generalizes Block2 to N tokens per forward. Two modes:

```
Cold start:  [t0, M, M, M, M]                → sample 1 clean + 2 spec, hold specs
Verify:      [pending, spec0, spec1, M, M]    → verify specs L→R, sample 3 new tokens
```

Verification is left-to-right: if `spec[i]` is rejected, all subsequent specs are discarded, and a corrected token is resampled from `max(0, p - q)`.

| Outcome | Output tokens | Trim | Advance |
|---------|--------------|------|---------|
| All N-1 specs accepted | N (verified specs + clean) | N-1 | N |
| Reject at index i | i+1 (accepted + corrected) | (N-1-i) + (N-1) | 1+i |
| Cold start | 2 (t0 + clean) | 2N-2 | 1 |

### 2.4 Speculative Verification (Lossless)

Standard speculative decoding acceptance criterion ensures the output distribution exactly matches autoregressive sampling:

```python
r = p(x) / q(x)    # p = clean distribution, q = draft distribution
if r >= 1.0 or random() < r:
    accept
else:
    reject, resample from max(0, p - q)
```

### 2.5 Deferred Carry (Force Next Token)

On spec reject, the corrected token is NOT re-processed immediately (avoiding wasted forward). Instead:

1. Output corrected token, set `_force_next_token[rpx] = corrected`
2. Next cold start uses forced token as `t0` (skip sampling)
3. Track `was_forced` to avoid double-output

This achieves **zero repetition** in output text.

## 3. System Optimizations

### 3.1 Paged-Only Attention (3 kernels → 1)

Standard SGLang extend uses cascade: ragged + paged + merge_state = 3 kernels/layer.
DLLM decode uses paged-only (`use_ragged=False`): 1 kernel/layer.

Mathematically equivalent with `causal=True` since new tokens attend causally to all prior tokens.

### 3.2 Batched Flashinfer Sampling

Replaced per-request Python sampling with flashinfer's fused `top_k_top_p_sampling_from_probs` kernel. Sampling cost reduced from ~0.86ms to ~0.24ms.

### 3.3 CUDA Graph Support

DLLM extend mode supports CUDA graph capture/replay. Model forward (p2) drops from ~13.6ms to ~0.5ms.

### 3.4 Causal Attention for SDAR Models

SDAR models with `use_regular_causal=True` require:
- **Prefill:** causal attention (matching training)
- **Decode:** `dllm_force_causal=True` so position 0 doesn't attend to MASK positions
- **Commit pass:** bidirectional (signaled by `dllm_is_commit=True`)

## 4. Performance

### 4.1 Single Request Latency Breakdown (H100, CUDA graph)

| Phase | Block2 | BlockN (N=3) | Description |
|-------|--------|-------------|-------------|
| p1 (pre-forward) | 0.04 ms | 0.04 ms | Case classification + input fill |
| p2 (model forward) | 0.52 ms | 0.48 ms | Transformer with CUDA graph |
| p3 (post-forward) | 7.31 ms | 7.06 ms | Sampling + verify + KV trim |
| **Total/forward** | **~7.9 ms** | **~7.6 ms** | |
| **TPF** | **1.53** | **2.19** | |
| **TPS** | **~183** | **~249** | |

### 4.2 Batch Scaling Comparison (H100, real prompts, max_tokens=1024)

| Batch Size | BlockN N=3 (tok/s) | Qwen3-8B AR (tok/s) | Speedup |
|-----------|-------------------|---------------------|---------|
| 1 | **261** | 151 | **1.73x** |
| 2 | **387** | 291 | **1.33x** |
| 4 | **622** | 581 | **1.07x** |
| 8 | 813 | **1,060** | 0.77x |
| 16 | 1,543 | **2,030** | 0.76x |
| 32 | 1,923 | **3,965** | 0.49x |

**Insight:** BlockN wins at low concurrency (bs=1-4) where per-request latency matters. AR wins at high concurrency because its decode step processes 1 token/request (more GPU-efficient) and has no p3 overhead.

### 4.3 Multi-GPU Evaluation Throughput

| Benchmark | GPUs | Total tok/s | Accuracy |
|-----------|------|------------|----------|
| GSM8K (Block2, 1319 problems) | 8 | 6,563 | 95.1% |
| IFEval (Block2, 541 prompts) | 8 | 12,731 | 80.4% prompt-strict |
| HumanEval (Block2, 164 problems) | 8 | 3,882 | 93.9% pass@1 |

## 5. Models

### SDAR-Qwen3-8B b1 (block_size=1, for Block2)
- Path: `/data/cxu/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b1-allmasked_fixed2_backup8000`
- Architecture: `SDARForCausalLM`, mask_id=151669
- Training: CE loss on clean region + teacher hidden-state distill, 100% mask rate

### SDAR-Qwen3-8B b2 (block_size=2, for BlockN)
- Path: `/data/cxu/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont_backup24000`
- Architecture: `SDARForCausalLM`, mask_id=151669
- Training: block_size=2, causal attention, continued training

## 6. Setup & Usage

### 6.1 Branch

```bash
git clone https://github.com/yjian-dev/sglang.git
git checkout jyq/dreamshift-block2-optimized
```

### 6.2 Environment

```bash
# Conda environment
conda activate sglang

# CUDA 12.9 required for flashinfer (not 13.1)
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

# If flashinfer JIT cache is stale:
rm -rf ~/.cache/flashinfer/
```

### 6.3 Launch Server

**DreamShiftBlock2 (N=2, b1 model):**

```bash
python -m sglang.launch_server \
  --model-path /data/cxu/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b1-allmasked_fixed2_backup8000 \
  --trust-remote-code --tp-size 1 \
  --mem-fraction-static 0.85 --max-running-requests 64 \
  --attention-backend flashinfer \
  --dllm-algorithm DreamShiftBlock2 \
  --dllm-algorithm-config dreamshift_block2_config.yaml \
  --dtype bfloat16 --port 30000
```

**DreamShiftBlockN (N=3, b2 model):**

```bash
python -m sglang.launch_server \
  --model-path /data/cxu/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont_backup24000 \
  --trust-remote-code --tp-size 1 \
  --mem-fraction-static 0.85 --max-running-requests 64 \
  --attention-backend flashinfer \
  --dllm-algorithm DreamShiftBlockN \
  --dllm-algorithm-config dreamshift_blockN_config.yaml \
  --dtype bfloat16 --port 30000
```

Add `--disable-cuda-graph` if CUDA graph capture fails (driver/runtime mismatch).

### 6.4 Query

```bash
# Streaming demo with TPS counter
python scripts/stream_demo.py --url http://localhost:30000 --prompt "Your prompt" --max-tokens 1024

# OpenAI-compatible API
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"sdar","messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'
```

### 6.5 Multi-GPU Evaluation

```bash
# Start 8 servers (one per GPU)
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$gpu python -m sglang.launch_server \
    --model-path <model> --trust-remote-code --tp-size 1 \
    --mem-fraction-static 0.85 --max-running-requests 64 \
    --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
    --dllm-algorithm-config dreamshift_blockN_config.yaml \
    --dtype bfloat16 --port $((30000+gpu)) &
done

# Run GSM8K eval
python test_gsm8k_0shot.py --port 30000 --num-questions 1319
```

### 6.6 Config Files

**dreamshift_block2_config.yaml** (for Block2, N=2):
```yaml
block_size: 3
confidence_threshold: 0.0
temperature: 1.0
top_k: 50
top_p: 0.95
use_spec_verify: true
```

**dreamshift_blockN_config.yaml** (for BlockN, N=3):
```yaml
block_size: 5
gen_block_size: 3
confidence_threshold: 0.0
temperature: 1.0
top_k: 50
top_p: 0.95
use_spec_verify: true
```

## 7. File Structure

```
python/sglang/srt/dllm/
├── algorithm/
│   ├── base.py                    # DllmAlgorithm base class
│   ├── dreamshift_block2.py       # Block2 algorithm (N=2, 497 lines)
│   ├── dreamshift_blockN.py       # BlockN algorithm (N=3+, 370 lines)
│   ├── joint_threshold.py         # Original DLLM algorithm
│   └── low_confidence.py          # Original DLLM algorithm
├── config.py                      # DllmConfig
└── mixin/
    ├── req.py                     # Per-request state
    └── scheduler.py               # DLLM scheduling

python/sglang/srt/
├── configs/sdar.py                # SDAR model config
├── models/sdar.py                 # SDARForCausalLM
├── layers/attention/flashinfer_backend.py  # Paged-only attention
├── layers/rotary_embedding/base.py        # Compatibility fix
├── managers/
│   ├── schedule_batch.py                  # prepare_for_dllm_decode()
│   ├── schedule_policy.py                 # block_size cap
│   └── scheduler_output_processor_mixin.py # KV trim + advance
└── mem_cache/chunk_cache.py               # kv_committed_len

scripts/stream_demo.py             # Streaming output demo
dreamshift_block2_config.yaml      # Block2 config
dreamshift_blockN_config.yaml      # BlockN config
```

## 8. Next Steps

### P0: Reduce p3 Bottleneck
Post-forward processing (p3 = 7ms) dominates latency and limits both single-request TPS and batch scaling. Key targets:
- **Fuse spec verification onto GPU:** Currently uses CPU-side `p(x)/q(x)` computation with GPU-CPU sync. A custom CUDA kernel could eliminate sync overhead.
- **Reduce Python loop overhead:** The per-request accept/reject/trim logic is in Python. Could be vectorized or moved to C++/Triton.
- **Batch KV trim:** Current per-request `req_to_token` lookup can be batched into a single tensor operation.

### P1: Improve Batch Throughput
At bs>=8, AR overtakes BlockN due to:
- AR decode processes 1 token/request (compute-efficient), BlockN processes 5 tokens/request
- p3 overhead is per-forward, not amortized across batch
- Solutions: overlap p3 with next p2 (pipeline), or fast decode path (reuse last_batch)

### P2: Support Larger N
BlockN is designed for arbitrary N. Testing with N=4,5 could increase TPF further, but requires models trained with larger block_size. The accept rate per spec token may decrease with more MASKs in context, yielding diminishing returns.

### P3: Full BlockN Evaluation
- Run complete GSM8K (1319 problems), IFEval (541 prompts), HumanEval (164 problems) with BlockN on 8 GPUs
- Compare accuracy and throughput against Block2 and AR baseline
- Test with different temperature/sampling configs

### P4: Fast Decode Path
Re-enable the fast decode path (reusing `last_batch` to skip scheduler overhead) for single-request latency. Currently disabled due to memory leak when requests finish in multi-request mode. Needs a clean reference-counting fix.
