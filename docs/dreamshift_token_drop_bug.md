# DreamShift Token Drop Bug: Analysis and Fix

## Problem

When running DreamShift models on sglang with `temperature=1.0`, some generated tokens are replaced by spaces or garbage characters. This causes incorrect outputs like `\boxed{3 **}` instead of `\boxed{35}`, or `\boxed{88 $}` instead of `\boxed{880}`.

This bug **does not occur** in the reference HuggingFace implementation (`generate.py`'s `block_diffusion_generate_with_shift`).

## Impact

On GSM8K 1319 questions (0-shot, temperature=1.0, top_k=50, top_p=0.95):

| Implementation | Accuracy | Token Drops |
|---|---|---|
| sglang (before fix) | 84.4% | 64/1319 |
| OpenCompass reference (same model) | 90.75% | 0/1319 |
| **sglang (after fix)** | **~98% on 50q** | **0** |

## Symptoms

### 1. Digit replacement in `\boxed{}` (43 cases)

| Question | Expected | sglang Output |
|----------|----------|---------------|
| Q32 | `\boxed{35}` | `\boxed{3 **}` |
| Q128 | `\boxed{880}` | `\boxed{88 $}` |
| Q141 | `\boxed{400}` | `\boxed{40 }` |
| Q149 | `\boxed{16}` | `\boxed{1 }` |

### 2. Generation loops (24 cases)

The model gets stuck in repetitive patterns with corrupted arithmetic:
```
4  4 is 16? , 4  4 is 16? , 4  4 is 16?    (should be "4 * 4 is 16")
5  8 is 13? , 5  8 is 13?                   (should be "5 + 8 is 13")
```

## Root Cause

**The commit pass in sglang incorrectly uses causal attention instead of bidirectional.**

### How it works

DreamShift generates tokens block-by-block (block_size=4). Each block goes through:
1. **Denoising steps** (4 forwards): iteratively unmask MASK tokens → bidirectional attention ✓
2. **Commit pass** (1 forward): write final KV cache, capture `_prev_last_logits` for next block

The `_prev_last_logits` from the commit pass feeds into the next block's logit shift (`shifted[0] = prev_last`). Its quality directly affects all subsequent generation.

### The bug

In `flashinfer_backend.py`, the attention mode detection uses `dllm_is_prefill`:

```python
# Detects "no MASK tokens in input_ids" → assumes STAGING_PREFILL → causal
dllm_is_prefill = not (forward_batch.input_ids == mask_id).any().item()
```

After denoising, all MASKs are filled → `dllm_is_prefill = True` → **commit pass uses causal attention**.

But in the reference `generate.py`:
```python
# Default causal_commit=False → commit uses BIDIRECTIONAL (same as denoising)
commit_attn_mask = cur_attn_mask  # bidirectional within current block
```

### Why this matters

With **causal** attention in the commit pass, the last token in the block only sees previous tokens. With **bidirectional**, it sees all tokens. This changes `_prev_last_logits`, which propagates through `shifted[0]` to the next block's denoising. With `temperature=1.0` sampling, the different logit distribution causes ~5% of answers to have digit→space token drops.

### Why it doesn't appear at temperature=0

At `temperature=0` (greedy/argmax), both causal and bidirectional produce the same argmax token because the differences are small. Only with `temperature=1.0` sampling do the probability differences matter enough to change which token gets sampled.

### The complication: STAGING_PREFILL also has no MASKs

Both "STAGING_PREFILL with prefix cache" and "commit pass" have:
- No MASK tokens in `input_ids`
- `extend_no_prefix=False` (has prefix)
- Enter the cascade attention path

But they need **different** attention:
- STAGING_PREFILL: **causal** (prompt tokens attend causally, matching training)
- Commit pass: **bidirectional** (denoised block tokens attend bidirectionally)

The original `dllm_is_prefill` detection can't distinguish them because both have no MASK tokens.

## Fix

### Approach: explicit commit signal from the algorithm

The DreamShift algorithm sets `forward_batch.dllm_is_commit = True` before the commit forward, signaling the attention backend to skip the causal override.

### Changes

**`python/sglang/srt/dllm/algorithm/dreamshift.py`:**

```python
# Before (commit pass uses whatever attention the backend decides):
out = model_runner.forward(forward_batch, pp_proxy_tensors=None)

# After (signal the backend this is a commit pass):
forward_batch.dllm_is_commit = True
out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
forward_batch.dllm_is_commit = False
```

**`python/sglang/srt/layers/attention/flashinfer_backend.py`:**

```python
# In init_forward_metadata, when detecting dllm_is_prefill:
dllm_is_prefill = False
if self.dllm_causal_prefill and forward_batch.forward_mode.is_dllm_extend():
    is_commit = getattr(forward_batch, "dllm_is_commit", False)
    if is_commit:
        dllm_is_prefill = False   # commit → bidirectional
    else:
        dllm_is_prefill = not (   # no MASK → prefill → causal
            forward_batch.input_ids == self.dllm_config.mask_id
        ).any().item()
```

This ensures:
- **STAGING_PREFILL** (no MASK, not commit): `dllm_is_prefill=True` → causal ✓
- **Denoising** (has MASK): `dllm_is_prefill=False` → bidirectional ✓
- **Commit pass** (no MASK, is commit): `dllm_is_prefill=False` → bidirectional ✓

### Failed approach: removing dllm_is_prefill entirely

Initially tried removing the `dllm_is_prefill` check from the cascade path. This fixed token drops but **broke STAGING_PREFILL** — prompt tokens were processed with bidirectional attention instead of causal, corrupting the KV cache and degrading generation quality from ~90% to 62%.

## Verification

### Token drop test (5 previously-affected questions, 10 runs each)

| Question | Before fix | After fix |
|----------|-----------|-----------|
| Q32 (gt=35) | 1/10 drops | **0/10** ✓ |
| Q128 (gt=880) | 1/10 drops | **0/10** ✓ |
| Q141 (gt=400) | — | **0/10** ✓ |
| Q149 (gt=16) | — | **0/10** ✓ |

### GSM8K accuracy (50 questions, 0-shot, temperature=1.0)

| Version | Accuracy |
|---------|----------|
| Before fix (causal commit) | 84% avg |
| Bad fix (no dllm_is_prefill) | 62% |
| **After fix (dllm_is_commit signal)** | **98%** |

### OpenCompass output validation

Our extraction logic on OpenCompass outputs reproduces exactly **90.75%** (1197/1319), confirming the evaluation pipeline is correct.

## Files Changed

| File | Change |
|------|--------|
| `python/sglang/srt/dllm/algorithm/dreamshift.py` | Set `forward_batch.dllm_is_commit = True` before commit forward |
| `python/sglang/srt/layers/attention/flashinfer_backend.py` | Check `dllm_is_commit` flag to skip causal override for commit pass |
