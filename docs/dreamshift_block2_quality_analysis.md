# DreamShift Block2 Quality Analysis: EXTEND vs DECODE Path

## Summary

Block2 (DLLM EXTEND mode) scores **~8% lower** on IFEval than the same model running in standard AR DECODE mode. This gap is structural — caused by the attention computation path, not sampling or speculation.

## Evidence

### IFEval Results (backup8000 checkpoint, 541 prompts, 8 GPUs)

| Mode | Prompt-strict | Sampling | Attention Path |
|------|--------------|----------|----------------|
| Qwen3 AR (sglang) | **84.29%** | sglang FlashInfer sampler | DECODE (paged attn) |
| Block2 conf=0.99, t=1.0 | 77.93% (5-run avg) | our `_sample()` | EXTEND (ragged+paged cascade) |
| Block2 conf=1.0, t=1.0 | 75.23% | our `_sample()` | EXTEND |
| Block2 conf=1.0, t=0 (greedy) | 76.16% | argmax | EXTEND |
| OpenCompass reference | 81.89% | HF `sample_with_temp...` | HF native (no cascade) |

### What We Ruled Out

1. **`_sample()` bug** — greedy (argmax, no randomness) also scores low (76.16%)
2. **Speculation quality** — conf=1.0 (no speculation, pure AR via DLLM) is *worse* (75.23%)
3. **Sampling params** — identical temp=1.0, top_k=50, top_p=0.95 across all modes
4. **top_p implementation** — verified numerically equivalent to reference
5. **Batching** — no-batch (max_req=1) scores same as batch (76.89% vs 77.93%)
6. **Chat template** — identical `apply_chat_template` with same tokenizer

### Root Cause: EXTEND vs DECODE Attention

The DLLM infrastructure uses `ForwardMode.DLLM_EXTEND` for all decode rounds:

```
DECODE mode (AR):     paged_attention(q, kv_cache) → hidden
EXTEND mode (DLLM):   ragged_attention(q, q) ⊕ paged_attention(q, kv_cache) → hidden
                      ↑ cascade merge via LogSumExp
```

Even for identical inputs, the two paths produce different hidden states because:
- **Ragged attention**: computes self-attention among new tokens (block_size=3 tokens)
- **Cascade merge**: combines ragged and paged results via LogSumExp, introducing numerical differences
- **FlashInfer kernel**: different CUDA kernels for DECODE vs EXTEND have different numerical precision

These differences are small per-token but compound over hundreds of tokens per response, degrading instruction-following accuracy.

### Why Speculation Helps

Counter-intuitively, conf=0.99 (with speculation) scores *higher* than conf=1.0 (no speculation):
- **77.93% (conf=0.99)** vs 75.23% (conf=1.0)

On accept, `prev_last_logits` comes from MASK's hidden state. The MASK at position P+1 sees the full context up to P and predicts P+2. This prediction, while speculative, provides a *different* (and apparently better) sampling distribution for the next token than the clean but EXTEND-degraded logits from the real token.

## GSM8K Results (for comparison)

| Mode | Accuracy | Notes |
|------|----------|-------|
| Block2 + CG (8 GPU, 1319 problems) | 95.4% | No significant quality loss |
| Qwen3 AR + CG (8 GPU, 100 problems) | 97% | Baseline |

GSM8K is less affected (~2% gap) because math reasoning is more robust to small logit perturbations than instruction-following format constraints.

## Follow-up: Paged-only Attention Experiment

We tested changing DLLM_EXTEND from ragged+paged cascade to paged-only (matching TARGET_VERIFY path):

| Mode | Prompt-strict |
|------|-------------|
| Ragged+Paged (default) | 77.93% avg |
| **Paged-only** | **78.74%** |
| Qwen3 AR (DECODE) | 84.29% |

Only ~1% improvement. The ragged cascade is NOT the main cause of the 8% gap.

The remaining gap likely comes from the EXTEND path's KV write behavior: in EXTEND mode, all block_size=3 tokens' KV is written in a single batch, while DECODE writes 1 token's KV per step. This affects how subsequent attention reads the KV cache — the paged attention kernel may index KV differently when the query spans multiple new positions vs just one.

## Recommendations

1. **For quality-sensitive tasks (IFEval)**: Use Qwen3 AR mode (`--json-model-override-args`)
2. **For math/reasoning tasks (GSM8K)**: Block2 is fine (95.4% vs 97%)
3. **Long-term fix**: Implement Block2 speculation on the DECODE attention path (requires multi-token DECODE kernel support or integration with speculative decoding framework)

## Appendix: Checkpoint Comparison (IFEval, Block2 conf=0.99, t=1.0)

| Checkpoint | Prompt-strict |
|-----------|--------------|
| backup8000 | **79.67%** |
| backup16000 | 76.52% |
| backup24000 | 78.37% |
| backup32000 | 75.05% |
| final | 79.11% |
