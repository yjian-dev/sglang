# Plan

## Status
**Phase 2 complete** — DreamShiftBlockN exceeds EAGLE3 throughput targets.

## Results Summary

### Throughput (1×H100, random input=256 output=256)

| Concurrency | verify=false | verify=true | EAGLE3 target |
|---|---|---|---|
| 16 | 3,732 | — | — |
| 32 | **5,690** | 4,208 | 5,103 |
| 48 | 6,825 | — | — |
| 64 | **7,460** | 5,713 | 5,569 |
| 96 | 7,592 | — | — |
| 128 | 7,597 | — | — |

- **verify=false (recommended for throughput)**: Exceeds EAGLE3 at conc=32 by 11.5%, conc=64 by 34%
- **verify=true (higher quality)**: Exceeds EAGLE3 at conc=64 (+2.6%), 82% at conc=32
- No KV memory leak (token_usage=0.00 after all benchmarks)
- CUDA graphs enabled (30% improvement at conc=32)

## Bugs Fixed This Iteration

### 1. Inline Prefill Stuck Bug (CRITICAL)
**Symptom**: Decode loop infinite loop — one request stuck in STAGING_PREFILL with output_ids=2 forever.

**Root cause**: When `_inline_absorb_new_requests` adds a request to the decode batch:
1. Algorithm doesn't distinguish inline prefill from decode in mixed batch → returns 2 tokens
2. `process_batch_result_dllm` adds tokens to output_ids instead of entering prefill→decode transition
3. On next step, `origin_remaining=0` but `_inline_prefill` flag still True → extends with 0 actual tokens
4. Repeats forever

**Fix** (in `scheduler_output_processor_mixin.py`):
- Force-treat `_inline_prefill=True` requests as pure prefill regardless of algorithm output
- In `prepare_for_dllm_decode`: when `origin_remaining <= 0`, transition to decode mode

### 2. bench_serving Compatibility
**Symptom**: bench_serving uses `/generate` endpoint with text prompts (ShareGPT-sampled).

**Fix**: No code change needed — the inline prefill fix resolved this. With `use_spec_verify=false`, bench_serving completes all 200+ requests.

### 3. `decode_reqs` NameError
Cleanup of debug logging accidentally removed the `decode_reqs = self.dllm_manager.get_decode_requests()` line. Fixed.

## Architecture

### Decode Loop Flow
```
event_loop_normal:
  recv_requests → process_input_requests → waiting_queue
  get_next_batch_to_run → get_new_batch_dllm (prefill 1 req)
  run_batch → process_batch_result_dllm (transition → STAGING_DECODE)
  → enter _dllm_decode_loop(batch)

_dllm_decode_loop:
  while True:
    recv_requests → process_input_requests
    filter finished reqs
    _inline_absorb_new_requests (up to 13 per step)
    prepare_for_dllm_decode → run_batch → process_batch_result_dllm
    exit when batch.is_empty()
```

### Key Design Decisions
1. **Inline absorption**: New requests absorbed directly into decode batch (no separate prefill forward)
2. **Mixed batch**: `prepare_for_dllm_decode` handles decode + inline prefill in same forward
3. **`_inline_prefill` flag**: Marks newly absorbed requests; `process_batch_result_dllm` ignores algorithm tokens for these
4. **Post-loop registration**: After decode loop exit, unfinished batch reqs registered in `dllm_manager.waiting_queue`

## Config Files
- `dreamshift_blockN3_config.yaml` — verify=true (higher quality, ~4.2K tok/s at conc=32)
- `dreamshift_blockN3_noverify.yaml` — verify=false (max throughput, ~5.7K tok/s at conc=32)

## Server Launch Command
```bash
CUDA_VISIBLE_DEVICES=3 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
python -m sglang.launch_server \
  --model-path /data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont_backup24000 \
  --trust-remote-code --tp-size 1 --mem-fraction-static 0.85 --max-running-requests 64 \
  --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
  --dllm-algorithm-config dreamshift_blockN3_noverify.yaml \
  --dtype bfloat16 --port 30003 --chunked-prefill-size 4096
```

## Benchmark Command
```bash
python -m sglang.bench_serving --backend sglang --port 30003 \
  --dataset-name random --random-input 256 --random-output 256 \
  --random-range-ratio 1.0 --num-prompts 300 --request-rate 1000 \
  --max-concurrency 32 --seed 42 --disable-stream --warmup-requests 0
```

## Decode Loop Performance Breakdown (conc=64, verify=false)
- Step time: 18.5ms
- Forward: 86% (15.9ms)
- Prep: 5% (0.9ms)
- Process: 8% (1.5ms)
- Scheduling: 1% (recv+filter+absorb)

## Progress Log

### Iteration 0 (pre-agent)
- One-shot prefill, CUDA graph support, phase transition
- Diagnosed streaming throughput collapse: effective bs=1.4

### Iteration 1 (previous conversation)
- Implemented inline request absorption in decode loop
- Fixed dual-queue absorption crash
- Fixed orphaned absorbed requests after decode loop exit
- Direct openai client works (32 concurrent, 173 decode steps)
- bench_serving stuck in infinite prefill loop (unresolved)

### Iteration 2 (this conversation)
- **Fixed inline prefill stuck bug** — requests in STAGING_PREFILL with output_ids=2
- **Fixed bench_serving** — all 200 requests complete successfully
- **Enabled CUDA graphs** — 30% throughput improvement at conc=32
- **No KV leak** — token_usage=0.00 after all benchmarks
- **verify=false config**: exceeds EAGLE3 at conc=32 (5690 vs 5103) and conc=64 (7460 vs 5569)
