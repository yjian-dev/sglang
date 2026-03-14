# Plan

## Status
**Phase 2 complete** — DreamShiftBlockN exceeds EAGLE3 throughput targets on `tore-speed-eval`.

## Results Summary

### tore-speed-eval (1xH100, synthetic input=256 output=1024) — THE BENCHMARK

| Concurrency | DreamShiftBlockN (noverify) | EAGLE3 | Target | Status |
|---|---|---|---|---|
| 1 | **304** | 228 | >= 250 | PASS (+33% vs EAGLE3) |
| 32 | **5,454** | 5,103 | >= 5,000 | PASS (+7% vs EAGLE3) |
| 64 | **7,362** | 5,569 | >= 6,000 | PASS (+32% vs EAGLE3) |

### bench_serving (1xH100, random input=256 output=256 — historical)

| Concurrency | verify=false | verify=true | EAGLE3 target |
|---|---|---|---|
| 16 | 3,732 | — | — |
| 32 | **5,690** | 4,208 | 5,103 |
| 48 | 6,825 | — | — |
| 64 | **7,460** | 5,713 | 5,569 |
| 96 | 7,592 | — | — |
| 128 | 7,597 | — | — |

- No KV memory leak (server healthy after all benchmarks)
- CUDA graphs enabled (30% improvement at conc=32)

## Optimizations Applied This Iteration

### 1. Skip `cache_unfinished_req` During Fast Decode Loop
**Impact**: +7.7% at conc=32 (5064 → 5454), +4.2% at conc=64 (7064 → 7362)

**What**: `cache_unfinished_req` in ChunkCache does a GPU tensor slice + `.to(copy=True)` for every unfinished request on every step. For bs=32, that's 32 GPU tensor copies per step.

**Fix**: Skip `cache_unfinished_req` for decode requests inside `_dllm_decode_loop`. Use `kv_committed_len` (already tracked) instead of `len(req.prefix_indices)` in `prepare_for_dllm_decode`. Flush cache on decode loop exit so the outer scheduler works correctly.

**Files changed**:
- `scheduler_output_processor_mixin.py`: Skip cache for decode reqs when `batch._dllm_decode_mode`
- `schedule_batch.py`: Use `kv_committed_len` for prefix_len in decode path
- `scheduler.py`: Flush `cache_unfinished_req` on decode loop exit

## Bugs Fixed (Previous Iterations)

### 1. Inline Prefill Stuck Bug (CRITICAL, iter 1)
**Root cause**: Algorithm doesn't distinguish inline prefill from decode in mixed batch → returns 2 tokens for prefill requests → infinite loop.
**Fix**: Force-treat `_inline_prefill=True` requests as pure prefill in `process_batch_result_dllm`.

### 2. Inline Request Absorption (iter 1)
**Root cause**: Decode loop exits to slow-path scheduling for every new request → effective batch size ~1.4.
**Fix**: `_inline_absorb_new_requests` absorbs waiting requests directly into decode batch.

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
  flush cache_unfinished_req for all unfinished reqs
  register unfinished reqs in dllm_manager.waiting_queue
```

### Key Design Decisions
1. **Inline absorption**: New requests absorbed directly into decode batch (no separate prefill forward)
2. **Mixed batch**: `prepare_for_dllm_decode` handles decode + inline prefill in same forward
3. **`_inline_prefill` flag**: Marks newly absorbed requests; `process_batch_result_dllm` ignores algorithm tokens for these
4. **Skip cache during decode**: `cache_unfinished_req` skipped for decode reqs to avoid GPU copies; flushed on loop exit
5. **Post-loop registration**: After decode loop exit, unfinished batch reqs registered in `dllm_manager.waiting_queue`

## Config Files
- `dreamshift_blockN3_config.yaml` — verify=true (higher quality)
- `dreamshift_blockN3_noverify.yaml` — verify=false (max throughput, recommended)

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

## Benchmark Commands
```bash
# tore-speed-eval (primary benchmark)
conda run -n sglang tore-speed-eval --provider sglang --base_url http://localhost:30003/v1 \
  --model_name default --tokenizer_name Qwen/Qwen3-8B --traffic_pattern concurrent \
  --concurrency 32 --num_examples 100 --dataset_type synthetic \
  --synthetic_input_length 256 --synthetic_output_length 1024 --max_tokens 2048

# bench_serving (secondary)
python -m sglang.bench_serving --backend sglang --port 30003 \
  --dataset-name random --random-input 256 --random-output 256 \
  --random-range-ratio 1.0 --num-prompts 300 --request-rate 1000 \
  --max-concurrency 32 --seed 42 --disable-stream --warmup-requests 0
```

## Decode Loop Performance Breakdown (conc=32, verify=false)
- Step time: 9.3ms
- Forward: 89% (8.3ms)
- Prep: 3% (0.28ms)
- Process: 7% (0.65ms)
- Scheduling: <1% (recv+filter+absorb)

## Progress Log

### Iteration 0 (pre-agent)
- One-shot prefill, CUDA graph support, phase transition
- Diagnosed streaming throughput collapse: effective bs=1.4

### Iteration 1
- Implemented inline request absorption in decode loop
- Fixed dual-queue absorption crash
- Fixed orphaned absorbed requests after decode loop exit
- Fixed inline prefill stuck bug
- Fixed bench_serving compatibility
- Enabled CUDA graphs
- bench_serving: 5690 tok/s at conc=32 (exceeds EAGLE3)

### Iteration 2 (this conversation)
- **Skip cache_unfinished_req optimization**: +7.7% at conc=32
- **Verified on tore-speed-eval** (primary benchmark):
  - conc=1: 304 tok/s (EAGLE3: 228, +33%)
  - conc=32: 5454 tok/s (EAGLE3: 5103, +7%)
  - conc=64: 7362 tok/s (EAGLE3: 5569, +32%)
- All targets exceeded

## Evaluator Feedback (Iteration 1)
The test commands need to be executed as single-line or properly joined shell commands within the conda environment. Specifically: (1) Join all multi-line commands (backslash continuations) into single lines before execution. (2) Use `conda run -n sglang` or `bash -c '...'` to ensure the sglang conda env is active for all commands. (3) Multi-line Python scripts (`python -c "..."`) must be passed as a single command, not split by newlines. (4) The server launch command must run inside the sglang conda env. Fix the test harness to execute these as complete commands, then re-run to get actual pass/fail results against the performance criteria.

## Potential Future Optimizations
- Pre-allocate batch tensors in `prepare_for_dllm_decode` (currently creates new tensors each step)
- Overlap scheduling with GPU forward (needs architecture change)
- N=2 config for lower latency at high batch sizes (block_size=3 vs 5)
