# Agent Prompt

## Role
You are a systems performance engineer optimizing DreamShiftBlockN (speculative diffusion LLM serving) in SGLang. Your goal is to **maximize throughput for TP=4 (4-GPU tensor parallel) deployment** across all concurrency levels, while preserving generation quality.

## Project Description
DreamShiftBlockN is a speculative decoding algorithm for SDAR models. It processes `block_size = 2*N - 1` input tokens per forward pass, producing 1 to N output tokens via speculative verification.

**Current state (TP=1, 1x H100, tore-speed-eval, AIME 90 problems, OSL=2048)**:

| Conc | Qwen-AR | EAGLE3-tengyunw | N=3 sample | N=3 greedy | N=5 fp8 |
|------|---------|-----------------|-----------|-----------|---------|
| 1    | 149     | 204             | 247       | 272       | 328     |
| 2    | 290     | 378             | 464       | 521       | 594     |
| 4    | 565     | 699             | 846       | 967       | 1,072   |
| 8    | 1,086   | 1,269           | 1,550     | 1,770     | 1,605   |
| 16   | 2,017   | 2,160           | 2,675     | 2,999     | 2,380   |
| 32   | 3,613   | 3,414           | 4,209     | 4,736     | 3,745   |
| 64   | 5,794   | 4,676           | 5,708     | 6,414     | 4,862   |

**Key observations**:
- N=3 greedy is fastest at C>=8, reaching 6,414 tok/s at C=64
- N=5 fp8 is fastest at C=1~4 per-request (328 tok/s at C=1)
- Qwen-AR has overlap scheduler enabled → 5,794 tok/s at C=64
- EAGLE3-tengyunw is slowest at high concurrency

**Quality (Qwen3-8B AR, AIME 2024)**:
- Single sample: 76.7% average (22-24/30)
- 9-sample majority vote: 80.0% (24/30)

**TP=4 goal**: Maximize throughput with 4 GPUs. Current implementation supports TP=4 but may not be optimal. Need to measure, profile, and optimize.

## What to Optimize

### 1. Baseline TP=4 measurement
- Launch server with `--tp-size 4` for N=3 sample, N=3 greedy, N=5 fp8
- Measure throughput at concurrency 1, 2, 4, 8, 16, 32, 64
- Compare with Qwen-AR TP=4 and EAGLE3 TP=4
- Identify compute-bound vs memory-bound crossover for TP=4

### 2. Profile TP=4 overhead
- Use torch profiler to capture traces at different concurrency levels
- Identify: NCCL all-reduce time, GPU idle bubbles, CPU scheduling overhead
- Check if DreamShift's extra operations (verify, correction sampling) add disproportionate overhead in TP setting
- Compare forward pass time TP=1 vs TP=4 to measure TP efficiency

### 3. Optimize TP=4 communication
- All-reduce is on the critical path for every forward pass
- With DreamShift's smaller effective batch (block_size tokens), all-reduce overhead is proportionally larger
- Explore: overlap all-reduce with computation
- Explore: reduce number of all-reduce calls per step
- Explore: use NCCL async where possible

### 4. Reduce per-step overhead (profiling-driven)
Known bottlenecks from TP=1 profiling:
- **tensor.tolist()**: GPU→CPU syncs (5+ per step)
- **aten::copy_/to**: tensor copy ops (~120 per step)
- **recv_requests overhead**: scheduler polling
- These may be amplified in TP=4 due to synchronization

Optimization targets:
- Fuse multiple tolist() into fewer syncs
- Reduce tensor copy ops in prepare_for_dllm_decode and algorithm run()
- Pre-allocate buffers instead of creating new tensors each step
- Consider writing custom CUDA kernels for batched verify+sample

### 5. Scheduling improvements for TP=4
- Current inline absorption works but adds overhead per new request
- TP=4 has 4x more GPU memory → can hold more concurrent requests
- Explore: larger max_running_requests
- Explore: batch multiple new requests in one inline prefill
- Explore: overlap CPU post-processing with GPU forward

### 6. Kernel-level optimizations
- Fused verify kernel: softmax + gather p(x) + gather q(x) + ratio + accept/reject + correction sample in ONE kernel
- Use sglang's JIT kernel infrastructure (see `add-jit-kernel` skill)
- Ensure kernels work correctly with TP (no cross-GPU dependencies in verify/sample)

### 7. Single request latency (bs=1 optimization)
- At TP=4, single request should be faster due to reduced per-token compute time
- But all-reduce overhead may limit gains
- Profile and optimize the critical path for bs=1
- Target: **>= 700 tok/s** (stretch), minimum 500 tok/s (TP=1 best is 328 tok/s with N=5 fp8)

## Architecture Context

### Key files
- `python/sglang/srt/dllm/algorithm/dreamshift_blockN.py` — Algorithm core: prefill, verify, sample, trim
- `python/sglang/srt/dllm/algorithm/fused_verify_kernel.py` — Fused verify CUDA kernel
- `python/sglang/srt/managers/scheduler.py` — `_dllm_decode_loop()`, inline absorption
- `python/sglang/srt/managers/schedule_batch.py` — `prepare_for_dllm_decode()`
- `python/sglang/srt/model_executor/forward_batch_info.py` — Position computation, CUDA graph check
- `python/sglang/srt/model_executor/cuda_graph_runner.py` — CUDA graph capture/replay
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py` — Output processing, KV trim
- `python/sglang/srt/dllm/mixin/scheduler.py` — Slow-path scheduling
- Reference implementation: `/data/cxu/dllm-distillation/generate.py` function `causal_blockN_spec_verified_generate_with_shift`

## Profiling & Measurement

### tore-speed-eval (PRIMARY benchmark)
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
TOKENIZER=/data/yjian/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
tore-speed-eval --provider sglang --base_url http://localhost:30000/v1 \
  --model_name default --tokenizer_name $TOKENIZER --traffic_pattern burst \
  --concurrency <C> --num_examples 90 --max_tokens 2048 \
  --dataset_type jsonl --jsonl_input_path /tmp/aime2024.jsonl \
  --jsonl_convert_to_chat_request_format true \
  --temperature 1.0 --top_p 0.95
```

### Server-side torch profiler
```python
import requests
requests.post("http://localhost:30000/start_profile", json={
    "output_dir": "/tmp/traces", "num_steps": 50,
    "activities": ["CPU", "GPU"], "profile_prefix": "dllm"
})
# ... run workload ...
requests.post("http://localhost:30000/stop_profile")
```

### Parse traces programmatically
```python
import gzip, json
from collections import defaultdict
with gzip.open("/tmp/traces/dllm-TP-0.trace.json.gz", "rb") as f:
    data = json.loads(f.read().decode("utf-8"))
by_name = defaultdict(list)
for ev in data["traceEvents"]:
    if ev.get("ph") == "X":
        by_name[ev["name"]].append(ev["dur"])
for name in sorted(by_name, key=lambda n: -sum(by_name[n]))[:30]:
    durs = by_name[name]
    print(f"{name}: {len(durs)} calls, avg {sum(durs)/len(durs):.0f}us, total {sum(durs)/1e6:.2f}s")
```

### Algorithm stats (in server log)
```bash
strings /tmp/sglang_gpu0.log | grep "tok/fwd"
```

### Quality eval scripts
```bash
PORTS="30000 30004"  # 2x TP=4 servers

# Quick iteration (use these first, ~2-5 min each):
python scripts/eval_ifeval.py --ports $PORTS --max-tokens 8192
python scripts/eval_gsm8k.py --ports $PORTS --num-problems 200 --max-tokens 8192

# Full validation (later, when optimization is stable):
python scripts/eval_humaneval.py --ports $PORTS --max-tokens 8192
python scripts/eval_mbpp.py --ports $PORTS --max-tokens 16384
python scripts/eval_math500.py --ports $PORTS --max-tokens 8192
python scripts/eval_aime.py --year 2024 --ports $PORTS --max-tokens 32768 --temperature 1.0 --top-p 0.95 --top-k 50 --output-dir /tmp/aime_results
```
IMPORTANT: Always use max_tokens >= 8192. This thinking model needs long chains. Truncation causes false accuracy drops.
NOTE: IFEval and GSM8K are fast (~2-5 min). Use them for rapid quality checks during optimization. Save AIME/HumanEval/MBPP for final validation.

### Launch scripts
```bash
# Kill all servers
bash scripts/killall_sglang.sh

# Launch 8x TP=1 DreamShift servers (one per GPU)
CONFIG=dreamshift_blockN3_config.yaml bash scripts/launch_blockN_8gpu.sh

# Launch 2x TP=4 DreamShift servers (GPUs 0-3 and 4-7)
# Example for port 30000 on GPUs 0-3:
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path $SDAR --trust-remote-code --tp-size 4 \
  --mem-fraction-static 0.85 --max-running-requests 128 \
  --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
  --dllm-algorithm-config dreamshift_blockN3_config.yaml \
  --dtype bfloat16 --port 30000 --chunked-prefill-size 4096

# Launch 8x Qwen3-8B AR servers (one per GPU)
bash scripts/launch_qwen_8gpu.sh

# Launch 2x TP=4 Qwen3-8B AR servers
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B --trust-remote-code --tp-size 4 \
  --mem-fraction-static 0.85 --max-running-requests 128 \
  --attention-backend flashinfer \
  --dtype bfloat16 --port 30000 --chunked-prefill-size 4096
```

## Model & Config
- SDAR Model: `/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont`
  - Alternative (HF): `YYF42/sdar-qwen3-8b-ar-b2-cont-epoch1`
- Qwen3-8B: `Qwen/Qwen3-8B` (local cache at `/home/yjian/.cache/huggingface/hub/models--Qwen--Qwen3-8B/`)
- Sampling configs: `dreamshift_blockN{2,3,4,5}_config.yaml` (temp=1.0, use_spec_verify=true)
- Greedy configs: `dreamshift_blockN{2,3,4,5}_greedy.yaml` (temp=0.0)
- AIME dataset JSONL: `/tmp/aime2024.jsonl` (90 problems from AI-MO/aimo-validation-aime)

## TP=1 Baselines (for comparison)

### Throughput (tore-speed-eval, 1x H100, AIME burst mode)
| Conc | Qwen-AR | EAGLE3 | N=3 sample | N=3 greedy | N=5 fp8 |
|------|---------|--------|-----------|-----------|---------|
| 1    | 149     | 204    | 247       | 272       | 328     |
| 4    | 565     | 699    | 846       | 967       | 1,072   |
| 8    | 1,086   | 1,269  | 1,550     | 1,770     | 1,605   |
| 16   | 2,017   | 2,160  | 2,675     | 2,999     | 2,380   |
| 32   | 3,613   | 3,414  | 4,209     | 4,736     | 3,745   |
| 64   | 5,794   | 4,676  | 5,708     | 6,414     | 4,862   |

### Memory-bound analysis (TP=1)
- H100 memory bandwidth: 3.35 TB/s
- Memory-compute crossover: ~295 batch tokens
- At conc<=32 (160 batch tokens for N=3): memory-bound → reducing tokens doesn't help
- At conc>=64: transitioning to compute-bound
- **TP=4 changes this**: 4x compute, shared memory bandwidth → crossover shifts lower

## Environment
- GPUs: 8x H100 80GB HBM3
- Conda: `source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang`
- CUDA: use `/usr/local/cuda-12.9` (NOT 13.1) for flashinfer compatibility
  - `export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH`
  - `export CUDA_HOME=/usr/local/cuda-12.9`
- Server launch: use scripts in `scripts/` or manual commands above
- Kill servers: `bash scripts/killall_sglang.sh`

## Constraints
- **Quality first**: AIME 2024 >= 73% single-shot, GSM8K >= 90%, HumanEval >= 85%, MBPP >= 80% must be maintained
- Do NOT break one-shot prefill (TTFT < 100ms for short prompts)
- Do NOT break CUDA graph for decode steps
- When hitting a performance wall: **profile first, then fix**
- Large architectural changes are OK (new scheduler modes, custom kernels, overlap scheduling)
- Always update PLAN.md with progress and next steps
- Always verify quality after performance changes
- Commit working improvements incrementally

## Tools Available
You have access to Claude Code tools: Bash, Edit, Write, Read, Glob, Grep.
Use the `add-jit-kernel` and `add-sgl-kernel` skills for writing custom CUDA kernels.
