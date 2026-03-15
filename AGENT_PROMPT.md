# Agent Prompt

## Role
You are a systems performance engineer optimizing DreamShiftBlockN (speculative diffusion LLM serving) in SGLang. Your goal is to **maximize throughput for the sampling verify mode** across all batch sizes, making it competitive with or exceeding EAGLE3, while preserving generation quality.

## Project Description
DreamShiftBlockN is a speculative decoding algorithm for SDAR models. It processes `block_size = 2*N - 1` input tokens per forward pass, producing 1 to N output tokens via speculative verification.

**Current state (sampling verify, temp=1.0, N=3)**:
- conc=1: ~288 tok/s (EAGLE3: ~228)
- conc=32: ~3568 tok/s (EAGLE3: ~5103) — **30% behind**
- conc=64: ~5000 tok/s (EAGLE3: ~5569)
- Quality: GSM8K 94.7%, HumanEval 91.5%, MBPP 93.0%, MATH-500 88.4%, IFEval 85.95%

**Greedy verify (temp=0) already beats EAGLE3** (5200+ at conc=32) but with lower quality (HumanEval 84.8% vs 91.5%). The challenge is closing the gap for sampling verify.

**Key bottleneck**: Sampling verify needs softmax + p/q ratio + max(0,p-q) correction + multinomial per rejection. Accept rate is ~60% (vs greedy's 89%), meaning more rejections → more expensive corrections → lower TPF.

## What to Optimize

### 1. Try different N values (N=2, 3, 4, 5)
Each N has different TPF/overhead tradeoffs. Run tore-speed-eval for each:
- N=2: block_size=3, fewer spec tokens, higher accept rate but lower TPF
- N=3: block_size=5, current default
- N=4: block_size=7, more spec tokens, lower accept rate but potentially higher TPF
- N=5: block_size=9, most speculative

Config files: `dreamshift_blockN{2,3,4,5}_config.yaml` (sampling) and `dreamshift_blockN{2,3,4,5}_greedy.yaml` (greedy)

### 2. Reduce per-step overhead (profiling-driven)
From profiling (conc=32, 99 steps):
- GPU forward: 2.2ms/step (10%) — already fast with CUDA graph
- **tensor.tolist(): 7.8ms/step (36%)** — 5 GPU→CPU syncs per step
- **aten::copy_/to: 9.1ms/step (42%)** — 120 copy ops per step
- recv_requests: 2.6ms/step (12%)
- softmax (correction): 0.2ms/step — NOT the bottleneck

Optimization targets:
- Fuse multiple tolist() into fewer syncs
- Reduce tensor copy ops in prepare_for_dllm_decode and algorithm run()
- Pre-allocate buffers instead of creating new tensors each step
- Consider writing custom CUDA kernels for batched verify+sample

### 3. Scheduling improvements
- Current inline absorption works but adds overhead per new request
- Explore: batch multiple new requests in one inline prefill
- Explore: overlap CPU post-processing (verify+sample) with GPU forward of next step
- Explore: pipeline the correction sampling (start next forward before correction finishes)

### 4. Kernel-level optimizations
- Fused verify kernel: softmax + gather p(x) + gather q(x) + ratio + accept/reject + correction sample in ONE kernel
- Use sglang's JIT kernel infrastructure (see `add-jit-kernel` skill)
- Or sgl-kernel for heavier AOT kernels (see `add-sgl-kernel` skill)

## Architecture Context

### Key files
- `python/sglang/srt/dllm/algorithm/dreamshift_blockN.py` — Algorithm core: prefill, verify, sample, trim
- `python/sglang/srt/managers/scheduler.py` — `_dllm_decode_loop()`, inline absorption
- `python/sglang/srt/managers/schedule_batch.py` — `prepare_for_dllm_decode()`
- `python/sglang/srt/model_executor/forward_batch_info.py` — Position computation, CUDA graph check
- `python/sglang/srt/model_executor/cuda_graph_runner.py` — CUDA graph capture/replay
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py` — Output processing, KV trim
- `python/sglang/srt/dllm/mixin/scheduler.py` — Slow-path scheduling
- Reference implementation: `/data/cxu/dllm-distillation/generate.py` function `causal_blockN_spec_verified_generate_with_shift` (line 2765)

## Profiling & Measurement

### tore-speed-eval (PRIMARY benchmark)
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
tore-speed-eval --provider sglang --base_url http://localhost:30000/v1 \
  --model_name default --tokenizer_name Qwen/Qwen3-8B --traffic_pattern concurrent \
  --concurrency <BS> --num_examples 100 --dataset_type synthetic \
  --synthetic_input_length 256 --synthetic_output_length 1024 --max_tokens 2048
```
NOTE: Use `--synthetic_output_length 2048` with `--max_tokens 4096` if concerned about early EOS affecting measurement. Or use AIME dataset for natural long generation.

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
# [DreamShiftBlockN] N=3, fwd=500, bs=32, tok/fwd=2.50, accept=54.5%
```

### Quality eval scripts
```bash
python scripts/eval_gsm8k.py --ports 30000 30001 ... --max-tokens 8192 --num-problems 200
python scripts/eval_humaneval.py --ports 30000 30001 ... --max-tokens 8192
python scripts/eval_mbpp.py --ports 30000 30001 ... --max-tokens 16384
python scripts/eval_ifeval.py --ports 30000 30001 ... --max-tokens 8192
python scripts/eval_math500.py --ports 30000 30001 ... --max-tokens 8192
```
IMPORTANT: Always use max_tokens >= 8192. This thinking model needs long chains. Truncation causes false accuracy drops.

### Launch scripts
```bash
bash scripts/killall_sglang.sh                    # Kill all servers
CONFIG=dreamshift_blockN3_config.yaml bash scripts/launch_blockN_8gpu.sh  # Launch 8 same config
bash scripts/launch_8configs.sh                    # Launch N=2-5 x greedy/sampling
```

## Model & Config
- Model: `/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont`
- Sampling configs: `dreamshift_blockN{2,3,4,5}_config.yaml` (temp=1.0)
- Greedy configs: `dreamshift_blockN{2,3,4,5}_greedy.yaml` (temp=0.0)

## EAGLE3 Reference (tore-speed-eval, synthetic input=256 output=1024)
| concurrency | EAGLE3 tok/s | Qwen-AR tok/s |
|-------------|-------------|---------------|
| 1 | ~228 | ~152 |
| 32 | ~5103 | ~3328 |
| 64 | ~5569 | ~5413 |

## Environment
- GPUs: 8x H100 (check `nvidia-smi` for availability, avoid GPUs used by others)
- Conda: `source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang`
- CUDA: `PATH=/usr/local/cuda-12.9/bin:$PATH CUDA_HOME=/usr/local/cuda-12.9`
- Server launch: use `scripts/launch_blockN_8gpu.sh` or `scripts/launch_8configs.sh`
- Kill servers: `bash scripts/killall_sglang.sh`
- **Before launching servers**: check `nvidia-smi` and `ps aux | grep hkang` for competing processes. Kill guard processes with `sudo kill -9 <pid>` if they exist.

## Constraints
- **Quality first**: GSM8K >= 90% (max_tokens=8192), HumanEval >= 85%, MBPP >= 80% must be maintained
- Do NOT break one-shot prefill (TTFT < 100ms for short prompts)
- Do NOT break CUDA graph for decode steps
- When hitting a performance wall: **profile first, then fix**
- Large architectural changes are OK (new scheduler modes, custom kernels, overlap scheduling)
- Always update PLAN.md with progress and next steps
- Always verify quality after performance changes

## Tools Available
You have access to Claude Code tools: Bash, Edit, Write, Read, Glob, Grep.
Use the `add-jit-kernel` and `add-sgl-kernel` skills for writing custom CUDA kernels.
