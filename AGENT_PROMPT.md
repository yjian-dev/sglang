# Agent Prompt — DreamShift vs SDAR Demo

## Role
You are a demo engineer building a side-by-side comparison demo of DreamShift (our ISD method on SGLang) vs SDAR baseline (on JetEngine). The demo will be screen-recorded as a video for a paper/presentation, showing our throughput advantage at various batch sizes.

## Goal
Create a polished terminal-based demo that:
1. Runs MBPP coding problems through both engines
2. Shows real-time streaming output and TPS comparison
3. Produces throughput comparison data and visualizations

## Models & Engines

### DreamShift N=4 (Ours) — SGLang server
- Checkpoint: `/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-causal-from-b2amc_fixed2_cont`
- Architecture: SDARForCausalLM (same 8B param count as baseline)
- Algorithm: `DreamShiftBlockN` with `dreamshift_blockN4_config.yaml`
- Config: block_size=7, gen_block_size=4, temperature=1.0, top_k=50, top_p=0.95, use_spec_verify=true
- Launch (use GPU 0):
```bash
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export PYTHONPATH=/data/yjian/code/sglang/python:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path /data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-causal-from-b2amc_fixed2_cont \
    --trust-remote-code --tp-size 1 --dtype bfloat16 \
    --mem-fraction-static 0.85 --max-running-requests 24 \
    --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
    --dllm-algorithm-config dreamshift_blockN4_config.yaml --port 30000
```
- API: standard OpenAI chat completions at `http://localhost:30000/v1/chat/completions`
- Supports streaming (`"stream": true`)
- Wait for server health: `curl http://localhost:30000/health`

### SDAR Baseline — JetEngine (offline batch engine)
- Model: `/data/shared/huggingface/SDAR-8B-Chat` (standard JetLM/SDAR-8B-Chat)
- mask_token_id: 151669, block_length: 4
- JetEngine installed at `/data/yjian/code/JetEngine` (already `pip install -e`'d in sglang env)
- IMPORTANT: JetEngine is an OFFLINE batch engine, NOT a server. You call `llm.generate()` directly.
- IMPORTANT: JetEngine requires `torchrun --nproc_per_node=1` to initialize torch.distributed.
- Use GPU 1: `CUDA_VISIBLE_DEVICES=1`
- API:
```python
from jetengine import LLM, SamplingParams
llm = LLM(model_path, enforce_eager=False, tensor_parallel_size=1,
          mask_token_id=151669, block_length=4, max_num_seqs=128,
          max_model_len=4096, gpu_memory_utilization=0.85)
sampling_params = SamplingParams(temperature=1.0, topk=0, topp=1.0,
    max_tokens=2048, remasking_strategy="low_confidence_dynamic",
    dynamic_threshold=0.9, block_length=4, denoising_steps=4)
outputs = llm.generate(prompt_ids_list, sampling_params)
# Each output: {'text': str, 'token_ids': list}
```
- For tokenization: both models use Qwen-based tokenizer with `enable_thinking=True` in chat template
- Existing benchmark data (SDAR-8B-Chat, 1 GPU H100, max_tokens=256):
  bs=1: 160, bs=2: 304, bs=4: 594, bs=8: 1140, bs=16: 1945, bs=32: 3132, bs=64: 4247 tok/s

## Input Data
Use MBPP sanitized test set:
```python
from datasets import load_dataset
ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
# Each item has: prompt, code, test_list, task_id
```
Format prompts like:
```
You are an expert Python programmer. Here is your task:
{prompt}
Your code should pass these tests:
{test_list joined by newlines}
```

## Fairness Requirements (CRITICAL)
- Both use TP=1, single GPU, bfloat16, NO quantization
- Both use temperature=1.0, top_p=0.95 (matching paper settings)
- Both process the EXACT same prompts with the same max_tokens
- SDAR-8B-Chat and our checkpoint are both SDAR architecture, same param count (~8B)
- Do NOT use any tricks to make one faster (no batching tricks, no special flags)

## Demo Design

The demo is a **live TPS traffic monitor** — like a network bandwidth graph — that runs both
engines simultaneously at a fixed concurrency and records per-second throughput over time.

### Core Concept
- Fix concurrency (e.g., 32) — always keep 32 requests in-flight
- Use MBPP problems as input, max_tokens=2048, temperature=1.0, top_p=0.95
- As requests complete, immediately send new ones to maintain concurrency
- Record tokens generated per second for the entire run duration (~60-120 seconds)
- Plot both curves on the same chart — our DreamShift curve climbs high, SDAR stays lower

### Component 1: Data Collection (scripts/demo/collect_tps_timeseries.py)

**DreamShift (SGLang server):**
- Use `tore-speed-eval` style: send streaming requests via aiohttp with fixed concurrency
- As each streaming chunk arrives, record timestamp + token count
- Aggregate into per-second buckets → TPS time series
- OR: poll `http://localhost:30000/get_server_info` → `last_gen_throughput` every 0.5s
  (this is the easiest approach — server already tracks this metric)
- `tore-speed-eval` is installed: `/home/yjian/miniconda3/envs/sglang/bin/tore-speed-eval`
  You can also just use it directly with `--traffic_pattern burst --concurrency 32`
  and parse the streaming output for per-second data.

**SDAR (JetEngine):**
- JetEngine is offline batch — it processes all prompts together, no streaming
- Approach: send batches of 32 prompts, measure total time per batch
- tokens_per_batch / time_per_batch = TPS for that interval
- Run multiple sequential batches to build a time series
- This runs as a separate process via `torchrun` on GPU 1

**Output:** JSON file with `{dreamshift: [{time, tps}, ...], sdar: [{time, tps}, ...]}`

### Component 2: Live TPS Area Chart (scripts/demo/live_tps_chart.py) [HIGH PRIORITY]

Renders the traffic monitor visualization. Reference: `Screenshot 2026-04-05 at 4.25.32 PM.png`

**Visual style:**
- Dark background (dark navy/black #0a0e27)
- Filled area chart with semi-transparent gradient fill under the curve
- DreamShift: bright cyan/blue (#00d4ff) with blue gradient fill
- SDAR baseline: dim red/orange (#ff4444) with subtle fill
- Y-axis: tokens/second, X-axis: time (seconds)
- Large bold text overlay: "Batch Size: 32" and "Throughput: X,XXX tokens/s"
- Dotted grid lines, subtle
- Title: "Large Batch" or "Throughput Comparison"

**Two rendering modes:**
1. **Animated (for video):** matplotlib FuncAnimation → save as MP4 or GIF
   - Chart builds up over time as data streams in
   - Final frame holds for a few seconds showing total throughput
2. **Static (for presentation):** single PNG with both curves overlaid

### Component 3: Single Request Demo (scripts/demo/stream_compare.py) [NICE TO HAVE]
Side-by-side terminal view of one MBPP problem showing text generation.
Lower priority than the TPS chart.

### Component 4: run_demo.sh
One-click script:
1. Start DreamShift server on GPU 0
2. Wait for health
3. Run data collection for both engines in parallel
4. Generate visualization
5. Print summary

## File Structure
```
scripts/demo/
  collect_tps_timeseries.py    # Collect per-second TPS data from both engines
  collect_jetengine_tps.py     # JetEngine worker (runs via torchrun on GPU 1)
  live_tps_chart.py            # Render the traffic monitor chart (animated + static)
  stream_compare.py            # Single-request side-by-side [nice to have]
  run_demo.sh                  # One-click launcher
  results/                     # Output directory
```

## Environment Setup
```bash
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export PYTHONPATH=/data/yjian/code/sglang/python:$PYTHONPATH
```

## Constraints
- All scripts go in `scripts/demo/` directory
- Always update PLAN.md with progress after completing each task
- Test each component individually before integrating
- If server fails, check logs and fix before proceeding
- GPU allocation: GPU 0 = DreamShift, GPU 1 = JetEngine SDAR
- The demo must be runnable from terminal (for screen recording on remote server)
- Keep output clean and visually appealing for recording
