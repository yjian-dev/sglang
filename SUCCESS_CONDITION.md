# Success Conditions

## Primary Metric: tore-speed-eval

All performance numbers come from `tore-speed-eval`. This is the only benchmark that matters.

## Criteria

### 0. We should use the verify version. The Non-verify is not our goal.

### 1. Throughput: Beat EAGLE3 at concurrency >= 32
- concurrency=32: DreamShiftBlockN Job-level tok/s **> 5100** (EAGLE3 = 5103)
- concurrency=64: DreamShiftBlockN Job-level tok/s **> 5600** (EAGLE3 ≈ 5569, compute-bound)
- Any N from 2-5 is acceptable; find the best one

### 2. Throughput: Maintain bs=1 advantage
- concurrency=1: DreamShiftBlockN Job-level tok/s **>= 200** (EAGLE3 = ~228, AR = ~152)

### 3. TTFT: Low latency
- TTFT median < 100ms at concurrency=1
- TTFT median < 500ms at concurrency=32
- TTFT P99 < 3000ms at concurrency=32

### 4. Correctness: Fluent generation (quality ≈ Qwen3-8B)
- `stream_demo.py --prompt "What is 15*23+7?"` produces correct answer (352)
- `stream_demo.py --prompt "Write a short poem about the ocean"` produces fluent, coherent text
- 0 failed requests in all tore-speed-eval runs
- Random 30 Question in GSM8K should have success rate > 90%

### 5. Algorithm health: TPF and accept rate
- N=3: TPF >= 2.0, accept rate >= 40% (check server log: `grep "tok/fwd"`)
- N=4: TPF >= 2.2, accept rate >= 40%
- These confirm the algorithm is working correctly, not just fast but wrong

## Test Commands

Each command below is a self-contained single-line `bash -lc` invocation. Execute them sequentially.
For step 4, please use 10-20 GSM8K math questions to verify.

### Step 1: Kill any existing server on port 30001
```bash
bash -lc 'lsof -ti :30001 | xargs -r kill -9 2>/dev/null; sleep 2; echo "port cleared"'
```

### Step 2: Start server (GPU 1, port 30001, N=3 verify mode)
```bash
bash -lc 'source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang && CUDA_VISIBLE_DEVICES=1 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 PATH=/usr/local/cuda-12.9/bin:$PATH CUDA_HOME=/usr/local/cuda-12.9 python -m sglang.launch_server --model-path /data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont --trust-remote-code --tp-size 1 --mem-fraction-static 0.85 --max-running-requests 64 --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN --dllm-algorithm-config dreamshift_blockN3_verify_fast7.yaml --dtype bfloat16 --port 30001 --chunked-prefill-size 4096 > /tmp/dllm_test_server.log 2>&1 &'
```

### Step 3: Wait for server health (up to 300s)
```bash
bash -lc 'for i in $(seq 1 300); do if curl -sf http://localhost:30001/health > /dev/null 2>&1; then echo "Server ready after ${i}s"; exit 0; fi; sleep 1; done; echo "TIMEOUT"; exit 1'
```

### Step 4: Correctness — math

### Step 5: Correctness — fluency
```bash
bash -lc 'source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang && python scripts/stream_demo.py --url http://localhost:30001 --prompt "Write a short poem about the ocean" --max-tokens 256 2>&1 | tee /tmp/test_poem.txt && echo "FLUENCY TEST: check output above for coherent poem"'
```

### Step 6: Algorithm health — TPF check
```bash
bash -lc 'python /tmp/check_tpf.py'
```

Where `/tmp/check_tpf.py` is:
```bash
bash -lc 'cat > /tmp/check_tpf.py << '"'"'PYEOF'"'"'
import re
with open("/tmp/dllm_test_server.log") as f:
    text = f.read()
lines = [l for l in text.split("\n") if "tok/fwd=" in l]
if not lines:
    print("No TPF stats yet (need more forwards). SKIP.")
    exit(0)
m = re.search(r"tok/fwd=([\d.]+)", lines[-1])
tpf = float(m.group(1))
m2 = re.search(r"accept=([\d.]+)%", lines[-1])
accept = float(m2.group(1))
print(f"TPF={tpf:.2f}, accept={accept:.1f}%")
assert tpf >= 2.0, f"TPF {tpf} < 2.0"
assert accept >= 40, f"Accept {accept}% < 40%"
print("Algorithm health OK")
PYEOF'
```

### Step 7: tore-speed-eval concurrency=1 (target: >= 200 tok/s)
```bash
bash -lc 'source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang && tore-speed-eval --provider sglang --base_url http://localhost:30001/v1 --model_name default --tokenizer_name Qwen/Qwen3-8B --traffic_pattern concurrent --concurrency 1 --num_examples 50 --dataset_type synthetic --synthetic_input_length 256 --synthetic_output_length 1024 --max_tokens 2048 2>&1 | tee /tmp/tore_result_1.txt'
```

### Step 8: Validate concurrency=1 result
```bash
bash -lc 'python /tmp/validate_tore.py /tmp/tore_result_1.txt 200'
```

### Step 9: tore-speed-eval concurrency=32 (target: >= 5100 tok/s)
```bash
bash -lc 'source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang && tore-speed-eval --provider sglang --base_url http://localhost:30001/v1 --model_name default --tokenizer_name Qwen/Qwen3-8B --traffic_pattern concurrent --concurrency 32 --num_examples 100 --dataset_type synthetic --synthetic_input_length 256 --synthetic_output_length 1024 --max_tokens 2048 2>&1 | tee /tmp/tore_result_32.txt'
```

### Step 10: Validate concurrency=32 result
```bash
bash -lc 'python /tmp/validate_tore.py /tmp/tore_result_32.txt 5100'
```

### Step 11: tore-speed-eval concurrency=64 (target: >= 5600 tok/s)
```bash
bash -lc 'source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang && tore-speed-eval --provider sglang --base_url http://localhost:30001/v1 --model_name default --tokenizer_name Qwen/Qwen3-8B --traffic_pattern concurrent --concurrency 64 --num_examples 100 --dataset_type synthetic --synthetic_input_length 256 --synthetic_output_length 1024 --max_tokens 2048 2>&1 | tee /tmp/tore_result_64.txt'
```

### Step 12: Validate concurrency=64 result
```bash
bash -lc 'python /tmp/validate_tore.py /tmp/tore_result_64.txt 5600'
```

### Step 13: Cleanup — kill server
```bash
bash -lc 'lsof -ti :30001 | xargs -r kill -9 2>/dev/null; echo "cleaned up"'
```

### Helper: Create validation script (run before Step 8)
```bash
bash -lc 'cat > /tmp/validate_tore.py << '"'"'PYEOF'"'"'
import re, sys
filename = sys.argv[1]
target = float(sys.argv[2])
with open(filename) as f:
    text = f.read()
m = re.search(r"Job-level tokens/s \(decode\):\s+([\d.]+)", text)
if not m:
    print(f"ERROR: No Job-level tokens/s found in {filename}")
    sys.exit(1)
tps = float(m.group(1))
print(f"Job-level tok/s: {tps:.0f} (target: >= {target:.0f})")
if tps >= target:
    print("PASSED")
else:
    print(f"FAILED: {tps:.0f} < {target:.0f}")
    sys.exit(1)
m2 = re.search(r"Num failed requests:\s+(\d+)", text)
if m2 and int(m2.group(1)) > 0:
    print(f"WARNING: {m2.group(1)} failed requests")
m3 = re.search(r"TTFT median \(ms\):\s+([\d.]+)", text)
if m3:
    print(f"TTFT median: {float(m3.group(1)):.0f}ms")
PYEOF'
```

## Debugging Methodology
When performance doesn't meet targets:
1. **Profile first**: Use `POST /start_profile` to capture GPU traces during the workload
2. **Parse traces**: Use the Python trace parser (see AGENT_PROMPT.md) to identify top time consumers
3. **Check batch utilization**: `strings /tmp/server.log | grep "DLLM decode loop"` — are exits frequent? Low step counts?
4. **Check effective batch size**: Count total steps vs total tokens in server log
5. **Check TPF**: `strings /tmp/server.log | grep "tok/fwd"` — if TPF drops, algorithm logic is broken
6. Only after understanding the bottleneck quantitatively should you attempt a fix

## Notes
- All test commands must exit with code 0 for success
- EAGLE3 reference: 5103 tok/s at concurrency=32, ~5569 at concurrency=64
- At concurrency>=48, EAGLE3 becomes compute-bound (1.03x over AR). DreamShiftBlockN should clearly win here.
- Model quality is assumed ≈ Qwen3-8B (distilled from it); no accuracy testing needed, just verify fluent coherent output
- If N=3 doesn't meet targets, try N=2 or N=4 — they have different TPF/overhead tradeoffs
- Config: `dreamshift_blockN3_verify_fast7.yaml` (verify mode with fast top-K check, recommended)
