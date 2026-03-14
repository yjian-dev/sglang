# Success Conditions

## Primary Metric: tore-speed-eval

All performance numbers come from `tore-speed-eval`. This is the only benchmark that matters.

```bash
conda run -n sglang tore-speed-eval --provider sglang --base_url http://localhost:30001/v1 \
  --model_name default --tokenizer_name Qwen/Qwen3-8B --traffic_pattern concurrent \
  --concurrency <CONCURRENCY> --num_examples 100 --dataset_type synthetic \
  --synthetic_input_length 256 --synthetic_output_length 1024 --max_tokens 2048
```

## Criteria

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

### 5. Algorithm health: TPF and accept rate
- N=3: TPF >= 2.0, accept rate >= 40% (check server log: `grep "tok/fwd"`)
- N=4: TPF >= 2.2, accept rate >= 40%
- These confirm the algorithm is working correctly, not just fast but wrong

## Test Commands

```bash
# Setup
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export CUDA_VISIBLE_DEVICES=1
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

# 1. Start server (N=3)
python -m sglang.launch_server \
  --model-path /data/cxu/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_backup32000 \
  --trust-remote-code --tp-size 1 --mem-fraction-static 0.85 --max-running-requests 64 \
  --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
  --dllm-algorithm-config dreamshift_blockN3_config.yaml \
  --dtype bfloat16 --port 30001 > /tmp/dllm_test_server.log 2>&1 &
for i in $(seq 1 80); do curl -sf http://localhost:30001/health && break; sleep 5; done

# 2. Correctness: math
python scripts/stream_demo.py --url http://localhost:30001 --prompt "What is 15*23+7?" --max-tokens 256 2>&1 | tee /tmp/test_math.txt
grep -q "352" /tmp/test_math.txt

# 3. Correctness: fluency
python scripts/stream_demo.py --url http://localhost:30001 --prompt "Write a short poem about the ocean" --max-tokens 256 2>&1 | tee /tmp/test_poem.txt
python -c "
with open('/tmp/test_poem.txt') as f: text = f.read()
import re; m = re.search(r'(\d+) tokens', text)
assert m and int(m.group(1)) >= 50, f'Too few tokens'
print('Fluency OK')
"

# 4. TPF check
python -c "
import subprocess, re
out = subprocess.check_output(['strings', '/tmp/dllm_test_server.log']).decode()
lines = [l for l in out.split('\n') if 'tok/fwd=' in l]
assert lines, 'No TPF stats in server log'
m = re.search(r'tok/fwd=([\d.]+)', lines[-1])
tpf = float(m.group(1))
m2 = re.search(r'accept=([\d.]+)%', lines[-1])
accept = float(m2.group(1))
print(f'TPF={tpf:.2f}, accept={accept:.1f}%')
assert tpf >= 2.0, f'TPF {tpf} < 2.0'
assert accept >= 40, f'Accept {accept}% < 40%'
print('Algorithm health OK')
"

# 5. tore-speed-eval concurrency=32 (PRIMARY)
conda run -n sglang tore-speed-eval --provider sglang --base_url http://localhost:30001/v1 \
  --model_name default --tokenizer_name Qwen/Qwen3-8B --traffic_pattern concurrent \
  --concurrency 32 --num_examples 100 --dataset_type synthetic \
  --synthetic_input_length 256 --synthetic_output_length 1024 --max_tokens 2048 2>&1 | \
  tee /tmp/tore_result_32.txt

python -c "
import re
with open('/tmp/tore_result_32.txt') as f: text = f.read()

# Check throughput
m = re.search(r'Job-level tokens/s \(decode\):\s+([\d.]+)', text)
assert m, 'No Job-level tokens/s found'
tps = float(m.group(1))
print(f'concurrency=32 tok/s: {tps:.0f}')
assert tps >= 5100, f'{tps:.0f} < 5100 (EAGLE3 baseline)'

# Check TTFT
m2 = re.search(r'TTFT median \(ms\):\s+([\d.]+)', text)
if m2:
    ttft = float(m2.group(1))
    print(f'TTFT median: {ttft:.0f}ms')
    assert ttft < 500, f'TTFT {ttft:.0f}ms > 500ms'

# Check no failures
m3 = re.search(r'Num. of failed requests:\s+(\d+)', text)
if m3:
    assert int(m3.group(1)) == 0, f'{m3.group(1)} failed requests'

print('concurrency=32 PASSED')
"

# 6. Cleanup
lsof -ti :30001 | xargs -r kill -9 2>/dev/null
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
