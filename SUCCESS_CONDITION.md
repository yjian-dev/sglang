# Success Conditions

## Primary Metric: tore-speed-eval (sampling verify mode)

All performance numbers come from `tore-speed-eval` with sampling verify configs (temperature=1.0).

```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
tore-speed-eval --provider sglang --base_url http://localhost:30000/v1 \
  --model_name default --tokenizer_name Qwen/Qwen3-8B --traffic_pattern concurrent \
  --concurrency <CONCURRENCY> --num_examples 100 --dataset_type synthetic \
  --synthetic_input_length 256 --synthetic_output_length 1024 --max_tokens 2048
```

## Criteria

### 1. Throughput: Match or exceed EAGLE3 with sampling verify
- concurrency=32: Job-level tok/s **>= 5000** (EAGLE3 = 5103, current sampling = 3568)
- concurrency=64: Job-level tok/s **>= 5600** (EAGLE3 = 5569)
- Any N from 2-5 is acceptable; find the optimal one
- Must use sampling verify (temperature=1.0, use_spec_verify=true)

### 2. Throughput: bs=1 advantage maintained
- concurrency=1: Job-level tok/s **>= 200** (current sampling = ~288)

### 3. TTFT: Low latency
- TTFT median < 100ms at concurrency=1
- TTFT median < 500ms at concurrency=32

### 4. Quality: Must be preserved (sampling verify)
- GSM8K (200 problems, max_tokens=8192): **>= 90%**
- HumanEval (164 problems, max_tokens=8192): **>= 85%**
- MBPP (257 problems, max_tokens=16384): **>= 80%**
- IFEval (541 prompts, max_tokens=8192): strict **>= 80%**
- 0 failed requests in all benchmarks
- IMPORTANT: always use max_tokens >= 8192 (16384 for MBPP). Truncation causes false failures.
- If accuracy seems low, check if outputs are truncated (no </think>, no \boxed{}) before assuming algorithm bug.

### 5. Algorithm health: TPF and accept rate
- Check server logs: `strings /tmp/sglang_gpu0.log | grep "tok/fwd"`
- N=2: TPF >= 1.3
- N=3: TPF >= 1.8
- N=4: TPF >= 2.0
- Accept rate >= 40% for all N values

## Test Commands

### Setup: Kill old servers, launch with chosen config
```bash
bash scripts/killall_sglang.sh
sleep 3
CONFIG=dreamshift_blockN3_config.yaml bash scripts/launch_blockN_8gpu.sh
```

### Quick sanity check
```bash
python scripts/stream_demo.py --url http://localhost:30000 --prompt 'What is 15*23+7?' --max-tokens 256
```

### Throughput benchmarks (vary concurrency)
```bash
for C in 1 4 8 16 32 64; do
  echo "=== concurrency=$C ==="
  tore-speed-eval --provider sglang --base_url http://localhost:30000/v1 \
    --model_name default --tokenizer_name Qwen/Qwen3-8B --traffic_pattern concurrent \
    --concurrency $C --num_examples 100 --dataset_type synthetic \
    --synthetic_input_length 256 --synthetic_output_length 1024 --max_tokens 2048
done
```

### Quality benchmarks (use all 8 GPUs)
```bash
PORTS="30000 30001 30002 30003 30004 30005 30006 30007"
python scripts/eval_gsm8k.py --ports $PORTS --num-problems 200 --max-tokens 8192
python scripts/eval_humaneval.py --ports $PORTS --max-tokens 8192
python scripts/eval_mbpp.py --ports $PORTS --max-tokens 16384
python scripts/eval_ifeval.py --ports $PORTS --max-tokens 8192
```

### TPF check
```bash
strings /tmp/sglang_gpu0.log | grep "tok/fwd" | tail -5
```

## Debugging Methodology
When performance doesn't meet targets:
1. **Profile first**: `POST /start_profile` to capture GPU traces
2. **Parse traces**: Use Python trace parser to find top time consumers
3. **Check batch utilization**: `strings /tmp/sglang_gpu0.log | grep "DLLM decode loop"`
4. **Check TPF**: If TPF drops, the algorithm logic may be broken
5. **Check quality**: Run GSM8K 30Q quick test before full benchmark
6. Only after quantitative understanding, attempt a fix

## Notes
- EAGLE3 becomes compute-bound at conc>=48 (1.03x over AR at conc=64). This is where DreamShiftBlockN should win.
- Sampling verify is slower than greedy due to: (a) lower accept rate 60% vs 89%, (b) softmax+multinomial per rejection
- The main optimization vectors are: reducing per-step CPU overhead, improving batch efficiency, writing fused kernels
- Large architectural changes are encouraged: new scheduler modes, overlap scheduling, custom kernels
- Reference implementation: `/data/cxu/dllm-distillation/generate.py` `causal_blockN_spec_verified_generate_with_shift`
