# Success Conditions

## Primary Metric: TP=4 throughput (tore-speed-eval)

All performance numbers come from `tore-speed-eval` with AIME dataset, burst mode.

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

## Criteria

### 1. TP=4 Throughput: DreamShift must beat Qwen-AR TP=4
- At ALL concurrency levels (1, 2, 4, 8, 16, 32, 64), DreamShift TP=4 tok/s > Qwen-AR TP=4 tok/s
- At low concurrency (C=1): significant speedup expected (2x+ over Qwen-AR)
- At high concurrency (C=64): at least match Qwen-AR

### 2. Single request latency: maximize bs=1 output tok/s
- TP=4 DreamShift at C=1 should significantly exceed TP=1 (current best: 328 tok/s with N=5 fp8)
- Target: **>= 700 tok/s** at C=1 (stretch goal)
- Minimum: >= 500 tok/s at C=1

### 3. Max throughput: find the peak
- Find the concurrency level that maximizes total throughput for TP=4 DreamShift
- Document the peak and compare with Qwen-AR TP=4 peak

### 4. Quality: Must be preserved
- **Quick iteration (use these first)**: IFEval strict >= 80%, GSM8K (200Q) >= 90%
- **Full validation (later)**: HumanEval >= 85%, MBPP >= 80%, AIME 2024 >= 73%
- IFEval and GSM8K are fast (~2-5 min); use them for rapid quality checks during optimization
- AIME takes 10+ minutes per run; save for final validation only
- 0 failed requests in all benchmarks
- IMPORTANT: always use max_tokens >= 8192 (32768 for AIME). Truncation causes false failures.

### 5. Algorithm health: TPF and accept rate
- Check server logs: `strings /tmp/sglang_gpu0.log | grep "tok/fwd"`
- N=3: TPF >= 1.8, accept rate >= 40%
- N=5: TPF >= 2.5, accept rate >= 30%

## Test Commands

### Setup: Launch TP=4 DreamShift server
```bash
bash scripts/killall_sglang.sh && sleep 3
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
SDAR=/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont

# TP=4 DreamShift N=3 sample on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path $SDAR --trust-remote-code --tp-size 4 \
  --mem-fraction-static 0.85 --max-running-requests 128 \
  --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
  --dllm-algorithm-config dreamshift_blockN3_config.yaml \
  --dtype bfloat16 --port 30000 --chunked-prefill-size 4096
```

### Throughput benchmarks
```bash
TOKENIZER=/data/yjian/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
for C in 1 2 4 8 16 32 64; do
  echo "=== concurrency=$C ==="
  tore-speed-eval --provider sglang --base_url http://localhost:30000/v1 \
    --model_name default --tokenizer_name $TOKENIZER --traffic_pattern burst \
    --concurrency $C --num_examples 90 --max_tokens 2048 \
    --dataset_type jsonl --jsonl_input_path /tmp/aime2024.jsonl \
    --jsonl_convert_to_chat_request_format true \
    --temperature 1.0 --top_p 0.95
done
```

### Quality benchmarks (use 2x TP=4 servers on ports 30000 and 30004)
```bash
PORTS="30000 30004"
# Quick iteration (use these first, ~2-5 min each):
python scripts/eval_ifeval.py --ports $PORTS --max-tokens 8192
python scripts/eval_gsm8k.py --ports $PORTS --num-problems 200 --max-tokens 8192

# Full validation (later, when optimization is stable):
python scripts/eval_humaneval.py --ports $PORTS --max-tokens 8192
python scripts/eval_mbpp.py --ports $PORTS --max-tokens 16384
python scripts/eval_aime.py --year 2024 --ports $PORTS --max-tokens 32768 --temperature 1.0 --top-p 0.95 --top-k 50 --output-dir /tmp/aime_tp4
```

### TPF check
```bash
strings /tmp/sglang_gpu0.log | grep "tok/fwd" | tail -5
```

## Debugging Methodology
When performance doesn't meet targets:
1. **Profile first**: `POST /start_profile` to capture GPU traces
2. **Parse traces**: Use Python trace parser to find top time consumers
3. **Check NCCL overhead**: Look for `nccl` in traces, measure all-reduce time vs compute time
4. **Check batch utilization**: `strings /tmp/sglang_gpu0.log | grep "DLLM decode loop"`
5. **Check TPF**: If TPF drops, the algorithm logic may be broken
6. **Check quality**: Run AIME quick test (--num-problems 10) before full benchmark
7. Only after quantitative understanding, attempt a fix

## Notes
- TP=4 reduces per-GPU compute by ~4x but adds NCCL all-reduce communication
- At low concurrency (memory-bound), TP=4 may not help much since the bottleneck is weight loading
- At high concurrency (compute-bound), TP=4 should scale well
- DreamShift's verify/sample operations happen AFTER forward pass — they don't need TP communication
- The `lm_head` output is already gathered on each rank — verify can happen on rank 0 only
- Reference implementation: `/data/cxu/dllm-distillation/generate.py` `causal_blockN_spec_verified_generate_with_shift`
