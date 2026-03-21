# Agent Prompt — b3 Checkpoint Benchmark Evaluation (N=4 Sampling)

## Role
Run the full benchmark suite on the b3 checkpoint using N=4 sampling across 8 GPUs, compare against N=3 reference results, and flag any anomalies.

## Model Under Test
```
/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-causal-from-b2amc_fixed2_cont_epoch2_lr1e-5_backup8000
```
- Config: `dreamshift_blockN4_config.yaml` (block_size=7, gen_block_size=4, sampling temp=1.0)
- 8×TP=1 servers on ports 30000-30007

## Environment
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export HF_HOME=/data/yjian/hf_cache
export FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache
export HF_TOKEN=<your_hf_token>
```

## Step 1: Kill Existing Servers + Launch b3 N=4
```bash
ps aux | grep "launch_server" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
sleep 5

MODEL=/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-causal-from-b2amc_fixed2_cont_epoch2_lr1e-5_backup8000
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache nohup python -m sglang.launch_server \
    --model-path $MODEL --trust-remote-code --tp-size 1 \
    --mem-fraction-static 0.85 --max-running-requests 32 \
    --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
    --dllm-algorithm-config dreamshift_blockN4_config.yaml \
    --dtype bfloat16 --port $((30000+i)) --chunked-prefill-size 4096 \
    --watchdog-timeout 1800 \
    > /tmp/sglang_b3n4_gpu${i}.log 2>&1 &
done
# Verify servers
for i in $(seq 0 7); do
  for j in $(seq 1 60); do
    curl -sf http://localhost:$((30000+i))/health > /dev/null 2>&1 && echo "GPU $i ready" && break; sleep 10
  done
done
```

## Step 2: Sanity Check Before Running Benchmarks
```bash
# Quick quality check: 2 test prompts
curl -s http://localhost:30000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"What is 2+3?"}],"max_tokens":64,"temperature":1.0}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"

# If output looks garbled or empty, servers have wrong model/config. Stop and debug.
```

## Step 3: Run Benchmarks (Easy → Hard)

```bash
PORTS="30000 30001 30002 30003 30004 30005 30006 30007"
mkdir -p bench_results/b3n4
```

### Tier 1 (run sequentially, fast):
```bash
python scripts/eval_arc_c.py --ports $PORTS --output-dir bench_results/b3n4
python scripts/eval_ifeval.py --ports $PORTS --max-tokens 32768 --max-workers 8 --output-dir bench_results/b3n4
python scripts/eval_gsm8k.py --ports $PORTS --output-dir bench_results/b3n4
python scripts/eval_math500.py --ports $PORTS --output-dir bench_results/b3n4
python scripts/eval_aime.py --year 2025 --ports $PORTS --output-dir bench_results/b3n4
python scripts/eval_humaneval.py --ports $PORTS --output-dir bench_results/b3n4
python scripts/eval_mbpp.py --ports $PORTS --output-dir bench_results/b3n4
```

### Tier 2 (medium):
```bash
python scripts/eval_lcb.py --version 6 --max-workers 16 --ports $PORTS --output-dir bench_results/b3n4
HF_TOKEN=$HF_TOKEN python scripts/eval_gpqa.py --subset main --ports $PORTS --output-dir bench_results/b3n4
```

### Tier 3 (slow, can run overnight):
```bash
python scripts/eval_mmlu_pro.py --ports $PORTS --output-dir bench_results/b3n4
python scripts/eval_mmlu.py --ports $PORTS --output-dir bench_results/b3n4
python scripts/eval_triviaqa.py --ports $PORTS --output-dir bench_results/b3n4
python scripts/eval_cmmlu.py --ports $PORTS --output-dir bench_results/b3n4
```

## Step 4: After Each Benchmark — Record and Compare

Update PLAN.md results table after each benchmark. Compare to N=3 reference:

| If b3 N=4 vs N=3 difference | Action |
|------------------------------|--------|
| Within ±5% | Normal, continue |
| > 5% lower | Log as anomaly, continue (flag for user review) |
| > 15% lower | STOP — likely config/model error. Debug before continuing |
| > 5% higher | Great! Log as improvement |

## Anomaly Debug Checklist
If score is suspiciously low:
1. Check server log: `tail -20 /tmp/sglang_b3n4_gpu0.log`
2. Verify model loaded: look for model path in server log
3. Verify config: `cat dreamshift_blockN4_config.yaml`
4. Test manually: send a simple math question, check output quality
5. Check for repetition/garbled output in wrong answers

## Notes
- All scripts default to --max-tokens 32768
- Use --max-workers 8 for DLLM (not 512 — causes OOM with bl=7 max_running_requests=32)
- LCB uses OC evaluator (reliability_guard fix) — each problem in subprocess
- GPQA needs HF_TOKEN
- b3 is trained on block_size=3 but N=4 means block_size=7 at inference — this is slightly OOD, expect small quality drop vs N=3 b2-cont
