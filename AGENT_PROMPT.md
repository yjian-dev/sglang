# Agent Prompt — OC-Aligned Benchmark: 3 Models × 8 Benchmarks × 2 Runs

## Role
Run 8 OC-aligned benchmarks on 3 models (N=3, Qwen3-8B, SDAR-8B-Chat), 2 runs each. Record results in PLAN.md.

## Environment
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export HF_HOME=/data/yjian/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/yjian/hf_cache/hub
export FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache
```

## Workflow: Run 3 models sequentially

### Phase 1: N=3 (2 runs)

Kill any existing servers, then launch:
```bash
ps aux | grep "launch_server" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null; sleep 5

MODEL=/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache nohup python -m sglang.launch_server \
    --model-path $MODEL --trust-remote-code --tp-size 1 \
    --mem-fraction-static 0.85 --max-running-requests 32 \
    --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
    --dllm-algorithm-config dreamshift_blockN3_config.yaml \
    --dtype bfloat16 --port $((30000+i)) --chunked-prefill-size 4096 \
    --watchdog-timeout 1800 \
    > /tmp/sglang_n3_gpu${i}.log 2>&1 &
done
for i in $(seq 0 7); do
  for j in $(seq 1 60); do curl -sf http://localhost:$((30000+i))/health > /dev/null 2>&1 && echo "GPU $i ready" && break; sleep 10; done
done
```

Run benchmarks (2 rounds):
```bash
PORTS="30000 30001 30002 30003 30004 30005 30006 30007"
for N in 1 2; do
  OUTDIR=bench_results/n3_oc_run${N}
  mkdir -p $OUTDIR

  # Fast
  python scripts/eval_arc_c.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
  python scripts/eval_mbpp.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
  python scripts/eval_gsm8k.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
  python scripts/eval_aime.py --year 2025 --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
  python scripts/eval_triviaqa.py --ports $PORTS --max-tokens 256 --output-dir $OUTDIR

  # Slow (sequential, NEVER concurrent)
  python scripts/eval_mmlu_pro.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
  python scripts/eval_mmlu.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
  python scripts/eval_cmmlu.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR

  # Update PLAN.md after each run
done
```

### Phase 2: Qwen3-8B (2 runs)

Kill N=3 servers, launch Qwen3-8B:
```bash
ps aux | grep "launch_server" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null; sleep 5

for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache nohup python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B --trust-remote-code --tp-size 1 \
    --mem-fraction-static 0.85 --max-running-requests 64 \
    --attention-backend flashinfer --dtype bfloat16 \
    --port $((30000+i)) --chunked-prefill-size 4096 --watchdog-timeout 1800 \
    > /tmp/sglang_qwen_gpu${i}.log 2>&1 &
done
for i in $(seq 0 7); do
  for j in $(seq 1 60); do curl -sf http://localhost:$((30000+i))/health > /dev/null 2>&1 && echo "GPU $i ready" && break; sleep 10; done
done
```

Run benchmarks (2 rounds), same scripts but:
- Output dir: `bench_results/qwen3_oc_run${N}`
- **Important**: Qwen3 is a thinking model, use `--timeout 600` for slow benchmarks

```bash
for N in 1 2; do
  OUTDIR=bench_results/qwen3_oc_run${N}
  mkdir -p $OUTDIR

  python scripts/eval_arc_c.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
  python scripts/eval_mbpp.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
  python scripts/eval_gsm8k.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
  python scripts/eval_aime.py --year 2025 --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
  python scripts/eval_triviaqa.py --ports $PORTS --max-tokens 256 --output-dir $OUTDIR

  python scripts/eval_mmlu_pro.py --ports $PORTS --max-tokens 32768 --timeout 600 --output-dir $OUTDIR
  python scripts/eval_mmlu.py --ports $PORTS --max-tokens 32768 --timeout 600 --output-dir $OUTDIR
  python scripts/eval_cmmlu.py --ports $PORTS --max-tokens 32768 --timeout 600 --output-dir $OUTDIR
done
```

### Phase 3: SDAR-8B-Chat (2 runs)

Kill Qwen3 servers, launch SDAR:
```bash
ps aux | grep "launch_server" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null; sleep 5

for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache nohup python -m sglang.launch_server \
    --model-path JetLM/SDAR-8B-Chat --dllm-algorithm LowConfidence \
    --tp-size 1 --trust-remote-code \
    --mem-fraction-static 0.85 --max-running-requests 32 \
    --attention-backend flashinfer --dtype bfloat16 \
    --port $((30000+i)) --watchdog-timeout 1800 \
    > /tmp/sglang_sdar_gpu${i}.log 2>&1 &
done
for i in $(seq 0 7); do
  for j in $(seq 1 60); do curl -sf http://localhost:$((30000+i))/health > /dev/null 2>&1 && echo "GPU $i ready" && break; sleep 10; done
done
```

Run benchmarks (2 rounds):
- Output dir: `bench_results/sdar_oc_run${N}`
- **Important**: SDAR is slow (block_size=32), use `--timeout 300` and `--max-workers 8`

```bash
for N in 1 2; do
  OUTDIR=bench_results/sdar_oc_run${N}
  mkdir -p $OUTDIR

  python scripts/eval_arc_c.py --ports $PORTS --max-tokens 32768 --max-workers 8 --timeout 300 --output-dir $OUTDIR
  python scripts/eval_mbpp.py --ports $PORTS --max-tokens 32768 --max-workers 8 --timeout 300 --output-dir $OUTDIR
  python scripts/eval_gsm8k.py --ports $PORTS --max-tokens 32768 --max-workers 8 --timeout 300 --output-dir $OUTDIR
  python scripts/eval_aime.py --year 2025 --ports $PORTS --max-tokens 32768 --max-workers 8 --timeout 300 --output-dir $OUTDIR
  python scripts/eval_triviaqa.py --ports $PORTS --max-tokens 256 --max-workers 8 --timeout 300 --output-dir $OUTDIR

  python scripts/eval_mmlu_pro.py --ports $PORTS --max-tokens 32768 --max-workers 8 --timeout 600 --output-dir $OUTDIR
  python scripts/eval_mmlu.py --ports $PORTS --max-tokens 32768 --max-workers 8 --timeout 600 --output-dir $OUTDIR
  python scripts/eval_cmmlu.py --ports $PORTS --max-tokens 32768 --max-workers 8 --timeout 600 --output-dir $OUTDIR
done
```

## Quality Checks
After each benchmark:
1. Errors < 5%. If higher → increase timeout, reduce max-workers, rerun.
2. Truncation < 10%.
3. Sample 3 wrong answers to verify no script bugs.

## Server Restart (if crashed)
Check: `for i in $(seq 0 7); do curl -sf http://localhost:$((30000+i))/health && echo "$i OK" || echo "$i DOWN"; done`
Restart crashed GPU: use the model-specific launch command for that GPU only.

## Critical Rules
- **NEVER run MMLU/MMLU-Pro/CMMLU concurrently** — causes OOM
- TriviaQA: always use `--max-tokens 256` (not 32768)
- Between models: kill ALL servers, wait 5s, then launch new model
- Update PLAN.md results table after EVERY benchmark completion
