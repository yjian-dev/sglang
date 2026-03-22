# Agent Prompt — N=3 Sampling 5-Run Stability Benchmark

## Role
Run all 13 benchmarks **5 times** on the N=3 sampling model to get mean ± std for each benchmark.

## Model
N=3 sampling (b2-allmasked-causal_fixed2_cont), 8×TP=1, ports 30000-30007.
**Servers are already running — do NOT restart unless they crash.**

## Environment
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export HF_HOME=/data/yjian/hf_cache
export FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache
export HF_TOKEN=<your_hf_token>
```

## Critical: Max Tokens
**Always use `--max-tokens 32768`** for every benchmark. Lower values cause systematic under-scoring.

## For Each Run (repeat 5 times: N=1,2,3,4,5)

```bash
PORTS="30000 30001 30002 30003 30004 30005 30006 30007"
N=<current_run_number>
OUTDIR=bench_results/n3_run${N}
mkdir -p $OUTDIR
```

### Step 1: Health check
```bash
for i in $(seq 0 7); do curl -sf http://localhost:$((30000+i))/health && echo "GPU $i OK" || echo "GPU $i DOWN"; done
```
If any server is down, restart it before proceeding.

### Step 2: Tier 1 — Fast benchmarks (run sequentially)
```bash
python scripts/eval_arc_c.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
python scripts/eval_ifeval.py --ports $PORTS --max-tokens 32768 --max-workers 8 --output-dir $OUTDIR
python scripts/eval_gsm8k.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
python scripts/eval_math500.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
python scripts/eval_aime.py --year 2025 --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
python scripts/eval_humaneval.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
python scripts/eval_mbpp.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
```

### Step 3: Tier 2 — Medium
```bash
python scripts/eval_lcb.py --version 6 --max-workers 16 --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
HF_TOKEN=$HF_TOKEN python scripts/eval_gpqa.py --subset main --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
```

### Step 4: Tier 3 — Slow (run concurrently to save time)
```bash
python scripts/eval_mmlu_pro.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR &
python scripts/eval_mmlu.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR &
python scripts/eval_triviaqa.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR &
wait
python scripts/eval_cmmlu.py --ports $PORTS --max-tokens 32768 --output-dir $OUTDIR
```

### Step 5: Record results in PLAN.md
After each run, extract all scores and fill in the Results table.
Use this to extract scores:
```bash
for f in $OUTDIR/*_summary.json; do echo $f; cat $f | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('accuracy', d.get('overall_acc', d.get('prompt_strict',''))))"; done
```

### Step 6: Check for issues
- Truncation > 10%: note but continue
- Extraction failures > 5%: investigate
- Score > 5pp off previous single-run reference: check for server crash or eval bug

## After All 5 Runs: Compute Stats
```python
import json, glob, numpy as np
benchmarks = ['arc_c', 'ifeval', 'gsm8k', 'math500', 'aime', 'humaneval', 'mbpp', 'lcb', 'gpqa', 'mmlu_pro', 'mmlu', 'triviaqa', 'cmmlu']
# Load all runs and compute mean ± std
```

## Server Restart (if crashed)
```bash
MODEL=/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont
CUDA_VISIBLE_DEVICES=$GPU_ID FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache python -m sglang.launch_server \
  --model-path $MODEL --trust-remote-code --tp-size 1 \
  --mem-fraction-static 0.85 --max-running-requests 32 \
  --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
  --dllm-algorithm-config dreamshift_blockN3_config.yaml \
  --dtype bfloat16 --port $((30000+GPU_ID)) --chunked-prefill-size 4096 \
  --watchdog-timeout 1800 \
  > /tmp/sglang_n3_gpu${GPU_ID}.log 2>&1 &
```
