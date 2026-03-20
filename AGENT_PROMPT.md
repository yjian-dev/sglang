# Agent Prompt — LLaDA2.1-mini Evaluation

## Role
You are a benchmark evaluation agent. Your primary goal is to make LLaDA2.1-mini match its claimed scores on IFEval (83.18% prompt-strict), then run the full benchmark suite on all remaining benchmarks in order of difficulty.

## Models
- **LLaDA2.1-mini**: `inclusionAI/LLaDA2.1-mini` (on HF, cached at `/data/yjian/hf_cache`)
- Servers: 8×TP=1 on ports 30000-30007 (launch fresh each session)

## Environment
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export HF_HOME=/data/yjian/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/yjian/hf_cache/hub
export FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache
export HF_TOKEN=<your_hf_token>
```

## Phase 0: Match Claimed IFEval Score (PREREQUISITE)

**Target: 83.18% prompt-strict** (LLaDA2.1 paper, Q Mode)

### Current Status
- bl=32, greedy → 62.7% (repetition/corruption bug in sglang)
- bl=4, threshold=0.95 → 68.2%
- bl=32, scheduled transfer (our fix) → ~80%+ (estimated, needs validation)

### Root Cause of Repetition
sglang's JointThreshold was missing **scheduled transfer** (`num_to_transfer` per step).
Official HF `generate()` distributes `block_length` tokens across `steps` steps (1 token/step for bl=32/steps=32).
Our fix: `_get_num_transfer_tokens(block_size, steps)` now in `joint_threshold.py`.

### Steps to Match Claimed Score
1. **Search for correct params**: Web search "LLaDA2.1-mini sglang JointThreshold IFEval parameters" and check:
   - https://huggingface.co/inclusionAI/LLaDA2.1-mini (model card)
   - https://github.com/sgl-project/sglang (issues/PRs about LLaDA2.1)
   - arxiv paper: https://arxiv.org/abs/2602.08676
2. **Key config file**: `llada21_quality.yaml` (currently: block_size=32, steps=32, threshold=0.7, edit_threshold=0.5)
3. **Test IFEval with 20 problems first**, check for repetition in outputs
4. If repetition still occurs, try:
   a. Increase `steps` (try 16, 8 instead of 32 per block_size)
   b. Adjust `threshold` (0.5, 0.6, 0.7)
   c. Check if `temperature=0.0` (greedy) vs `temperature=1.0` matters
5. **Only run full 541 IFEval once repetition < 5% of responses**

### Launch Command
```bash
MODEL_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('inclusionAI/LLaDA2.1-mini'))")
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache nohup python -m sglang.launch_server \
    --model-path inclusionAI/LLaDA2.1-mini \
    --dllm-algorithm JointThreshold \
    --dllm-algorithm-config llada21_quality.yaml \
    --tp-size 1 --trust-remote-code \
    --mem-fraction-static 0.8 --max-running-requests 1 \
    --attention-backend flashinfer \
    --port $((30000+i)) --watchdog-timeout 1800 \
    > /tmp/sglang_llada21_gpu${i}.log 2>&1 &
done
# Wait for all 8
for i in $(seq 0 7); do
  for j in $(seq 1 60); do
    curl -sf http://localhost:$((30000+i))/health > /dev/null 2>&1 && break; sleep 10
  done
done
```

### Quick Repetition Check
Before running full IFEval, test 5 examples and check output quality:
```bash
python scripts/eval_ifeval.py --ports 30000 --num-problems 5 --max-tokens 16384 --max-workers 1
cat output_ifeval/ifeval_details.json | python3 -c "
import json, sys
for d in json.load(sys.stdin)[:5]:
    print('PASS:', d['strict_pass'])
    print('OUT:', d['prediction_stripped'][:200])
    print()
"
```
If output contains repetition like "of of of of" or "the the the", params need adjustment.

---

## Phase 1: Benchmarks (Easy → Hard), run AFTER IFEval matches claimed score

Run in this order (quick first, slow last). Use 8 servers ports 30000-30007, max_tokens=32768.

### Tier 1: Very Fast (< 10 min each)
```bash
PORTS="30000 30001 30002 30003 30004 30005 30006 30007"
python scripts/eval_arc_c.py --ports $PORTS
python scripts/eval_gsm8k.py --ports $PORTS
python scripts/eval_ifeval.py --ports $PORTS --max-tokens 16384 --max-workers 8
python scripts/eval_aime.py --year 2025 --ports $PORTS
python scripts/eval_aime.py --year 2024 --ports $PORTS
python scripts/eval_math500.py --ports $PORTS
```

### Tier 2: Medium (10-30 min each)
```bash
python scripts/eval_humaneval.py --ports $PORTS
python scripts/eval_mbpp.py --ports $PORTS
python scripts/eval_mathbench.py --ports $PORTS  # circular, ~30 min
HF_TOKEN=$HF_TOKEN python scripts/eval_gpqa.py --subset diamond --ports $PORTS
HF_TOKEN=$HF_TOKEN python scripts/eval_gpqa.py --subset main --ports $PORTS
```

### Tier 3: Slow (> 1 hour each)
```bash
python scripts/eval_triviaqa.py --ports $PORTS
python scripts/eval_mmlu.py --ports $PORTS
python scripts/eval_mmlu_pro.py --ports $PORTS
python scripts/eval_cmmlu.py --ports $PORTS
python scripts/eval_lcb.py --version 6 --max-workers 8 --ports $PORTS
```

---

## Quality Check After Each Benchmark
1. Check truncation rate (should be < 15% at 32k tokens)
2. Check extraction failures (should be < 5%)
3. **Check for repetition in wrong answers** — sample 3 failed responses
4. If repetition > 10%: go back to Phase 0, adjust params

## Config Files
- `llada21_quality.yaml` — Quality Mode (current best params)
- `llada21_b4_t95.yaml` — Speed variant (bl=4)

## Notes
- max-running-requests=1 is required for LLaDA2.1-mini (MoE model, high memory per request)
- Model is ~15GB, all 8 GPUs should fit
- Do NOT download to home folder (disk quota exceeded) — always use HF_HOME=/data/yjian/hf_cache
- IFEval: use --max-workers 8 (not 541) to avoid OOM with bl=32
