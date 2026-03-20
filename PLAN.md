# Plan — LLaDA2.1-mini Benchmark Evaluation

## Status
Phase 0 NEARLY COMPLETE — 82.07% vs 83.18% target (1.1% gap)
Phase 1 READY TO START — servers running with bl=4 config

## Model
**LLaDA2.1-mini** (`inclusionAI/LLaDA2.1-mini`)
- MoE architecture, ~8B params
- Servers: 8×TP=1, ports 30000-30007, max-running-requests=1
- Config: `llada21_quality.yaml` (bl=4, steps=4, threshold=0.95, edit_threshold=0.9)

## Claimed Scores (Paper)
| Benchmark | Q Mode | S Mode |
|-----------|--------|--------|
| IFEval (prompt-strict) | **83.18%** | 81.33% |
| GSM8K | 86.55% | 85.88% |
| Math500 | — | — |
| HumanEval+ | 82.93% | 80.49% |
| MBPP+ | 74.07% | 73.28% |
| LCB-v6 | 30.40% | 28.85% |
| GPQA-diamond | 53.28% | 48.36% |

## Phase 0: Match IFEval Claimed Score ← NEARLY DONE

### Progress
- [x] bl=32, greedy (broken): 62.7% — repetition bug
- [x] bl=4, threshold=0.95 (old code): 68.2%
- [x] bl=32 + scheduled transfer fix, threshold=0.7: 62.48% — still corrupted
- [x] bl=32 + scheduled transfer fix, threshold=0.95: ~40% — worse
- [x] **bl=4, threshold=0.95, edit_threshold=0.9: 82.07%** ← current best
- [ ] Optional: try to close remaining 1.1% gap

### Key Finding: bl=32 Denoising Quality Bug
sglang's JointThreshold with bl=32 produces corrupted/repetitive output:
- 32 denoising iterations per block compound numerical errors
- bl=4 (4 iterations) avoids this compounding
- Root cause likely in ragged attention KV rewrite during iteration loop

### Config (sglang vs official HF defaults)
| Parameter | Old sglang | Official | Final sglang |
|-----------|-----------|----------|-------------|
| block_size | 32 | 32 | **4** |
| steps | 32 | 32 | **4** |
| threshold | 0.7 | 0.95 | **0.95** |
| edit_threshold | 0.5 | 0.9 | **0.9** |
| temperature | 1.0 | 0.0 | **0.0** |

## Phase 1: Benchmark Suite ← NEXT

| Benchmark | Status | Score | Notes |
|-----------|--------|-------|-------|
| **IFEval** | ✅ Done | **82.07%** | target 83.18% |
| ARC-C | ⬜ Next | | fast |
| GSM8K | ⬜ Next | | fast, target 86.55% |
| Math500 | ⬜ Next | | fast |
| AIME-2024 | ⬜ | | fast |
| AIME-2025 | ⬜ | | fast |
| HumanEval+ | ⬜ | | fast, target 82.93% |
| MBPP+ | ⬜ | | fast, target 74.07% |
| MathBench | ⬜ | | medium (~30 min) |
| GPQA-diamond | ⬜ | | medium, target 53.28% |
| GPQA-main | ⬜ | | medium |
| TriviaQA | ⬜ | | slow |
| MMLU | ⬜ | | slow |
| MMLU-Pro | ⬜ | | slow |
| CMMLU | ⬜ | | slow |
| LCB-v6 | ⬜ | | slow, target 30.40% |

## Next Steps (for next iteration)
1. Run Tier 1 fast benchmarks: ARC-C, GSM8K, Math500, AIME-2024, AIME-2025
2. Run Tier 2: HumanEval+, MBPP+, MathBench, GPQA
3. Run Tier 3: TriviaQA, MMLU, MMLU-Pro, CMMLU, LCB

## Environment
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export HF_HOME=/data/yjian/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/yjian/hf_cache/hub
export FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache
```

## Server Launch (bl=4)
```bash
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
```

## Progress Log
- Fixed JointThreshold repetition bug (missing scheduled transfer)
- Discovered bl=32 has fundamental quality issue in sglang (error compounding in denoising loop)
- Updated thresholds to match official HF defaults (threshold=0.95, edit_threshold=0.9)
- **bl=4, threshold=0.95, edit_threshold=0.9: 82.07% IFEval** (target: 83.18%)
- Only 2/97 failures have severe repetition (2.1%)
- Throughput: 369.6 tok/s on 8 GPUs
