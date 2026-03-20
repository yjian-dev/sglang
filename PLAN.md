# Plan — LLaDA2.1-mini Benchmark Evaluation

## Status
Phase 0 in progress — matching claimed IFEval score

## Model
**LLaDA2.1-mini** (`inclusionAI/LLaDA2.1-mini`)
- MoE architecture, ~8B params
- Servers: 8×TP=1, ports 30000-30007, max-running-requests=1
- Config: `llada21_quality.yaml` (bl=32, steps=32, threshold=0.7)

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

## Phase 0: Match IFEval Claimed Score ← CURRENT

### Progress
- [x] bl=32, greedy (broken): 62.7% — repetition bug
- [x] bl=4, threshold=0.95: 68.2% — better but not there yet
- [x] bl=32 + scheduled transfer fix: ~80% (5-sample test), needs full run
- [ ] **Full IFEval with scheduled transfer fix** — target: 83.18%
- [ ] If not matching: research correct params, try variations

### Config to Try (in order)
1. `llada21_quality.yaml`: bl=32, steps=32, t=0.7, et=0.5, temp=1.0 ← current
2. If repetition: try steps=16 (2 tokens/step)
3. If still low: web search for correct sglang params

## Phase 1: Benchmark Suite (after Phase 0 complete)

| Benchmark | Status | Score | Notes |
|-----------|--------|-------|-------|
| **IFEval** | Phase 0 | | target 83.18% |
| ARC-C | | | fast |
| GSM8K | | | fast, target 86.55% |
| Math500 | | | fast |
| AIME-2024 | | | fast |
| AIME-2025 | | | fast |
| HumanEval+ | | | fast, target 82.93% |
| MBPP+ | | | fast, target 74.07% |
| MathBench | | | medium (~30 min) |
| GPQA-diamond | | | medium, target 53.28% |
| GPQA-main | | | medium |
| TriviaQA | | | slow |
| MMLU | | | slow |
| MMLU-Pro | | | slow |
| CMMLU | | | slow |
| LCB-v6 | | | slow, target 30.40% |

## Key Issue: Repetition Bug
The repetition ("of of of of") was caused by sglang's JointThreshold missing scheduled transfer.
**Fix applied**: `joint_threshold.py` now has `_get_num_transfer_tokens(block_size, steps)`.
Config: add `steps: 32` to use 1-token-per-step schedule (matches official HF generate).

## Environment
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export HF_HOME=/data/yjian/hf_cache
export HUGGINGFACE_HUB_CACHE=/data/yjian/hf_cache/hub
export FLASHINFER_CACHE_DIR=/tmp/flashinfer_cache
export HF_TOKEN=<your_hf_token>
```

## Progress Log
- Fixed JointThreshold repetition bug (missing scheduled transfer)
- bl=32 + steps=32: 5/5 IFEval sample test passed 80%, no visible repetition
- Need full IFEval run to confirm 83%+
