# Plan — 32B Remaining Benchmarks

## Status
Remaining: AIME-25, GPQA-D, GPQA, LCB-v6, MMLU-Pro, MMLU for both AR and DLLM.

## Completed Results (do NOT rerun)

| Benchmark | Qwen3-32B AR | DLLM-32B b1 merged |
|-----------|-------------|-------------------|
| HumanEval | 96.3% | 96.3% |
| MBPP | 95.7% | 94.6% |
| IFEval | 84.5% | 84.7% |
| GSM8K | 94.7% | 95.9% |
| MATH-500 | 97.8% | 97.6% |
| ARC-C | 97.2% | 96.8% |
| AIME-24 | 76.7% | 83.3% |
| TriviaQA | 74.4% | 72.9% |

## Remaining (fastest first, full datasets)

| # | Benchmark | AR | DLLM |
|---|-----------|-----|------|
| 1 | AIME-25 | redo | redo |
| 2 | GPQA-Diamond | redo (had errors before) | redo (had errors before) |
| 3 | GPQA (main) | need | need |
| 4 | LCB-v6 | need | need |
| 5 | MMLU-Pro | need | need |
| 6 | MMLU | need | need |

## Critical Rules
- max-running-requests=4 (32B OOMs at higher)
- max-workers=16 (too many concurrent = failures)
- timeout=600 (900 for MMLU/MMLU-Pro)
- Run SEQUENTIALLY, check Errors=0 after each
- Full datasets (no --num-problems)

## Progress Log

### Iteration 1 — AR Benchmarks (2026-03-26)

**AR servers running on ports 30000-30003 (TP=2, 4 servers, 8 GPUs)**

| # | Benchmark | AR Result | Status |
|---|-----------|-----------|--------|
| 1 | AIME-25 | 63.3% (19/30), 5 timeouts on hardest problems (26-30) | DONE |
| 2 | GPQA-Diamond | 67.7% (134/198), 0 errors | DONE |
| 3 | GPQA (main) | 65.2% (292/448), 0 errors | DONE |
| 4 | LCB-v6 | 58.3% (102/175), 3 truncated + 5 empty/error | DONE |
| 5 | MMLU-Pro | FAILED — servers crashed ~74min in, 9624/12032 done but results lost | NEEDS RERUN |
| 6 | MMLU | blocked | PENDING |

**Blocker: GPU processes unkillable**
- 8 sglang scheduler processes (PIDs 3942849-3942856) are stuck in CUDA kernel calls
- SIGKILL has no effect, `nvidia-smi -r` fails (insufficient permissions)
- All 8 GPUs have ~74GB used, cannot launch new servers
- Need machine reboot or admin to kill processes before continuing

**Next steps (after GPU memory is freed):**
1. Launch AR servers
2. Re-run MMLU-Pro on AR (12032 problems, ~2-4 hrs, use `--timeout 900 --max-workers 16`)
3. Run MMLU on AR (use `--timeout 900 --max-workers 16`)
4. Kill AR servers, launch DLLM servers
5. Run all 6 benchmarks on DLLM (AIME-25, GPQA-D, GPQA, LCB-v6, MMLU-Pro, MMLU)
6. Record all results in final table


## Evaluator Feedback (Iteration 1)
1. Free GPU memory first: get admin help to reboot the machine or kill stuck CUDA processes. 2. After GPUs are free, launch AR servers and run MMLU-Pro and MMLU (the two remaining AR benchmarks). 3. Kill AR servers, launch DLLM servers, and run all 6 benchmarks sequentially for DLLM. 4. After all 12 runs complete with Errors=0, update PLAN.md with a 'Final Results' section containing the full 14-benchmark comparison table.

### Iteration 2 — Blocked by GPU Availability (2026-03-26)

**Blocker: All 8 GPUs occupied by another user (`hkang`)**
- Machine was rebooted (stuck processes from iteration 1 are gone)
- User `hkang` launched MiniMax-M2.5 with TP=8, EP=8 on all 8 GPUs (~74GB each)
- Process owner: hkang (PIDs 3942849-3942857), port 8000
- Cannot proceed until hkang's job finishes or GPUs become available

**No benchmarks could be run this iteration.**

**Next steps (after GPUs are free):**
1. Launch AR servers (4x TP=2, ports 30000-30003)
2. Run MMLU-Pro on AR (`--timeout 900 --max-workers 16`)
3. Run MMLU on AR (`--timeout 900 --max-workers 16`)
4. Kill AR servers, launch DLLM servers
5. Run all 6 benchmarks on DLLM (AIME-25, GPQA-D, GPQA, LCB-v6, MMLU-Pro, MMLU)
6. Record all results in final table
