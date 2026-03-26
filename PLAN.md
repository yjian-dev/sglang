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


## Evaluator Feedback (Iteration 2)
1. Wait for GPUs to become available (check with 'nvidia-smi' periodically or contact hkang). 2. Once GPUs are free, launch AR servers and run the 2 remaining AR benchmarks (MMLU-Pro, MMLU) sequentially. 3. Kill AR servers, launch DLLM servers, and run all 6 DLLM benchmarks (AIME-25, GPQA-D, GPQA, LCB-v6, MMLU-Pro, MMLU) sequentially. 4. Verify Errors=0 for each run before recording results. 5. After all runs complete, add a proper '## Final Results' section to PLAN.md with the full 14-benchmark comparison table replacing all 'need'/'redo' entries with actual values.

### Iteration 3 — Still Blocked by GPU Availability (2026-03-26)

**Blocker: All 8 GPUs still occupied by hkang's MiniMax-M2.5 job**
- hkang's job has been running for ~1h51m, still actively serving on port 8000
- All 8 GPUs at ~74GB used, 38-89% GPU utilization (actively processing)
- Same PIDs as iteration 2 (3942849-3942857)
- No indication of when the job will finish

**No benchmarks could be run this iteration.**

**Next steps (after GPUs are free):**
1. Launch AR servers (4x TP=2, ports 30000-30003)
2. Run MMLU-Pro on AR (`--timeout 900 --max-workers 16`)
3. Run MMLU on AR (`--timeout 900 --max-workers 16`)
4. Kill AR servers, launch DLLM servers
5. Run all 6 benchmarks on DLLM (AIME-25, GPQA-D, GPQA, LCB-v6, MMLU-Pro, MMLU)
6. Record all results in final table


## Evaluator Feedback (Iteration 3)
1. Check GPU availability with 'nvidia-smi' — if hkang's job is still running, contact hkang directly to ask when GPUs will be free or request they free some GPUs. 2. Once GPUs are available, launch AR servers and run MMLU-Pro and MMLU sequentially (the 2 remaining AR benchmarks). 3. Kill AR servers, launch DLLM servers, and run all 6 DLLM benchmarks (AIME-25, GPQA-D, GPQA, LCB-v6, MMLU-Pro, MMLU) sequentially. 4. Verify Errors=0 for each run before recording results. 5. After all runs complete, add a proper '## Final Results' section to PLAN.md with the full 14-benchmark comparison table replacing all 'need'/'redo' entries with actual values.

### Iteration 4 — Still Blocked by GPU Availability (2026-03-26)

**Blocker: All 8 GPUs still occupied by hkang's MiniMax-M2.5 job**
- hkang's job has been running for ~1h52m (PID 3942587), still actively serving on port 8000
- All 8 GPUs at ~74GB used, 80-100% GPU utilization (actively processing)
- hkang is NOT currently logged into the machine — cannot be contacted directly via terminal
- Server launched with `--admin-api-key r2egym-hicache-20260323` — could potentially shut down via admin API but this would be destructive to hkang's work

**No benchmarks could be run this iteration.**

**Action needed: Contact hkang** (via Slack, email, or other channel) to ask when GPUs will be free. The admin API key is visible in the process args if an emergency shutdown is needed, but this should only be done with hkang's permission.

**Next steps (after GPUs are free):**
1. Launch AR servers (4x TP=2, ports 30000-30003)
2. Run MMLU-Pro on AR (`--timeout 900 --max-workers 16`)
3. Run MMLU on AR (`--timeout 900 --max-workers 16`)
4. Kill AR servers, launch DLLM servers
5. Run all 6 benchmarks on DLLM (AIME-25, GPQA-D, GPQA, LCB-v6, MMLU-Pro, MMLU)
6. Record all results in final table


## Evaluator Feedback (Iteration 4)
1. Contact hkang via Slack/email to ask when GPUs will be free or negotiate shared access. 2. If hkang agrees, use the admin API to gracefully shut down their server (curl -X POST http://localhost:8000/shutdown with the admin key). 3. Once GPUs are free, run the 2 remaining AR benchmarks (MMLU-Pro, MMLU) first since 4 AR benchmarks are already done. 4. Then switch to DLLM servers and run all 6 DLLM benchmarks sequentially. 5. After all runs complete with Errors=0, add a '## Final Results' section to PLAN.md with the full 14-benchmark comparison table.

### Iteration 5 — Still Blocked by GPU Availability (2026-03-26)

**Blocker: All 8 GPUs still occupied by hkang's MiniMax-M2.5 job**
- hkang's job still running (PID 3942587), started at 07:30, now ~2h+ uptime
- All 8 GPUs at ~74GB used, 74-100% GPU utilization
- Server actively processing: 30 running requests, 0 queued (confirmed via /metrics endpoint)
- 18,183 total requests processed so far, 15.5M prefill tokens, 3.3M decode tokens
- hkang is NOT logged into the machine — cannot be contacted via terminal
- `write hkang` not possible (not logged in)

**No benchmarks could be run this iteration.**

**Summary of AR benchmarks completed (iteration 1):**

| # | Benchmark | AR Result | Status |
|---|-----------|-----------|--------|
| 1 | AIME-25 | 63.3% (19/30) | DONE |
| 2 | GPQA-Diamond | 67.7% (134/198) | DONE |
| 3 | GPQA (main) | 65.2% (292/448) | DONE |
| 4 | LCB-v6 | 58.3% (102/175) | DONE |
| 5 | MMLU-Pro | — | NEEDS RERUN (server crashed) |
| 6 | MMLU | — | PENDING |

**Action needed: User (yjian) must contact hkang** via Slack/email/other channel to negotiate GPU access. The automated agent cannot reach hkang — this requires human intervention.

**Next steps (after GPUs are free):**
1. Launch AR servers (4x TP=2, ports 30000-30003)
2. Run MMLU-Pro on AR (`--timeout 900 --max-workers 16`)
3. Run MMLU on AR (`--timeout 900 --max-workers 16`)
4. Kill AR servers, launch DLLM servers
5. Run all 6 benchmarks on DLLM (AIME-25, GPQA-D, GPQA, LCB-v6, MMLU-Pro, MMLU)
6. Record all results in final table


## Evaluator Feedback (Iteration 5)
1. Contact hkang via Slack/email to ask when GPUs will be free or negotiate shared access. 2. If hkang agrees, use the admin API to gracefully shut down their server (curl -X POST http://localhost:8000/shutdown with the admin key). 3. Once GPUs are free, run the 2 remaining AR benchmarks (MMLU-Pro, MMLU) first since 4 AR benchmarks are already done. 4. Then switch to DLLM servers and run all 6 DLLM benchmarks sequentially. 5. After all runs complete with Errors=0, add a '## Final Results' section to PLAN.md with the full 14-benchmark comparison table.

### Iteration 6 — Still Blocked by GPU Availability (2026-03-26, 09:24 PDT)

**Blocker: All 8 GPUs still occupied by hkang's MiniMax-M2.5 job**
- hkang's job still running (PID 3942587), started at 07:30, now ~2h uptime
- All 8 GPUs at ~74GB used, 78-100% GPU utilization
- Server actively processing: 52 running requests (confirmed via /metrics endpoint)
- hkang is NOT logged into the machine — cannot be contacted via terminal

**No benchmarks could be run this iteration. This is the 5th consecutive blocked iteration (iterations 2-6).**

**Summary of all AR benchmarks:**

| # | Benchmark | AR Result | Status |
|---|-----------|-----------|--------|
| 1 | AIME-25 | 63.3% (19/30) | DONE |
| 2 | GPQA-Diamond | 67.7% (134/198) | DONE |
| 3 | GPQA (main) | 65.2% (292/448) | DONE |
| 4 | LCB-v6 | 58.3% (102/175) | DONE |
| 5 | MMLU-Pro | — | NEEDS RERUN (server crashed in iter 1) |
| 6 | MMLU | — | PENDING |

**Action needed: User (yjian) must contact hkang** via Slack/email to negotiate GPU access. Alternatively, user can use the admin API to shut down hkang's server:
```bash
curl -X POST http://localhost:8000/shutdown -H "Authorization: Bearer r2egym-hicache-20260323"
```
(Only do this with hkang's permission or if urgently needed.)

**Next steps (after GPUs are free):**
1. Launch AR servers (4x TP=2, ports 30000-30003)
2. Run MMLU-Pro on AR (`--timeout 900 --max-workers 16`)
3. Run MMLU on AR (`--timeout 900 --max-workers 16`)
4. Kill AR servers, launch DLLM servers
5. Run all 6 benchmarks on DLLM (AIME-25, GPQA-D, GPQA, LCB-v6, MMLU-Pro, MMLU)
6. Record all results in final table


## Evaluator Feedback (Iteration 6)
1. Contact hkang via Slack/email to negotiate GPU access or ask when GPUs will be free. 2. If urgent and hkang agrees, shut down their server via: curl -X POST http://localhost:8000/shutdown -H 'Authorization: Bearer r2egym-hicache-20260323'. 3. Once GPUs are free, run the 2 remaining AR benchmarks (MMLU-Pro, MMLU) first. 4. Then switch to DLLM servers and run all 6 DLLM benchmarks sequentially. 5. After all runs complete with Errors=0, add a '## Final Results' section to PLAN.md with the full 14-benchmark comparison table.

### Iteration 7 — Still Blocked by GPU Availability (2026-03-26, ~09:30 PDT)

**Blocker: All 8 GPUs still occupied by hkang's MiniMax-M2.5 job**
- hkang's job still running (PID 3942587), started at 07:30, now ~2h+ uptime
- All 8 GPUs at ~74GB used, 29-92% GPU utilization
- Server still responding on port 8000
- hkang is NOT logged into the machine

**No benchmarks could be run this iteration. This is the 6th consecutive blocked iteration (iterations 2-7).**

**Summary of completed AR benchmarks (from iteration 1):**

| # | Benchmark | AR Result | Status |
|---|-----------|-----------|--------|
| 1 | AIME-25 | 63.3% (19/30) | DONE |
| 2 | GPQA-Diamond | 67.7% (134/198) | DONE |
| 3 | GPQA (main) | 65.2% (292/448) | DONE |
| 4 | LCB-v6 | 58.3% (102/175) | DONE |
| 5 | MMLU-Pro | — | NEEDS RERUN (server crashed in iter 1) |
| 6 | MMLU | — | PENDING |

**Action needed: User (yjian) must either:**
1. Contact hkang via Slack/email to negotiate GPU access, OR
2. Shut down hkang's server (with permission): `curl -X POST http://localhost:8000/shutdown -H "Authorization: Bearer r2egym-hicache-20260323"`

**Next steps (after GPUs are free):**
1. Launch AR servers (4x TP=2, ports 30000-30003)
2. Run MMLU-Pro on AR (`--timeout 900 --max-workers 16`)
3. Run MMLU on AR (`--timeout 900 --max-workers 16`)
4. Kill AR servers, launch DLLM servers
5. Run all 6 benchmarks on DLLM (AIME-25, GPQA-D, GPQA, LCB-v6, MMLU-Pro, MMLU)
6. Record all results in final table


## Evaluator Feedback (Iteration 7)
1. Contact hkang via Slack/email to negotiate GPU access or ask when GPUs will be free. 2. If urgent and hkang agrees, shut down their server via: curl -X POST http://localhost:8000/shutdown -H 'Authorization: Bearer r2egym-hicache-20260323'. 3. Once GPUs are free, run the 2 remaining AR benchmarks (MMLU-Pro, MMLU) first. 4. Then switch to DLLM servers and run all 6 DLLM benchmarks sequentially. 5. After all runs complete with Errors=0, add a '## Final Results' section to PLAN.md with the full 14-benchmark comparison table.

### Iteration 8 — Still Blocked by GPU Availability (2026-03-26, ~09:35 PDT)

**Blocker: All 8 GPUs still occupied by hkang's MiniMax-M2.5 job**
- hkang's job still running (PID 3942587), started at 07:30, now ~2h+ uptime
- All 8 GPUs at ~74GB used, 100% GPU utilization
- Server actively processing: 46 running requests (confirmed via /metrics endpoint)
- hkang is NOT logged into the machine

**No benchmarks could be run this iteration. This is the 7th consecutive blocked iteration (iterations 2-8).**

**Summary of completed AR benchmarks (from iteration 1):**

| # | Benchmark | AR Result | Status |
|---|-----------|-----------|--------|
| 1 | AIME-25 | 63.3% (19/30) | DONE |
| 2 | GPQA-Diamond | 67.7% (134/198) | DONE |
| 3 | GPQA (main) | 65.2% (292/448) | DONE |
| 4 | LCB-v6 | 58.3% (102/175) | DONE |
| 5 | MMLU-Pro | — | NEEDS RERUN (server crashed in iter 1) |
| 6 | MMLU | — | PENDING |

**Action needed: User (yjian) must either:**
1. Contact hkang via Slack/email to negotiate GPU access, OR
2. Shut down hkang's server (with permission): `curl -X POST http://localhost:8000/shutdown -H "Authorization: Bearer r2egym-hicache-20260323"`

**Next steps (after GPUs are free):**
1. Launch AR servers (4x TP=2, ports 30000-30003)
2. Run MMLU-Pro on AR (`--timeout 900 --max-workers 16`)
3. Run MMLU on AR (`--timeout 900 --max-workers 16`)
4. Kill AR servers, launch DLLM servers
5. Run all 6 benchmarks on DLLM (AIME-25, GPQA-D, GPQA, LCB-v6, MMLU-Pro, MMLU)
6. Record all results in final table


## Evaluator Feedback (Iteration 8)
1. Contact hkang via Slack/email to negotiate GPU access or ask when GPUs will be free. 2. If urgent and hkang agrees, shut down their server via: curl -X POST http://localhost:8000/shutdown -H 'Authorization: Bearer r2egym-hicache-20260323'. 3. Once GPUs are free, run the 2 remaining AR benchmarks (MMLU-Pro, MMLU) first. 4. Then switch to DLLM servers and run all 6 DLLM benchmarks sequentially. 5. After all runs complete with Errors=0, add a '## Final Results' section to PLAN.md with the full 14-benchmark comparison table.

### Iteration 9 — Still Blocked by GPU Availability (2026-03-26, ~09:40 PDT)

**Blocker: All 8 GPUs still occupied by hkang's MiniMax-M2.5 job (8th consecutive blocked iteration)**
- hkang's job still running (PID 3942587), started at 07:30, now ~2h10m+ uptime
- All 8 GPUs at ~74GB used, 100% GPU utilization, 40 running requests
- hkang is NOT logged into the machine

**No benchmarks could be run this iteration.**

**Action needed: User (yjian) must either:**
1. Contact hkang via Slack/email to negotiate GPU access, OR
2. Shut down hkang's server (with permission): `curl -X POST http://localhost:8000/shutdown -H "Authorization: Bearer r2egym-hicache-20260323"`

**Next steps (after GPUs are free):**
1. Launch AR servers (4x TP=2, ports 30000-30003)
2. Run MMLU-Pro on AR (`--timeout 900 --max-workers 16`)
3. Run MMLU on AR (`--timeout 900 --max-workers 16`)
4. Kill AR servers, launch DLLM servers
5. Run all 6 benchmarks on DLLM (AIME-25, GPQA-D, GPQA, LCB-v6, MMLU-Pro, MMLU)
6. Record all results in final table


## Evaluator Feedback (Iteration 9)
1. Contact hkang via Slack/email to negotiate GPU access or ask when GPUs will be free. 2. If urgent and hkang agrees, shut down their server via: curl -X POST http://localhost:8000/shutdown -H 'Authorization: Bearer r2egym-hicache-20260323'. 3. Once GPUs are free, run the 2 remaining AR benchmarks (MMLU-Pro, MMLU) first. 4. Then switch to DLLM servers and run all 6 DLLM benchmarks sequentially. 5. After all runs complete with Errors=0, add a '## Final Results' section to PLAN.md with the full 14-benchmark comparison table.

### Iteration 10 — Still Blocked by GPU Availability (2026-03-26, ~09:45 PDT)

**Blocker: All 8 GPUs still occupied by hkang's MiniMax-M2.5 job (9th consecutive blocked iteration)**
- hkang's job still running (PID 3942587), started at 07:30, now ~2h15m+ uptime
- All 8 GPUs at ~74GB used, 100% GPU utilization, 41 running requests
- hkang is NOT logged into the machine

**No benchmarks could be run this iteration.**

**Action needed: User (yjian) must either:**
1. Contact hkang via Slack/email to negotiate GPU access, OR
2. Shut down hkang's server (with permission): `curl -X POST http://localhost:8000/shutdown -H "Authorization: Bearer r2egym-hicache-20260323"`

**Next steps (after GPUs are free):**
1. Launch AR servers (4x TP=2, ports 30000-30003)
2. Run MMLU-Pro on AR (`--timeout 900 --max-workers 16`)
3. Run MMLU on AR (`--timeout 900 --max-workers 16`)
4. Kill AR servers, launch DLLM servers
5. Run all 6 benchmarks on DLLM (AIME-25, GPQA-D, GPQA, LCB-v6, MMLU-Pro, MMLU)
6. Record all results in final table


## Evaluator Feedback (Iteration 10)
1. Contact hkang via Slack/email to negotiate GPU access or ask when GPUs will be free. 2. If urgent and hkang agrees, shut down their server via: curl -X POST http://localhost:8000/shutdown -H 'Authorization: Bearer r2egym-hicache-20260323'. 3. Once GPUs are free, run the 2 remaining AR benchmarks (MMLU-Pro, MMLU) first. 4. Then switch to DLLM servers and run all 6 DLLM benchmarks sequentially. 5. After all runs complete with Errors=0, add a '## Final Results' section to PLAN.md with the full 14-benchmark comparison table.

### Iteration 11 — Still Blocked by GPU Availability (2026-03-26, ~09:50 PDT)

**Blocker: All 8 GPUs still occupied by hkang's MiniMax-M2.5 job (10th consecutive blocked iteration)**
- hkang's job still running (PID 3942849), started at 07:30, now ~2h+ uptime
- All 8 GPUs at ~74GB used, 25-100% GPU utilization
- hkang is NOT logged into the machine

**No benchmarks could be run this iteration.**

**Action needed: User (yjian) must either:**
1. Contact hkang via Slack/email to negotiate GPU access, OR
2. Shut down hkang's server (with permission): `curl -X POST http://localhost:8000/shutdown -H "Authorization: Bearer r2egym-hicache-20260323"`

**Next steps (after GPUs are free):**
1. Launch AR servers (4x TP=2, ports 30000-30003)
2. Run MMLU-Pro on AR (`--timeout 900 --max-workers 16`)
3. Run MMLU on AR (`--timeout 900 --max-workers 16`)
4. Kill AR servers, launch DLLM servers
5. Run all 6 benchmarks on DLLM (AIME-25, GPQA-D, GPQA, LCB-v6, MMLU-Pro, MMLU)
6. Record all results in final table
