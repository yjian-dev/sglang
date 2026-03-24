# Success Conditions — Iteration 3

## Criteria
1. `tab:throughput` in experiments.tex has real TPS numbers for at least 3 configurations at C=1,4,8,16,32,64
2. `conclusion.tex` has meaningful content (not empty)
3. No remaining XX placeholders in analysis.tex

## Test Commands
The following commands must all pass (exit code 0):

```bash
# tab:throughput should have real numbers (not XX) for at least the first row
! grep -A10 'tab:throughput' docs/Dllm_colm_2026/sections/experiments.tex | grep -c 'XX' | grep -q '^0$' || grep -A10 'tab:throughput' docs/Dllm_colm_2026/sections/experiments.tex | grep -v 'XX' | grep -c '[0-9]' | grep -qv '^0$'

# conclusion.tex should have content beyond just the section header
test $(wc -l < docs/Dllm_colm_2026/sections/conclusion.tex) -gt 5

# analysis.tex should have no XX placeholders
! grep 'XX' docs/Dllm_colm_2026/sections/analysis.tex
```

## Notes
- Throughput measurements require concurrent benchmark (multiple simultaneous requests)
- AR baseline: launch Qwen3-8B without --dllm-algorithm flag
- Use `python -m sglang.bench_serving` for concurrent benchmarks
- If bench_serving is complex, use a simple Python script with ThreadPoolExecutor
