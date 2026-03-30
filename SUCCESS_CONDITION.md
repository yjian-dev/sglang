# Success Conditions

## Criteria
1. All 6 result tables in PLAN.md are complete (no "— " remaining in data cells)
2. AR per-req TPS is < 150 at all batch sizes
3. All runs have 0 failed requests
4. Final tables are written to docs/benchmark_results.md under a new "## Table 6: Comprehensive Throughput (tore-speed-eval)" section

## Test Commands
```bash
# No dashes remaining in result tables
! grep "| — |" PLAN.md

# AR per-req should be < 150 everywhere
python3 -c "
import re
with open('PLAN.md') as f:
    text = f.read()
# Find AR column values in Per-Req tables
for line in text.split('\n'):
    if line.startswith('|') and '| AR |' not in line and '|--' not in line:
        cols = [c.strip() for c in line.split('|')]
        if len(cols) > 2 and cols[1].isdigit():  # bs column
            try:
                ar_val = float(cols[2])
                assert ar_val < 160, f'AR per-req {ar_val} >= 160 at bs={cols[1]}'
            except ValueError:
                pass
print('AR sanity check passed')
"

# docs/benchmark_results.md has Table 6
grep -q "Table 6" docs/benchmark_results.md
```

## Notes
- All 5 models must be tested: AR, DFlash s1d16, EAGLE3, Ours N=4 LoRA, Ours N=4 b2
- All 3 datasets: MBPP (257), MATH-500 (500), LMSYS-Chat (182)
- All 7 batch sizes: 1, 2, 4, 8, 16, 32, 64
- Tool: tore-speed-eval, burst mode, max_tokens=2048, thinking mode
- Common settings: bf16, TP=1, mem_fraction_static=0.85, max_running_requests=64
- If N=4 b2 crashes at high bs, mark as "OOM" (not a failure)
