# Success Condition

## Done When
All 3 models × 8 benchmarks × 2 runs = **48 results** filled in PLAN.md.

## Quality Gates
- Errors < 5% per benchmark run
- No server crashes left unresolved
- Means computed for each model × benchmark

## Test Command
```bash
for i in $(seq 0 7); do curl -sf http://localhost:$((30000+i))/health && echo "GPU $i OK" || echo "GPU $i DOWN"; done
```
