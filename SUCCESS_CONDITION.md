# Success Conditions

## Criteria
1. All 15 benchmarks have results with 0 errors
2. Final Results table in PLAN.md is complete (no "—" remaining in Score column)
3. Any benchmark deviating >5pp from reference has been investigated (notes in Progress Log)

## Test Commands
```bash
# No dashes remaining in Score column (skip header and divider lines)
! grep "| — |" PLAN.md | grep -v "^|.*#\|Benchmark"
```

## Notes
- Run benchmarks SEQUENTIALLY (one at a time)
- Always verify Errors=0 before recording
- If errors > 0, reduce --max-workers to 32 and rerun
- Thinking mode: max-tokens=32768, timeout=1200
- If score deviates >5pp from reference, investigate wrong answers before continuing
