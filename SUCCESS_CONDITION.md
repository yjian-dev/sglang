# Success Conditions

## Criteria
1. All 6 remaining benchmarks have results for BOTH AR and DLLM with 0 errors
2. Final comparison table in PLAN.md with all 14 benchmarks filled

## Test Commands
```bash
# PLAN.md has Final Results section
grep "Final Results" PLAN.md

# All benchmarks have values (no "need" or "redo" remaining)
! grep -E "need|redo" PLAN.md | grep -v "Completed\|do NOT"
```

## Notes
- Run benchmarks SEQUENTIALLY (one at a time)
- Always verify Errors=0 before recording
- If errors > 0, reduce --max-workers to 8 and rerun
- max-running-requests=4 is mandatory for 32B TP=2
- Full datasets only (no --num-problems)
