# Success Condition

## Done When
All 13 benchmarks in PLAN.md results table have scores for b3 N=4 sampling.

## Anomaly Gate
- If any benchmark is > 15pp lower than N=3 reference: STOP and report to user
- Otherwise: complete all benchmarks and report final comparison table

## Test Commands
```bash
# Verify servers running
for i in $(seq 0 7); do curl -sf http://localhost:$((30000+i))/health && echo "GPU $i OK" || echo "GPU $i DOWN"; done

# Sanity test
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
python scripts/eval_gsm8k.py --num-problems 5 --ports 30000
```

## Final Report Format
Print a table comparing b3 N=4 vs b2 N=3 with delta for each benchmark.
