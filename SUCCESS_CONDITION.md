# Success Condition

## Done When
All 13 benchmarks × 5 runs completed. PLAN.md Results table fully filled with mean ± std for each benchmark.

## Minimum Requirement
- 5 complete runs for each of the 13 benchmarks
- Mean and std computed for each benchmark
- Any benchmark with std > 3pp flagged as high-variance

## Test Command
```bash
# Verify servers
for i in $(seq 0 7); do curl -sf http://localhost:$((30000+i))/health && echo "GPU $i OK" || echo "GPU $i DOWN"; done

# Quick sanity (5 problems of GSM8K)
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
python scripts/eval_gsm8k.py --num-problems 5 --ports 30000 --max-tokens 32768
```

## Output
Results saved in `bench_results/n3_run1/` through `bench_results/n3_run5/`.
Final summary table in PLAN.md.
