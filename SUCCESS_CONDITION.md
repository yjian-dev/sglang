# Success Conditions

## Criteria
1. All 8 configs have throughput data for BOTH datasets (AIME + ShareGPT) at ALL 8 concurrency levels (1,2,4,8,16,32,48,64)
2. All 8 configs have quality data for ALL 5 benchmarks (GSM8K, HumanEval, IFEval, MBPP, MATH-500)
3. Sampling configs (N=3/4/5 sampling) also have temp=0.6 quality data
4. All results recorded in `bench_results/comprehensive_benchmark.md`
5. No suspiciously low quality scores (< 50% on GSM8K) — if found, script was fixed and re-run

## Test Commands
```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
test -f bench_results/comprehensive_benchmark.md && echo "Results file exists" || exit 1
grep -c "N=3 sampling" bench_results/comprehensive_benchmark.md | grep -v "^0$" || exit 1
grep -c "EAGLE3" bench_results/comprehensive_benchmark.md | grep -v "^0$" || exit 1
grep -c "ShareGPT" bench_results/comprehensive_benchmark.md | grep -v "^0$" || exit 1
grep -c "GSM8K" bench_results/comprehensive_benchmark.md | grep -v "^0$" || exit 1
echo "All checks passed"
```

## Notes
- Each config runs on 8x TP=1 servers (one per GPU, ports 30000-30007)
- Throughput uses tore-speed-eval burst mode against port 30000
- Quality uses eval scripts from scripts/ with max_tokens=16384, distributed across 8 ports
- If eval scripts have parsing bugs causing low scores, fix and re-run
