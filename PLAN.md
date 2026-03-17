# Plan

## Status
Starting comprehensive benchmark run.

## Goal
Measure throughput and quality for 8 configs × 2 datasets × 8 concurrency levels + quality benchmarks. Record everything in `bench_results/comprehensive_benchmark.md`.

## Config Order (run sequentially)
1. [ ] N=3 sampling (temp=1.0)
2. [ ] N=3 greedy (temp=0.0)
3. [ ] N=4 sampling (temp=1.0)
4. [ ] N=4 greedy (temp=0.0)
5. [ ] N=5 sampling (temp=1.0)
6. [ ] N=5 greedy (temp=0.0)
7. [ ] Qwen3-8B AR
8. [ ] EAGLE3 (Tengyunw)

## Per Config Steps
- [ ] Kill old servers, launch 8x TP=1 servers
- [ ] Wait for all 8 healthy
- [ ] Throughput: AIME C=1,2,4,8,16,32,48,64
- [ ] Throughput: ShareGPT C=1,2,4,8,16,32,48,64
- [ ] Quality: GSM8K, HumanEval, IFEval, MBPP, MATH-500 (max_tokens=16384)
- [ ] For sampling configs: also test temp=0.6 quality
- [ ] Record all results in bench_results/comprehensive_benchmark.md
- [ ] Update PLAN.md progress

## Progress Log
<!-- Agent updates this -->


## Evaluator Feedback (Iteration 1)
Could not parse evaluator response.


## Evaluator Feedback (Iteration 2)
Could not parse evaluator response.


## Evaluator Feedback (Iteration 3)
Could not parse evaluator response.
