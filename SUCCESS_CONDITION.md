# Success Condition

## Done When
All 15 rows in PLAN.md results table are filled for both DLLM N=3 and Qwen3-8B, with quality checks passed.

## Quality Requirements Per Benchmark
- Truncation rate < 15% (at 32k tokens, higher is acceptable for thinking-heavy tasks)
- Extraction failure < 5%
- Prompt verified against OpenCompass implementation

## Test Command
```bash
# Verify DLLM servers running
for i in $(seq 0 7); do curl -sf http://localhost:$((30000+i))/health > /dev/null 2>&1 && echo "GPU $i OK" || echo "GPU $i DOWN"; done

# Quick sanity check
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
python scripts/eval_gsm8k.py --num-problems 5 --ports 30000
```

## Completion Criteria
- [ ] All 15 DLLM results recorded
- [ ] All 15 Qwen3-8B results recorded
- [ ] Each result has quality check notes
- [ ] No systematic extraction bugs remaining
