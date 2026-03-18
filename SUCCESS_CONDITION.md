# Success Condition

## Done When
All rows in PLAN.md results table are filled for both **Ours (DLLM N=3)** and **Qwen3-8B**.

| Benchmark | Ours | Qwen3-8B |
|-----------|------|----------|
| ARC-C | ✓ filled | ✓ filled |
| TriviaQA | ✓ filled | ✓ filled |
| MMLU | ✓ filled | ✓ filled |
| MMLU-Pro | ✓ filled (random 2k) | ✓ filled |
| GPQA-Diamond | 59.1% ✓ | 48.01% ✓ |
| IFEval | 87.4% ✓ | ✓ filled |
| GSM8K | 96% ✓ | ✓ filled |
| Math500 | 95.2% ✓ | ✓ filled |
| MathBench | ✓ filled | ✓ filled |
| AIME-2025 | 61.04% ✓ | ✓ filled |
| HumanEval+ | 93.9% ✓ | ✓ filled |
| MBPP+ | 91.8% ✓ | ✓ filled |
| HumanEval-X | 86.6% ✓ | ✓ filled |
| LCB-v6 | 45.1% ✓ | ✓ filled |
| CMMLU | ✓ filled | ✓ filled |
| MMMLU-lite | ✓ filled | ✓ filled |

## Quality Check
- Each result diagnosed for extraction/truncation issues
- Extraction failures < 5% per benchmark
- Truncation failures < 10% (if higher, re-run with larger max_tokens)

## Test Commands
```bash
# Verify DLLM servers are running
for i in $(seq 0 7); do curl -sf http://localhost:$((30000+i))/health && echo "GPU $i OK"; done

# Quick sanity check
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
python scripts/eval_gsm8k.py --num-problems 5 --ports 30000
```
