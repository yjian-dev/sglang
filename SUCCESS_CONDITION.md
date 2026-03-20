# Success Condition

## Phase 0 Done When
IFEval prompt-strict accuracy ≥ **80%** (within 3pp of claimed 83.18%).
- Zero servers crashed during eval
- Repetition rate in outputs < 5%
- No systematic extraction failures

## Phase 1 Done When
All 16 benchmarks in PLAN.md results table have scores for LLaDA2.1-mini.

## Quality Gates
- Repetition check: sample 5 wrong answers per benchmark, < 2 should show "of of of" style corruption
- Extraction: pred='?' < 5%
- Truncation: finish_reason='length' < 15%

## Test Commands
```bash
# Health check
for i in $(seq 0 7); do curl -sf http://localhost:$((30000+i))/health && echo "GPU $i OK"; done

# Quick IFEval sanity (5 problems)
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
python scripts/eval_ifeval.py --ports 30000 --num-problems 5 --max-tokens 16384 --max-workers 1

# Check output quality manually
cat output_ifeval/ifeval_details.json | python3 -c "
import json, sys
for d in json.load(sys.stdin)[:5]:
    rep = 'REPETITION' if any(w*3 in d['prediction_stripped'] for w in d['prediction_stripped'].split()[:20] if len(w) > 2) else 'OK'
    print(rep, d['strict_pass'], d['prediction_stripped'][:100])
"
```
