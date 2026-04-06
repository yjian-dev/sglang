# Success Conditions

## Criteria
1. DreamShift SGLang server runs on GPU 0 and produces correct code output for MBPP problems
2. JetEngine SDAR runs on GPU 1 and produces correct code output for the same MBPP problems
3. Throughput benchmark data collected for both at bs=1,4,8,16,32,64 — saved to JSON
4. A comparison plot (PNG) showing TPS vs batch_size for both engines exists
5. A side-by-side streaming demo script exists and runs cleanly
6. DreamShift shows meaningful speedup over SDAR at large batch sizes (bs>=16)
7. All comparisons are fair: same TP=1, same GPU type, same dtype, same prompts, same max_tokens

## Test Commands
The following commands must all pass (exit code 0):

```bash
# Check demo scripts exist
test -f scripts/demo/bench_comparison.py
test -f scripts/demo/bench_jetengine_worker.py
test -f scripts/demo/plot_comparison.py

# Check results exist
test -f scripts/demo/results/throughput_comparison.json
test -f scripts/demo/results/throughput_comparison.png

# Check results have data for both engines
python3 -c "
import json
data = json.load(open('scripts/demo/results/throughput_comparison.json'))
assert 'dreamshift' in data or 'DreamShift' in str(data), 'Missing DreamShift data'
assert 'sdar' in data or 'SDAR' in str(data) or 'jetengine' in str(data), 'Missing SDAR data'
print('Results validated OK')
"
```

## Notes
- All test commands must exit with code 0 for success
- The demo should be visually clean enough for screen recording
- Fairness is paramount — no tricks to inflate numbers
