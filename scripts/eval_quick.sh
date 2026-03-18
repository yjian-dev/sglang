#!/bin/bash
# Quick evaluation suite for DLLM models
# Runs fast benchmarks across 8 sglang servers (ports 30000-30007)
#
# Usage:
#   bash scripts/eval_quick.sh                    # all benchmarks
#   bash scripts/eval_quick.sh gsm8k              # single benchmark
#   bash scripts/eval_quick.sh gsm8k humaneval    # specific benchmarks
#   TAG=n3_sampling bash scripts/eval_quick.sh    # with custom tag

set -e

PORTS="30000 30001 30002 30003 30004 30005 30006 30007"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTDIR="bench_results/eval_${TAG}"
mkdir -p "$OUTDIR"

# Default: run all fast benchmarks
BENCHMARKS="${@:-gsm8k aime humaneval mbpp math500}"

echo "============================================"
echo "Quick Eval Suite"
echo "Tag: $TAG"
echo "Output: $OUTDIR"
echo "Benchmarks: $BENCHMARKS"
echo "Ports: $PORTS"
echo "============================================"

for bench in $BENCHMARKS; do
  echo ""
  echo ">>> Running: $bench"
  echo "--------------------------------------------"

  case $bench in
    gsm8k)
      # GSM8K: 1319 problems, ~2-5 min
      python scripts/eval_gsm8k.py \
        --num-problems 1319 \
        --ports $PORTS \
        --max-tokens 8192 \
        --temperature 1.0 --top-p 0.95 --top-k 50 \
        --output-dir "$OUTDIR" --tag "$TAG" \
        2>&1 | tee "$OUTDIR/gsm8k.log"
      ;;

    gsm8k_quick)
      # GSM8K quick: 200 problems
      python scripts/eval_gsm8k.py \
        --num-problems 200 \
        --ports $PORTS \
        --max-tokens 8192 \
        --temperature 1.0 --top-p 0.95 --top-k 50 \
        --output-dir "$OUTDIR" --tag "$TAG" \
        2>&1 | tee "$OUTDIR/gsm8k_quick.log"
      ;;

    aime)
      # AIME 2025: 30 problems, ~1-2 min
      python scripts/eval_aime.py \
        --year 2025 \
        --num-problems 30 \
        --num-samples 1 \
        --ports $PORTS \
        --max-tokens 16384 \
        --temperature 1.0 --top-p 0.95 --top-k 50 \
        --output-dir "$OUTDIR" \
        2>&1 | tee "$OUTDIR/aime2025.log"
      ;;

    aime_vote)
      # AIME 2025 with majority voting: 30 problems x 5 samples
      python scripts/eval_aime.py \
        --year 2025 \
        --num-problems 30 \
        --num-samples 5 \
        --ports $PORTS \
        --max-tokens 16384 \
        --temperature 1.0 --top-p 0.95 --top-k 50 \
        --output-dir "$OUTDIR" \
        2>&1 | tee "$OUTDIR/aime2025_vote5.log"
      ;;

    humaneval)
      # HumanEval: 164 problems, ~1 min
      python scripts/eval_humaneval.py \
        --num-problems 164 \
        --ports $PORTS \
        --max-tokens 2048 \
        --temperature 1.0 --top-p 0.95 --top-k 50 \
        --output-dir "$OUTDIR" \
        2>&1 | tee "$OUTDIR/humaneval.log"
      ;;

    mbpp)
      # MBPP sanitized: ~257 problems, ~1-2 min
      python scripts/eval_mbpp.py \
        --ports $PORTS \
        --max-tokens 2048 \
        --temperature 1.0 --top-p 0.95 --top-k 50 \
        --output-dir "$OUTDIR" \
        2>&1 | tee "$OUTDIR/mbpp.log"
      ;;

    math500)
      # MATH-500: 500 problems, ~2-3 min
      python scripts/eval_math500.py \
        --num-problems 500 \
        --ports $PORTS \
        --max-tokens 8192 \
        --temperature 1.0 --top-p 0.95 --top-k 50 \
        --output-dir "$OUTDIR" \
        2>&1 | tee "$OUTDIR/math500.log"
      ;;

    ifeval)
      # IFEval: 541 prompts, ~2-3 min
      python scripts/eval_ifeval.py \
        --ports $PORTS \
        --max-tokens 4096 \
        --temperature 1.0 --top-p 0.95 --top-k 50 \
        --output-dir "$OUTDIR" \
        2>&1 | tee "$OUTDIR/ifeval.log"
      ;;

    mmlu)
      # MMLU via simple-eval: slower, ~10 min
      python -m sglang.test.run_eval \
        --port 30000 \
        --eval-name mmlu \
        --num-examples 500 \
        2>&1 | tee "$OUTDIR/mmlu.log"
      ;;

    *)
      echo "Unknown benchmark: $bench"
      echo "Available: gsm8k gsm8k_quick aime aime_vote humaneval mbpp math500 ifeval mmlu"
      ;;
  esac
done

echo ""
echo "============================================"
echo "All benchmarks done! Results in: $OUTDIR"
echo "============================================"
