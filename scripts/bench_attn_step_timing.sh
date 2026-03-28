#!/bin/bash
# Benchmark per-step timing: paged-only vs cascade attention at C=1..64
# Outputs step time breakdown for each concurrency level.
set -euo pipefail

MODEL="/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-causal-from-b2amc_fixed2_cont"
ALGO_CONFIG="dreamshift_blockN3_config.yaml"
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
PYTHON=/home/yjian/miniconda3/envs/sglang/bin/python
OUT="ablation_step_timing_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do kill -TERM "-$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Step Timing: Paged-only vs Cascade ==="

# Launch paged-only on GPU 4
echo ">>> Launching paged-only (GPU 4, port 31601)..."
(
    export CUDA_VISIBLE_DEVICES=4
    $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" --trust-remote-code --tp-size 1 \
        --port 31601 --mem-fraction-static 0.85 --max-running-requests 128 \
        --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
        --dllm-algorithm-config "$ALGO_CONFIG" --dtype bfloat16 \
        --decode-log-interval 1 2>&1
) > "$OUT/paged_server.log" 2>&1 &
PIDS+=($!)

# Wait for paged-only (JIT cache)
echo "  Waiting for paged-only..."
for i in $(seq 1 200); do
    curl -s -o /dev/null -w "%{http_code}" "http://localhost:31601/health" 2>/dev/null | grep -q "200" && echo "  READY" && break
    sleep 3
done

# Launch cascade on GPU 5
echo ">>> Launching cascade (GPU 5, port 31602)..."
(
    export CUDA_VISIBLE_DEVICES=5
    export SGLANG_ABLATION_DLLM_USE_RAGGED=1
    $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" --trust-remote-code --tp-size 1 \
        --port 31602 --mem-fraction-static 0.85 --max-running-requests 128 \
        --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
        --dllm-algorithm-config "$ALGO_CONFIG" --dtype bfloat16 \
        --decode-log-interval 1 2>&1
) > "$OUT/cascade_server.log" 2>&1 &
PIDS+=($!)

echo "  Waiting for cascade..."
for i in $(seq 1 200); do
    curl -s -o /dev/null -w "%{http_code}" "http://localhost:31602/health" 2>/dev/null | grep -q "200" && echo "  READY" && break
    sleep 3
done

# Warmup both
echo ">>> Warmup..."
for port in 31601 31602; do
    $PYTHON -c "
import asyncio, aiohttp
async def warmup():
    url = 'http://localhost:$port/v1/chat/completions'
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for j in range(5):
            body = {'model': 'test', 'messages': [{'role': 'user', 'content': f'What is {j+1}*{j+2}?'}], 'max_tokens': 256, 'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}
            async with session.post(url, json=body) as resp:
                await resp.json()
asyncio.run(warmup())
"
done
echo "  Done."

# Benchmark at each concurrency level
# Use enough requests so that the decode loop runs at steady state
for c in 1 2 4 8 16 32 64; do
    # Requests: enough to get ~500+ decode steps at steady state
    if [ $c -le 4 ]; then
        n=$((c * 4))
        [ $n -lt 8 ] && n=8
        maxtok=1024
    elif [ $c -le 16 ]; then
        n=$((c * 2))
        maxtok=512
    else
        n=$((c * 2))
        [ $n -gt 128 ] && n=128
        maxtok=512
    fi

    echo ""
    echo "--- C=$c (n=$n, max_tokens=$maxtok) ---"

    # Record log positions before benchmark
    PAGED_BEFORE=$(wc -l < "$OUT/paged_server.log")
    CASCADE_BEFORE=$(wc -l < "$OUT/cascade_server.log")

    # Run both in parallel
    $PYTHON scripts/bench_concurrent_throughput.py --port 31601 --model test \
        --concurrency $c --num-requests $n --max-tokens $maxtok --skip-warmup \
        2>&1 | grep "TPS=" | sed "s/^/  [paged] /" &
    PID1=$!
    $PYTHON scripts/bench_concurrent_throughput.py --port 31602 --model test \
        --concurrency $c --num-requests $n --max-tokens $maxtok --skip-warmup \
        2>&1 | grep "TPS=" | sed "s/^/  [cascade] /" &
    PID2=$!
    wait $PID1 $PID2

    sleep 2

    # Extract step timing from new log lines
    for mode in paged cascade; do
        if [ "$mode" = "paged" ]; then
            logfile="$OUT/paged_server.log"
            before=$PAGED_BEFORE
        else
            logfile="$OUT/cascade_server.log"
            before=$CASCADE_BEFORE
        fi

        # Get decode loop summary lines (has step=XXms and total time)
        tail -n +$((before+1)) "$logfile" | grep "DLLM decode loop" | tail -3 > "$OUT/${mode}_c${c}_loops.txt"

        # Get step profile lines (every 500 steps)
        tail -n +$((before+1)) "$logfile" | grep "DLLM step profile" | tail -3 > "$OUT/${mode}_c${c}_profile.txt"

        # Parse average step time from decode loop summary
        avg_step=$(tail -n +$((before+1)) "$logfile" | grep "DLLM decode loop" | \
            grep -o "step=[0-9.]*ms" | awk -F'[=m]' '{sum+=$2; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}')

        # Parse forward % from decode loop summary
        fwd_pct=$(tail -n +$((before+1)) "$logfile" | grep "DLLM decode loop" | \
            grep -o "fwd=[0-9]*%" | tail -1 | grep -o "[0-9]*")

        # Parse step profile for detailed breakdown
        profile_line=$(tail -n +$((before+1)) "$logfile" | grep "DLLM step profile" | tail -1)
        if [ -n "$profile_line" ]; then
            step_us=$(echo "$profile_line" | grep -o "step=[0-9]*" | head -1 | cut -d= -f2)
            fwd_us=$(echo "$profile_line" | grep -o "forward_call=[0-9]*" | head -1 | cut -d= -f2)
            prep_us=$(echo "$profile_line" | grep -o "prep=[0-9]*" | head -1 | cut -d= -f2)
            proc_us=$(echo "$profile_line" | grep -o "process=[0-9]*" | head -1 | cut -d= -f2)
            bs=$(echo "$profile_line" | grep -o "bs=[0-9]*" | head -1 | cut -d= -f2)
            echo "  [$mode] C=$c bs=$bs step=${step_us}us fwd=${fwd_us}us prep=${prep_us}us proc=${proc_us}us (loop avg: ${avg_step}ms)"
        else
            echo "  [$mode] C=$c step=${avg_step}ms fwd=${fwd_pct:-?}%"
        fi
    done
done

echo ""
echo "=== Final Summary ==="
echo ""
printf "%-5s | %-35s | %-35s\n" "C" "Paged-only (step_us / fwd_us)" "Cascade (step_us / fwd_us)"
printf "%-5s-+-%-35s-+-%-35s\n" "-----" "-----------------------------------" "-----------------------------------"

for c in 1 2 4 8 16 32 64; do
    paged_info=""
    cascade_info=""
    for mode in paged cascade; do
        logfile="$OUT/${mode}_server.log"
        profile_line=$(grep "DLLM step profile" "$logfile" | grep "bs=$c " | tail -1)
        if [ -z "$profile_line" ]; then
            # Try without exact bs match (use last profile before the concurrency changed)
            profile_line=$(cat "$OUT/${mode}_c${c}_profile.txt" 2>/dev/null | tail -1)
        fi
        if [ -n "$profile_line" ]; then
            step_us=$(echo "$profile_line" | grep -o "step=[0-9]*" | head -1 | cut -d= -f2)
            fwd_us=$(echo "$profile_line" | grep -o "forward_call=[0-9]*" | head -1 | cut -d= -f2)
            bs=$(echo "$profile_line" | grep -o "bs=[0-9]*" | head -1 | cut -d= -f2)
            info="bs=${bs} step=${step_us}us fwd=${fwd_us}us"
        else
            # Fall back to loop summary
            loop_line=$(cat "$OUT/${mode}_c${c}_loops.txt" 2>/dev/null | tail -1)
            if [ -n "$loop_line" ]; then
                avg_step=$(echo "$loop_line" | grep -o "step=[0-9.]*ms" | cut -d= -f2 | cut -dm -f1)
                info="step=${avg_step}ms"
            else
                info="N/A"
            fi
        fi
        if [ "$mode" = "paged" ]; then paged_info="$info"; else cascade_info="$info"; fi
    done
    printf "%-5s | %-35s | %-35s\n" "$c" "$paged_info" "$cascade_info"
done

echo ""
echo "Results saved to $OUT/"
