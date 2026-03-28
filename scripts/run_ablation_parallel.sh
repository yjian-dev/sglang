#!/bin/bash
# Parallel infrastructure ablation: 6 configs on 6 GPUs simultaneously.
#
# Phase 1: Launch all 6 servers + warmup
# Phase 2: Run real benchmarks on all servers
# Phase 3: Collect and merge results
#
# Usage:
#   bash scripts/run_ablation_parallel.sh

set -euo pipefail

MODEL="/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-causal-from-b2amc_fixed2_cont"
ALGO_CONFIG="dreamshift_blockN3_config.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="ablation_results_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# CUDA 12.9
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

# Use sglang conda env python + add its bin to PATH for ninja etc.
PYTHON=/home/yjian/miniconda3/envs/sglang/bin/python
export PATH=/home/yjian/miniconda3/envs/sglang/bin:$PATH

# GPU assignment: 6 free GPUs for 6 configs
GPUS=(1 2 4 5 6 7)
PORTS=(31101 31102 31104 31105 31106 31107)

# Config names (matching ABLATION_CONFIGS indices 0-5)
CONFIG_NAMES=(
    "0_naive"
    "1_cuda_graph"
    "2_decode_loop"
    "3_argmax_proposals"
    "4_fused_verify"
    "5_full_system"
)

# Env vars for each config (space-separated KEY=VAL pairs)
CONFIG_ENVS=(
    # 0: Naive — all disabled
    "SGLANG_ABLATION_NO_DECODE_LOOP=1 SGLANG_ABLATION_NO_ARGMAX_PROPOSALS=1 SGLANG_ABLATION_NO_FUSED_VERIFY=1 SGLANG_ABLATION_NO_DEFERRED_STREAM=1"
    # 1: + CUDA graph
    "SGLANG_ABLATION_NO_DECODE_LOOP=1 SGLANG_ABLATION_NO_ARGMAX_PROPOSALS=1 SGLANG_ABLATION_NO_FUSED_VERIFY=1 SGLANG_ABLATION_NO_DEFERRED_STREAM=1"
    # 2: + Decode loop
    "SGLANG_ABLATION_NO_ARGMAX_PROPOSALS=1 SGLANG_ABLATION_NO_FUSED_VERIFY=1 SGLANG_ABLATION_NO_DEFERRED_STREAM=1"
    # 3: + Argmax proposals
    "SGLANG_ABLATION_NO_FUSED_VERIFY=1 SGLANG_ABLATION_NO_DEFERRED_STREAM=1"
    # 4: + Fused verify
    "SGLANG_ABLATION_NO_DEFERRED_STREAM=1"
    # 5: Full system
    ""
)

# Extra CLI flags per config
CONFIG_CLI=(
    "--disable-cuda-graph"   # 0: no CUDA graph
    ""                       # 1: CUDA graph enabled
    ""                       # 2
    ""                       # 3
    ""                       # 4
    ""                       # 5
)

PIDS=()

cleanup() {
    echo ""
    echo ">>> Cleaning up all servers..."
    for pid in "${PIDS[@]}"; do
        kill -TERM "-$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

echo "=============================================="
echo "Parallel Ablation: 6 configs × 6 H100 GPUs"
echo "Output: $OUTPUT_DIR/"
echo "=============================================="

# ══════════════════════════════════════════════════
# Phase 1: Launch servers (first one alone to build JIT cache, then rest)
# ══════════════════════════════════════════════════

launch_server() {
    local idx=$1
    local gpu=${GPUS[$idx]}
    local port=${PORTS[$idx]}
    local name=${CONFIG_NAMES[$idx]}
    local envs=${CONFIG_ENVS[$idx]}
    local cli=${CONFIG_CLI[$idx]}

    echo "  [$name] GPU=$gpu PORT=$port"

    # Build env
    local env_cmd=""
    for kv in $envs; do
        env_cmd="$env_cmd export $kv;"
    done

    # Launch server in background, log to file
    (
        eval "$env_cmd"
        export CUDA_VISIBLE_DEVICES=$gpu
        $PYTHON -m sglang.launch_server \
            --model-path "$MODEL" \
            --trust-remote-code \
            --tp-size 1 \
            --port $port \
            --mem-fraction-static 0.85 \
            --max-running-requests 64 \
            --attention-backend flashinfer \
            --dllm-algorithm DreamShiftBlockN \
            --dllm-algorithm-config "$ALGO_CONFIG" \
            --dtype bfloat16 \
            $cli \
            2>&1
    ) > "$OUTPUT_DIR/${name}_server.log" 2>&1 &
    PIDS+=($!)
}

echo ""
echo ">>> Phase 1a: Launch first server (builds flashinfer JIT cache)..."
launch_server 0

# Wait for first server to be ready (JIT compiled)
echo "  Waiting for first server to build JIT cache..."
FIRST_TIMEOUT=600
FIRST_START=$(date +%s)
while true; do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORTS[0]}/health" 2>/dev/null | grep -q "200"; then
        echo "  [${CONFIG_NAMES[0]}] READY — JIT cache built"
        break
    fi
    ELAPSED=$(( $(date +%s) - FIRST_START ))
    if [ $ELAPSED -gt $FIRST_TIMEOUT ]; then
        echo "  ERROR: First server timed out. Check $OUTPUT_DIR/${CONFIG_NAMES[0]}_server.log"
        tail -30 "$OUTPUT_DIR/${CONFIG_NAMES[0]}_server.log"
        exit 1
    fi
    sleep 3
done

echo ""
echo ">>> Phase 1b: Launch remaining 5 servers (JIT cache already built)..."
for i in 1 2 3 4 5; do
    launch_server $i
done

# ══════════════════════════════════════════════════
# Wait for all servers to be ready
# ══════════════════════════════════════════════════
echo ""
echo ">>> Waiting for all servers to be ready (timeout 5min)..."

READY=()
for i in "${!GPUS[@]}"; do
    port=${PORTS[$i]}
    name=${CONFIG_NAMES[$i]}
    if [ "$i" == "0" ]; then
        READY+=("1")  # First server already ready
    else
        READY+=("0")
    fi
done

TIMEOUT=300
START_TIME=$(date +%s)
while true; do
    ALL_READY=true
    for i in "${!PORTS[@]}"; do
        if [ "${READY[$i]}" == "0" ]; then
            if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORTS[$i]}/health" 2>/dev/null | grep -q "200"; then
                READY[$i]="1"
                echo "  [${CONFIG_NAMES[$i]}] READY (port ${PORTS[$i]})"
            else
                ALL_READY=false
            fi
        fi
    done

    if $ALL_READY; then
        echo "  All servers ready!"
        break
    fi

    ELAPSED=$(( $(date +%s) - START_TIME ))
    if [ $ELAPSED -gt $TIMEOUT ]; then
        echo "  ERROR: Timeout waiting for servers!"
        for i in "${!PORTS[@]}"; do
            if [ "${READY[$i]}" == "0" ]; then
                echo "    [${CONFIG_NAMES[$i]}] NOT READY — check $OUTPUT_DIR/${CONFIG_NAMES[$i]}_server.log"
            fi
        done
        exit 1
    fi
    sleep 3
done

# ══════════════════════════════════════════════════
# Phase 2a: Warmup all servers (5 requests each)
# ══════════════════════════════════════════════════
echo ""
echo ">>> Phase 2a: Warming up all servers (5 requests each)..."

for i in "${!PORTS[@]}"; do
    port=${PORTS[$i]}
    name=${CONFIG_NAMES[$i]}
    (
        $PYTHON -c "
import asyncio, aiohttp, time

async def warmup():
    url = 'http://localhost:${port}/v1/chat/completions'
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for j in range(5):
            body = {
                'model': 'test',
                'messages': [{'role': 'user', 'content': f'What is {j+1}+{j+1}? Think step by step.'}],
                'max_tokens': 256,
                'temperature': 1.0, 'top_k': 50, 'top_p': 0.95,
            }
            async with session.post(url, json=body) as resp:
                data = await resp.json()
                tokens = data.get('usage', {}).get('completion_tokens', 0)
    print(f'  [$name] warmup done')

asyncio.run(warmup())
" 2>&1
    ) &
done
wait
echo "  All warmup done."

# Extended warmup skipped — short warmup above is sufficient

# ══════════════════════════════════════════════════
# Phase 3: Real benchmarks — C=1 and C=32
# ══════════════════════════════════════════════════
echo ""
echo ">>> Phase 3: Real benchmarks..."

# C=1: 8 requests, max_tokens=1024
echo ""
echo "--- Concurrency = 1 (8 requests, max_tokens=1024) ---"
for i in "${!PORTS[@]}"; do
    port=${PORTS[$i]}
    name=${CONFIG_NAMES[$i]}
    (
        $PYTHON scripts/bench_concurrent_throughput.py \
            --port $port --model test \
            --concurrency 1 --num-requests 8 --max-tokens 1024 \
            --skip-warmup \
            2>&1 | tee "$OUTPUT_DIR/${name}_c1.log" | tail -3 | sed "s/^/  [$name] /"
    ) &
done
wait

# C=8: 16 requests
echo ""
echo "--- Concurrency = 8 (16 requests, max_tokens=1024) ---"
for i in "${!PORTS[@]}"; do
    port=${PORTS[$i]}
    name=${CONFIG_NAMES[$i]}
    (
        $PYTHON scripts/bench_concurrent_throughput.py \
            --port $port --model test \
            --concurrency 8 --num-requests 16 --max-tokens 1024 \
            --skip-warmup \
            2>&1 | tee "$OUTPUT_DIR/${name}_c8.log" | tail -3 | sed "s/^/  [$name] /"
    ) &
done
wait

# C=32: 64 requests
echo ""
echo "--- Concurrency = 32 (64 requests, max_tokens=1024) ---"
for i in "${!PORTS[@]}"; do
    port=${PORTS[$i]}
    name=${CONFIG_NAMES[$i]}
    (
        $PYTHON scripts/bench_concurrent_throughput.py \
            --port $port --model test \
            --concurrency 32 --num-requests 64 --max-tokens 1024 \
            --skip-warmup \
            2>&1 | tee "$OUTPUT_DIR/${name}_c32.log" | tail -3 | sed "s/^/  [$name] /"
    ) &
done
wait

# ══════════════════════════════════════════════════
# Phase 4: Collect results
# ══════════════════════════════════════════════════
echo ""
echo "=============================================="
echo ">>> Phase 4: Collecting results..."
echo "=============================================="

$PYTHON -c "
import re, os, json

output_dir = '$OUTPUT_DIR'
configs = $( printf '%s\n' "${CONFIG_NAMES[@]}" | $PYTHON -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin]))" )
concurrencies = [1, 8, 32]
results = []

for name in configs:
    for c in concurrencies:
        logfile = os.path.join(output_dir, f'{name}_c{c}.log')
        if not os.path.exists(logfile):
            continue
        with open(logfile) as f:
            text = f.read()
        # Parse TPS from bench_concurrent_throughput.py output
        m = re.search(r'TPS=([0-9.]+)', text)
        tps = float(m.group(1)) if m else 0
        m2 = re.search(r'tokens=(\d+)', text)
        tokens = int(m2.group(1)) if m2 else 0
        m3 = re.search(r'wall=([0-9.]+)', text)
        wall = float(m3.group(1)) if m3 else 0
        m4 = re.search(r'avg_tok/req=(\d+)', text)
        avg_tok = int(m4.group(1)) if m4 else 0
        results.append({
            'config': name, 'concurrency': c,
            'tps': tps, 'tokens': tokens, 'wall_time': wall,
            'avg_tok_per_req': avg_tok,
        })

# Save JSON
with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
labels = {
    '0_naive': 'Naive (eager, full sched)',
    '1_cuda_graph': '+ CUDA graph',
    '2_decode_loop': '+ Decode inner loop',
    '3_argmax_proposals': '+ Argmax proposals',
    '4_fused_verify': '+ Fused verify kernel',
    '5_full_system': 'Full system',
}

for c in concurrencies:
    print(f'\n--- Concurrency = {c} ---')
    print(f\"  {'Config':<35} {'TPS':>8} {'Δ':>8}\")
    print(f\"  {'-'*51}\")
    prev = None
    for name in configs:
        row = next((r for r in results if r['config'] == name and r['concurrency'] == c), None)
        if row:
            tps = row['tps']
            if prev:
                delta = f'+{(tps-prev)/prev*100:.0f}%'
            else:
                delta = '---'
            label = labels.get(name, name)
            print(f'  {label:<35} {tps:>8.1f} {delta:>8}')
            prev = tps

# Generate LaTeX
print('\n--- LaTeX Table ---')
print(r'\begin{table}[t]')
print(r'\centering')
print(r'\caption{\textbf{Infrastructure optimization ablation} (\$N{=}3\$, 1\$\\times\$H100, bf16). Each row cumulatively enables one optimization.}')
print(r'\label{tab:infra_ablation}')
print(r'\small')
cstr = ' & '.join([f'\$C{{=}}{c}\$' for c in concurrencies])
print(f'\\\\begin{{tabular}}{{l{\"c\" * len(concurrencies)}}}')
print(r'\toprule')
print(f'\\\\textbf{{Configuration}} & {cstr} \\\\\\\\')
print(r'\midrule')
for name in configs:
    label = labels.get(name, name)
    vals = []
    for c in concurrencies:
        row = next((r for r in results if r['config'] == name and r['concurrency'] == c), None)
        vals.append(f'{row[\"tps\"]:.0f}' if row and row['tps'] > 0 else '---')
    print(f'{label} & {\" & \".join(vals)} \\\\\\\\')
print(r'\bottomrule')
print(r'\end{tabular}')
print(r'\end{table}')
"

echo ""
echo "=============================================="
echo "All results saved to: $OUTPUT_DIR/"
echo "=============================================="
ls -la "$OUTPUT_DIR/"
