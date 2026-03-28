#!/bin/bash
# DFlash comprehensive benchmark: 7 batch sizes × 3 datasets on 7 H100 GPUs
# Measures per-request latency and total throughput with thinking mode enabled.
set -euo pipefail

export PATH=/home/yjian/miniconda3/envs/dflash/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export PYTHON=/home/yjian/miniconda3/envs/dflash/bin/python
export SGLANG_PYTHON=/home/yjian/miniconda3/envs/sglang/bin/python  # for datasets

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="dflash_bench_${TIMESTAMP}"
mkdir -p "$OUT"

GPUS=(0 1 2 3 4 5 6)
BS_LIST=(1 2 4 8 16 32 64)
PORTS=(32001 32002 32003 32004 32005 32006 32007)

PIDS=()
cleanup() {
    echo ">>> Cleaning up..."
    for pid in "${PIDS[@]}"; do kill -TERM "-$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT

echo "=============================================="
echo "DFlash Benchmark: bs={1,2,4,8,16,32,64}"
echo "Datasets: AIME25, ShareGPT, Random (speed eval)"
echo "Thinking mode: enabled"
echo "Output: $OUT/"
echo "=============================================="

# ══════════════════════════════════════════════════
# Phase 1: Launch 7 DFlash servers
# ══════════════════════════════════════════════════
echo ""
echo ">>> Phase 1: Launch first server (GPU 0) for JIT cache..."

export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_DFLASH_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

launch_dflash() {
    local idx=$1
    local gpu=${GPUS[$idx]}
    local port=${PORTS[$idx]}
    local bs=${BS_LIST[$idx]}
    local max_req=$((bs + 8))  # headroom
    [ $max_req -lt 16 ] && max_req=16

    echo "  [bs=$bs] GPU=$gpu PORT=$port max_running=$max_req"
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        export SGLANG_ENABLE_SPEC_V2=1
        export SGLANG_ENABLE_DFLASH_SPEC_V2=1
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        $PYTHON -m sglang.launch_server \
            --model-path Qwen/Qwen3-8B \
            --speculative-algorithm DFLASH \
            --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
            --tp-size 1 --dtype bfloat16 --attention-backend fa3 \
            --mem-fraction-static 0.75 --trust-remote-code \
            --port $port --max-running-requests $max_req \
            2>&1
    ) > "$OUT/server_bs${bs}.log" 2>&1 &
    PIDS+=($!)
}

# First server for JIT
launch_dflash 0
echo "  Waiting for first server (JIT cache)..."
for i in $(seq 1 200); do
    curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORTS[0]}/health" 2>/dev/null | grep -q "200" && echo "  READY" && break
    sleep 3
done

echo ""
echo ">>> Launching remaining 6 servers..."
for i in 1 2 3 4 5 6; do
    launch_dflash $i
done

echo ">>> Waiting for all servers..."
for i in "${!PORTS[@]}"; do
    port=${PORTS[$i]}
    bs=${BS_LIST[$i]}
    for j in $(seq 1 200); do
        curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health" 2>/dev/null | grep -q "200" && echo "  [bs=$bs] READY" && break
        sleep 3
    done
done

# ══════════════════════════════════════════════════
# Phase 2: Warmup all servers
# ══════════════════════════════════════════════════
echo ""
echo ">>> Phase 2: Warmup..."
for i in "${!PORTS[@]}"; do
    port=${PORTS[$i]}
    bs=${BS_LIST[$i]}
    (
        $PYTHON -c "
import asyncio, aiohttp
async def warmup():
    url = 'http://localhost:$port/v1/chat/completions'
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as s:
        for j in range(3):
            body = {'model':'test','messages':[{'role':'user','content':f'What is {j+1}+{j+1}? Think step by step.'}],'max_tokens':256,'temperature':1.0,'top_k':50,'top_p':0.95,'chat_template_kwargs':{'enable_thinking':True}}
            async with s.post(url, json=body) as r: await r.json()
asyncio.run(warmup())
" 2>&1 && echo "  [bs=$bs] warmup done"
    ) &
done
wait
echo "  All warmup done."

# ══════════════════════════════════════════════════
# Phase 3: Benchmarks
# ══════════════════════════════════════════════════

# --- 3a: AIME 2025 ---
echo ""
echo "=============================================="
echo ">>> Dataset 1: AIME 2025 (thinking mode, max_tokens=2048)"
echo "=============================================="

for i in "${!BS_LIST[@]}"; do
    bs=${BS_LIST[$i]}
    port=${PORTS[$i]}
    # For AIME: use all 30 problems, C=bs
    n=$((bs * 3))
    [ $n -lt 16 ] && n=16
    [ $n -gt 60 ] && n=60
    (
        $SGLANG_PYTHON -c "
import asyncio, aiohttp, time
from datasets import load_dataset

ds = load_dataset('MathArena/aime_2025', split='test')
problems = [d['problem'] for d in ds][:$n]

async def send_one(session, url, prompt, sem):
    body = {'model':'test','messages':[{'role':'user','content':prompt}],'max_tokens':2048,'temperature':1.0,'top_k':50,'top_p':0.95,'chat_template_kwargs':{'enable_thinking':True}}
    async with sem:
        t0 = time.time()
        try:
            async with session.post(url, json=body) as resp:
                data = await resp.json()
                elapsed = time.time() - t0
                return data.get('usage',{}).get('completion_tokens',0), elapsed
        except Exception as e:
            return 0, time.time() - t0

async def run():
    url = 'http://localhost:$port/v1/chat/completions'
    sem = asyncio.Semaphore($bs)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        tasks = [send_one(session, url, p, sem) for p in problems]
        wall_start = time.time()
        results = await asyncio.gather(*tasks)
        wall_time = time.time() - wall_start
    total_tok = sum(r[0] for r in results)
    lats = [r[1] for r in results if r[0] > 0]
    tps = total_tok / wall_time if wall_time > 0 else 0
    avg_lat = sum(lats)/len(lats) if lats else 0
    per_req_tps = sum(r[0]/r[1] for r in results if r[0]>0 and r[1]>0) / len(lats) if lats else 0
    print(f'bs=$bs total_tps={tps:.1f} per_req_tps={per_req_tps:.1f} avg_lat={avg_lat:.1f}s tokens={total_tok} wall={wall_time:.1f}s n={len(problems)}')

asyncio.run(run())
" 2>&1 | tee "$OUT/aime25_bs${bs}.log"
    ) &
done
wait

# --- 3b: ShareGPT ---
echo ""
echo "=============================================="
echo ">>> Dataset 2: ShareGPT (thinking mode, burst, max_tokens=2048)"
echo "=============================================="

for i in "${!BS_LIST[@]}"; do
    bs=${BS_LIST[$i]}
    port=${PORTS[$i]}
    n=$((bs * 4))
    [ $n -lt 16 ] && n=16
    [ $n -gt 128 ] && n=128
    (
        $PYTHON -m sglang.bench_serving \
            --backend sglang \
            --port $port \
            --dataset-name sharegpt \
            --dataset-path /data/yjian/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
            --num-prompts $n \
            --request-rate inf \
            --max-concurrency $bs \
            --sharegpt-output-len 2048 \
            --output-file "$OUT/sharegpt_bs${bs}.jsonl" \
            2>&1 | tail -20 | tee "$OUT/sharegpt_bs${bs}.log"
    ) &
done
wait

# --- 3c: Random (speed eval) ---
echo ""
echo "=============================================="
echo ">>> Dataset 3: Random speed eval (burst, input=256, output=2048)"
echo "=============================================="

for i in "${!BS_LIST[@]}"; do
    bs=${BS_LIST[$i]}
    port=${PORTS[$i]}
    n=$((bs * 4))
    [ $n -lt 16 ] && n=16
    [ $n -gt 128 ] && n=128
    (
        $PYTHON -m sglang.bench_serving \
            --backend sglang \
            --port $port \
            --dataset-name random \
            --num-prompts $n \
            --request-rate inf \
            --max-concurrency $bs \
            --random-input-len 256 \
            --random-output-len 2048 \
            --output-file "$OUT/random_bs${bs}.jsonl" \
            2>&1 | tail -20 | tee "$OUT/random_bs${bs}.log"
    ) &
done
wait

# ══════════════════════════════════════════════════
# Phase 4: Collect and format results
# ══════════════════════════════════════════════════
echo ""
echo "=============================================="
echo ">>> Phase 4: Results Summary"
echo "=============================================="

$SGLANG_PYTHON -c "
import re, os, json, glob

out = '$OUT'
bs_list = [1, 2, 4, 8, 16, 32, 64]

# Parse AIME25
print('=== AIME 2025 (thinking mode) ===')
print(f'{\"bs\":>4} {\"Total TPS\":>10} {\"Per-Req TPS\":>12} {\"Avg Lat(s)\":>11} {\"Tokens\":>8}')
print('-' * 55)
for bs in bs_list:
    f = os.path.join(out, f'aime25_bs{bs}.log')
    if os.path.exists(f):
        text = open(f).read()
        m = re.search(r'total_tps=([0-9.]+).*per_req_tps=([0-9.]+).*avg_lat=([0-9.]+).*tokens=(\d+)', text)
        if m:
            print(f'{bs:>4} {float(m.group(1)):>10.1f} {float(m.group(2)):>12.1f} {float(m.group(3)):>11.1f} {int(m.group(4)):>8}')

# Parse bench_serving output (ShareGPT + Random)
for dataset in ['sharegpt', 'random']:
    name = 'ShareGPT' if dataset == 'sharegpt' else 'Random (speed eval)'
    print(f'\n=== {name} (burst mode) ===')
    print(f'{\"bs\":>4} {\"Out TPS\":>10} {\"Total TPS\":>10} {\"Avg Lat(s)\":>11} {\"TTFT(ms)\":>9} {\"ITL(ms)\":>8}')
    print('-' * 62)
    for bs in bs_list:
        f = os.path.join(out, f'{dataset}_bs{bs}.log')
        if os.path.exists(f):
            text = open(f).read()
            # Parse output token throughput
            m_out = re.search(r'Output token throughput.*?([0-9.]+)\s*tok', text)
            m_tot = re.search(r'Total token throughput.*?([0-9.]+)\s*tok', text)
            m_lat = re.search(r'Mean E2E Latency.*?([0-9.]+)', text)
            m_ttft = re.search(r'Mean TTFT.*?([0-9.]+)', text)
            m_itl = re.search(r'Mean ITL.*?([0-9.]+)', text)
            if not m_lat:
                m_lat = re.search(r'Average latency.*?([0-9.]+)', text)
            out_tps = float(m_out.group(1)) if m_out else 0
            tot_tps = float(m_tot.group(1)) if m_tot else 0
            lat = float(m_lat.group(1)) if m_lat else 0
            ttft = float(m_ttft.group(1)) if m_ttft else 0
            itl = float(m_itl.group(1)) if m_itl else 0
            print(f'{bs:>4} {out_tps:>10.1f} {tot_tps:>10.1f} {lat:>11.2f} {ttft:>9.1f} {itl:>8.1f}')
"

echo ""
echo "Results saved to $OUT/"
ls -la "$OUT/"
