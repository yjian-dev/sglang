#!/bin/bash
# DFlash benchmark using tore-speed-eval, burst mode
# 7 servers on 7 GPUs, one per batch size
set -euo pipefail

export PATH=/home/yjian/miniconda3/envs/dflash/bin:/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
DFLASH_PYTHON=/home/yjian/miniconda3/envs/dflash/bin/python
SGLANG_PYTHON=/home/yjian/miniconda3/envs/sglang/bin/python
TORE=/home/yjian/miniconda3/envs/sglang/bin/tore-speed-eval

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="dflash_tore_${TIMESTAMP}"
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
echo "DFlash Benchmark (tore-speed-eval, burst)"
echo "bs: ${BS_LIST[*]}"
echo "Datasets: AIME25, ShareGPT, Synthetic (speed eval)"
echo "Output: $OUT/"
echo "=============================================="

# ── Launch 7 DFlash servers ──
echo ""
echo ">>> Launching first server (GPU 0, JIT cache)..."

launch_dflash() {
    local idx=$1
    local gpu=${GPUS[$idx]}
    local port=${PORTS[$idx]}
    local bs=${BS_LIST[$idx]}
    local max_req=$((bs + 8)); [ $max_req -lt 16 ] && max_req=16
    echo "  [bs=$bs] GPU=$gpu PORT=$port"
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        export SGLANG_ENABLE_SPEC_V2=1
        export SGLANG_ENABLE_DFLASH_SPEC_V2=1
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        $DFLASH_PYTHON -m sglang.launch_server \
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

launch_dflash 0
for i in $(seq 1 200); do
    curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORTS[0]}/health" 2>/dev/null | grep -q "200" && echo "  READY" && break
    sleep 3
done

echo ">>> Launching remaining 6 servers..."
for i in 1 2 3 4 5 6; do launch_dflash $i; done

echo ">>> Waiting for all servers..."
for i in "${!PORTS[@]}"; do
    for j in $(seq 1 200); do
        curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORTS[$i]}/health" 2>/dev/null | grep -q "200" && echo "  [bs=${BS_LIST[$i]}] READY" && break
        sleep 3
    done
done

# ── Benchmark ──
# Dataset 1: AIME 2025
echo ""
echo "=============================================="
echo ">>> AIME 2025 (thinking, burst)"
echo "=============================================="
for i in "${!BS_LIST[@]}"; do
    bs=${BS_LIST[$i]}; port=${PORTS[$i]}
    n=$((bs * 3)); [ $n -lt 16 ] && n=16; [ $n -gt 60 ] && n=60
    (
        $TORE --provider sglang \
            --base_url "http://localhost:$port/v1" \
            --model_name "Qwen/Qwen3-8B" \
            --dataset_type hf \
            --hf_dataset "MathArena/aime_2025" \
            --hf_dataset_split "train" \
            --hf_dataset_column_name "problem" \
            --num_examples $n \
            --max_tokens 2048 \
            --traffic_pattern burst \
            --concurrency $bs \
            --chat True \
            --stream True \
            --temperature 1.0 --top_p 0.95 \
            --evaluation_output_path "$OUT/aime25_bs${bs}.csv" \
            --generation_output_path "$OUT/aime25_bs${bs}_gen.jsonl" \
            2>&1 | tee "$OUT/aime25_bs${bs}.log"
    ) &
done
wait

# Dataset 2: ShareGPT
echo ""
echo "=============================================="
echo ">>> ShareGPT (burst)"
echo "=============================================="
for i in "${!BS_LIST[@]}"; do
    bs=${BS_LIST[$i]}; port=${PORTS[$i]}
    n=$((bs * 4)); [ $n -lt 16 ] && n=16; [ $n -gt 128 ] && n=128
    (
        $TORE --provider sglang \
            --base_url "http://localhost:$port/v1" \
            --model_name "Qwen/Qwen3-8B" \
            --dataset_type sharegpt \
            --num_examples $n \
            --max_tokens 2048 \
            --traffic_pattern burst \
            --concurrency $bs \
            --chat True \
            --stream True \
            --temperature 1.0 --top_p 0.95 \
            --evaluation_output_path "$OUT/sharegpt_bs${bs}.csv" \
            --generation_output_path "$OUT/sharegpt_bs${bs}_gen.jsonl" \
            2>&1 | tee "$OUT/sharegpt_bs${bs}.log"
    ) &
done
wait

# Dataset 3: Synthetic (speed eval)
echo ""
echo "=============================================="
echo ">>> Synthetic speed eval (in=256, out=2048, burst)"
echo "=============================================="
for i in "${!BS_LIST[@]}"; do
    bs=${BS_LIST[$i]}; port=${PORTS[$i]}
    n=$((bs * 4)); [ $n -lt 16 ] && n=16; [ $n -gt 128 ] && n=128
    (
        $TORE --provider sglang \
            --base_url "http://localhost:$port/v1" \
            --model_name "Qwen/Qwen3-8B" \
            --dataset_type synthetic \
            --synthetic_input_length 256 \
            --synthetic_output_length 2048 \
            --num_examples $n \
            --max_tokens 2048 \
            --traffic_pattern burst \
            --concurrency $bs \
            --chat True \
            --stream True \
            --ignore_eos True \
            --temperature 1.0 --top_p 0.95 \
            --evaluation_output_path "$OUT/synthetic_bs${bs}.csv" \
            --generation_output_path "$OUT/synthetic_bs${bs}_gen.jsonl" \
            2>&1 | tee "$OUT/synthetic_bs${bs}.log"
    ) &
done
wait

# ── Collect Results ──
echo ""
echo "=============================================="
echo ">>> Results"
echo "=============================================="
for ds in aime25 sharegpt synthetic; do
    echo ""
    echo "--- $ds ---"
    for bs in 1 2 4 8 16 32 64; do
        f="$OUT/${ds}_bs${bs}.csv"
        if [ -f "$f" ]; then
            echo "  bs=$bs: $(head -2 "$f" | tail -1 | cut -d, -f1-8)"
        else
            # Try parsing from log
            f="$OUT/${ds}_bs${bs}.log"
            if [ -f "$f" ]; then
                tps=$(grep -i "output.*throughput\|tok/s\|tps" "$f" | tail -1)
                echo "  bs=$bs: $tps"
            fi
        fi
    done
done

echo ""
echo "All results in $OUT/"
ls -la "$OUT/"
