#!/bin/bash
# DreamShift vs SDAR Demo — One-click launcher
# Usage: bash scripts/demo/run_demo.sh

set -e

# Environment
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export PYTHONPATH=/data/yjian/code/sglang/python:$PYTHONPATH

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
mkdir -p "$RESULTS_DIR"

# Config
CONCURRENCY=32
DURATION=90
MAX_TOKENS=2048
BATCH_SIZE=32
NUM_BATCHES=8
PORT=30000
MODEL_PATH="/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-causal-from-b2amc_fixed2_cont"
SDAR_MODEL="/data/shared/huggingface/SDAR-8B-Chat"

echo "========================================="
echo "  DreamShift vs SDAR Demo"
echo "========================================="
echo ""

# Step 1: Start DreamShift server
echo "[1/5] Starting DreamShift SGLang server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --trust-remote-code --tp-size 1 --dtype bfloat16 \
    --mem-fraction-static 0.85 --max-running-requests 24 \
    --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
    --dllm-algorithm-config "$ROOT_DIR/dreamshift_blockN4_config.yaml" \
    --port $PORT &
SERVER_PID=$!

echo "  Server PID: $SERVER_PID"
echo "  Waiting for server health..."

for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "  Server ready! (took ${i}s)"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "  ERROR: Server process died!"
        exit 1
    fi
    sleep 1
done

if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "  ERROR: Server failed to start within 120s"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Step 2: Run DreamShift TPS collection
echo ""
echo "[2/5] Collecting DreamShift TPS data (concurrency=$CONCURRENCY, ${DURATION}s)..."
cd "$ROOT_DIR"
python scripts/demo/collect_tps_timeseries.py \
    --server-url "http://localhost:$PORT" \
    --concurrency $CONCURRENCY \
    --max-tokens $MAX_TOKENS \
    --duration $DURATION \
    --output "$RESULTS_DIR/dreamshift_tps.json" &
DS_PID=$!

# Step 3: Run SDAR JetEngine TPS collection (in parallel on GPU 1)
echo "[3/5] Collecting SDAR baseline TPS data (batch=$BATCH_SIZE, $NUM_BATCHES batches)..."
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 scripts/demo/collect_jetengine_tps.py \
    --model-path "$SDAR_MODEL" \
    --batch-size $BATCH_SIZE \
    --max-tokens $MAX_TOKENS \
    --num-batches $NUM_BATCHES \
    --output "$RESULTS_DIR/sdar_tps.json" &
SDAR_PID=$!

# Wait for both
echo "  Waiting for both engines..."
wait $DS_PID
echo "  DreamShift collection done."
wait $SDAR_PID
echo "  SDAR collection done."

# Step 4: Generate visualization
echo ""
echo "[4/5] Generating visualization..."
python scripts/demo/live_tps_chart.py \
    --dreamshift-data "$RESULTS_DIR/dreamshift_tps.json" \
    --sdar-data "$RESULTS_DIR/sdar_tps.json" \
    --output-png "$RESULTS_DIR/tps_comparison.png" \
    --output-gif "$RESULTS_DIR/tps_comparison.gif" \
    --title "Throughput Comparison" \
    --batch-label "Batch Size: $BATCH_SIZE" \
    --max-time $DURATION

# Step 5: Summary
echo ""
echo "[5/5] Done!"
echo "========================================="
echo "  Results in: $RESULTS_DIR/"
echo "  - dreamshift_tps.json"
echo "  - sdar_tps.json"
echo "  - tps_comparison.png"
echo "========================================="

# Cleanup server
echo ""
echo "Stopping DreamShift server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
echo "Server stopped."
