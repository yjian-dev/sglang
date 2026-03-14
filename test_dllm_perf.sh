#!/bin/bash
# DreamShiftBlockN performance test script
# Usage: bash test_dllm_perf.sh
# Requires: conda env 'sglang', GPU 1 free, model at specified path
set -e

# Configuration
GPU=1
PORT=30001
MODEL="/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont_backup24000"
CONFIG="dreamshift_blockN3_noverify.yaml"
URL="http://localhost:${PORT}"

# Activate conda
source /home/yjian/miniconda3/etc/profile.d/conda.sh
conda activate sglang
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

# Kill any existing server on this port
lsof -ti :${PORT} | xargs -r kill -9 2>/dev/null || true
sleep 2

echo "=== Starting DreamShiftBlockN server on GPU ${GPU}, port ${PORT} ==="
CUDA_VISIBLE_DEVICES=${GPU} SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 python -m sglang.launch_server --model-path ${MODEL} --trust-remote-code --tp-size 1 --mem-fraction-static 0.85 --max-running-requests 64 --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN --dllm-algorithm-config ${CONFIG} --dtype bfloat16 --port ${PORT} --chunked-prefill-size 4096 > /tmp/dllm_server.log 2>&1 &
SERVER_PID=$!

# Wait for server to be ready (up to 5 minutes)
echo "Waiting for server to start (PID: ${SERVER_PID})..."
for i in $(seq 1 300); do
    if curl -sf ${URL}/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "Server process died. Check /tmp/dllm_server.log"
        tail -50 /tmp/dllm_server.log
        exit 1
    fi
    sleep 1
done

if ! curl -sf ${URL}/health > /dev/null 2>&1; then
    echo "Server failed to start within 300s"
    kill ${SERVER_PID} 2>/dev/null || true
    tail -50 /tmp/dllm_server.log
    exit 1
fi

# Quick sanity check
echo "=== Sanity check ==="
python scripts/stream_demo.py --url ${URL} --prompt "What is 15*23+7?" --max-tokens 64

echo ""
echo "=== tore-speed-eval concurrency=1 ==="
tore-speed-eval --provider sglang --base_url ${URL}/v1 --model_name default --tokenizer_name Qwen/Qwen3-8B --traffic_pattern concurrent --concurrency 1 --num_examples 50 --dataset_type synthetic --synthetic_input_length 256 --synthetic_output_length 1024 --max_tokens 2048

echo ""
echo "=== tore-speed-eval concurrency=32 ==="
tore-speed-eval --provider sglang --base_url ${URL}/v1 --model_name default --tokenizer_name Qwen/Qwen3-8B --traffic_pattern concurrent --concurrency 32 --num_examples 100 --dataset_type synthetic --synthetic_input_length 256 --synthetic_output_length 1024 --max_tokens 2048

echo ""
echo "=== tore-speed-eval concurrency=64 ==="
tore-speed-eval --provider sglang --base_url ${URL}/v1 --model_name default --tokenizer_name Qwen/Qwen3-8B --traffic_pattern concurrent --concurrency 64 --num_examples 100 --dataset_type synthetic --synthetic_input_length 256 --synthetic_output_length 1024 --max_tokens 2048

echo ""
echo "=== Cleanup ==="
kill ${SERVER_PID} 2>/dev/null || true
lsof -ti :${PORT} | xargs -r kill -9 2>/dev/null || true
echo "Done."
