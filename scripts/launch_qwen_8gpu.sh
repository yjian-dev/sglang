#!/bin/bash
# Launch 8 Qwen3-8B AR servers (1 per GPU)
# Usage: bash scripts/launch_qwen_8gpu.sh

cd "$(dirname "$0")/.."

export PYTHONPATH=python
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
PYTHON=/home/yjian/miniconda3/envs/sglang/bin/python

MODEL=${MODEL_PATH:-Qwen/Qwen3-8B}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
BASE_PORT=${BASE_PORT:-30000}

echo "=== Launching Qwen3-8B AR servers ==="
echo "Model: $MODEL"
echo "GPUs: $GPUS"

for gpu in $GPUS; do
  port=$((BASE_PORT + gpu))
  nohup env CUDA_VISIBLE_DEVICES=$gpu PATH=$PATH CUDA_HOME=$CUDA_HOME \
    $PYTHON -m sglang.launch_server \
    --model-path $MODEL \
    --trust-remote-code --tp-size 1 --mem-fraction-static 0.85 --max-running-requests 64 \
    --attention-backend flashinfer \
    --dtype bfloat16 --port $port --chunked-prefill-size 4096 > /tmp/sglang_gpu${gpu}.log 2>&1 &
  echo "  GPU $gpu -> port $port (PID=$!)"
done

echo ""
echo "=== Waiting for servers ==="
for gpu in $GPUS; do
  port=$((BASE_PORT + gpu))
  for i in $(seq 1 120); do
    r=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health 2>/dev/null)
    if [ "$r" = "200" ]; then echo "  GPU $gpu port $port ready"; break; fi
    if [ $i -eq 120 ]; then echo "  GPU $gpu port $port TIMEOUT (check /tmp/sglang_gpu${gpu}.log)"; fi
    sleep 2
  done
done
echo "=== Done ==="
