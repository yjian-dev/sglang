#!/bin/bash
# Launch 8 servers: N=2/3/4/5 x greedy/sampling on GPU 0-7
# GPU 0: N=2 sampling  (port 30000)
# GPU 1: N=2 greedy    (port 30001)
# GPU 2: N=3 sampling  (port 30002)
# GPU 3: N=3 greedy    (port 30003)
# GPU 4: N=4 sampling  (port 30004)
# GPU 5: N=4 greedy    (port 30005)
# GPU 6: N=5 sampling  (port 30006)
# GPU 7: N=5 greedy    (port 30007)

cd "$(dirname "$0")/.."

export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=false
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
PYTHON=/home/yjian/miniconda3/envs/sglang/bin/python

MODEL=${MODEL_PATH:-/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont}

CONFIGS=(
  "dreamshift_blockN2_config.yaml"       # GPU 0: N=2 sampling
  "dreamshift_blockN2_greedy.yaml"        # GPU 1: N=2 greedy
  "dreamshift_blockN3_config.yaml"        # GPU 2: N=3 sampling
  "dreamshift_blockN3_greedy.yaml"        # GPU 3: N=3 greedy
  "dreamshift_blockN4_config.yaml"        # GPU 4: N=4 sampling
  "dreamshift_blockN4_greedy.yaml"        # GPU 5: N=4 greedy
  "dreamshift_blockN5_config.yaml"        # GPU 6: N=5 sampling
  "dreamshift_blockN5_greedy.yaml"        # GPU 7: N=5 greedy
)

LABELS=(
  "N=2 sampling" "N=2 greedy"
  "N=3 sampling" "N=3 greedy"
  "N=4 sampling" "N=4 greedy"
  "N=5 sampling" "N=5 greedy"
)

BASE_PORT=${BASE_PORT:-30000}

echo "=== Launching 8 servers ==="
echo "Model: $MODEL"

for gpu in 0 1 2 3 4 5 6 7; do
  port=$((BASE_PORT + gpu))
  config=${CONFIGS[$gpu]}
  label=${LABELS[$gpu]}
  nohup env CUDA_VISIBLE_DEVICES=$gpu PATH=$PATH CUDA_HOME=$CUDA_HOME \
    $PYTHON -m sglang.launch_server \
    --model-path $MODEL \
    --trust-remote-code --tp-size 1 --mem-fraction-static 0.85 --max-running-requests 64 \
    --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
    --dllm-algorithm-config $config \
    --dtype bfloat16 --port $port --chunked-prefill-size 4096 > /tmp/sglang_gpu${gpu}.log 2>&1 &
  echo "  GPU $gpu -> port $port: $label ($config) PID=$!"
done

echo ""
echo "=== Waiting for servers ==="
for gpu in 0 1 2 3 4 5 6 7; do
  port=$((BASE_PORT + gpu))
  label=${LABELS[$gpu]}
  for i in $(seq 1 120); do
    r=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health 2>/dev/null)
    if [ "$r" = "200" ]; then echo "  GPU $gpu port $port ready ($label)"; break; fi
    if [ $i -eq 120 ]; then echo "  GPU $gpu port $port TIMEOUT ($label)"; fi
    sleep 2
  done
done
echo "=== Done ==="
echo ""
echo "Port mapping:"
echo "  30000: N=2 sampling    30001: N=2 greedy"
echo "  30002: N=3 sampling    30003: N=3 greedy"
echo "  30004: N=4 sampling    30005: N=4 greedy"
echo "  30006: N=5 sampling    30007: N=5 greedy"
