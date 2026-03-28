#!/bin/bash
# Run full infrastructure ablation for COLM paper.
#
# Usage:
#   bash scripts/run_ablation.sh [GPU_ID] [MODEL_PATH]
#
# Defaults:
#   GPU_ID=0
#   MODEL_PATH=/data/cxu/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b1-allmasked_fixed2_backup8000

set -euo pipefail

GPU="${1:-0}"
MODEL="${2:-/data/cxu/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b1-allmasked_fixed2_backup8000}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="ablation_results_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Strided DLLM Infrastructure Ablation"
echo "=============================================="
echo "GPU: $GPU"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR/"
echo "=============================================="

# Ensure CUDA 12.9
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9

# Kill any existing sglang servers
pkill -f "sglang.launch_server" 2>/dev/null || true
sleep 2

# ── Non-LoRA ablation (N=3, C=1 and C=32) ──
echo ""
echo ">>> Running Non-LoRA ablation (N=3, C=1 and C=32)..."
python scripts/ablation_infra.py \
    --model-path "$MODEL" \
    --gpu "$GPU" \
    --tp-size 1 \
    --port 31100 \
    --algorithm-config dreamshift_blockN3_config.yaml \
    --concurrency 1 32 \
    --max-tokens 2048 \
    --max-running-requests 64 \
    --output "$OUTPUT_DIR/ablation_nonlora_n3.json" \
    --latex-output "$OUTPUT_DIR/ablation_nonlora_n3.tex" \
    2>&1 | tee "$OUTPUT_DIR/ablation_nonlora_n3.log"

# ── Non-LoRA ablation (N=5, C=1) ──
echo ""
echo ">>> Running Non-LoRA ablation (N=5, C=1)..."
python scripts/ablation_infra.py \
    --model-path "$MODEL" \
    --gpu "$GPU" \
    --tp-size 1 \
    --port 31100 \
    --algorithm-config dreamshift_blockN5_config.yaml \
    --concurrency 1 \
    --max-tokens 2048 \
    --max-running-requests 64 \
    --output "$OUTPUT_DIR/ablation_nonlora_n5.json" \
    --latex-output "$OUTPUT_DIR/ablation_nonlora_n5.tex" \
    --configs 0 5 \
    2>&1 | tee "$OUTPUT_DIR/ablation_nonlora_n5.log"

echo ""
echo "=============================================="
echo "Ablation complete. Results in: $OUTPUT_DIR/"
echo "=============================================="
ls -la "$OUTPUT_DIR/"
