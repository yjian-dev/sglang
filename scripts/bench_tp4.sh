#!/bin/bash
# Benchmark TP=4 servers at various concurrency levels
# Usage: bash scripts/bench_tp4.sh

source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang

TOKENIZER=/data/yjian/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
JSONL=/tmp/aime2024.jsonl
OUTPUT_DIR=/tmp/tp4_bench_results
mkdir -p $OUTPUT_DIR

CONCURRENCIES="1 2 4 8 16 32 64"

bench_server() {
    local name=$1
    local port=$2
    local outfile=$OUTPUT_DIR/${name}_results.txt

    echo "=== Benchmarking $name on port $port ===" | tee $outfile
    echo "" >> $outfile

    for C in $CONCURRENCIES; do
        echo "--- $name C=$C ---" | tee -a $outfile
        tore-speed-eval --provider sglang --base_url http://localhost:${port}/v1 \
            --model_name default --tokenizer_name $TOKENIZER --traffic_pattern burst \
            --concurrency $C --num_examples 90 --max_tokens 2048 \
            --dataset_type jsonl --jsonl_input_path $JSONL \
            --jsonl_convert_to_chat_request_format true \
            --temperature 1.0 --top_p 0.95 2>&1 | tee -a $outfile
        echo "" >> $outfile
    done

    echo "=== $name complete ===" | tee -a $outfile
}

# Run benchmarks
if [ "$1" = "dreamshift" ] || [ "$1" = "both" ] || [ -z "$1" ]; then
    bench_server "dreamshift_n3_tp4" 30000
fi

if [ "$1" = "qwen" ] || [ "$1" = "both" ] || [ -z "$1" ]; then
    bench_server "qwen_ar_tp4" 30004
fi

echo "Results saved to $OUTPUT_DIR/"
