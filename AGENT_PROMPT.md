# Agent Prompt — COLM Throughput Benchmark

## Role
You are a benchmark agent. You run throughput benchmarks using tore-speed-eval across 5 models, 3 datasets, and 7 batch sizes. You record results in `docs/benchmark_results.md`.

## 5 Models (already launched on GPUs)

| Port | GPU | Label | Model | Backend |
|------|-----|-------|-------|---------|
| 39001 | 4 | AR | Qwen/Qwen3-8B | sglang (our branch) |
| 39002 | 1 | DFlash s1d16 | Qwen/Qwen3-8B + z-lab/Qwen3-8B-DFlash-b16 (steps=1 draft=16) | dflash env (PR #20547) |
| 39003 | 2 | EAGLE3 | Qwen/Qwen3-8B + Tengyunw/qwen3_8b_eagle3 (steps=3 topk=1 draft=4) | dflash env (PR #20547) |
| 39004 | 3 | Ours N=4 LoRA | b3-allmasked-causal + lora128_epoch2, DreamShiftBlockN gen_bs=4 block=7 | sglang (our branch) |
| 38001 | 0 | Ours N=4 b2 | b2-allmasked-causal_fixed2_cont, DreamShiftBlockN gen_bs=4 block=7 | sglang (our branch) |

## Server Launch Commands (if servers are down)

```bash
source /home/yjian/miniconda3/etc/profile.d/conda.sh && conda activate sglang
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/home/yjian/miniconda3/envs/dflash/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
SGLANG_P=/home/yjian/miniconda3/envs/sglang/bin/python
DFLASH_P=/home/yjian/miniconda3/envs/dflash/bin/python

# AR (GPU 4)
CUDA_VISIBLE_DEVICES=4 nohup $SGLANG_P -m sglang.launch_server --model-path Qwen/Qwen3-8B --tp-size 1 --dtype bfloat16 --mem-fraction-static 0.85 --max-running-requests 64 --port 39001 > /dev/null 2>&1 &

# DFlash s1d16 (GPU 1)
CUDA_VISIBLE_DEVICES=1 SGLANG_ENABLE_SPEC_V2=1 SGLANG_ENABLE_DFLASH_SPEC_V2=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 nohup $DFLASH_P -m sglang.launch_server --model-path Qwen/Qwen3-8B --speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 16 --tp-size 1 --dtype bfloat16 --attention-backend fa3 --mem-fraction-static 0.85 --trust-remote-code --port 39002 --max-running-requests 64 > /dev/null 2>&1 &

# EAGLE3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 nohup $DFLASH_P -m sglang.launch_server --model-path Qwen/Qwen3-8B --speculative-algorithm EAGLE3 --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --tp-size 1 --dtype bfloat16 --mem-fraction-static 0.85 --trust-remote-code --port 39003 --max-running-requests 64 > /dev/null 2>&1 &

# Ours N=4 LoRA (GPU 3)
CUDA_VISIBLE_DEVICES=3 nohup $SGLANG_P -m sglang.launch_server --model-path /data/cxu/dllm-distillation/training/model/Qwen3-8B-b3-allmasked-causal --trust-remote-code --tp-size 1 --dtype bfloat16 --mem-fraction-static 0.85 --max-running-requests 64 --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN --dllm-algorithm-config dreamshift_blockN4_config.yaml --enable-lora --lora-paths "b3lora=/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-causal-from-b2amc-lora128_fixed2_epoch2" --max-lora-rank 128 --port 39004 > /dev/null 2>&1 &

# Ours N=4 b2 (GPU 0)
CUDA_VISIBLE_DEVICES=0 nohup $SGLANG_P -m sglang.launch_server --model-path /data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont --trust-remote-code --tp-size 1 --dtype bfloat16 --mem-fraction-static 0.85 --max-running-requests 64 --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN --dllm-algorithm-config dreamshift_blockN4_config.yaml --port 38001 > /dev/null 2>&1 &
```

## Datasets (jsonl files, create if missing)

```bash
# MBPP (257 problems)
python3 -c "
from datasets import load_dataset; import json
ds = load_dataset('google-research-datasets/mbpp', 'sanitized', split='test')
with open('/tmp/mbpp.jsonl', 'w') as f:
    for d in ds: f.write(json.dumps({'prompt': d['prompt']}) + '\n')
print(f'MBPP: {len(ds)}')
"

# MATH-500 (500 problems)
python3 -c "
from datasets import load_dataset; import json
ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
with open('/tmp/math500.jsonl', 'w') as f:
    for d in ds: f.write(json.dumps({'prompt': d['problem']}) + '\n')
print(f'MATH-500: {len(ds)}')
"

# LMSYS-Chat (182 problems)
python3 -c "
from datasets import load_dataset; import json
ds = load_dataset('lmsys/lmsys-chat-1m', split='train[:200]')
with open('/tmp/lmsys_chat.jsonl', 'w') as f:
    for d in ds:
        msg = next((t['content'] for t in d['conversation'] if t['role'] == 'user'), None)
        if msg and len(msg) > 10: f.write(json.dumps({'prompt': msg}) + '\n')
print('LMSYS done')
"
```

## Benchmark Command Template

```bash
TORE=/home/yjian/miniconda3/envs/sglang/bin/tore-speed-eval

$TORE --provider sglang --base_url "http://localhost:$PORT/v1" \
    --model_name "test" --tokenizer_name "Qwen/Qwen3-8B" \
    --dataset_type jsonl --jsonl_input_path $JSONL_PATH \
    --jsonl_dataset_column_name prompt --jsonl_convert_to_chat_request_format true \
    --num_examples $N --max_tokens 2048 \
    --traffic_pattern burst --concurrency $BS \
    --chat True --stream True --temperature 1.0 --top_p 0.95 \
    $LORA_ARGS \
    --evaluation_output_path "$OUTDIR/${DS}_${MODEL}_bs${BS}.csv"
```

For N=4 LoRA, add: `--lora_names b3lora --lora_ratio 1.0`

## num_examples per batch size
- bs=1,2: 50 examples
- bs=4: 100 examples (or all if dataset < 100)
- bs>=8: ALL examples from dataset

## Batch sizes to test
1, 2, 4, 8, 16, 32, 64

## Critical Rules
1. **Before starting**: verify all 5 servers are healthy (`curl http://localhost:$PORT/health`)
2. **Before starting**: verify AR per-req TPS < 150 at bs=1 (send 5 test requests)
3. **All 5 models run in PARALLEL** for each (dataset, bs) combination — saves time
4. **Check decode length = 2048.0 ± 0.0** — if not, results are invalid (requests ended early)
5. **Check failed = 0** — if any failures, server may have crashed, restart and rerun
6. **If N=4 b2 crashes** on a dataset at bs=64, it's likely OOM from long inputs. Note as "OOM" in results.
7. **Record results in PLAN.md** after each dataset completes
8. **Final results** go into `docs/benchmark_results.md`

## Output Format
Parse CSV: `summary_job_level_tps` (Job TPS) and `user_tps_mean` (Per-Req TPS)

## Environment
```bash
export PATH=/home/yjian/miniconda3/envs/sglang/bin:/home/yjian/miniconda3/envs/dflash/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
```
