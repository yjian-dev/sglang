# COLM Benchmark Data Package

## Data Files

### `data_5models.csv`
5 models comparison across 3 datasets × 7 batch sizes.
- **AR**: Qwen3-8B autoregressive baseline
- **DFlash s1d16**: Qwen3-8B + DFlash draft (steps=1, draft=16)
- **EAGLE3**: Qwen3-8B + EAGLE3 draft (steps=3, topk=1, draft=4)
- **Ours N=4 LoRA**: Strided DLLM b3-base + LoRA r=128 (gen_bs=4, block=7, lossless)
- **Ours N=4 b2**: Strided DLLM b2-cont (gen_bs=4, block=7, sampling)

Columns: `dataset, bs, {model}_job, {model}_per_req` (job=total throughput tok/s, per_req=per-request tok/s)

### `data_dllm_baselines.csv`
Our method vs prior DLLM baselines across 3 datasets × 7 batch sizes.
- **Ours N=4 b2**: Same as above
- **LLaDA 2.1-mini**: 16B MoE, JointThreshold block=4
- **SDAR**: 8B, LowConfidence block=4, denoising_steps=4

## Plots

### 5-model comparison (Total TPS vs Per-Req TPS)
- `colm_tvl_all.png` — 1×3 all datasets
- `colm_tvl_mbpp.png` / `colm_tvl_math_500.png` / `colm_tvl_lmsys_chat.png` — individual
- `colm_speedup_over_ar.png` — speedup vs batch size

### DLLM baseline comparison
- `dllm_comparison_all.png` — 1×3 all datasets
- `dllm_cmp_mbpp.png` / `dllm_cmp_math_500.png` / `dllm_cmp_lmsys_chat.png` — individual

## Plot Scripts
- `plot_5models.py` — generates 5-model plots (reads data inline, edit to use CSV)
- `plot_dllm_baselines.py` — generates DLLM comparison plots

## Benchmark Settings
- Tool: tore-speed-eval, burst mode
- Hardware: 1×H100 80GB, bf16, TP=1
- mem_fraction_static=0.85, max_running_requests=64
- max_tokens=2048, temperature=1.0, top_p=0.95, thinking mode (Qwen3 chat template)
- bs=1,2: 50 examples; bs=4: 100 examples; bs≥8: all examples

## Datasets
- **MBPP**: 257 problems (google-research-datasets/mbpp, sanitized test)
- **MATH-500**: 500 problems (HuggingFaceH4/MATH-500, test)
- **LMSYS-Chat**: 182 problems (lmsys/lmsys-chat-1m, first 200 user turns)
