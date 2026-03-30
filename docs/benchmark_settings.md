# Benchmark Settings & Results

## 1. Hardware & Common Settings

- **GPU**: 1× NVIDIA H100 80GB HBM3
- **dtype**: bfloat16, TP=1
- **mem_fraction_static**: 0.85 (all methods)
- **max_running_requests**: 24
- **CUDA**: 12.9 (`PATH=/usr/local/cuda-12.9/bin:$PATH`)
- **Conda envs**: `sglang` (our branch), `dflash` (PR #20547 for EAGLE3/DFlash)

---

## 2. Models & Checkpoints

### Our Models
| | Base Model | LoRA | Config YAML |
|---|---|---|---|
| **N=3 non-LoRA** | `sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont` at `/data/cxu/keep/dllm_experiments/` | — | `dreamshift_blockN3_config.yaml` |
| **N=4 non-LoRA** | Same as N=3 (b3-cont, forced to N=4) | — | `dreamshift_blockN4_config.yaml` |
| **N=3 LoRA** | `Qwen3-8B-b3-allmasked-causal` at `/data/cxu/dllm-distillation/training/model/` | `b3-causal-from-b2amc-lora128_fixed2_epoch2` (r=128) at `/data/cxu/keep/dllm_experiments/` | `dreamshift_blockN3_config.yaml` |
| **N=4 LoRA** | Same base as N=3 LoRA | Same LoRA | `dreamshift_blockN4_config.yaml` |

### Algorithm Config YAMLs
```yaml
# dreamshift_blockN3_config.yaml
block_size: 5
gen_block_size: 3
confidence_threshold: 0.0
temperature: 1.0
top_k: 50
top_p: 0.95
use_spec_verify: true

# dreamshift_blockN4_config.yaml
block_size: 7
gen_block_size: 4
confidence_threshold: 0.0
temperature: 1.0
top_k: 50
top_p: 0.95
use_spec_verify: true
```

### Baseline Models
| Method | Target Model | Draft Model |
|---|---|---|
| **AR** | `Qwen/Qwen3-8B` | — |
| **EAGLE3** | `Qwen/Qwen3-8B` | `Tengyunw/qwen3_8b_eagle3` |
| **DFlash** | `Qwen/Qwen3-8B` | `z-lab/Qwen3-8B-DFlash-b16` |

---

## 3. Fair Compute Matching

### Our ISD per-step breakdown
```
N=3: [pending, spec0, spec1, MASK, MASK]     block_size=5, specs_verified=2
N=4: [pending, spec0, spec1, spec2, M, M, M] block_size=7, specs_verified=3
```
- `specs_verified = N - 1` (speculative tokens from previous round)
- `target_tokens = 2N - 1` (includes N-1 MASK positions for next-round proposals)
- The MASK positions are our "proposal cost" (extra target model compute)

### EAGLE3 per-step breakdown (topk=1, linear chain)
```
steps=S, topk=1, draft_tokens=D:
  - Draft model forwards: S - 1 (last step doesn't forward)
  - Candidate pool: S tokens
  - Selected for verify: D - 1 (top from pool)
  - Target model verifies: D tokens total (D-1 draft + 1 verified)
  - specs_verified = D - 1
```
Draft model forwards are EAGLE3's "proposal cost" (extra small-model compute).

### DFlash per-step breakdown
```
steps=1, draft_tokens=D:
  - Draft model (block diffusion): 1 forward, generates D tokens
  - Target model verifies: D tokens
  - specs_verified = D (all are speculative)
```

### Matching Strategy: by specs_verified
Match number of speculative tokens verified per step (determines acceptance rate and TPF).
Both sides have their own proposal overhead: we use extra MASK target tokens, they use draft model forwards.

| Our Method | Specs | Target Tokens | Matched EAGLE3 | EAGLE3 Target | Matched DFlash |
|---|---|---|---|---|---|
| **N=3** | **2** | 5 | `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3` (2 specs, 3 target) | 3 | `--speculative-num-steps 1 --speculative-num-draft-tokens 3` (2 specs, 3 target) |
| **N=4** | **3** | 7 | `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` (3 specs, 4 target) = **(3,1,4)** | 4 | `--speculative-num-steps 1 --speculative-num-draft-tokens 4` (3 specs, 4 target) |

### Alternative: Matching by target model tokens
Match total tokens processed by target model per forward (determines target model compute cost).
This is more favorable to EAGLE3/DFlash because they get more specs verified with fewer target tokens.

| Our Method | Target Tokens | Matched EAGLE3 | Matched DFlash |
|---|---|---|---|
| **N=3** (target=5) | 5 | `--speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5` (4 specs) | `--speculative-num-steps 1 --speculative-num-draft-tokens 5` |
| **N=4** (target=7) | 7 | `--speculative-num-steps 7 --speculative-eagle-topk 1 --speculative-num-draft-tokens 7` (6 specs) | `--speculative-num-steps 1 --speculative-num-draft-tokens 7` |

### Compute overhead comparison
| Method | Target tokens | Draft model cost | Total proposal cost |
|---|---|---|---|
| **Ours N=3** | 5 (2 specs + 1 pending + 2 MASKs) | **0** | 2 MASK target tokens |
| **EAGLE3 (3,1,3)** | 3 (2 specs + 1 verified) | 2 draft forwards | 2 draft forwards |
| **EAGLE3 (3,1,4)** | 4 (3 specs + 1 verified) | 2 draft forwards | 2 draft forwards |
| **Ours N=4** | 7 (3 specs + 1 pending + 3 MASKs) | **0** | 3 MASK target tokens |

### DFlash env vars (required)
```bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_DFLASH_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
```

---

## 4. Server Launch Commands

### AR
```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B --tp-size 1 --dtype bfloat16 \
    --mem-fraction-static 0.85 --max-running-requests 24 --port PORT
```

### Ours N=3 non-LoRA
```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path /data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b2-allmasked-causal_fixed2_cont \
    --trust-remote-code --tp-size 1 --dtype bfloat16 \
    --mem-fraction-static 0.85 --max-running-requests 24 \
    --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
    --dllm-algorithm-config dreamshift_blockN3_config.yaml --port PORT
```

### Ours LoRA N=4
```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path /data/cxu/dllm-distillation/training/model/Qwen3-8B-b3-allmasked-causal \
    --trust-remote-code --tp-size 1 --dtype bfloat16 \
    --mem-fraction-static 0.85 --max-running-requests 24 \
    --attention-backend flashinfer --dllm-algorithm DreamShiftBlockN \
    --dllm-algorithm-config dreamshift_blockN4_config.yaml \
    --enable-lora --lora-paths "b3lora=/data/cxu/keep/dllm_experiments/sdar_qwen3_8b_dreamshift_ar_b3-causal-from-b2amc-lora128_fixed2_epoch2" \
    --max-lora-rank 128 --port PORT
```

### EAGLE3 (3,1,3) — matching N=3 by specs (2 specs verified)
```bash
# Use dflash conda env
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 \
    --tp-size 1 --dtype bfloat16 --mem-fraction-static 0.85 \
    --max-running-requests 24 --trust-remote-code --port PORT
```

### EAGLE3 (3,1,4) — matching N=4 by specs (3 specs verified)
```bash
# Use dflash conda env
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --tp-size 1 --dtype bfloat16 --mem-fraction-static 0.85 \
    --max-running-requests 24 --trust-remote-code --port PORT
```

### DFlash d=2 — matching N=3 by specs (2 specs verified)
```bash
# Use dflash conda env
SGLANG_ENABLE_SPEC_V2=1 SGLANG_ENABLE_DFLASH_SPEC_V2=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
    --speculative-num-steps 1 --speculative-num-draft-tokens 2 \
    --tp-size 1 --dtype bfloat16 --attention-backend fa3 \
    --mem-fraction-static 0.85 --max-running-requests 24 \
    --trust-remote-code --port PORT
```

### DFlash d=3 — matching N=4 by specs (3 specs verified)
```bash
# Use dflash conda env
SGLANG_ENABLE_SPEC_V2=1 SGLANG_ENABLE_DFLASH_SPEC_V2=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
    --speculative-num-steps 1 --speculative-num-draft-tokens 3 \
    --tp-size 1 --dtype bfloat16 --attention-backend fa3 \
    --mem-fraction-static 0.85 --max-running-requests 24 \
    --trust-remote-code --port PORT
```

---

## 5. Benchmark Methods

### 5a. Fixed Concurrency (C=24)
24 concurrent requests, `ignore_eos=True`, `max_tokens=2048`, thinking mode, 2-3 rounds average.

```python
body = {
    "model": "test",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 2048,
    "temperature": 1.0, "top_p": 0.95,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": True},
    "ignore_eos": True,
}
# For LoRA: add "lora_path": "b3lora"
```

**Results (matched by target tokens — old, may re-run with specs matching):**

N=3 group (target=5):

| | Job TPS | Per-Req TPS | vs AR |
|---|---|---|---|
| AR | 2,833 | 121 | — |
| EAGLE3 d=5 | 3,940 | 223 | +39% |
| DFlash d=5 | 4,253 | 241 | +50% |
| **Ours N=3** | **4,495** | **215** | **+59%** |

N=4 group (target=7):

| | Job TPS | Per-Req TPS | vs AR |
|---|---|---|---|
| AR | 2,833 | 121 | — |
| DFlash d=7 | 3,766 | 243 | +33% |
| EAGLE3 d=7 | 3,927 | 233 | +39% |
| **Ours LoRA N=4** | **4,214** | **220** | **+49%** |

### 5b. tore-speed-eval Burst Benchmarks
Used `tore-speed-eval` (installed from `/data/yjian/code/tore-speed-eval`) with 4 datasets:
- **AIME 2022-2024**: `/tmp/aime2224.jsonl` (90 problems from `AI-MO/aimo-validation-aime`)
- **MATH-500**: `/tmp/math500.jsonl` (500 problems from `HuggingFaceH4/MATH-500`)
- **ShareGPT**: `/tmp/sharegpt.jsonl` (200 prompts from `/data/yjian/datasets/ShareGPT_V3_unfiltered_cleaned_split.json`)
- **Synthetic**: `--dataset_type synthetic --synthetic_input_length 256 --synthetic_output_length 2048`

```bash
tore-speed-eval --provider sglang --base_url "http://localhost:PORT/v1" \
    --model_name "Qwen/Qwen3-8B" --tokenizer_name "Qwen/Qwen3-8B" \
    --dataset_type jsonl --jsonl_input_path FILE.jsonl \
    --jsonl_dataset_column_name prompt --jsonl_convert_to_chat_request_format true \
    --num_examples N --max_tokens 2048 \
    --traffic_pattern burst --concurrency BS \
    --chat True --stream True --temperature 1.0 --top_p 0.95 \
    --evaluation_output_path OUTPUT.csv
# For LoRA: add --lora_names b3lora --lora_ratio 1.0
```

Tested at bs=1,2,4,8,16,32,64. For bs≤2: num_examples=10, bs≤4: 20, else: larger.

### 5c. Real Traffic Replay (Azure Trace)

**Trace source**: `assets/AzureLLMInferenceTrace_code_1week.csv` (Microsoft Azure LLM Inference Trace, 16.8M requests, 7 days)

**How to create the trace:**
```python
import pandas as pd
from datetime import timedelta

df = pd.read_csv('assets/AzureLLMInferenceTrace_code_1week.csv')
df['ts'] = pd.to_datetime(df['TIMESTAMP'])

# Find peak day
peak_day = df.groupby(df['ts'].dt.floor('h')).size().idxmax().date()
day_df = df[df['ts'].dt.date == peak_day]

# Sample every Nth to control QPS, compress time to target duration
step = 10000  # adjust for desired request count
sampled = day_df.iloc[::step]
# Compress timestamps to fit target duration (e.g., 60s)
sampled['relative_time'] = (unix - unix.min()) / (unix.max() - unix.min()) * 60
```

**Controlling peak QPS:**
- Subsample (every Nth) to thin requests while preserving arrival pattern
- Compress time to target duration
- `target-peak-qps` flag in `trace_replay_timeseries.py` auto-downsamples

**Scripts:**
- `scripts/trace_replay.py` — Basic trace replay with summary stats
- `scripts/trace_replay_timeseries.py` — Records per-second TPS for plotting
- `scripts/plot_trace_tps.py` — Plots TPS over time for multiple models

**Example (peak concurrent ~16, 256 tokens, 60s):**
```bash
# Create trace: every 10000th request from peak day, compressed to 60s
# Result: 320 requests, QPS 1.2→11.8 variation

python scripts/trace_replay_timeseries.py \
    --trace /tmp/azure_trace_test60s.csv --port PORT --duration 60 \
    --max-tokens 256 --label "Model Name" \
    --output trace_ts/model.json
# For LoRA: add --lora-path b3lora
```

**Pre-extracted trace files:**
- `/tmp/azure_trace_peak5min.csv` — 5 min around peak hour (24828 requests)
- `/tmp/azure_trace_subsample.csv` — 1/64 subsample (388 requests, QPS 0.1-3.8)
- `/tmp/azure_trace_test60s.csv` — 320 requests in 60s, QPS 1.2→11.8
- `/tmp/azure_trace_test60s_light.csv` — 160 requests in 60s, QPS 0→7 (peak concurrent ~16)
- `/tmp/azure_trace_daily_5min.csv` — Full day compressed to 5 min (3192 requests, QPS 2.4→26)

**Plotting:**
```bash
python scripts/plot_trace_tps.py \
    --inputs trace_ts/AR.json trace_ts/N3.json trace_ts/EAGLE3.json trace_ts/DFlash.json \
    --output trace_ts/comparison.png \
    --title "Per-Request TPS Under Varying Load"
```

---

## 6. Infrastructure Ablation

### Ablation toggles (env vars)
| Env Var | Disables |
|---|---|
| `--disable-cuda-graph` | CUDA graph capture/replay |
| `SGLANG_ABLATION_NO_DECODE_LOOP=1` | Stationary-batch decode loop |
| `SGLANG_ABLATION_NO_ARGMAX_PROPOSALS=1` | Argmax for draft positions |
| `SGLANG_ABLATION_NO_FUSED_VERIFY=1` | Fused Triton verify kernel |
| `SGLANG_ABLATION_NO_DEFERRED_STREAM=1` | Deferred stream_output |
| `SGLANG_ABLATION_DLLM_USE_RAGGED=1` | Forces cascade attention (3 kernels/layer) |

Note: These toggles are in a git stash. Restore with `git stash pop` (stash message: "ablation toggles and paper updates").

### Ablation results (scaled to match full system = 282 at C=1)

| Configuration | C=1 | Δ | C=8 | Δ | C=32 | Δ |
|---|---|---|---|---|---|---|
| Naive (cascade, eager, full sched) | 111 | — | 829 | — | 2,791 | — |
| + Paged-only attention | 126 | +14% | 915 | +10% | 3,058 | +10% |
| + CUDA graph | 222 | +76% | 1,501 | +64% | 4,344 | +42% |
| + Decode inner loop | 246 | +11% | 1,744 | +16% | 5,264 | +21% |
| + Argmax proposals | 273 | +11% | 1,998 | +15% | 5,846 | +11% |
| + Fused verify | 277 | +1% | 2,000 | +0% | 5,895 | +1% |
| Full system | 282 | +2% | 2,084 | +4% | 5,882 | -0% |
| **Total** | **2.54×** | | **2.51×** | | **2.11×** | |

---

## 7. Key Metrics from Original Code (no ablation toggles)

- **TPF**: 2.28 (N=3, sampling, measured from server step logs)
- **Accept rate**: 92.9% per spec position
- **Step time (C=1)**: 8.5ms, forward 97% of step
- **Advance distribution**: adv=3 (58.9%), adv=2 (9.8%), adv=1 (31.2%)
