# Strided DLLM 推理系统优化详解（Presentation 用）

## 背景：为什么 DLLM 在 AR 推理系统上这么难？

现有 LLM serving 系统（SGLang、vLLM）的整个 stack 都是围绕 AR 的 "每步 1 个 token" 假设深度优化的。DLLM 要在这套系统上跑，面临 **四个根本性的不兼容**：

### 问题 1：Attention 模式不兼容

AR 的 decode mode 假设每步只有 **1 个新 query token**，attention kernel 和 metadata 都针对这个 pattern 深度优化（metadata 构建只需 0.07ms）。

DLLM 每步处理一整个 block 的 token（比如 N=5 就是 9 个），这些新 token 之间还需要互相 attend（双向或 block-causal attention）。AR 的 decode kernel 根本不支持——必须用 extend mode（metadata 重建 0.43ms，贵 6 倍）或者自己写 attention kernel。

更糟的是，之前的 DLLM（LLaDA 用双向、SDAR 用 block-causal）的 attention pattern 跟 AR 的 causal mask 完全不同，没法复用 flashinfer/flashattention 的任何优化路径。

### 问题 2：KV Cache 管理模式不兼容

AR 的 KV cache **只增不减**——每步 append 一个 entry，paged memory 管理非常简单。整个 vLLM/SGLang 的 paged attention 都建立在这个假设上。

DLLM 的情况完全不同：
- **Iterative denoising**（LLaDA/SDAR）：每轮 denoise 都要 **overwrite** MASK 位置的 KV（hidden state 在变），同一个 slot 被反复写入
- **Speculative verify**（我们的 ISD）：verify 后要 **释放** 被拒绝的 spec token 和 MASK 位置的 KV slot（trim 操作），AR 系统没有 "释放中间 KV slot" 的概念
- 两种情况都打破了 "KV 只增长" 的假设

### 问题 3：Batching / Continuous Batching 不兼容

AR 的 continuous batching 非常优雅：每个请求每步都产出 1 个 token，所有请求 "形状" 一样（decode mode, 1 token），可以无缝合批，随时插入新请求、移除完成的请求。

DLLM 的 batching 有三层困难：

**a) 请求之间 denoising 进度不同步。** DLLM 每个 block 要跑 T 步 denoising（SDAR 4 步，LLaDA 更多）。batch 里不同请求可能在第 1 步、第 3 步等不同阶段，每步的 mask pattern 不同（第 1 步几乎全 MASK，最后一步几乎全 clean）。要么强制对齐（浪费算力），要么搞复杂的 per-request mask routing。

**b) 每步的 token 数量不统一。** AR decode 每个请求都是 1 token，天然对齐。DLLM 不同请求的 block size 可能不同，有的在 prefill 有的在 decode，extend length 不一致，很难高效 batch。

**c) 新请求插入时机受限。** AR 可以在任意 decode step 之间插入新请求。DLLM 如果在 denoising 中间插入，要重建整个 batch 的 attention metadata，而且新请求的生命周期（prefill → denoising loop）跟正在 decode 的请求完全不同。

### 问题 4：CPU-GPU Overlap Scheduling 不兼容

AR 系统的杀手锏是 **overlap scheduling**：CPU 在调度下一批请求的同时，GPU 在跑当前 forward。CPU 开销完全被隐藏（~0.2ms，但用户感知不到）。

DLLM 有严格的依赖链，**每一步的 CPU 工作都依赖上一步的 GPU 结果**：

```
forward(t) → 需要 logits → verify(t) → 需要 accept/reject →
trim(t) → 需要 trim count → prepare(t+1) → forward(t+1)
```

无法把任何 CPU 工作提前到 GPU forward 期间执行。结果：AR 的 CPU overhead ~0.2ms（被隐藏），DLLM 的 ~1.5ms **全部加到延迟上**。

### 现有 DLLM 的做法 vs 我们的做法

**之前的 DLLM（LLaDA、SDAR）**：放弃 AR infra，从零搭自己的推理系统。丢掉了 paged KV cache、continuous batching、CUDA graph、overlap scheduling 这些多年积累的优化。

**我们的 key insight**：用 **strict causal attention + logit shift**，让每个 ISD step 变成 SGLang 原生的 extend 操作。然后：
- 问题 1 → causal attention 可以直接用 flashinfer 的 paged-only kernel，不需要自定义 attention
- 问题 2 → 加一个 trim-and-commit cycle 管理 KV 释放
- 问题 3 → ISD 没有 iterative denoising（只有 1 步 verify），所有请求统一 block_size，batching 退化成标准的 extend batching
- 问题 4 → 用 stationary-batch inner loop 最小化 CPU overhead，虽然不能 overlap 但把 overhead 从 4ms 压到 1.5ms

---

## 总览：优化前 vs 优化后

| 配置 | TPS | 对比 AR 149 tok/s |
|------|-----|-------------------|
| 朴素实现（eager forward + 完整调度） | ~100 | 0.67x（比 AR 还慢！） |
| + CUDA graph | ~180 | 1.2x |
| + 所有调度优化 | ~287 (N=3) / ~313 (N=5) | 1.9x / 2.1x |
| + LoRA（lossless 模式） | ~212 | 1.4x |

---

## 优化 1：把 DLLM 映射到 AR 的 Extend 模式

### 问题
其他 DLLM（LLaDA、SDAR）用双向注意力，没法复用 AR 系统的任何东西，必须从零搭 serving stack。

### 我们的做法
因为我们用的是 **strict causal attention**（跟 AR 完全一样的因果注意力），所以每个 ISD 步骤可以直接映射成 SGLang 的 **extend** 操作——就是往 prefix 后面追加多个 token。

这意味着：
- ✅ Paged KV cache 直接用
- ✅ Continuous batching 直接用
- ✅ Tensor parallelism 直接用
- ✅ CUDA graph 可以 capture

不需要写一行自定义的 attention kernel。

---

## 优化 2：CUDA Graph Capture（最大的单项加速：eager 14.7ms → graph 7.1ms）

### 问题
bs=1 时 GPU 利用率只有 ~15%，大部分时间花在 kernel launch 的 overhead 上。Qwen3-8B 有 36 层，每层要 launch 十几个 kernel（GEMM、attention、norm 等），总共几千个 kernel launch。

### 做法
把整个 DLLM extend forward（包括 attention + lm_head）capture 成一个 CUDA graph。每步 replay 时只需要更新 `input_ids` 和 attention metadata（page table、seq_lens），其他所有 kernel launch 都是 replay。

### 关键挑战
DLLM extend 的 attention metadata **每步都变**（sequence length 在增长），而 CUDA graph 要求 kernel 参数固定。我们的解法：
- 在 capture 时预分配最大尺寸的 metadata buffer
- 每步 replay 前，只更新变化的字段（kv_indptr、kv_indices、qo_indptr）
- 用 `init_forward_metadata_replay_cuda_graph()` 做这个增量更新

### 效果
- Forward time: 14.7ms → 7.1ms（**2x 加速**）
- 这是 TP=1 bs=1 下最大的单项优化

---

## 优化 3：Paged-Only Attention（3 个 kernel → 1 个）

### 问题
SGLang 标准的 extend 用 **cascade attention**：
1. Ragged attention（处理新 token 之间的注意力）
2. Paged attention（处理新 token 对 prefix 的注意力）
3. Merge state（合并两者的结果）

每层 3 个 kernel，36 层 = 108 个 kernel launch。

### 做法
因为 DLLM decode 时每个请求都有 cached prefix，我们直接用 **paged-only attention**（`use_ragged=False`），一个 flashinfer kernel 同时处理新 token 之间的 causal attention 和对 prefix 的 attention。

### 效果
每层 3 → 1 个 kernel，省掉 72 个 kernel launch/step。

---

## 优化 4：Stationary-Batch Decode Loop（+134% at bs=32）

### 问题
SGLang 每一步都重新走完整的调度流程：
1. `cache_unfinished_req()` — 每个请求做一次 GPU tensor copy（复制 KV indices）
2. 过滤完成的请求、合并新请求
3. 从头创建 batch 对象、重建 attention metadata
4. GPU forward
5. 处理结果、发送 output

问题：步骤 1 每步对每个请求执行 **2 次**（进出各一次），bs=32 时就是 64 次 GPU tensor copy + 大量 Python 开销 = **每步 4ms CPU 开销**。

而且 AR 模型可以用 overlap scheduling（CPU 和 GPU 并行），但 ISD 不行——因为 verify 需要 GPU forward 的 logits，下一步的 batch 准备又需要 verify 的结果。严格的依赖链：

```
forward(t) → verify(t) → trim(t) → prepare(t+1) → forward(t+1)
```

每毫秒的 CPU 开销都直接加到总延迟上。

### 做法
写了一个 **tight inner loop** (`_dllm_decode_loop`)，跳过整个调度流程：

- **复用 batch 对象**：不再每步创建新 batch，直接复用上一步的
- **轻量 batch 准备** (`prepare_for_dllm_decode`)：
  - KV 分配用一次 batched scatter
  - `seq_lens` 用算术更新：`prev + bs × block_size`（不重新计算）
  - `req_pool_indices` 和 `position_offsets` 是常量，cache 起来复用
  - `input_ids` buffer 预分配，fill with MASK，algorithm 只覆盖特定位置
- **非关键工作延迟执行**：`stream_output`（发 token 给 detokenizer，~120μs）推迟到下一步的 GPU forward 期间执行
- 只有新请求到达或 batch 空了才退出 inner loop 回到完整调度

### 效果
| Batch Size | 优化前 | 优化后 | 提升 |
|-----------|--------|--------|------|
| bs=1 | ~139 tok/s | ~238 tok/s | +71% |
| bs=32 | ~1,923 tok/s | ~4,509 tok/s | **+134%** |

这是所有优化里对高 batch size 影响最大的。

---

## 优化 5：Argmax-for-Proposals（accept rate 68% → 78%，+10% TPS）

### 问题
标准 speculative decoding 用 top-k/top-p 采样生成 draft token。但 ISD 有一个 AR spec decoding 没有的性质：**输出质量跟 proposal 质量无关**。因为每个被接受的 token 都经过 p/q 验证，数学上保证 follow base AR 分布；被拒绝的 token 会从 max(0, p-q) 重采样。

### 做法
对 proposal 位置用 **argmax** 而不是采样：`x_k = argmax(q_k)`。Clean 位置（位置 1）仍然用用户指定的 temperature/top-p 采样。

### 为什么不影响多样性？
output 的多样性完全由 acceptance criterion `min(1, p/q)` 决定。argmax 只是让 proposal 更容易被接受（因为 p 和 q 通过共享训练对齐了，最可能的 token 最容易被接受）。被拒绝时的重采样跟 proposal 方法完全无关。

### 效果
- Accept rate: 68% → 78%
- TPF: 2.17 → 2.37
- TPS: +10%

---

## 优化 6：Fused Triton Verify Kernel（省 4+ kernel launches）

### 问题
验证步骤需要：
1. 对 anchor logits 做 softmax → p(x)
2. Gather p(x_k)（被提议的 token 的概率）
3. Gather q(x_k)（draft 概率）
4. 计算 p/q ratio
5. 跟随机数比较，决定接受/拒绝
6. 如果拒绝：从 max(0, p-q) 重采样（又是一次 softmax + multinomial）

朴素实现：7+ 个 GPU kernel launch。

### 做法
写了一个 **单 Triton kernel**，two-pass early-exit 设计：

**Pass 1（每次都跑）：**
- 用 online softmax 一次遍历整个 vocabulary（152K tokens）
- 维护 running max 和 exp-sum，streaming 计算
- 最后得到 p(x_k)，跟 q(x_k) 比较
- 如果 **接受**（78% 的情况）：**直接返回，不跑 Pass 2**

**Pass 2（只有拒绝时才跑，~22%）：**
- 再遍历一次 vocabulary
- 用 Gumbel-max trick 采样 corrected token：`argmax(log(max(0, p-q)) + Gumbel noise)`
- 一次遍历就完成，不需要显式 normalize + multinomial

### 额外优化：Scalar Draft Probabilities
不存完整的 draft 分布 q ∈ ℝ^V（bs=32 时 ~39MB），只存 q(x_k) 这一个 scalar（4 bytes）。拒绝时直接从 p 重采样（而不是 max(0,p-q)），质量影响可忽略。

### 效果
- 省 4+ kernel launches
- ~0.15ms/step
- 内存 39MB → 几乎为零

---

## 优化 7：KV Trim（ISD 独有的 KV 管理）

### 为什么需要 trim？
AR 模型的 KV cache 只增不减。但 ISD 每步处理 2N-1 个位置，其中只有被接受的 token 的 KV 是有效的，MASK 位置和被拒绝的 spec token 的 KV 需要**释放**。

### 做法
每步 verify 后：
1. 算出 `trim_count`（要释放多少个 KV slot）和 `advance`（要 commit 多少个）
2. 查出要释放的 KV indices（batched GPU 操作，不是逐请求 sync）
3. 调用 `token_to_kv_pool.free()` 释放
4. 更新 `kv_committed_len`

这个 trim-and-commit cycle 是每步都要跑的，所以 batch 化很重要。

---

## LoRA 优化（5 步，177 → 213 tok/s）

### 背景
Lossless ISD 需要 **conditional LoRA**：MASK 位置用 base+LoRA（生成高质量 proposal），verify 位置用 base-only（保证精确的 AR 分布）。同一个 forward pass 里不同位置要不同的权重。

### 步骤 1：Conditional Mask（实现正确性）

给 LoRA residual 加一个 per-token 的 binary mask：

```
output = W·x + (mask ⊙ A·x) · B^T · α
```

mask[i] = 0（verify 位置）→ 不加 LoRA = base-only
mask[i] = 1（MASK 位置）→ 加 LoRA = base+LoRA

mask 是预分配的 GPU tensor，每步 in-place 更新内容（不重新分配），所以可以被 CUDA graph capture。

**效果**：228 → 194 tok/s（**下降**是因为 verify 位置现在正确地用 base-only 了，accept rate 从 80% 降到 71%）

### 步骤 2：CUDA Graph Capture with Real LoRA Weights

**问题**：SGLang 默认 capture CUDA graph 时用空的 LoRA ID（`lora_ids=[None]`），导致 LoRA kernel 被 capture 成 no-op。

**做法**：
- Graph capture **前**先 pre-load LoRA adapter 到 GPU 内存
- Capture 时传真实的 adapter ID，让 cuBLAS 操作用真实的 weight pointer 录制
- Replay 时 **跳过 `prepare_lora_batch()`**（graph 里的操作已经 baked-in 了），只更新 mask

**效果**：eager 13.4ms → graph replay 8.3ms

### 步骤 3：cuBLAS 替换 csgmv（177 → 228 tok/s）

**问题**：SGLang 默认的 LoRA kernel 是 Triton 写的 csgmv（chunked segmented GEMV），针对大 batch 优化。但 ISD 的 M=5（只有 5 个 token），csgmv 每个 kernel 11-19μs，288 个 kernel/forward = 4ms overhead。

| | csgmv (Triton) | cuBLAS (torch.mm) |
|---|---|---|
| Registers | 48 | 168 |
| Shared memory | 32KB | 160KB |
| Per-kernel | 11-19μs | 4.5-6.5μs |

**做法**：用 `torch.mm`（shrink）和 `torch.addmm_`（expand）替换 csgmv。对多 slice 的层（QKV=3 slices，GateUp=2 slices）分别做 per-slice `addmm_`。

**效果**：LoRA overhead 4.0ms → 1.3ms，TPS 177 → 228（+29%）

### 步骤 4：Two-Stream Overlap（194 → 212 tok/s）

**问题**：bs=1 时 base matmul 只用了 H100 的 <15% SMs，大量算力闲置。

**做法**：LoRA shrink（`mm + mask`）放到专门的 CUDA stream 上，跟 base matmul 并行执行：

```
lora_stream:  mm(x, A^T) → mul_(mask)     ← 跟 base 并行
main_stream:  mm(x, W^T)                   ← 跟 shrink 并行
              ↓ sync
main_stream:  addmm_(shrink_out, B^T)      ← 两个都完成后才执行
```

shrink output buffer 预分配（按 slice 数 1/2/3 分别缓存），避免在 side stream 上分配内存。

**效果**：隐藏了 57% 的 LoRA shrink 延迟，TPS 194 → 212

### 步骤 5：Deferred Draft Softmax（212 → 213 tok/s）

**问题**：标准路径每步对 draft logits 做 2 次 `F.softmax`（一次算 draft prob 存起来，一次在 verify 时用）。

**做法**：fused verify kernel 直接接收 raw logits，在 acceptance check 的 Pass 1 里 on-the-fly 算 softmax（反正已经在遍历 vocabulary 了）。省掉 2 次独立的 softmax call。

**效果**：+1 tok/s（很小但 free）

### LoRA 优化汇总

| 优化 | TPS | 说明 |
|------|-----|------|
| 原始 csgmv（CUDA graph） | 177 | 288 个 Triton kernel，每个 11-19μs |
| + cuBLAS 替换 | 228 | torch.mm/addmm_，每个 4.5-6.5μs |
| + Conditional mask | 194 | verify=base, MASK=LoRA（accept 下降是正确行为） |
| + Two-stream overlap | 212 | shrink 隐藏在 base matmul 背后 |
| + Deferred softmax | 213 | 省 2 次 F.softmax |
| **Non-LoRA 参考** | **267** | 无 adapter overhead |

---

## 剩余瓶颈：为什么 TP=4 时 AR 在高并发下赢了？

| | Strided DLLM | AR |
|---|---|---|
| Attention mode | Extend（2N-1 tokens） | Decode（1 token） |
| Metadata 成本 | 0.43ms | 0.07ms |
| CPU overhead | ~1.5ms（**串行**） | ~0.2ms（**被 overlap 隐藏**） |
| TP=1→4 scaling | 1.5x | 2.15x |

核心原因：ISD 的依赖链让 CPU 和 GPU 无法重叠。TP=4 时 GPU forward 变快（3.5ms），但 CPU overhead 不变（1.5ms），占比从 5% 涨到 30%。

### 理论上限
- GPU forward: 3.5ms，TPF: 2.70
- 理论 max: 2.70 / 3.5ms = **771 tok/s**
- 当前: 453 tok/s = **59%** of theoretical
- Gap: 1.9ms serial CPU overhead

### 未来方向
1. **Speculative metadata pre-computation**：78% 的 proposal 被接受，可以提前预测下一步的 seq_lens，在 GPU forward 期间预算 flashinfer plan
2. **Hybrid decode+extend attention**：prefix 部分用 decode attention（0.07ms metadata），新 token 部分用 local causal kernel
3. **Pipelined verification**：forward(t+1) 跟 verify(t) 并行执行，22% misprediction 时 rollback
