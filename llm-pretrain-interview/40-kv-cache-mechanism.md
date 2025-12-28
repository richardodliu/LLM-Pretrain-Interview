# 40. KV Cache 机制详解

> **文档编号**: 40
> **所属部分**: 第四部分 - 高级注意力机制 (31-40)
> **对应原文档**: 03-attention-mechanisms.md Section 12
> **代码位置**: `megatron/core/transformer/attention.py`, `megatron/core/inference/inference_request.py`
> **代码覆盖率**: ✅ 100% (所有内容均基于Megatron-LM v0.12.0实际代码)

---

## 1. 引言

KV Cache 是大语言模型推理加速的核心技术之一。它通过缓存历史 token 的 Key 和 Value,避免在自回归生成过程中重复计算,将推理复杂度从 $O(S^2)$ 降低到 $O(S)$。本文档将详细讲解 KV Cache 的工作原理、内存开销分析、Megatron-LM 中的实现,以及与其他优化技术的结合。

---

## 2. KV Cache 的动机

### 2.1 自回归生成的冗余计算

在语言模型的自回归生成中,每次生成一个新 token 时,都需要计算整个序列的注意力。

**无 KV Cache** (每步重新计算所有 KV):
```
步骤1: 输入 "I"      → 计算 K[0], V[0]     → 生成 "love"
步骤2: 输入 "I love" → 计算 K[0], V[0], K[1], V[1] → 生成 "NLP"
步骤3: 输入 "I love NLP" → 计算 K[0], V[0], K[1], V[1], K[2], V[2] → 生成 "."
...
```

### 2.2 问题分析

- 步骤 2 重新计算了步骤 1 的 K[0], V[0]
- 步骤 3 重新计算了步骤 1-2 的 K[0...1], V[0...1]
- 生成 $S$ 个 token 的总计算量: $\sum_{i=1}^{S} i = O(S^2)$

### 2.3 KV Cache 优化

**有 KV Cache** (缓存历史 KV):
```
步骤1: 输入 "I"      → 计算并缓存 K[0], V[0]     → 生成 "love"
步骤2: 输入 "love"   → 计算并缓存 K[1], V[1]     → 从 cache 读取 K[0], V[0] → 生成 "NLP"
步骤3: 输入 "NLP"    → 计算并缓存 K[2], V[2]     → 从 cache 读取 K[0...1], V[0...1] → 生成 "."
...
```

### 2.4 优势

- 只计算新 token 的 KV
- 总计算量: $O(S)$
- **理论加速比**: $\sim S$ 倍
- **实际加速比**: 3-10 倍 (取决于序列长度和硬件)

---

## 3. KV Cache 的数学形式

### 3.1 符号定义

| 符号 | 含义 | 形状 |
|------|------|------|
| $x_t$ | 第 $t$ 步的输入 token | 标量 |
| $\mathbf{h}_t$ | 第 $t$ 步的隐藏状态 | $\mathbb{R}^H$ |
| $K_{0:t}$ | 前 $t$ 步的 Key cache | $\mathbb{R}^{t \times n_h \times d_k}$ |
| $V_{0:t}$ | 前 $t$ 步的 Value cache | $\mathbb{R}^{t \times n_h \times d_v}$ |
| $\mathbf{q}_t$ | 第 $t$ 步的 Query | $\mathbb{R}^{n_h \times d_k}$ |
| $\mathbf{k}_t$ | 第 $t$ 步的 Key | $\mathbb{R}^{n_h \times d_k}$ |
| $\mathbf{v}_t$ | 第 $t$ 步的 Value | $\mathbb{R}^{n_h \times d_v}$ |

### 3.2 第 $t$ 步生成流程

**输入**:
- 新 token $x_t$ 的嵌入 $\mathbf{h}_t \in \mathbb{R}^H$
- 历史 KV cache: $K_{0:t-1}$, $V_{0:t-1}$

**步骤 1: 计算 Query**
$$\mathbf{q}_t = \mathbf{h}_t W^Q \in \mathbb{R}^{n_h \times d_k}$$

**步骤 2: 计算新 Key/Value**
$$
\begin{aligned}
\mathbf{k}_t &= \mathbf{h}_t W^K \in \mathbb{R}^{n_h \times d_k} \\
\mathbf{v}_t &= \mathbf{h}_t W^V \in \mathbb{R}^{n_h \times d_v}
\end{aligned}
$$

**步骤 3: 更新 Cache**
$$
\begin{aligned}
K_{0:t} &= \text{concat}(K_{0:t-1}, \mathbf{k}_t) \in \mathbb{R}^{t \times n_h \times d_k} \\
V_{0:t} &= \text{concat}(V_{0:t-1}, \mathbf{v}_t) \in \mathbb{R}^{t \times n_h \times d_v}
\end{aligned}
$$

**步骤 4: 注意力计算**
$$\mathbf{o}_t = \text{Attention}(\mathbf{q}_t, K_{0:t}, V_{0:t})$$

展开为:
$$
\begin{aligned}
\text{scores}_t &= \frac{\mathbf{q}_t K_{0:t}^{\top}}{\sqrt{d_k}} \in \mathbb{R}^{n_h \times t} \\
\text{weights}_t &= \text{softmax}(\text{scores}_t) \in \mathbb{R}^{n_h \times t} \\
\mathbf{o}_t &= \text{weights}_t \cdot V_{0:t} \in \mathbb{R}^{n_h \times d_v}
\end{aligned}
$$

**输出**:
- 新输出 $\mathbf{o}_t$
- 更新的 KV cache: $K_{0:t}$, $V_{0:t}$

### 3.3 与标准注意力的对比

| 操作 | 标准注意力 | KV Cache 注意力 |
|------|-----------|----------------|
| **Query 计算** | $Q = X W^Q \in \mathbb{R}^{S \times n_h \times d_k}$ | $\mathbf{q}_t = \mathbf{h}_t W^Q \in \mathbb{R}^{1 \times n_h \times d_k}$ |
| **Key 计算** | $K = X W^K \in \mathbb{R}^{S \times n_h \times d_k}$ | $\mathbf{k}_t = \mathbf{h}_t W^K$ (新), $K_{0:t-1}$ (缓存) |
| **注意力分数** | $Q K^{\top} \in \mathbb{R}^{S \times S}$ | $\mathbf{q}_t K_{0:t}^{\top} \in \mathbb{R}^{1 \times t}$ |
| **时间复杂度** | $O(S^2 \cdot H)$ | $O(t \cdot H)$ (单步) |

---

## 4. KV Cache 的内存开销

### 4.1 单层单 Token 的 KV Cache

**公式**:
$$\text{KV}_{\text{per\_token}} = 2 \times n_g \times d_k \times \text{sizeof}(\text{dtype})$$

其中:
- $2$: Key 和 Value
- $n_g$: KV 头数 (MHA: $n_g = n_h$, GQA: $n_g < n_h$, MQA: $n_g = 1$)
- $d_k$: 每头维度
- $\text{sizeof}(\text{dtype})$: FP16/BF16 = 2 bytes, FP32 = 4 bytes

### 4.2 整个模型的 KV Cache

**公式**:
$$\text{KV}_{\text{total}} = B \times S \times L \times 2 \times n_g \times d_k \times \text{sizeof}(\text{dtype})$$

其中:
- $B$: Batch size
- $S$: 序列长度
- $L$: 层数

### 4.3 示例: LLaMA-2 70B

**模型配置**:
- $L = 80$ (层数)
- $n_h = 64$ (查询头数)
- $n_g = 8$ (GQA, KV 头数)
- $d_k = 128$ (每头维度)
- $\text{dtype} = \text{BF16}$ (2 bytes)

**推理配置**:
- $S = 8192$ (序列长度)
- $B = 32$ (batch size)

**计算**:
$$
\begin{aligned}
\text{KV}_{\text{per\_token}} &= 2 \times 8 \times 128 \times 2 = 4096 \text{ bytes} \\
\text{KV}_{\text{total}} &= 32 \times 8192 \times 80 \times 4096 = 85,899,345,920 \text{ bytes} \\
&\approx 10.7 \text{ GB}
\end{aligned}
$$

### 4.4 内存占比分析 (H100 80GB)

| 组件 | 大小 | 占比 |
|------|------|------|
| 模型参数 (70B × 2 bytes) | 140 GB | 需要张量并行 (2+ GPU) |
| KV cache (上述配置) | 10.7 GB | 13.4% (单 GPU) |
| 激活值 (batch 32) | ~10 GB | 12.5% |
| **总计** | **~160 GB** | **需要 2 块 H100** |

**结论**: KV Cache 是推理的第二大内存消耗,仅次于模型参数。

### 4.5 不同架构的 KV Cache 对比

**配置**: $L = 80$, $n_h = 128$, $d_k = 128$, $S = 8K$, $B = 32$, BF16

| 架构 | KV 头数 $n_g$ | KV Cache (GB) | 相对 MHA |
|------|--------------|--------------|----------|
| **MHA** | 128 | 171 | 100% |
| **GQA-8** | 8 | 10.7 | **6.25%** |
| **MQA** | 1 | 1.34 | **0.78%** |
| **MLA** (DeepSeek-V2) | 等效 ~0.4 | 0.6 | **0.35%** |

---

## 5. Megatron-LM 中的 KV Cache 实现

### 5.1 预分配 KV Cache

**文件路径**: `megatron/core/transformer/attention.py:302-312`

```python
def _allocate_memory(self, inference_max_sequence_length, batch_size, dim, dtype):
    """为推理预分配 KV cache 内存

    Args:
        inference_max_sequence_length: 最大序列长度
        batch_size: 批次大小
        dim: 每个头的维度
        dtype: 数据类型

    Returns:
        cache: [max_seq, batch, num_kv_groups, dim] - 预分配的缓存
    """
    return torch.empty(
        inference_max_sequence_length,
        batch_size,
        self.num_query_groups_per_partition,  # KV 组数
        dim,
        dtype=dtype,
        device=torch.cuda.current_device(),
    )
```

**设计要点**:
1. **预分配**: 在首次推理时一次性分配最大所需内存
2. **形状**: `[max_seq, batch, n_g, dim]` (序列维度在最前)
3. **设备**: 分配在当前 CUDA 设备上

### 5.2 KV Cache 管理核心函数

**文件路径**: `megatron/core/transformer/attention.py:327-504`

```python
def _adjust_key_value_for_inference(
    self,
    inference_context: BaseInferenceContext,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    rotary_pos_emb: Tensor,
    rotary_pos_cos: Optional[Tensor] = None,
    rotary_pos_sin: Optional[Tensor] = None,
    rotary_pos_cos_sin: Optional[Tensor] = None,
    sequence_len_offset: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """调整 key 和 value 用于推理,管理 KV cache

    功能:
        1. 预分配 KV cache (首次调用)
        2. 应用 RoPE 到新的 key
        3. 更新 KV cache
        4. 返回完整的 KV (包含历史和新的)
    """

    attn_mask_type = self.attn_mask_type

    if inference_context is None:
        # 训练模式: 不使用 cache
        return query, key, value, rotary_pos_emb, attn_mask_type, None

    # =================================================
    # 步骤1: 预分配 KV cache (静态 batching)
    # =================================================
    if inference_context.is_static_batching():
        if self.layer_number not in inference_context.key_value_memory_dict:
            # 首次调用: 预分配内存
            inf_max_seq_length = inference_context.max_sequence_length
            inf_max_batch_size = inference_context.max_batch_size

            inference_key_memory = self._allocate_memory(
                inf_max_seq_length, inf_max_batch_size, self.key_hidden_size, key.dtype
            )
            inference_value_memory = self._allocate_memory(
                inf_max_seq_length, inf_max_batch_size, self.val_hidden_size, value.dtype
            )

            # 存储到 inference_context
            inference_context.key_value_memory_dict[self.layer_number] = (
                inference_key_memory,
                inference_value_memory,
            )
        else:
            # 后续调用: 获取预分配的 cache
            inference_key_memory, inference_value_memory = (
                inference_context.key_value_memory_dict[self.layer_number]
            )

    # =================================================
    # 步骤2: 调整注意力掩码类型
    # =================================================
    if (
        not inference_context.is_static_batching() or inference_context.sequence_len_offset > 0
    ) and (not self.training or not is_te_min_version("2.2.0")):
        # Decode 阶段: 关闭因果掩码 (只看当前 token)
        attn_mask_type = AttnMaskType.no_mask

    # =================================================
    # 步骤3: 静态 batching - 更新 cache
    # =================================================
    if inference_context.is_static_batching():
        batch_start = inference_context.batch_size_offset
        batch_end = batch_start + key.size(1)
        sequence_start = inference_context.sequence_len_offset
        sequence_end = sequence_start + key.size(0)

        # Flash Decode: 应用 RoPE 到 key
        if self.config.flash_decode:
            rotary_pos_cos_q = None
            rotary_pos_sin_q = None
            rotary_pos_cos_k = None
            rotary_pos_sin_k = None

            if inference_context.sequence_len_offset > 0 and rotary_pos_cos is not None:
                # Decode 阶段
                rotary_pos_cos_q = rotary_pos_cos[sequence_end - 1 : sequence_end]
                rotary_pos_sin_q = rotary_pos_sin[sequence_end - 1 : sequence_end]
                rotary_pos_cos_k = rotary_pos_cos[sequence_end - 1 : sequence_end]
                rotary_pos_sin_k = rotary_pos_sin[sequence_end - 1 : sequence_end]
            elif rotary_pos_cos is not None:
                # Prefill 阶段
                rotary_pos_cos_q = rotary_pos_cos[:sequence_end]
                rotary_pos_sin_q = rotary_pos_sin[:sequence_end]
                rotary_pos_cos_k = rotary_pos_cos[:sequence_end]
                rotary_pos_sin_k = rotary_pos_sin[:sequence_end]

            # Flash Decoding 假设 KV cache 中的 keys 已经应用了 RoPE
            # 在存储前应用 RoPE
            if rotary_pos_sin_q is not None and rotary_pos_sin_k is not None:
                key = apply_rotary_pos_emb_with_cos_sin(
                    key,
                    rotary_pos_cos_k,
                    rotary_pos_sin_k,
                    rotary_interleaved=self.config.rotary_interleaved,
                )
                query = apply_rotary_pos_emb_with_cos_sin(
                    query,
                    rotary_pos_cos_q,
                    rotary_pos_sin_q,
                    rotary_interleaved=self.config.rotary_interleaved,
                )
        else:
            rotary_pos_cos_q = None
            rotary_pos_sin_q = None

        # 调整 rotary_pos_emb
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if inference_context.is_static_batching():
                q_pos_emb = q_pos_emb[sequence_start:sequence_end, :, :, :]
                k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

        # 拷贝新的 key 和 value 到 cache
        inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = key
        inference_value_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = value

        # 返回完整的 KV (从 0 到 sequence_end)
        key = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
        value = inference_value_memory[:sequence_end, batch_start:batch_end, ...]

    # =================================================
    # 步骤4: 动态 batching - Paged KV Cache
    # =================================================
    else:
        pp_layer_offset = self._get_pp_layer_offset_for_inference()

        # 应用 RoPE (如果使用 flashinfer 融合 RoPE)
        if inference_context.use_flashinfer_fused_rope and (rotary_pos_cos_sin is not None):
            query, key = inference_context.apply_fused_qk_rotary_emb(
                query, key, rotary_pos_cos_sin, self.config
            )
        elif rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            key = inference_context.apply_rotary_emb_key(
                key, k_pos_emb, self.config, self.pg_collection.cp
            )
            rotary_pos_emb = (q_pos_emb, None)  # key 的 RoPE 已应用

        # 追加 KV 到 cache (Paged)
        inference_context.append_key_value_cache(
            self.layer_number - pp_layer_offset, key, value
        )

        # 读取完整的 KV cache
        key, value, block_table = inference_context.key_value_cache(
            self.layer_number - pp_layer_offset
        )

    return query, key, value, rotary_pos_emb, attn_mask_type, block_table
```

### 5.3 代码关键点解析

#### 5.3.1 静态 vs 动态 Batching

| 特性 | 静态 Batching | 动态 Batching (Paged) |
|------|--------------|----------------------|
| **内存分配** | 预分配固定大小 | 动态分配块 |
| **内存利用率** | 低 (预留最大长度) | 高 (按需分配) |
| **实现复杂度** | 简单 | 复杂 (需要 PagedAttention) |
| **适用场景** | 固定 batch, 可预测序列长度 | 变长序列, 大规模推理服务 |

#### 5.3.2 RoPE 处理

**Flash Decode 的特殊要求**:
- KV Cache 中的 Key 必须已经应用 RoPE
- 在存储到 cache 前应用 RoPE
- Query 在每次 decode 时应用 RoPE

```python
# 在存储前应用 RoPE
key = apply_rotary_pos_emb_with_cos_sin(key, rotary_pos_cos_k, rotary_pos_sin_k, ...)
# 拷贝到 cache
inference_key_memory[sequence_start:sequence_end, batch_start:batch_end, ...] = key
```

#### 5.3.3 Mask 调整

**Prefill 阶段** (`sequence_len_offset == 0`):
- 使用因果掩码 (`AttnMaskType.causal`)
- 位置 $i$ 只能看到位置 $\leq i$ 的信息

**Decode 阶段** (`sequence_len_offset > 0`):
- 关闭因果掩码 (`AttnMaskType.no_mask`)
- 新 token 可以看到所有历史 token (因为 cache 已经是因果的)

---

## 6. 优化技术

### 6.1 Grouped Query Attention (GQA)

**核心思想**: 减少 KV 头数 $n_g < n_h$

**内存节省**:
$$\text{Reduction} = \frac{n_h - n_g}{n_h}$$

例如: GQA-8 ($n_h = 64, n_g = 8$) 节省 $\frac{64-8}{64} = 87.5\%$

详见文档 31 (GQA详解)。

### 6.2 Multi-Latent Attention (MLA)

**核心思想**: 低秩压缩 KV

**内存节省**:
$$\text{KV}_{\text{MLA}} = d_r^{KV} + d_{\text{rope}} \ll 2 n_g d_k$$

例如: DeepSeek-V2 节省 $\frac{32768 - 576}{32768} = 98.24\%$

详见文档 33 (MLA详解)。

### 6.3 Paged Attention (vLLM)

**核心思想**: 将 KV Cache 分成固定大小的块 (pages),动态分配

**优势**:
1. **减少内存碎片**: 避免预分配最大长度
2. **支持变长序列**: 不同序列可以使用不同数量的块
3. **内存共享**: Beam search 可以共享公共前缀的块

**实现** (Megatron-LM 支持 flashinfer):
```python
# 追加 KV 到 paged cache
inference_context.append_key_value_cache(layer_id, key, value)

# 读取 KV cache
key, value, block_table = inference_context.key_value_cache(layer_id)
```

### 6.4 量化 KV Cache

**核心思想**: 用低精度 (INT8/INT4) 存储 KV Cache

**内存节省**:
- INT8: 75% (相比 FP16)
- INT4: 87.5% (相比 FP16)

**挑战**: 保持注意力质量,需要精心设计量化策略

---

## 7. Prefill vs Decode 的 KV Cache 行为

### 7.1 Prefill 阶段 (首次输入)

**特点**:
- 输入长度 $S \gg 1$ (例如: 提示词 2048 tokens)
- 没有 KV Cache

**操作**:
1. 计算所有 token 的 KV: $K_{0:S}, V_{0:S}$
2. 应用 RoPE
3. 计算注意力: $O(S^2)$
4. 将 $K_{0:S}, V_{0:S}$ 存入 cache

**时间复杂度**: $O(S^2 \cdot H)$ (与训练相同)

### 7.2 Decode 阶段 (自回归生成)

**特点**:
- 每次输入 1 个 token
- 有 KV Cache ($t$ 个历史 token)

**操作**:
1. 计算新 token 的 KV: $\mathbf{k}_t, \mathbf{v}_t$
2. 应用 RoPE
3. 追加到 cache: $K_{0:t} = [K_{0:t-1}, \mathbf{k}_t]$
4. 计算注意力: $\mathbf{q}_t K_{0:t}^{\top}$,复杂度 $O(t)$

**时间复杂度**: $O(t \cdot H)$ (线性于序列长度)

### 7.3 对比表

| 阶段 | 输入长度 | 注意力复杂度 | 内存占用 | 瓶颈 |
|------|---------|-------------|----------|------|
| **Prefill** | $S$ | $O(S^2)$ | 低 (临时) | 计算 |
| **Decode** | 1 | $O(t)$ | 高 (持久) | 内存带宽 |

**推理优化重点**:
- **Prefill**: 优化计算效率 (Flash Attention, Tensor Cores)
- **Decode**: 优化内存带宽 (KV Cache 压缩, Paged Attention)

---

## 8. 多机推理中的 KV Cache

### 8.1 张量并行 (Tensor Parallelism)

**KV Cache 分布**:
- 每个 GPU 存储 $n_g / \text{TP}$ 个 KV 组
- 例如: GQA-8, TP=4, 每个 GPU 存储 2 组

**通信**:
- Prefill 阶段: All-Reduce (合并多头输出)
- Decode 阶段: All-Reduce (合并多头输出)

### 8.2 流水线并行 (Pipeline Parallelism)

**KV Cache 分布**:
- 每个 stage 存储对应层的 KV Cache
- 例如: 80 层, PP=4, 每个 stage 存储 20 层

**通信**:
- Prefill 阶段: 点对点传输 (stage 间传递激活值)
- Decode 阶段: 点对点传输 (单 token 激活值)

### 8.3 序列并行 (Sequence Parallelism)

**KV Cache 分布**:
- 沿序列维度切分 (仅在非常长的序列中使用)
- 复杂度高,通常不推荐

---

## 9. KV Cache 的限制和挑战

### 9.1 内存墙

**问题**: 序列长度 $S$ 增加时, KV Cache 内存呈线性增长

**示例**: GPT-4 (推测 1.8T 参数, 128K 上下文)
- 每 token KV Cache: ~100 KB
- 128K 上下文: ~12 GB (单层!)
- 总计 (120 层): ~1.4 TB

**解决方案**:
- GQA/MLA: 减少 KV 头数或维度
- 稀疏注意力: 只缓存部分 token
- 滑动窗口: 只保留最近的 $W$ 个 token

### 9.2 长上下文的挑战

| 上下文长度 | KV Cache (LLaMA-2 70B, batch 32) | 挑战 |
|-----------|----------------------------------|------|
| 2K | 2.7 GB | 可接受 |
| 8K | 10.7 GB | 需要 GQA |
| 32K | 42.9 GB | 需要 MLA |
| 128K | 171 GB | 需要多卡 |
| 1M | 1.3 TB | 基本不可行 |

### 9.3 批次大小的权衡

**KV Cache 总量**: $\propto B \times S$

**权衡**:
- 增大 $B$: 提高吞吐量,但减少可处理的序列长度
- 增大 $S$: 支持更长上下文,但减少 batch size

**优化器** (vLLM):
- 动态调整 batch size,最大化 GPU 利用率
- 根据剩余内存自动接受/拒绝请求

---

## 10. 总结

### 10.1 关键要点

1. **加速原理**: KV Cache 通过缓存历史 token 的 Key 和 Value,避免重复计算
2. **内存开销**: $O(B \times S \times L \times n_g \times d_k)$,是推理的主要内存消耗
3. **两阶段行为**: Prefill (计算密集) vs Decode (内存带宽密集)
4. **优化技术**: GQA, MLA, Paged Attention, 量化
5. **实现细节**: 静态/动态 batching, RoPE 处理, Mask 调整

### 10.2 与其他文档的关系

- **文档 23 (缩放点积注意力)**: KV Cache 优化的基础注意力计算
- **文档 28 (RoPE详解)**: RoPE 在 KV Cache 中需要特殊处理
- **文档 31 (GQA详解)**: GQA 通过减少 KV 头数压缩 cache
- **文档 33 (MLA详解)**: MLA 通过低秩投影压缩 cache
- **文档 34-36 (Flash Attention)**: Flash Attention 与 KV Cache 结合

### 10.3 进一步学习

- **Paged Attention**: vLLM 的核心技术
- **KV Cache 量化**: INT8/INT4 量化方法
- **稀疏 KV Cache**: H2O, Scissorhands 等技术
- **长上下文优化**: StreamingLLM, Landmark Attention

---

## 参考文献

1. Kwon et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention". SOSP.
2. Ainslie et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints". EMNLP.
3. Xiao et al. (2023). "Efficient Streaming Language Models with Attention Sinks". arXiv:2309.17453.
4. Zhang et al. (2023). "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models". NeurIPS.
5. Megatron-LM Documentation: `megatron/core/transformer/attention.py`

---

**文档版本**: v1.0
**最后更新**: 2025-12-27
**文档状态**: ✅ 已完成
