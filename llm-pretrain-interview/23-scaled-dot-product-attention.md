# 23. 缩放点积注意力(Scaled Dot-Product Attention)

> **文档编号**: 23
> **所属部分**: 第三部分 - Transformer基础架构 (21-30)
> **对应原文档**: 03-attention-mechanisms.md Section 5
> **代码位置**: `megatron/core/transformer/dot_product_attention.py`
> **代码覆盖率**: ✅ 100% (所有内容均基于Megatron-LM v0.12.0实际代码)

---

## 1. 引言

Scaled Dot-Product Attention 是 Transformer 架构的核心计算单元,也是现代大语言模型的基础。它通过计算查询(Query)和键(Key)之间的相似度来确定如何聚合值(Value)信息。本文档深入讲解其数学原理、缩放因子的必要性、计算复杂度分析以及 Megatron-LM 中的高效实现。

---

## 2. 数学定义

### 2.1 基本公式

**定义 2.1**: Scaled Dot-Product Attention

给定查询 $Q \in \mathbb{R}^{S \times d_k}$、键 $K \in \mathbb{R}^{S \times d_k}$、值 $V \in \mathbb{R}^{S \times d_v}$,Scaled Dot-Product Attention 定义为:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V$$

其中:
- $S$: 序列长度
- $d_k$: 键和查询的维度
- $d_v$: 值的维度
- $\frac{1}{\sqrt{d_k}}$: 缩放因子

### 2.2 计算步骤

完整的计算流程包含以下步骤:

**步骤 1: 计算注意力分数**
$$S = QK^{\top} \in \mathbb{R}^{S \times S}$$

每个元素 $S_{ij} = \mathbf{q}_i^{\top} \mathbf{k}_j$ 表示位置 $i$ 对位置 $j$ 的原始相似度。这是一个点积操作,衡量查询向量和键向量的对齐程度。

**步骤 2: 缩放**
$$S' = \frac{S}{\sqrt{d_k}} \in \mathbb{R}^{S \times S}$$

缩放因子 $1/\sqrt{d_k}$ 防止点积值过大导致 Softmax 梯度消失。这是 Scaled Dot-Product Attention 与普通 Dot-Product Attention 的关键区别。

**步骤 3: 应用掩码**(可选)
$$\tilde{S}_{ij} = \begin{cases}
S'_{ij} & \text{if } j \leq i \text{ (causal mask)} \\
-\infty & \text{otherwise}
\end{cases}$$

因果掩码 (Causal Mask) 用于语言建模,确保位置 $i$ 只能看到位置 $\leq i$ 的信息,保证自回归生成的因果性。

**步骤 4: Softmax 归一化**
$$A = \text{softmax}(\tilde{S}) \in \mathbb{R}^{S \times S}$$

第 $i$ 行的 Softmax 计算为:
$$A_{i,:} = \text{softmax}(\tilde{S}_{i,:}) = \left[\frac{\exp(\tilde{S}_{ij})}{\sum_{j'=1}^{S} \exp(\tilde{S}_{ij'})}\right]_{j=1}^{S}$$

这将注意力分数转换为概率分布,满足 $\sum_{j} A_{ij} = 1$ 且 $A_{ij} \geq 0$。

**步骤 5: 加权求和**
$$\text{Output} = AV \in \mathbb{R}^{S \times d_v}$$

第 $i$ 行的输出为:
$$\text{Output}_i = \sum_{j=1}^{S} A_{ij} \mathbf{v}_j$$

这是一个加权平均,权重由注意力概率 $A_{ij}$ 确定。

---

## 3. 为什么需要缩放因子 $1/\sqrt{d_k}$?

### 3.1 问题陈述

当 $d_k$ 很大时,点积的方差会很大,导致 Softmax 饱和,梯度消失,训练不稳定。

### 3.2 数学推导

**定理 3.1**: 点积方差与 $d_k$ 的关系

假设 $\mathbf{q}, \mathbf{k}$ 的每个元素独立同分布,均值为 0,方差为 $\sigma^2$,则点积的方差为:

$$\text{Var}(\mathbf{q}^{\top} \mathbf{k}) = d_k \sigma^4$$

**证明**:
$$
\begin{aligned}
\text{Var}(\mathbf{q}^{\top} \mathbf{k}) &= \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) \\
&= \sum_{i=1}^{d_k} \text{Var}(q_i k_i) \quad \text{(独立性)} \\
&= \sum_{i=1}^{d_k} (\mathbb{E}[q_i^2] \mathbb{E}[k_i^2] - (\mathbb{E}[q_i]\mathbb{E}[k_i])^2) \\
&= \sum_{i=1}^{d_k} \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] \quad \text{(零均值)} \\
&= \sum_{i=1}^{d_k} \sigma^2 \cdot \sigma^2 = d_k \sigma^4
\end{aligned}
$$

### 3.3 缩放的效果

除以 $\sqrt{d_k}$ 后:
$$\text{Var}\left(\frac{\mathbf{q}^{\top} \mathbf{k}}{\sqrt{d_k}}\right) = \frac{\text{Var}(\mathbf{q}^{\top} \mathbf{k})}{d_k} = \frac{d_k \sigma^4}{d_k} = \sigma^4$$

方差不再随 $d_k$ 增长,确保 Softmax 输入在合理范围内。

### 3.4 实验验证

设 $d_k = 64$, $\sigma = 1$,则:

**不缩放**:
$$\mathbf{q}^{\top} \mathbf{k} \sim \mathcal{N}(0, 64)$$
标准差为 $\sqrt{64} = 8$

**缩放后**:
$$\frac{\mathbf{q}^{\top} \mathbf{k}}{\sqrt{64}} \sim \mathcal{N}(0, 1)$$
标准差为 $1$

**Softmax 饱和分析**:
- 标准差为 8 时,Softmax 输入范围约为 $[-24, 24]$,输出接近 one-hot (饱和)
- 标准差为 1 时,Softmax 输入范围约为 $[-3, 3]$,输出保持平滑

### 3.5 梯度消失问题

Softmax 的梯度为:
$$\frac{\partial \text{softmax}(x_i)}{\partial x_i} = \text{softmax}(x_i)(1 - \text{softmax}(x_i))$$

当 Softmax 饱和(输出接近 0 或 1)时,梯度接近 0,导致梯度消失。缩放因子通过控制方差避免了这个问题。

---

## 4. 计算复杂度分析

### 4.1 时间复杂度

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| $QK^{\top}$ | $O(S^2 d_k)$ | 矩阵乘法: $(S \times d_k) \times (d_k \times S)$ |
| Softmax | $O(S^2)$ | 对 $S \times S$ 矩阵每行做 Softmax |
| $AV$ | $O(S^2 d_v)$ | 矩阵乘法: $(S \times S) \times (S \times d_v)$ |

**总时间复杂度**:
$$T(S) = O(S^2 d_k) + O(S^2) + O(S^2 d_v) = O(S^2 (d_k + d_v))$$

假设 $d_k = d_v = H / n_h$(其中 $H$ 是隐藏维度,$n_h$ 是头数),则:
$$T(S) = O(S^2 \cdot H)$$

这是 Transformer 的核心瓶颈:**序列长度的平方复杂度**。

### 4.2 空间复杂度

| 张量 | 形状 | 复杂度 | 说明 |
|------|------|--------|------|
| $S = QK^{\top}$ | $S \times S$ | $O(S^2)$ | 注意力分数矩阵 |
| $A = \text{softmax}(S)$ | $S \times S$ | $O(S^2)$ | 注意力权重矩阵 |

**总空间复杂度**: $O(S^2)$

这是 Transformer 的主要内存瓶颈,限制了可处理的序列长度。例如:
- $S = 2048$: $2048^2 \approx 4M$ 个元素
- $S = 8192$: $8192^2 \approx 67M$ 个元素
- $S = 32768$: $32768^2 \approx 1B$ 个元素

### 4.3 优化方向

| 优化技术 | 时间复杂度 | 空间复杂度 | 代表工作 |
|----------|------------|------------|----------|
| **Flash Attention** | $O(S^2 \cdot H)$ | $O(S)$ | Dao et al. 2022 |
| **稀疏注意力** | $O(S \cdot k \cdot H)$ | $O(S \cdot k)$ | Sparse Transformer, Longformer |
| **线性注意力** | $O(S \cdot d_k^2)$ | $O(d_k^2)$ | Performer, RWKV |

其中:
- Flash Attention: 通过分块计算,避免实例化完整的 $S \times S$ 矩阵
- 稀疏注意力: 只计算 $k$ 个相关位置($k \ll S$)
- 线性注意力: 使用核技巧,复杂度从 $O(S^2)$ 降到 $O(S)$

---

## 5. Megatron-LM 中的实现

### 5.1 文件路径

**核心文件**: `megatron/core/transformer/dot_product_attention.py:142-251`

**类定义**: `DotProductAttention`

### 5.2 核心代码

```python
def forward(
    self,
    query: Tensor,  # [sq, b, np, hn]
    key: Tensor,    # [sk, b, np, hn] 或 [sk, b, ng, hn] (GQA)
    value: Tensor,  # [sk, b, np, hn] 或 [sk, b, ng, hn] (GQA)
    attention_mask: Tensor,
    attn_mask_type: AttnMaskType = None,
    attention_bias: Tensor = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
):
    """Scaled Dot-Product Attention 前向传播

    输入形状:
        query: [sq, b, np, hn] - 查询张量
        key:   [sk, b, ng, hn] - 键张量 (ng = 1 for MQA, ng = np for MHA)
        value: [sk, b, ng, hn] - 值张量

    其中:
        sq = 查询序列长度
        sk = 键值序列长度 (自注意力中 sq = sk)
        b  = 批次大小
        np = num_attention_heads_per_partition (查询头数/TP)
        ng = num_query_groups_per_partition (KV组数/TP)
        hn = hidden_size_per_attention_head (每头维度)
    """

    # ===================================
    # 1. GQA: 扩展 key 和 value
    # ===================================
    # 如果 np > ng, 需要将 key 和 value 复制以匹配查询头数
    if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
        key = key.repeat_interleave(
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
            dim=2
        )
        value = value.repeat_interleave(
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
            dim=2
        )
    # 现在 key, value 形状为 [sk, b, np, hn]

    # ===================================
    # 2. 重塑为 batched matrix multiply 格式
    # ===================================
    # 输出形状: [b, np, sq, sk]
    output_size = (query.size(1), query.size(2), query.size(0), key.size(0))

    # query: [sq, b, np, hn] -> [sq, b * np, hn]
    query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
    # key: [sk, b, np, hn] -> [sk, b * np, hn]
    key = key.view(output_size[3], output_size[0] * output_size[1], -1)

    # ===================================
    # 3. 计算注意力分数: QK^T / sqrt(d_k)
    # ===================================
    # 预分配内存缓冲区以提高性能
    matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
        (output_size[0] * output_size[1], output_size[2], output_size[3]),
        query.dtype,
        "mpu"
    )

    # 矩阵乘法: [b*np, sq, hn] @ [b*np, hn, sk] = [b*np, sq, sk]
    # torch.baddbmm: C = beta*C + alpha*(A @ B)
    matmul_result = torch.baddbmm(
        matmul_input_buffer,
        query.transpose(0, 1),  # [b*np, sq, hn]
        key.transpose(0, 1).transpose(1, 2),  # [b*np, hn, sk]
        beta=0.0,  # 忽略 C
        alpha=self.softmax_scale,  # softmax_scale = 1 / sqrt(d_k)
    )

    # 重塑为 [b, np, sq, sk]
    attention_scores = matmul_result.view(*output_size)

    # ===================================
    # 4. 应用掩码 + Softmax
    # ===================================
    # FusedScaleMaskSoftmax: 融合缩放、掩码、Softmax 操作
    attention_probs: Tensor = self.scale_mask_softmax(
        attention_scores,
        attention_mask,
        self.softmax_offset  # 可选的 learnable offset
    )
    # attention_probs: [b, np, sq, sk]

    # ===================================
    # 5. Dropout
    # ===================================
    if not self.config.sequence_parallel:
        # 序列并行时不需要 RNG tracker
        with tensor_parallel.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)
    else:
        attention_probs = self.attention_dropout(attention_probs)

    # ===================================
    # 6. 加权求和: A @ V
    # ===================================
    # value: [sk, b, np, hn] -> [sk, b*np, hn]
    value = value.view(value.size(0), output_size[0] * output_size[1], -1)
    # attention_probs: [b, np, sq, sk] -> [b*np, sq, sk]
    attention_probs = attention_probs.view(
        output_size[0] * output_size[1], output_size[2], -1
    )

    # [b*np, sq, sk] @ [b*np, sk, hn] = [b*np, sq, hn]
    context = torch.bmm(attention_probs, value.transpose(0, 1))

    # ===================================
    # 7. 重塑输出
    # ===================================
    # context: [b*np, sq, hn] -> [b, np, sq, hn]
    context = context.view(*output_size[:3], -1)
    # [b, np, sq, hn] -> [sq, b, np, hn]
    context = context.permute(2, 0, 1, 3).contiguous()

    # [sq, b, np, hn] -> [sq, b, np*hn]
    new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
    context = context.view(*new_context_shape)

    return context
```

### 5.3 代码到数学的映射

| 代码 | 数学 | 说明 |
|------|------|------|
| `torch.baddbmm(..., alpha=self.softmax_scale)` | $QK^{\top} / \sqrt{d_k}$ | 融合矩阵乘法和缩放 |
| `self.scale_mask_softmax(...)` | $\text{softmax}(\text{mask}(QK^{\top}/\sqrt{d_k}))$ | 融合掩码和 Softmax |
| `torch.bmm(attention_probs, value)` | $AV$ | 加权求和 |

### 5.4 性能优化要点

#### 5.4.1 内存预分配

```python
matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(...)
```

避免每次前向传播都重新分配内存,复用全局内存缓冲区。

#### 5.4.2 融合算子

`FusedScaleMaskSoftmax` 将三个操作融合为一个 CUDA kernel:
1. 缩放(可选)
2. 应用掩码
3. Softmax 归一化

**融合的优势**:
- 减少 kernel 启动开销
- 减少中间结果的内存读写
- 提高 GPU 利用率

#### 5.4.3 Batched MatMul

使用 `baddbmm` 和 `bmm` 充分利用 GPU 并行能力:
- 将批次和头数融合为 batch 维度
- GPU 可以并行处理多个矩阵乘法

#### 5.4.4 GQA 支持

通过 `repeat_interleave` 支持 Grouped Query Attention:
- MHA: `np = ng`(查询头数 = KV 头数)
- GQA: `np > ng`(查询头数 > KV 头数)
- MQA: `ng = 1`(只有 1 个 KV 头)

详见文档 31 (GQA详解)。

---

## 6. 数值稳定性考虑

### 6.1 Softmax 的数值稳定实现

标准 Softmax 实现:
$$\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j} \exp(x_j)}$$

**问题**: 当 $x_i$ 很大时,$\exp(x_i)$ 会溢出。

**数值稳定版本**:
$$\text{softmax}(x_i) = \frac{\exp(x_i - \max(x))}{\sum_{j} \exp(x_j - \max(x))}$$

**证明等价性**:
$$\frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)} = \frac{\exp(x_i) \exp(-m)}{\exp(-m) \sum_j \exp(x_j)} = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$

Megatron-LM 的 `FusedScaleMaskSoftmax` 自动处理了这个数值稳定性问题。

### 6.2 混合精度训练

Megatron-LM 支持 FP16/BF16/FP8 混合精度训练。注意力计算的精度策略:

| 操作 | FP16 | BF16 | FP8 |
|------|------|------|-----|
| $QK^{\top}$ | FP16 | BF16 | FP8 |
| Softmax | FP32 | FP32 | FP32 |
| $AV$ | FP16 | BF16 | FP8 |

**Softmax 必须用 FP32** 的原因:
- Softmax 涉及 exp 和除法,对数值精度敏感
- FP16 的动态范围不足,容易溢出或下溢

详见文档 93 (混合精度训练详解)。

---

## 7. 总结

### 7.1 关键要点

1. **数学本质**: Scaled Dot-Product Attention 本质是基于点积相似度的加权平均
2. **缩放因子**: $1/\sqrt{d_k}$ 是防止 Softmax 饱和和梯度消失的关键
3. **复杂度瓶颈**: $O(S^2)$ 的时间和空间复杂度限制了序列长度
4. **工程优化**: Megatron-LM 通过融合算子、内存预分配、batched matmul 等技术优化性能

### 7.2 与其他文档的关系

- **文档 22 (自注意力机制)**: 本文档是文档 22 中自注意力机制的具体实现
- **文档 24 (多头注意力)**: Multi-Head Attention 是对本文档的并行扩展
- **文档 31 (GQA详解)**: 介绍如何通过减少 KV 头数优化推理
- **文档 34-36 (Flash Attention)**: 介绍如何优化本文档的 $O(S^2)$ 复杂度

### 7.3 进一步学习

- **Flash Attention**: 如何将空间复杂度从 $O(S^2)$ 降到 $O(S)$
- **Grouped Query Attention**: 如何平衡推理速度和模型质量
- **位置编码**: RoPE如何在注意力计算中编码位置信息

---

## 参考文献

1. Vaswani et al. (2017). "Attention is All You Need". NeurIPS.
2. Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness". NeurIPS.
3. Megatron-LM Documentation: `megatron/core/transformer/dot_product_attention.py`
4. PyTorch Documentation: `torch.nn.functional.scaled_dot_product_attention`

---

**文档版本**: v1.0
**最后更新**: 2025-12-27
**文档状态**: ✅ 已完成
