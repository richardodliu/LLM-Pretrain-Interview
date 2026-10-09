# 23. 缩放点积注意力(Scaled Dot-Product Attention)

> **文档编号**: 23
> **所属部分**: 第三部分 - Transformer基础架构 (21-30)
> **对应原文档**: 03-attention-mechanisms.md Section 5
> **代码位置**: `megatron/core/transformer/dot_product_attention.py`
> **代码锚点**: 基于 Megatron-LM 当前仓库的缩放点积注意力实现，并结合 Transformer 论文背景说明

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

**核心文件**: `megatron/core/transformer/dot_product_attention.py`

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

## 8. 消融研究

### 8.1 缩放因子消融

移除 $1/\sqrt{d_k}$ 会导致 logits 方差随 head dimension 增大而上升，Softmax 更容易饱和，训练早期梯度更不稳定。

### 8.2 Mask与Dropout消融

- 去掉 causal mask 会破坏自回归训练目标。
- attention dropout 过大可能削弱长程依赖，过小则降低正则化效果。

## 9. 超参数分析

### 9.1 Head Dimension

常见 $d_k$ 为64或128。较大 $d_k$ 提高单头表达能力，但增加 $QK^\top$ 与 $AV$ 的计算量。

### 9.2 Attention Dropout

预训练常用较小 dropout 或关闭 dropout，具体取决于数据规模、模型规模和过拟合风险。

## 10. 深入探讨

### 10.1 与FlashAttention的关系

FlashAttention不改变 scaled dot-product attention 的数学定义，而是改变计算顺序，避免显式物化 $S \times S$ attention matrix。

### 10.2 与GQA/MQA的关系

GQA/MQA减少 KV 头数，但每个查询头内部仍执行同样的缩放点积注意力。

## 11. 总结与最佳实践

- 始终保留 $1/\sqrt{d_k}$ 缩放。
- Softmax应使用数值稳定实现。
- 长序列场景优先使用FlashAttention或分块注意力实现。
- 推理场景结合KV Cache、GQA或MLA降低内存带宽压力。

## 12. 参考文献

1. Vaswani et al. (2017). "Attention is All You Need". NeurIPS.
2. Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness". NeurIPS.
3. Megatron-LM Documentation: `megatron/core/transformer/dot_product_attention.py`
4. PyTorch Documentation: `torch.nn.functional.scaled_dot_product_attention`

---

## 附录 A：形状检查清单

Scaled Dot-Product Attention 的公式很短，但实现中最常见的错误来自张量形状和布局。建议按以下顺序检查：

| 项目 | 期望 | 常见错误 |
|------|------|----------|
| Query | `[S_q, B, H_q, D]` 或等价布局 | 把 batch 和 head 维度交换 |
| Key | `[S_k, B, H_k, D]` | GQA下 `H_k` 未扩展或未被kernel识别 |
| Value | `[S_k, B, H_k, D_v]` | `D_v` 和 `D` 混用 |
| Mask | 可广播到 `[B, H_q, S_q, S_k]` | causal mask 方向反了 |
| Softmax dim | `S_k` 维 | 对 head 维或 feature 维 softmax |
| Dropout | 只作用于 attention prob | 作用到 logits 后改变分布 |
| 输出 | `[S_q, B, H_q, D_v]` | 合并 head 前后顺序不一致 |

一个可靠的检查方法是写出每一步的维度：

```text
Q: [S_q, B, H_q, D]
K: [S_k, B, H_k, D]
V: [S_k, B, H_k, D_v]
scores = Q @ K^T: [B, H_q, S_q, S_k]
probs = softmax(scores + mask): [B, H_q, S_q, S_k]
context = probs @ V: [S_q, B, H_q, D_v]
```

在 GQA/MQA 中，`H_k` 可以小于 `H_q`。此时要么在通用路径中显式扩展 KV，要么由融合 kernel 在逻辑上处理组映射。

## 附录 B：数值稳定实现细节

注意力 logits 的稳定性主要受三个因素影响：

1. 点积维度 $D$。
2. 输入激活的范数。
3. mask 和 dtype 的处理方式。

稳定 softmax 通常使用：

$$
\text{softmax}(z)_i =
\frac{\exp(z_i - \max_j z_j)}
{\sum_k \exp(z_k - \max_j z_j)}
$$

实现审查点：

| 审查项 | 正确做法 | 风险 |
|--------|----------|------|
| 缩放 | 在 softmax 前乘 $1/\sqrt{D}$ | logits 方差随 D 增大 |
| mask 值 | 使用 dtype 安全的极小值 | FP16 中 `-inf` 传播异常 |
| max-subtraction | softmax kernel 内执行 | exp overflow |
| dropout | softmax 后执行 | 改变 mask 语义 |
| accumulation | 必要时使用 FP32 累积 | BF16/FP16 误差放大 |

如果出现 NaN，优先检查：

- attention logits 是否已经有 NaN/Inf。
- mask 后是否整行都被屏蔽。
- dropout 是否在训练/推理模式切换时一致。
- RoPE 或 QK LayerNorm 是否改变了 logits 范围。
- FP8/FP16 scale 是否在注意力层附近异常。

## 附录 C：复杂度口径

对单层单个 micro-batch，忽略常数项：

| 阶段 | 时间复杂度 | 显存/临时内存 | 说明 |
|------|------------|---------------|------|
| QK 点积 | $O(B H S_q S_k D)$ | logits | prefill 最重 |
| Softmax | $O(B H S_q S_k)$ | probs 或在线统计 | FlashAttention可避免完整物化 |
| AV 点积 | $O(B H S_q S_k D_v)$ | context | decode时受KV读带宽影响 |
| Mask | $O(B H S_q S_k)$ | 可广播 | 长上下文下不可忽略 |
| Dropout | $O(B H S_q S_k)$ | dropout mask | 推理关闭 |

Prefill 与 decode 的差异：

| 阶段 | $S_q$ | $S_k$ | 主瓶颈 |
|------|-------|-------|--------|
| Prefill | 输入长度 | 输入长度 | 计算和临时显存 |
| Decode | 1 或很小 | 历史长度 | KV Cache 带宽 |

因此，同一个注意力公式在训练、prefill、decode 三个阶段的优化方向并不一样。

## 附录 D：Megatron 代码审查点

在 Megatron-LM 中审查 scaled dot-product attention 时，建议从以下路径进入：

| 目标 | 文件 |
|------|------|
| 标准 attention 计算 | `megatron/core/transformer/dot_product_attention.py` |
| SelfAttention 中的 QKV 拆分 | `megatron/core/transformer/attention.py` |
| RoPE 应用时机 | `megatron/core/transformer/attention.py` |
| Transformer 配置字段 | `megatron/core/transformer/transformer_config.py` |
| FlashAttention/TE 后端接入 | `megatron/core/extensions/transformer_engine.py` |

审查问题：

1. `num_attention_heads` 是否能被 tensor parallel size 整除。
2. `kv_channels` 是否与 Q/K 最后一维一致。
3. `num_query_groups` 是否改变 K/V 头数。
4. causal mask 是否与 sequence offset 一致。
5. inference context 是否在 cache 写入前应用 RoPE。
6. attention backend 是否支持当前 mask、GQA、window 或 FP8 配置。

## 附录 E：最小单元测试设计

Scaled Dot-Product Attention 的单元测试应覆盖数学正确性、mask、dtype 和并行语义：

| 测试 | 输入 | 通过标准 |
|------|------|----------|
| 小矩阵手算 | `S=2, H=1, D=2` | 与手算 softmax 一致 |
| causal mask | `S_q=S_k=4` | 未来 token 概率为 0 |
| padding mask | batch 内不同长度 | padding 不影响有效 token |
| GQA | `H_q=8, H_k=2` | 输出 shape 正确 |
| dropout off | eval mode | 多次输出一致 |
| mixed precision | BF16/FP16 | 无 NaN，误差在容忍范围 |
| long sequence | 大 S | 不 OOM 或按预期使用 FlashAttention |

调试时可使用三个不变量：

- softmax 概率沿 key 维求和应接近 1。
- 完全 mask 的行必须有明确定义的处理策略。
- 在没有 dropout 时，同一输入应确定性输出。

## 附录 F：常见面试追问

**为什么要除以 $\sqrt{d_k}$？**

如果 Q 和 K 的各维独立且方差接近 1，点积方差约为 $d_k$。不缩放会让 logits 方差随 head dimension 增大，softmax 更容易饱和，梯度变小。

**为什么不是除以 $d_k$？**

除以 $d_k$ 会把方差压到 $1/d_k$，logits 过小，attention 分布过于平滑。$1/\sqrt{d_k}$ 对应方差归一化。

**mask 应该在缩放前还是缩放后加？**

通常先缩放 logits，再加 mask，再 softmax。只要 mask 的极小值足够安全，缩放和 mask 的相对顺序在未 mask 位置不改变；工程上更重要的是避免被 mask 位置在 softmax 中泄漏概率。

**FlashAttention 是否改变数学结果？**

它实现的是精确注意力的等价重排，主要改变 IO 和中间张量物化方式。数值上可能有浮点舍入差异，但不是近似注意力。

**GQA 是否改变 scaled dot-product attention？**

每个 query head 内部仍是同一个公式；改变的是 K/V 头的来源和共享关系。

## 附录 G：实验记录模板

注意力层实验不要只记录 throughput。建议至少记录：

```text
model:
  hidden_size:
  num_attention_heads:
  num_query_groups:
  kv_channels:
sequence:
  seq_length:
  micro_batch_size:
backend:
  attention_backend:
  flash_attention:
  transformer_engine:
precision:
  bf16:
  fp16:
  fp8:
metrics:
  tokens_per_second:
  peak_memory:
  attention_time:
  loss:
  grad_norm:
```

对比实验要固定：

- 相同模型结构。
- 相同 tokenizer 和数据 batch。
- 相同 dropout/eval 模式。
- 相同并行配置。
- 相同 warmup 步数。

## 附录 H：故障排查表

| 症状 | 可能原因 | 检查方法 |
|------|----------|----------|
| attention 输出全零 | mask 全部生效 | 打印 mask 有效比例 |
| softmax 后 NaN | logits overflow 或整行 mask | 检查 max/min logits |
| 长序列 OOM | 显式物化 $S^2$ 矩阵 | 启用 FlashAttention |
| decode 吞吐低 | KV Cache 带宽瓶颈 | profile KV 读写 |
| GQA 结果 shape 错 | KV 头扩展错误 | 检查 `num_query_groups` |
| 推理重复 token | cache position 错位 | 检查 sequence offset |
| 训练 loss 抖动 | logits 尺度异常 | 检查 Q/K norm |
| TP rank 不一致 | head 切分错误 | 比对各 rank shape |

## 附录 I：与其他优化的边界

| 技术 | 是否改变数学注意力 | 主要改变 |
|------|--------------------|----------|
| FlashAttention | 否 | IO 和分块计算 |
| GQA/MQA | 部分改变 K/V 参数化 | KV 头共享 |
| MLA | 改变 KV 参数化 | 低秩 cache |
| Sliding Window | 是 | 限制可见 token |
| Sparse Attention | 是 | 稀疏连接模式 |
| KV Quantization | 近似 | cache 存储精度 |
| RoPE | 不改变注意力公式 | 改变 Q/K 表示 |

理解边界很重要：如果 loss 或质量变化，先判断改动是“精确实现优化”还是“模型结构近似”。前者通常应接近等价，后者需要重新训练或消融验证。

## 附录 J：上线前检查

1. `attention_mask` 与训练/推理任务一致。
2. `softmax_scale` 等于 $1/\sqrt{d_k}$ 或有明确理由。
3. Q/K/V dtype 与 backend 支持矩阵一致。
4. GQA 配置下 KV 头数和 TP 切分一致。
5. RoPE 在 cache 写入前后语义一致。
6. 长序列使用不会显式保存完整 attention probs。
7. dropout 在 eval/inference 中关闭。
8. 单元测试覆盖 causal mask、padding mask、GQA 和混合精度。
9. 性能测试区分 prefill 与 decode。
10. 文档中所有 benchmark 数值都标明硬件、backend 和 batch。

## 附录 K：边界条件与反例

Scaled Dot-Product Attention 的公式默认每个 query 至少能看到一个 key。以下边界条件必须显式处理：

| 边界条件 | 风险 | 推荐处理 |
|----------|------|----------|
| `S_k=0` | softmax 空输入 | 上游禁止或返回空输出 |
| 整行被 mask | softmax 产生 NaN | 使用安全 masked softmax |
| `d_k=0` | 缩放因子无定义 | 配置校验禁止 |
| 极大 `d_k` | logits 方差大 | 保留缩放并监控 Q/K norm |
| 极长 `S_k` | 临时矩阵 OOM | FlashAttention 或分块 |
| FP16 mask | 极小值溢出 | 使用 dtype 安全 mask 常量 |
| dropout=1 | attention 全丢弃 | 配置校验禁止 |
| query/key dtype不同 | kernel fallback | 显式 cast 或禁止 |

一个常见反例是“只要 softmax 稳定，就不需要缩放”。稳定 softmax 只避免指数溢出，不能避免分布过尖导致梯度集中。缩放和稳定 softmax 解决的是两个不同问题。

另一个反例是“FlashAttention 后就没有 $S^2$ 成本”。FlashAttention 不保存完整 attention matrix，但精确注意力仍然需要读取和组合 $S_q \times S_k$ 的相互作用；它主要降低 IO 和峰值中间显存。

## 附录 L：阅读源码时的变量对照

不同实现中同一概念可能有不同命名：

| 数学符号 | 常见变量名 | 说明 |
|----------|------------|------|
| $Q$ | `query`, `query_layer` | query heads |
| $K$ | `key`, `key_layer` | key heads |
| $V$ | `value`, `value_layer` | value heads |
| $d_k$ | `hidden_size_per_attention_head`, `kv_channels` | 每头维度 |
| $n_h$ | `num_attention_heads` | 总 query head 数 |
| $n_g$ | `num_query_groups` | GQA 的 KV group 数 |
| $S_q$ | `query_seq_len` | query长度 |
| $S_k$ | `key_seq_len` | key/cache长度 |
| mask | `attention_mask` | causal/padding mask |

源码阅读建议：

1. 先确认张量布局是 `sbhd`、`bshd` 还是 `thd`。
2. 再确认 head 维在 tensor parallel 后是全局还是 per-rank。
3. 最后确认 backend 是否会重新排列布局。

## 附录 M：候选人答题评分点

| 题目 | 合格答案 | 高质量答案 |
|------|----------|------------|
| 为什么缩放 | 点积方差随维度增大 | 推导方差并解释softmax饱和 |
| mask位置 | softmax前加mask | 说明dtype安全和整行mask |
| FlashAttention | 节省显存 | 说明IO-aware和精确性 |
| GQA关系 | 共享KV | 区分cache、参数、kernel收益 |
| NaN排查 | 看logits/mask | 给出完整排查顺序 |
| decode瓶颈 | KV Cache | 区分prefill和decode |
| dropout语义 | 作用于prob | 说明训练/推理切换 |
| TP影响 | head切分 | 说明per-rank heads与groups |

这类题目的关键不是背公式，而是能把数学、张量形状、dtype、mask 和系统瓶颈连接起来。

## 附录 N：最小复现实验

为了验证一个 attention backend 是否正确，可以构造以下最小实验：

```text
batch_size = 1
num_heads = 2
query_length = 3
key_length = 3
head_dim = 4
dropout = 0
mask = causal
dtype = fp32
```

实验步骤：

1. 固定随机种子生成 Q/K/V。
2. 用朴素矩阵乘手写 attention。
3. 用目标 backend 计算 attention。
4. 比较 logits、prob 和 context。
5. 再切换到 BF16/FP16，记录误差变化。
6. 再加入 padding mask，验证 masked token 概率为 0。
7. 最后把 `key_length` 增大，观察显存和时间趋势。

建议记录：

```text
max_abs_score_diff:
max_abs_prob_diff:
max_abs_context_diff:
prob_row_sum_min:
prob_row_sum_max:
masked_prob_max:
backend:
dtype:
```

如果 FP32 朴素实现和 backend 不一致，不要先怀疑浮点误差，应优先检查 shape、mask、scale 和布局。

## 附录 O：最终自检

提交前自问：

1. 是否区分了数学公式和 kernel 实现。
2. 是否区分了 prefill 与 decode。
3. 是否说明了 mask 的广播维度。
4. 是否说明了缩放和稳定 softmax 的不同作用。
5. 是否避免把 FlashAttention 写成近似算法。
6. 是否避免给出无硬件背景的吞吐数字。
7. 是否写清 GQA/MLA 只是改变 K/V 表示或缓存，不改变 softmax 的基本语义。
8. 是否把 NaN 排查顺序写成可执行步骤。

---

**文档版本**: v1.0
**最后更新**: 2026-05-10
**文档状态**: ✅ 已完成
