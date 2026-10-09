# 24. 多头注意力机制(Multi-Head Attention)

> **文档编号**: 24
> **所属部分**: 第三部分 - Transformer基础架构 (21-30)
> **对应原文档**: 03-attention-mechanisms.md Section 6
> **代码位置**: `megatron/core/transformer/attention.py`
> **代码锚点**: 基于 Megatron-LM 当前仓库的注意力实现，并结合多头注意力论文背景说明

---

## 1. 引言

Multi-Head Attention (MHA) 是 Transformer 架构的核心创新之一。它通过并行运行多个注意力"头",使模型能够在不同的表示子空间中学习不同类型的依赖关系。本文档深入讲解 MHA 的数学原理、为什么需要多头、如何高效实现以及 Megatron-LM 中的工程优化。

---

## 2. 数学定义

### 2.1 基本公式

**定义 2.1**: Multi-Head Attention (MHA)

给定输入 $X \in \mathbb{R}^{S \times H}$,Multi-Head Attention 定义为:

$$
\begin{aligned}
\text{MultiHead}(X) &= \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_{n_h}) W^O \\
\text{where } \text{head}_i &= \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)
\end{aligned}
$$

其中:
- $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{H \times d_k}$: 第 $i$ 个头的投影矩阵
- $W^O \in \mathbb{R}^{n_h \cdot d_k \times H}$: 输出投影矩阵
- $d_k = H / n_h$: 每个头的维度
- $n_h$: 头的数量(通常为 8, 16, 32)

### 2.2 完整展开

将上述定义完全展开:

$$
\begin{aligned}
Q_i &= XW_i^Q \in \mathbb{R}^{S \times d_k} \\
K_i &= XW_i^K \in \mathbb{R}^{S \times d_k} \\
V_i &= XW_i^V \in \mathbb{R}^{S \times d_k} \\
\text{head}_i &= \text{softmax}\left(\frac{Q_i K_i^{\top}}{\sqrt{d_k}}\right) V_i \in \mathbb{R}^{S \times d_k} \\
\text{Output} &= [\text{head}_1, \text{head}_2, \ldots, \text{head}_{n_h}] W^O \in \mathbb{R}^{S \times H}
\end{aligned}
$$

**矩阵维度追踪**:
- 输入: $X \in \mathbb{R}^{S \times H}$
- 每个头的 QKV: $\mathbb{R}^{S \times d_k}$
- 每个头的输出: $\mathbb{R}^{S \times d_k}$
- 拼接后: $\mathbb{R}^{S \times (n_h \cdot d_k)} = \mathbb{R}^{S \times H}$
- 最终输出: $\mathbb{R}^{S \times H}$

输入和输出维度相同,这使得 Transformer 层可以堆叠。

### 2.3 参数量分析

| 组件 | 形状 | 参数量 |
|------|------|--------|
| $W_i^Q$ (所有 $n_h$ 个头) | $n_h \times (H \times d_k)$ | $H \cdot n_h \cdot d_k = H^2$ |
| $W_i^K$ (所有 $n_h$ 个头) | $n_h \times (H \times d_k)$ | $H \cdot n_h \cdot d_k = H^2$ |
| $W_i^V$ (所有 $n_h$ 个头) | $n_h \times (H \times d_k)$ | $H \cdot n_h \cdot d_k = H^2$ |
| $W^O$ | $H \times H$ | $H^2$ |
| **总计** | - | **$4H^2$** |

**关键观察**: MHA 的参数量 $4H^2$ 与单头注意力相同,但表达能力更强。

---

## 3. 为什么需要多头?

### 3.1 直觉理解

**单头注意力的局限**:
- 单头只能学习一种类型的依赖关系
- 例如,只能关注语法关系或语义关系,无法同时捕获多种模式

**多头注意力的优势**:
不同的头可以在不同的表示子空间中学习不同类型的依赖关系:

| 头编号 | 学习的关系类型 | 示例 |
|--------|----------------|------|
| Head 1 | 语法关系 | 主语-谓语-宾语 |
| Head 2 | 语义关系 | 同义词、反义词 |
| Head 3 | 共指关系 | 代词指代 |
| Head 4 | 位置关系 | 相邻词、固定搭配 |
| Head 5 | 长程依赖 | 从句引导词与主句的关系 |
| Head 6 | 实体关系 | 命名实体之间的关系 |

### 3.2 表达能力的数学分析

**定理 3.1**: 多头注意力的表达能力

对于固定的参数量,多头注意力的表达能力强于单头注意力。

**证明思路**:

1. **参数量相同**:
   - 单头注意力: $4H^2$ 参数
   - $n_h$ 头注意力: $4H^2$ 参数

2. **表达能力不同**:
   - 单头: 学习 $H \times H$ 的注意力矩阵,秩最多为 $H$
   - 多头: 学习 $n_h$ 个 $d_k \times d_k$ 的注意力矩阵,每个秩最多为 $d_k$
   - 拼接后的输出空间维度为 $n_h \cdot d_k = H$,但是由 $n_h$ 个独立子空间组成

3. **子空间独立性**:
   - 每个头在其子空间中独立学习
   - 不同头可以捕获不同频率/模式的信息
   - 类似于傅里叶变换的不同频率分量

**实验证据** (Vaswani et al., 2017):

| 配置 | 参数量 | BLEU (WMT En-De) |
|------|--------|------------------|
| 1 head × 512-dim | $4 \times 512^2$ | 25.3 |
| 8 heads × 64-dim | $4 \times 512^2$ | 26.2 (+0.9) |

在该机器翻译实验中，相同参数量下多头配置优于单头配置；这个结果说明多头分解有效，但具体增益不能直接外推到所有模型规模和任务。

### 3.3 子空间分解的几何直觉

将 $H$ 维空间分解为 $n_h$ 个 $d_k$ 维子空间:
$$\mathbb{R}^H = \mathbb{R}^{d_k} \oplus \mathbb{R}^{d_k} \oplus \cdots \oplus \mathbb{R}^{d_k}$$

每个头在其子空间中计算注意力:
- Head $i$ 关注子空间 $\mathbb{R}^{d_k}_i$ 中的信息
- 不同子空间可以编码不同类型的特征
- 最后通过 $W^O$ 融合所有子空间的信息

这类似于信号处理中的**滤波器组**(filter bank):
- 每个头是一个滤波器,提取特定频率/模式的信息
- 所有头的输出组合成完整的信号表示

---

## 4. 多头注意力的实现优化

### 4.1 朴素实现的问题

**朴素实现**(循环计算每个头):

```python
heads = []
for i in range(num_heads):
    Q_i = X @ W_Q[i]  # [S, H] @ [H, d_k] = [S, d_k]
    K_i = X @ W_K[i]
    V_i = X @ W_V[i]
    head_i = attention(Q_i, K_i, V_i)  # [S, d_k]
    heads.append(head_i)
output = concat(heads) @ W_O  # [S, n_h*d_k] @ [n_h*d_k, H] = [S, H]
```

**问题**:
1. **循环开销**: $n_h$ 次循环,无法充分利用 GPU 并行
2. **Kernel 启动开销**: $n_h$ 次矩阵乘法需要 $n_h$ 次 kernel 启动
3. **内存碎片**: 多次分配中间张量

### 4.2 优化策略: QKV 融合

**关键思想**: 将所有头的 QKV 投影合并为一次大矩阵乘法。

**优化后的实现**:

```python
# 1. 融合 QKV 投影矩阵
W_QKV = concat([W_Q, W_K, W_V], dim=1)  # [H, 3*n_h*d_k]
QKV = X @ W_QKV  # [S, H] @ [H, 3*H] = [S, 3*H]

# 2. 分离 Q, K, V
Q, K, V = split(QKV, dim=-1)  # 3 × [S, n_h*d_k]

# 3. 重塑为多头格式
Q = Q.view(S, n_h, d_k)  # [S, n_h, d_k]
K = K.view(S, n_h, d_k)
V = V.view(S, n_h, d_k)

# 4. 批量注意力计算 (将 n_h 视为批次维度)
attn_output = batched_attention(Q, K, V)  # [S, n_h, d_k]

# 5. 拼接并投影
attn_output = attn_output.view(S, n_h * d_k)  # [S, H]
output = attn_output @ W_O  # [S, H] @ [H, H] = [S, H]
```

### 4.3 性能对比

| 实现方式 | Kernel 启动次数 | GPU 利用率 | 相对速度 |
|----------|-----------------|------------|----------|
| 朴素循环 | $3n_h + 1$ (QKV投影 + Output) | 低 | 1.0x |
| QKV 融合 | $2$ (融合QKV + Output) | 高 | ~$n_h$ x |

**加速原因**:
1. **减少 Kernel 启动**: 从 $3n_h + 1$ 降到 $2$
2. **充分利用 GPU 并行**: 批量矩阵乘法同时计算所有头
3. **更好的内存访问模式**: 连续内存访问,减少 cache miss

---

## 5. Megatron-LM 中的实现

### 5.1 QKV 融合投影

**文件路径**: `megatron/core/transformer/attention.py`

```python
# 融合的 QKV 线性层
self.linear_qkv = build_module(
    submodules.linear_qkv,
    self.config.hidden_size,  # 输入维度 H
    self.query_projection_size + 2 * self.kv_projection_size,  # 输出维度
    config=self.config,
    init_method=self.config.init_method,
    gather_output=False,  # 输出保持分片（张量并行）
    bias=self.config.add_bias_linear or self.config.add_qkv_bias,
    skip_bias_add=False,
    is_expert=False,
    tp_comm_buffer_name='qkv',
    tp_group=self.pg_collection.tp,
)
```

### 5.2 维度计算

**MHA 情况** ($n_g = n_h$):

```python
query_projection_size = kv_channels * num_attention_heads = d_k * n_h = H
kv_projection_size = kv_channels * num_query_groups = d_k * n_h = H
output_size = H + 2*H = 3H
```

**GQA 情况** ($n_g < n_h$):

```python
query_projection_size = d_k * n_h = H
kv_projection_size = d_k * n_g < H
output_size = H + 2*(d_k * n_g) < 3H
```

GQA 通过减少 KV 头数降低参数量和计算量,详见文档 31。

### 5.3 QKV 分离与重塑

**文件路径**: `megatron/core/transformer/attention.py`

```python
def get_query_key_value_tensors(self, hidden_states, key_value_states=None, split_qkv=True):
    """从 hidden_states 计算 Q, K, V

    Args:
        hidden_states: [sq, b, h] - 输入隐藏状态
        split_qkv: 是否分离 QKV (False 时返回融合的 mixed_qkv)

    Returns:
        query: [sq, b, np, hn]
        key:   [sq, b, ng, hn]
        value: [sq, b, ng, hn]
    """
    # 1. 融合 QKV 投影
    # hidden_states: [sq, b, h]
    # mixed_qkv: [sq, b, ng * (np/ng + 2) * hn]
    mixed_qkv, _ = self.linear_qkv(hidden_states)

    # 2. GQA 特殊处理: 当 num_query_groups < world_size 时
    # 需要先 All-Gather 再索引
    if self.config.num_query_groups < self.world_size:
        # 假设 num_query_groups=2, TP=4, 则每个 TP rank 负责 0.5 个 KV 组
        # 通过 All-Gather 得到完整的 2 个 KV 组,然后索引到自己负责的部分
        mixed_qkv = all_gather_last_dim_from_tensor_parallel_region(mixed_qkv)
        idx = get_tensor_model_parallel_rank() // (
            self.world_size // self.config.num_query_groups
        )
        size = mixed_qkv.size()[-1] // self.config.num_query_groups
        mixed_qkv = mixed_qkv[:, :, idx * size : (idx + 1) * size]

    # 3. 重塑为 [sq, b, ng, (np/ng + 2) * hn]
    new_tensor_shape = mixed_qkv.size()[:-1] + (
        self.num_query_groups_per_partition,  # ng
        (
            (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
            * self.hidden_size_per_attention_head
        ),  # (np/ng + 2) * hn
    )
    mixed_qkv = mixed_qkv.view(*new_tensor_shape)

    # 4. 分离 Q, K, V
    split_arg_list = [
        (self.num_attention_heads_per_partition // self.num_query_groups_per_partition)
        * self.hidden_size_per_attention_head,  # Q 的维度: (np/ng) * hn
        self.hidden_size_per_attention_head,     # K 的维度: hn
        self.hidden_size_per_attention_head,     # V 的维度: hn
    ]

    if not split_qkv:
        return mixed_qkv, split_arg_list

    # 使用 Transformer Engine 的优化分离函数（如果可用）
    if SplitAlongDim is not None:
        (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
    else:
        (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

    # 5. 重塑 query: [sq, b, ng, (np/ng)*hn] -> [sq, b, np, hn]
    query = query.reshape(
        query.size(0), query.size(1), -1, self.hidden_size_per_attention_head
    )

    # 6. GQA 后续处理: 从 (np/ng) 个查询头中选择属于本 TP rank 的
    if self.config.num_query_groups < self.world_size:
        idx = get_tensor_model_parallel_rank() % (
            self.world_size // self.config.num_query_groups
        )
        size = self.num_attention_heads_per_partition // (
            self.world_size // self.config.num_query_groups
        )
        query = query[:, :, idx * size : (idx + 1) * size, :]

    # 7. 可选的 QK LayerNorm
    if self.q_layernorm is not None:
        query = self.q_layernorm(query)
    if self.k_layernorm is not None:
        key = self.k_layernorm(key)

    return query, key, value
```

### 5.4 关键工程技巧

#### 5.4.1 融合投影

一次矩阵乘法计算所有 QKV,减少 kernel 启动开销:
- 单次 `linear_qkv` 调用替代 3 次独立调用
- Kernel 启动次数: $3 \to 1$
- 通常能提升 GPU 利用率，具体幅度取决于矩阵尺寸、并行度和后端 kernel

#### 5.4.2 GQA 支持

通过调整输出维度支持 Grouped Query Attention:
- MHA: `output_size = 3H`
- GQA: `output_size = H + 2(d_k * n_g)`
- 参数量降低: $4H^2 \to H^2(1 + n_h/n_g + 2)$

#### 5.4.3 张量并行集成

每个 TP rank 只计算部分头:
- `num_attention_heads_per_partition = num_attention_heads / TP`
- 通过索引选择属于本 rank 的头
- 详见文档 58 (注意力层的张量并行)

#### 5.4.4 可选 QK LayerNorm

DeepSeek-V2 引入的技术,提高训练稳定性:
```python
if self.q_layernorm is not None:
    query = self.q_layernorm(query)
if self.k_layernorm is not None:
    key = self.k_layernorm(key)
```

在 Q 和 K 上应用 LayerNorm,缓解注意力分数的数值不稳定。

---

## 6. 参数量与计算量对比

### 6.1 不同配置的参数量

| 配置 | 头数 $n_h$ | 每头维度 $d_k$ | 参数量 |
|------|-----------|---------------|--------|
| 小模型 | 12 | 64 | $4 \times 768^2 \approx 2.4M$ |
| 中模型 | 16 | 64 | $4 \times 1024^2 \approx 4.2M$ |
| 大模型 | 32 | 128 | $4 \times 4096^2 \approx 67M$ |
| LLaMA-7B | 32 | 128 | $4 \times 4096^2 \approx 67M$ |
| LLaMA-70B | 64 | 128 | $4 \times 8192^2 \approx 268M$ |

**关键观察**: 在很多 Transformer 配置中，注意力层的参数量不是最大头寸；端到端计算占比则会随序列长度、FFN扩展倍数、GQA配置和FlashAttention实现显著变化。

### 6.2 计算量分析

**单个 Transformer 层的 FLOPs**:

| 组件 | FLOPs |
|------|-------|
| QKV 投影 | $6BSH^2$ |
| 注意力计算 | $4BS^2H$ |
| Output 投影 | $2BSH^2$ |
| MLP | $16BSH^2$ (hidden_dim = 4H) |
| **总计** | $24BSH^2 + 4BS^2H$ |

**比例分析**:
- 当 $S \ll H$ 时,MLP 主导 (24 vs 8)
- 当 $S \approx H$ 时,注意力和 MLP 相当
- 当 $S \gg H$ 时,注意力主导 (4S vs 24H)

这解释了为什么长序列训练的瓶颈在注意力。

---

## 7. 多头注意力的可视化与分析

### 7.1 注意力模式的多样性

不同的头学习到不同的注意力模式,例如(来自 BERT 分析):

**Head 1: 局部注意力**
```
The cat sat on the mat
 ↓   ↓   ↓  ↓   ↓  ↓
The cat sat on the mat
```
主要关注相邻词。

**Head 2: 语法依赖**
```
The cat sat on the mat
    ↓   ↑        ↑
    sat <------ the
```
主谓宾关系。

**Head 3: 全局注意力**
```
The cat sat on the mat
 ↓   ↓   ↓  ↓   ↓  ↓
[CLS]
```
所有词都关注 [CLS] token。

### 7.2 头的冗余性

研究发现(Michel et al., 2019):
- 不是所有头都同等重要
- 部分任务和层上可以剪枝若干头而不立即造成明显退化
- GQA/MQA 借鉴了“头存在冗余”的观察，但是否能保持质量仍取决于继续训练和目标任务

---

## 8. 总结

### 8.1 关键要点

1. **数学本质**: MHA 通过多个并行的注意力头捕获不同子空间的信息
2. **表达能力**: 相同参数量下,多头比单头表达能力更强
3. **实现优化**: QKV 融合和批量计算是高效实现的关键
4. **参数-性能权衡**: 可以通过 GQA/MQA 降低参数量和推理成本

### 8.2 与其他文档的关系

- **文档 23 (Scaled Dot-Product Attention)**: MHA 的基础单元
- **文档 31 (GQA详解)**: MHA 的参数高效变体
- **文档 32 (MQA详解)**: GQA 的极端情况
- **文档 33 (MLA详解)**: MHA 的另一种压缩方法
- **文档 58 (注意力层的张量并行)**: MHA 在分布式训练中的实现

### 8.3 进一步学习

- **GQA/MQA**: 如何通过减少 KV 头数加速推理
- **MLA**: 如何通过低秩压缩进一步优化 KV Cache
- **张量并行**: 如何将 MHA 分布到多个 GPU

---

## 9. 超参数分析

### 9.1 头数与Head Dimension

`num_attention_heads` 需要与 `hidden_size` 和 tensor parallel size 协同选择。头数过少会限制子空间多样性，头数过多会降低每头维度并增加调度开销。

### 9.2 Query Group数量

当 `num_query_groups < num_attention_heads` 时，模型从MHA过渡到GQA/MQA，推理KV Cache显著降低，但需要验证质量损失。

## 10. 深入探讨

### 10.1 为什么QKV通常融合

QKV融合把三次线性投影合并为一次大GEMM，提升GPU利用率，并减少kernel launch和内存读写。

### 10.2 多头冗余与剪枝

多头注意力中的头存在冗余，这解释了GQA/MQA和head pruning的可行性，但剪枝或共享KV必须通过目标任务验证。

## 11. 总结与最佳实践

- MHA是Transformer注意力层的标准并行扩展。
- QKV融合和输出投影是高性能实现关键。
- 推理优化优先考虑GQA/MQA，训练并行优先检查TP head整除关系。

## 12. 参考文献

1. Vaswani et al. (2017). "Attention is All You Need". NeurIPS.
2. Michel et al. (2019). "Are Sixteen Heads Really Better than One?". NeurIPS.
3. Ainslie et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints". EMNLP.
4. Megatron-LM Documentation: `megatron/core/transformer/attention.py`

---

## 附录 A：多头配置推导

设计 MHA 时，首先要让隐藏维度、头数、每头维度和 tensor parallel size 自洽：

| 符号 | 含义 | 约束 |
|------|------|------|
| $H$ | hidden size | 模型主宽度 |
| $n_h$ | attention heads | 通常要求 $H \mod n_h = 0$ |
| $d_h$ | head dimension | $d_h = H/n_h$ |
| $N_t$ | tensor parallel size | 通常要求 $n_h \mod N_t = 0$ |
| $n_g$ | query groups | GQA时要求与TP策略兼容 |

配置推导示例：

```text
hidden_size = 4096
num_attention_heads = 32
head_dim = 128
tensor_model_parallel_size = 4
heads_per_rank = 8
```

若切换到 GQA：

```text
num_query_groups = 8
kv_heads_per_rank = 2
q_heads_per_kv_group = 4
```

如果 `num_query_groups < tensor_model_parallel_size`，Megatron-LM 会走更复杂的 gather/slice 路径；这类配置应单独做 shape 和性能验证。

## 附录 B：QKV 融合的工程收益

MHA 的投影可以写成三次矩阵乘：

$$
Q=XW^Q,\quad K=XW^K,\quad V=XW^V
$$

工程上通常合并为：

$$
[Q,K,V]=XW^{QKV}
$$

收益来自：

| 来源 | 解释 |
|------|------|
| 更大的 GEMM | GPU 对大矩阵乘更容易达到高利用率 |
| 更少 kernel launch | 三次小操作变一次大操作 |
| 更少读输入 | $X$ 只从显存读一次 |
| 更易融合 bias | bias add 可跟投影输出融合 |
| 更易配合 TP | 每个 rank 拥有连续分片 |

风险和限制：

- GQA 下 Q 与 KV 输出维度不同，拆分逻辑必须精确。
- QK LayerNorm、LoRA 或 adapter 可能要求投影后插入额外操作。
- 量化训练/推理中，QKV 共享一个大权重可能带来 scale 分组问题。
- checkpoint 转换时必须知道 fused 权重的排列顺序。

## 附录 C：头数选择的经验规则

头数并非越多越好。常见权衡如下：

| 选择 | 收益 | 风险 |
|------|------|------|
| 更多头、更小 head dim | 子空间更多 | 单头容量下降，调度开销上升 |
| 更少头、更大 head dim | 单头容量强 | 注意力模式多样性下降 |
| head dim 64 | 经典 Transformer 配置 | 对大模型可能偏小 |
| head dim 128 | LLM 常见配置 | QK logits 范围更需缩放和稳定化 |
| head dim >128 | 表达能力强 | kernel支持、稳定性、显存都需验证 |

调参时建议固定 $H$，只改变 `num_attention_heads`，并记录：

```text
head_dim
attention_time
mlp_time
grad_norm
validation_loss
tokens_per_second
peak_memory
```

如果端到端速度几乎不变，说明瓶颈可能不在 attention head 切分，而在 MLP、通信或数据输入。

## 附录 D：Megatron 实现路径

| 目标 | 文件 | 审查点 |
|------|------|--------|
| SelfAttention 初始化 | `megatron/core/transformer/attention.py` | head 数、query group 数、投影维度 |
| QKV 分离 | `megatron/core/transformer/attention.py` | fused tensor 的 reshape/split |
| DotProductAttention | `megatron/core/transformer/dot_product_attention.py` | GQA repeat、mask、softmax |
| RoPE | `megatron/core/transformer/attention.py` | Q/K 应用位置 |
| Transformer 配置 | `megatron/core/transformer/transformer_config.py` | `num_attention_heads`, `num_query_groups` |

代码审查时要特别关注“维度命名是否保持一致”。同一个实现中可能同时出现：

- total heads。
- heads per tensor-parallel rank。
- query groups。
- query groups per rank。
- heads per query group。
- hidden size per attention head。

如果日志或注释没有说明是哪一种口径，就容易在排障时误判。

## 附录 E：参数量与 FLOPs 口径

标准 MHA 单层参数量：

| 模块 | 参数量 |
|------|--------|
| Q projection | $H^2$ |
| K projection | $H^2$ |
| V projection | $H^2$ |
| O projection | $H^2$ |
| 合计 | $4H^2$ |

注意力计算 FLOPs 还包含 $S^2$ 项：

| 项 | 复杂度 |
|----|--------|
| QKV projection | $O(B S H^2)$ |
| QK scores | $O(B H_n S^2 D)$ |
| Softmax | $O(B H_n S^2)$ |
| AV | $O(B H_n S^2 D)$ |
| Output projection | $O(B S H^2)$ |

短序列时，投影和 MLP 往往更重要；长序列时，$S^2$ 注意力项迅速变成瓶颈。FlashAttention降低的是 IO 和中间显存，不消除 $S^2$ 精确注意力计算量。

## 附录 F：MHA、GQA、MQA 对比

| 机制 | Q头数 | KV头数 | KV Cache | 质量风险 | 实现复杂度 |
|------|-------|--------|----------|----------|------------|
| MHA | $n_h$ | $n_h$ | 最高 | 最低 | 标准 |
| GQA | $n_h$ | $n_g$ | 中 | 中 | 中 |
| MQA | $n_h$ | 1 | 最低 | 最高 | 简单到中等 |

选择建议：

1. 训练新模型时，如果目标是长上下文或高并发推理，优先从 GQA 设计开始，而不是训练后再转换。
2. 从 MHA checkpoint 转 GQA/MQA 时，要继续训练或蒸馏恢复质量。
3. 如果模型主要用于短上下文低并发，MHA 的简单性仍有价值。
4. MoE 模型中，attention 不是唯一瓶颈；GQA收益需要和专家通信一起评估。

## 附录 G：调试可视化

多头注意力可视化不能只看漂亮的 heatmap。建议同时看：

| 图 | 目的 |
|----|------|
| head entropy | 判断头是否过度尖锐或退化 |
| average attention distance | 判断长程依赖 |
| per-head norm | 判断某些头是否异常 |
| Q/K norm | 判断 logits 尺度 |
| attention backend time | 判断性能瓶颈 |
| per-layer head similarity | 判断冗余 |

头冗余不等于可以直接剪枝。剪枝前需要确认：

- 对 validation loss 的影响。
- 对下游任务的影响。
- 对不同层的影响是否一致。
- 剪枝后是否继续训练。
- 推理 kernel 是否真的从剪枝中获益。

## 附录 H：常见故障

| 症状 | 可能原因 | 排查动作 |
|------|----------|----------|
| reshape 报错 | `hidden_size` 不能整除 heads | 检查 head_dim |
| TP 下 shape 不一致 | heads 不能整除 TP | 检查 heads per rank |
| GQA 下输出错 | query group split 错 | 检查 Q/K/V 切片 |
| 长序列 OOM | attention probs 物化 | 检查 backend |
| loss spike | QK logits 过大 | 检查 scaling、QK norm |
| 推理质量下降 | MHA->GQA 转换不足 | 继续训练或调大 group |
| checkpoint 加载失败 | fused QKV 排列不同 | 写转换脚本并验证 |
| 性能低于预期 | 小GEMM或通信瓶颈 | profile QKV GEMM/attention |

## 附录 I：面试题

**MHA 的参数量为什么和单头同阶？**

因为每个头的维度通常设为 $H/n_h$，所有头的 Q/K/V 投影参数相加仍是 $H^2$ 级别，再加输出投影为 $4H^2$。

**多头是否一定学到不同语义？**

不一定。多头提供了结构上的子空间分解能力，但不同头是否分工明确取决于数据、层数、训练目标和正则。

**为什么很多 LLM 使用 head dim 128？**

这是表达能力、kernel效率和数值稳定性之间的折中。head dim 过大时 logits 方差和计算成本增加；过小时单头容量可能不足。

**GQA 是不是只减少参数？**

不是。GQA 更关键的收益是减少推理阶段持久 KV Cache 和 decode 带宽。参数量下降只是副作用之一。

**QKV 融合会影响模型数学吗？**

不会。它只是把三次线性层合并成一次大线性层；只要权重排列和拆分正确，数学等价。

## 附录 J：上线前检查

1. `hidden_size / num_attention_heads` 是整数。
2. `num_attention_heads / tensor_model_parallel_size` 是整数，或有明确特殊处理。
3. GQA 下 `num_query_groups` 与 TP 配置兼容。
4. fused QKV 权重排列在训练、保存、加载、推理中一致。
5. RoPE 只作用于 Q/K，不误作用到 V。
6. attention mask 与任务一致。
7. FlashAttention/TE backend 支持当前 GQA、mask、dtype。
8. profile 区分 QKV projection、attention core、output projection。
9. 文档中的性能数字都标注硬件和 backend。
10. MHA/GQA/MQA 对比实验使用相同 token budget。

## 附录 K：Checkpoint 与权重排列

MHA 的 checkpoint 常把 QKV 融合权重保存在一个张量里。迁移或改结构时必须明确排列顺序：

| 布局 | 含义 | 风险 |
|------|------|------|
| `[Q, K, V]` 连续 | 最常见 fused QKV | 切错会立即破坏模型 |
| 按 head 交错 | 每个 head 的 QKV 相邻 | 转换脚本更复杂 |
| TP 分片后保存 | 每个 rank 只保存部分列/行 | 需要知道并行度 |
| GQA 布局 | Q 多头、KV 少头 | 不能按 MHA 直接读取 |

转换 checkpoint 前要做两个测试：

1. 随机小张量 round-trip：拆分、合并后逐元素一致。
2. 模型 logits 对齐：转换前后在同一输入上的 logits 差异符合预期。

如果是 MHA 到 GQA，logits 不会完全一致，因为 K/V 参数被合并；这时要记录转换策略和继续训练 token 数。

## 附录 L：层级差异

不同层的注意力头作用可能不同：

| 层位置 | 常见倾向 | GQA/剪枝风险 |
|--------|----------|--------------|
| 低层 | 局部和词法模式 | 过度共享会影响基础表示 |
| 中层 | 句法和实体关系 | 需要看任务 |
| 高层 | 语义和任务相关模式 | 对下游质量敏感 |
| 长上下文层 | 远距离检索 | 对KV共享更敏感 |

因此，不建议只看全模型平均 head entropy。更稳妥的做法是按层统计：

- attention entropy。
- average attention distance。
- per-head output norm。
- head similarity。
- ablation 后 validation loss。

## 附录 M：设计评审问题

在确定 MHA/GQA 配置前，评审应回答：

1. 模型主要面向训练吞吐还是推理并发。
2. 目标上下文长度是多少。
3. serving backend 是否支持 GQA。
4. TP size 是否会导致特殊 gather 路径。
5. checkpoint 是否需要与其他框架互转。
6. 是否有足够 token 进行结构变更后的继续训练。
7. 质量评估是否覆盖长上下文。
8. 是否保留更大 KV group 的回滚 checkpoint。

## 附录 N：最小配置回归

每次修改注意力头数、GQA 或 TP 配置，都建议保留一组最小回归：

```text
hidden_size = 128
num_attention_heads = 4
num_query_groups = 4 or 2
tensor_model_parallel_size = 1 or 2
sequence_length = 16
micro_batch_size = 2
```

回归项目：

| 项目 | 通过标准 |
|------|----------|
| forward | 输出shape正确 |
| backward | 梯度非NaN |
| checkpoint | save/load后logits一致 |
| TP=1 vs TP=2 | 聚合输出接近 |
| MHA vs GQA | shape和cache符合预期 |
| eval mode | dropout关闭后确定性 |

这个小回归无法证明质量，但能快速发现权重排列、head切分和checkpoint转换错误。

---

**文档版本**: v1.0
**最后更新**: 2026-05-10
**文档状态**: ✅ 已完成
